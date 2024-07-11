import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class QaImageOutput:
    q_input_ids: torch.long
    pixel_values: torch.long
    a_input_ids: torch.long


class LlavaDataset(Dataset):
    def __init__(self, data_path: str, json_name: str, image_folder: str) -> None:
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(data_path, json_name, image_folder)

    def build_dataset(self, data_path: str, json_name: str, image_folder: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_path)
        chat_file = data_dir.joinpath(json_name)
        image_dir = data_dir.joinpath(image_folder)

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        image_path = self.image_dir.joinpath(cur_data.get("image"))
        return human_input, chatbot_output, image_path


def build_qaimage(
    processor: AutoProcessor, q_text: str, a_text: str, image_path: Path, model_max_length: int
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_file = image_path
    raw_image = Image.open(image_file).convert("RGB")
    inputs = processor(prompt, raw_image, return_tensors="pt")

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=model_max_length,
    )["input_ids"].long()

    q_input_ids = inputs.get("input_ids")[:, :model_max_length]
    pixel_values = inputs.get("pixel_values")

    res = QaImageOutput(
        q_input_ids=q_input_ids,
        pixel_values=pixel_values,
        a_input_ids=a_input_ids,
    )
    return res


class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int, model_max_length: int) -> None:
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX
        self.model_max_length = model_max_length

    def convert_one_piece(
        self,
        q_input_ids: torch.long,
        a_input_ids: torch.long,
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id, dtype=torch.long).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index, dtype=torch.long),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id, dtype=torch.long).reshape(1, -1),
            ],
            axis=1,
        )
        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]

        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []
        image_paths = []

        for feature in features:
            qaimage_output = build_qaimage(
                self.processor, feature[0], feature[1], feature[2], self.model_max_length
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)
            image_paths.append(feature[2])

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, self.model_max_length - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,dtype=torch.long
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, self.model_max_length - max_input_len_list[index]),
                            self.ingnore_index, dtype=torch.long
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids.long() == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }


if __name__ == "__main__":
    data_path = "/home/models/weight/data/liuhaotianLLaVA-Pretrain"
    json_name = "blip_laion_cc_sbu_558k_cleaned_v2.json"
    image_folder = "images"

    llavadataset = LlavaDataset(data_path, json_name, image_folder)
    print(len(llavadataset))
    print(llavadataset[100])
