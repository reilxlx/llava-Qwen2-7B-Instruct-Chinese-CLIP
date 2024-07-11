import copy
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from train_llava.custom_trainer import WebTrainer
from train_llava.data import LlavaDataset, TrainLLavaModelCollator
from train_llava.data_websend import DatasetReceiveByWeb, TrainLlavaModelCollatorByWeb
from train_llava.util import print_trainable_parameters, print_trainable_parameters_name

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="test_model/model001")
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora:使用lora训练,
            2. none:全量参数训练;
            3. freeze_vision:只冻结vision_tower进行训练
            """
        },
    )
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    low_lr: Optional[float] =  field(
        default=2e-5,
        metadata={
            "help": "使用lora进行训练模型的lr"
        }
    )
    high_lr: Optional[float] =  field(
        default=1e-3,metadata={
            "help": "multi_modal_projector层训练的lr"
        }
    )


@dataclass
class DataArguments:
    build_data_from_web: bool = field(
        default=False, metadata={"help": "是否使用web获得数据"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    web_host_ip: str = field(default="0.0.0.0", metadata={"help": "web端的数据ip"})
    model_max_length: int = field(default=4096)
    json_name: str = field(
        default=None, metadata={"help": "Path to the training data json path."}
    )
    image_folder: str = field(
        default=None, metadata={"help": "Path to the training data images path."}
    )
    gpu_nums: int = field(
        default=8, metadata={"help": "Number of GPUs to use for training."}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    
    warmup_steps: int = field(
        default=190,
        metadata={"help": "Number of warmup steps for the learning rate scheduler."}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Number of warmup ratio for the learning rate scheduler."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for the optimizer."}
    )

def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
    )
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        LORA_R = modelargs.lora_r
        LORA_ALPHA = modelargs.lora_alpha
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)
        high_lr = modelargs.high_lr
        low_lr = modelargs.low_lr
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in config.modules_to_save)
                ],
                "lr": high_lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in config.modules_to_save)
                ],
                "lr": low_lr,
            },
        ]

    elif modelargs.train_type == "none":
        logging.warning("使用全量参数进行训练")
        pass
        
    elif modelargs.train_type == "freeze_vision":
        logging.warning("冻结vision_tower网络层，剩下的网络权重进行训练")

        for param in model.vision_tower.parameters():
            param.requires_grad = False

    print_trainable_parameters(model)
    return model, processor, optimizer_grouped_parameters


def load_dataset_collator(processor, data_args: DataArguments):
    if data_args.build_data_from_web:
        llava_dataset = DatasetReceiveByWeb(
            data_args.web_host_ip,
        )
        logging.warning("从网络层进行数据初始化")

        if len(llava_dataset) <= 0:
            raise ValueError("数据出现问题，无法进行web数据初始化")
        data_collator = TrainLlavaModelCollatorByWeb(processor, -100, model_max_length=data_args.model_max_length)
    else:

        llava_dataset = LlavaDataset(
            data_path = data_args.data_path,
            json_name = data_args.json_name,
            image_folder = data_args.image_folder
        )
        data_collator = TrainLLavaModelCollator(processor, -100, model_max_length=data_args.model_max_length)

    return llava_dataset, data_collator

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.optimizer_grouped_parameters = kwargs.pop("optimizer_grouped_parameters", None)
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps):
        if self.optimizer_grouped_parameters is not None:
            if self.args.warmup_ratio is not None:
                warmup_steps = int(self.args.warmup_ratio * num_training_steps)
            else:
                warmup_steps = self.args.warmup_steps

            self.optimizer = torch.optim.AdamW(self.optimizer_grouped_parameters, lr=self.args.learning_rate)
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        else:
            super().create_optimizer_and_scheduler(num_training_steps)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        high_lr = self.optimizer.param_groups[0]['lr']
        low_lr = self.optimizer.param_groups[1]['lr']
        self.log({"high_lr": high_lr, "low_lr": low_lr})
        return loss

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor, optimizer_grouped_parameters = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    num_training_steps = len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size *
        training_args.gradient_accumulation_steps * data_args.gpu_nums)

    if data_args.build_data_from_web:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            optimizer_grouped_parameters=optimizer_grouped_parameters,
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            optimizer_grouped_parameters=optimizer_grouped_parameters,
        )

    trainer.train()
    trainer.save_state()
    if model_args.train_type == "use_lora":
        model.save_pretrained(training_args.output_dir) 
    else:
        trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
