from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

raw_model_name_or_path = "/保存的完整模型路径"
model = LlavaForConditionalGeneration.from_pretrained(raw_model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(raw_model_name_or_path)
model.eval()

def build_model_input(model, processor):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<image>\n 使用中文描述图片中的信息"}
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open("01.PNG")
    inputs = processor(text=prompt, images=image, return_tensors="pt", return_token_type_ids=False)
    
    for tk in inputs.keys():
        inputs[tk] = inputs[tk].to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=200)
    
    generate_ids = [
        oid[len(iids):] for oid, iids in zip(generate_ids, inputs.input_ids)
    ]
    gen_text = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    return gen_text
build_model_input(model, processor)