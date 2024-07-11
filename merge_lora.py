import torch
from peft import PeftModel, LoraConfig
from transformers import LlavaForConditionalGeneration
model_name = "/替换为你的基础模型路径"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["multi_modal_projector"],
) 
model = LlavaForConditionalGeneration.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, "/替换为你的lora模型路径", config=lora_config, adapter_name='lora')

model.cpu()
model.eval()
base_model = model.get_base_model()
base_model.eval()
model.merge_and_unload()

base_model.save_pretrained("/保存的完整模型路径")