## 模型 llava-Qwen2-7B-Instruct-Chinese-CLIP 增强中文文字识别能力和表情包内涵识别能力，达到gpt4o、claude-3.5-sonnet的水平！

### 完整模型下载</br>
https://huggingface.co/REILX/llava-Qwen2-7B-Instruct-Chinese-CLIP

### 模型结构</br>
llava-Qwen2-7B-Instruct-Chinese-CLIP = Qwen/Qwen2-7B-Instruct + multi_modal_projector + OFA-Sys/chinese-clip-vit-large-patch14-336px</br>

### 微调模块
- vision_tower和language_model的q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj部分进行lora训练</br>
- mmp层全量训练</br>

### 微调</br>
参数：lora_r=32,lora_alpha=64,num_train_epochs=5,per_device_train_batch_size=1,gradient_accumulation_steps=8,high_lr=1e-3,low_lr=2e-5,model_max_length=2048.</br>
设备：8*A800</br>
训练时长：5小时12分钟

### 数据集</br>
使用gemini-1.5-pro, gemini-1.5-flash, yi-vision, gpt4o,claude-3.5-sonnet模型描述emo-visual-data和ChineseBQB数据集。</br>
文本描述信息通过[text-description-of-the-meme](https://huggingface.co/datasets/REILX/text-description-of-the-meme) 下载</br>
图像可通过[emo-visual-data](https://github.com/LLM-Red-Team/emo-visual-data), [ChineseBQB](https://github.com/zhaoolee/ChineseBQB)下载</br>
图片数据总量1.8G，约10835张中文表情包图片。文字总量42Mb，约24332个图像文本对描述信息。

### 效果展示</br>
以下测试结果显示模型能识别图像中的文字信息，且能正确识别表情包想要表达的内涵。对比[REILX/llava-1.5-7b-hf-meme-lora](https://huggingface.co/REILX/llava-1.5-7b-hf-meme-lora)模型中也测试了原始llava-1.5-7b-hf模型的输出，模型无法正确识别图像中的文本信息。</br>
以下6张图片为llava-Qwen2-7B-Instruct-Chinese-CLIP模型的识别效果：</br>
<img src="./images/llava-qwen-2-7b-OFA-Syschinese-clip-memechinesebqb_merged_0708_fp16/llava-qwen2-7b-OFA-Syschinese-clip-fp16-01.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-OFA-Syschinese-clip-memechinesebqb_merged_0708_fp16/llava-qwen2-7b-OFA-Syschinese-clip-fp16-02.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-OFA-Syschinese-clip-memechinesebqb_merged_0708_fp16/llava-qwen2-7b-OFA-Syschinese-clip-fp16-03.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-OFA-Syschinese-clip-memechinesebqb_merged_0708_fp16/llava-qwen2-7b-OFA-Syschinese-clip-fp16-04.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-OFA-Syschinese-clip-memechinesebqb_merged_0708_fp16/llava-qwen2-7b-OFA-Syschinese-clip-fp16-05.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-OFA-Syschinese-clip-memechinesebqb_merged_0708_fp16/llava-qwen2-7b-OFA-Syschinese-clip-fp16-06.PNG" width="600" height="200">
</br></br></br>
**以下6张图片为Qwen/Qwen2-7B-Instruct + multi_modal_projector + openai/clip-vit-large-patch14-336模型训练后的识别效果**</br>
<img src="./images/llava-qwen-2-7b-openaiclip-memechinesebqb_merged_0709_fp16/llava-qwen2-7b-openaiclipL14-336-fp16-01.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-openaiclip-memechinesebqb_merged_0709_fp16/llava-qwen2-7b-openaiclipL14-336-fp16-02.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-openaiclip-memechinesebqb_merged_0709_fp16/llava-qwen2-7b-openaiclipL14-336-fp16-03.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-openaiclip-memechinesebqb_merged_0709_fp16/llava-qwen2-7b-openaiclipL14-336-fp16-04.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-openaiclip-memechinesebqb_merged_0709_fp16/llava-qwen2-7b-openaiclipL14-336-fp16-05.PNG" width="600" height="200">
<img src="./images/llava-qwen-2-7b-openaiclip-memechinesebqb_merged_0709_fp16/llava-qwen2-7b-openaiclipL14-336-fp16-06.PNG" width="600" height="200">
</br></br></br>
**以下3张图为gpt4o的识别效果**</br>
<img src="./images/gpt4o-01.JPG" width="600" height="400">
<img src="./images/gpt4o-02.JPG" width="600" height="400">
<img src="./images/gpt4o-03.JPG" width="600" height="400">

## 项目代码：</br>
基础模型生成代码和训练代码可参考项目：https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava</br>
本项目基于原代码进行优化，主要改进包括：
1. **新增warmup功能，并针对不同模块采用差异化学习率，加速了multi_modal_projector的收敛速度。**  这一改进有效提升了模型训练效率。
2. **针对训练过程中出现的报错情况，对部分代码进行了优化，增强了代码稳定性。**  
3. **用户只需修改run.sh中的参数，即可直接执行训练。**  简化了操作流程，提升了用户体验。

合并模型代码，合并模型之后将add_tokens.json,merge.txt,preprocessor_config.json,specital_token_map.json,tokenizer.json,vocab.json文件复制到"/保存的完整模型路径"。
```python
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
```

推理代码
```python
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
```

### TODO
- llava项目基于二阶段训练，先进行Pretraining，再进行Fine-tuning。Pretraining训练的部分：multi_modal_projector，
Fine-tuning训练的部分：language_model + multi_modal_projector。本项目后续基于上述思路进行优化，比较两者差异。
- 对llava-Qwen2-7B-Instruct-Chinese-CLIP模型进行更全面测评。

### 致谢

本项目受益于 [train_llava](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava)、[ChineseBQB](https://github.com/zhaoolee/ChineseBQB)、[emo-visual-data](https://github.com/LLM-Red-Team/emo-visual-data)，感谢以上诸位作者的付出。
