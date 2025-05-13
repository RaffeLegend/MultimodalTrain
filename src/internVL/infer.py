import os
import torch
from swift.llm import get_model_tokenizer, get_template, get_tokenizer
from swift.tuners import Swift
from transformers import GenerationConfig

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 模型相关参数
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "/root/autodl-tmp/biqwen/"  # 训练输出路径
checkpoint_dir = output_dir  # 可以设置为包含最后checkpoint的路径

# 加载模型和 tokenizer
model = Swift.from_pretrained(base_model_id, checkpoint_dir, torch_dtype="bfloat16", device_map="auto")
tokenizer = get_tokenizer(base_model_id)
template = get_template(model.model_meta.template, tokenizer)

# 加载 LoRA 微调权重
model = Swift.from_pretrained(model, checkpoint_dir)

# 设置模板为推理模式
template.set_mode('eval')

# 推理函数
def infer(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    # 构造 prompt 和 tokenized 输入
    input_data = {'input': prompt}
    input_dict = template.encode(input_data, add_special_tokens=True)
    input_ids = torch.tensor([input_dict['input_ids']], device=model.device)

    # 设置生成配置
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 生成输出
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, generation_config=generation_config)[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text

# 示例推理
from data_parser import user_prompt, system_message
import json
if __name__ == "__main__":
    dataset = json.load(open("training_with_caption_path.json", "r", encoding="utf-8"))
    for sample in dataset:
        image_path = sample.get("captioned_path")
        # 构造 assistant 的 response 内容
        bbox_values = ", ".join(str(x) for x in sample["bbox"]) if sample["bbox"] else ""
        assistant_response = f"<think></think><answer>{{\"classification\": {json.dumps(sample['label'])}, \"region\": {{\"bbox\":[{bbox_values}]}}}}</answer>"
        
        # 组装最终结构
        messages_data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
            "images": [image_path]  # 顶层放图像路径，List[str]
        }
        response = infer(messages_data)
        print("result\n", response)