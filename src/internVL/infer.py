import os
import torch
from swift.llm import get_model_tokenizer, get_template, get_tokenizer
from swift.tuners import Swift
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template
from transformers import GenerationConfig

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 模型相关参数
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "/root/autodl-tmp/biqwen/"  # 训练输出路径
checkpoint_dir = output_dir  # 可以设置为包含最后checkpoint的路径

model_id_or_path = 'Qwen/Qwen2.5-3B-Instruct'  # model_id or model_path
system = 'You are a helpful assistant.'
infer_backend = 'pt'

# 生成参数
max_new_tokens = 512
temperature = 0
stream = True

# 加载模型和 tokenizer
engine = PtEngine(model_id_or_path, adapters=[checkpoint_dir])
template = get_template(engine.model.model_meta.template, engine.tokenizer, default_system=system)
# 这里对推理引擎的默认template进行修改，也可以在`engine.infer`时进行传入
engine.default_template = template

def infer(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')

# 示例推理
from data_parser import user_prompt, system_message
import json
if __name__ == "__main__":
    dataset = json.load(open("training_with_caption_path.json", "r", encoding="utf-8"))
    for sample in dataset:
        image_path = sample.get("captioned_path")
        # 构造 assistant 的 response 内容
        # bbox_values = ", ".join(str(x) for x in sample["bbox"]) if sample["bbox"] else ""
        # assistant_response = f"<think></think><answer>{{\"classification\": {json.dumps(sample['label'])}, \"region\": {{\"bbox\":[{bbox_values}]}}}}</answer>"
        
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
        response = infer(engine, InferRequest(messages=messages_data['messages'], images=messages_data['images']))
        print("result\n", response)