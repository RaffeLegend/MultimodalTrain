import os
import torch
from swift.tuners import Swift
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template, get_model_tokenizer
from transformers import GenerationConfig
import json
# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 模型相关参数
model_id = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "/root/autodl-tmp/biqwen/checkpoint-200"  # 训练输出路径
checkpoint_dir = output_dir  # 可以设置为包含最后checkpoint的路径

infer_backend = 'pt'

# 生成参数
max_new_tokens = 512
temperature = 0
stream = True

# 加载模型和 tokenizer
model, tokenizer = get_model_tokenizer(model_id)
model = Swift.from_pretrained(model, checkpoint_dir)

template_type = model.model_meta.template
template = get_template(template_type, tokenizer)
engine = PtEngine.from_model_template(model, template, max_batch_size=2)

engine.default_template = template

def infer(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')

system_message = "You are an expert in misinformation detection area. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>reasoning process here</think><answer>answer here</answer>"

user_prompt = """
    "Please analyze the image <image> with both Chinese and English subtitles. Complete the following three tasks:\n\n"
    "1.Classification Task: choose one of six labels:\n"
    "- All Consistent: image and both subtitles are accurate.\n"
    "- Image Manipulated: only image is fake; subtitles are real.\n"
    "- Both Misaligned: image is real; both subtitles are misleading.\n"
    "- Chinese Misaligned: image is real; Chinese subtitle is misleading.\n"
    "- English Misaligned: image is real; English subtitle is misleading.\n"
    "- All Inconsistent: image and both subtitles are misleading.\n\n"
    "2.Manipulation Detection: if the image is manipulated, return a bounding box as {\"bbox\":[x_min, y_min, x_max, y_max]}. Otherwise, return an empty list.\n\n"
    "3.Decision Explanation: before the result, explain your reasoning inside <think>...</think>. Return your output as:\n"
    "<think>EXPLANATION</think><answer>{\"classification\": RESULT, \"region\": {\"bbox\":[X_MIN,Y_MIN,X_MAX,Y_MAX]}}</answer>"
"""

# 示例推理
if __name__ == "__main__":
    dataset = json.load(open("../../training_with_caption_path.json", "r", encoding="utf-8"))
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
