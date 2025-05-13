import json
import os

input_json_path = "../../data_tools/training_with_caption_path.json"  # 含参考描述字段 reference
output_json_path = "converted_dialog.jsonl"  # 输出的 JSON 文件路径
root_path = "/root/autodl-tmp/"  # 图像文件根路径

# system + user prompt
# system_message = "You are an expert in misinformation detection area. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>reasoning process here</think><answer>answer here</answer>"
system_prompt = """You are an expert in misinformation detection area. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags.Please analyze the image <image> with both Chinese and English subtitles. Complete the following three tasks:Classification Task: choose one of six labels: All Consistent: image and both subtitles are accurate.Image Manipulated: only image is fake; subtitles are real.Both Misaligned: image is real; both subtitles are misleading.Chinese Misaligned: image is real; Chinese subtitle is misleading.English Misaligned: image is real; English subtitle is misleading.All Inconsistent: image and both subtitles are misleading.Manipulation Detection: if the image is manipulated, return a bounding box as {\"bbox\":[x_min, y_min, x_max, y_max]}. Otherwise, return an empty list.<think>EXPLANATION</think><answer>{\"classification\": RESULT, \"region\": {\"bbox\":[X_MIN,Y_MIN,X_MAX,Y_MAX]}}</answer>"""
user_prompt = """Please analyze the given image <image>"""
# 构造 messages
with open(input_json_path, "r", encoding="utf-8") as f_in, open(output_json_path, "w", encoding="utf-8") as f_out:
    samples = json.load(f_in)

    for sample in samples:
        image_path = sample.get("captioned_path")

        # 构造 assistant 的 response 内容
        bbox_values = ", ".join(str(x) for x in sample["bbox"]) if sample["bbox"] else ""
        assistant_response = """
            "<think></think>"
            f"<answer>{{\"classification\": {json.dumps(sample['label'])}, "
            f"\"region\": {{\"bbox\":[{bbox_values}]}}}}</answer>"
        """

        # 组装最终结构
        messages_data = {
            "messages": [
                # {
                #     "role": "system",
                #     "content": system_message
                # },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ],
            "images": [image_path]  # 顶层放图像路径，List[str]
        }

        f_out.write(json.dumps(messages_data, ensure_ascii=False) + "\n")