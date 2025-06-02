import torch
from sklearn.metrics import accuracy_score, f1_score

from load_eval import user_prompt, system_message
from load import process_vision_info
from PIL import Image

from utils import extract, iou
import json
import os
# 配置路径
json_path = "training_with_caption_path.json"  # 含参考描述字段 reference
root_path = "/root/autodl-tmp/"      # 图像文件根路径

import torch
from intern_utils import split_model, load_image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


definition = "Assign one of the following six categories based on their mutual alignment: All Consistent: The image aligns well with both Chinese and English captions. No signs of manipulation or misalignment. Image Manipulated: The image is manipulated. Both captions truthfully describe the manipulated image. Both Misaligned: Both Chinese and English captions are manipulated. Neither caption correctly describes the image. Chinese Misaligned: Only the Chinese caption is manipulated. The English caption aligns correctly with the image. English Misaligned: Only the English caption is manipulated. The Chinese caption aligns correctly with the image. All Inconsistent: The image, Chinese caption, and English caption are all manipulated and mutually inconsistent. "

system_prompt = system_message + definition + user_prompt

user_prompt = "please analyze the <image> with label "

path = 'OpenGVLab/InternVL3-14B'
device_map = split_model('InternVL3-14B', path)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False)

# 生成单条样本描述
def generate(sample):
    # single-image single-round conversation (单图单轮对话)
    pixel_values = load_image(os.path.join(root_path, sample['captioned_path']), max_num=1).to(torch.bfloat16).cuda()
    label = sample['label']
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    question = system_prompt + user_prompt + label
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')
    
    return response

# 评估函数（输入：数据集子集）
def evaluate_model(ref_samples):
    predictions = []
    detections = []
    prediction_ref = []
    detections_ref = []
    
    for sample in ref_samples[:200]:
        print("-------------------------sample------------------")
        print(sample)
        print("-------------------------------------------------")
        pred = generate(sample)
        print("-------------------------prediction--------------")
        print(pred)
        print("--------------------------------------------------")
        pixel_values = Image.open(os.path.join(root_path, sample['captioned_path']))
        image_width, image_height = pixel_values.size
        classification, bbox = extract(pred, image_height, image_width)
        print(classification, bbox)
        if classification is not None:
            prediction_ref.append(sample["label"])
            predictions.append(classification)
        if bbox is not None:
            detections.append(bbox)
            detections_ref.append(sample["bbox"])

    # 适用于分类任务
    print(predictions, prediction_ref)
    accuracy = accuracy_score(predictions, prediction_ref)
    f1 = f1_score(predictions, prediction_ref, average="macro")  # 或 micro, weighted 等
    # 计算平均IoU
    ious = [iou(p, r) for p, r in zip(detections, detections_ref)]
    if len(ious) == 0:
        mean_iou = 0
    else:
        mean_iou = sum(ious) / len(ious)

    return accuracy, f1, mean_iou

# 执行评估
samples = json.load(open(json_path, 'r', encoding='utf-8'))
accuracy, f1, mean_iou = evaluate_model(samples[:1000])

print("Evaluation Results:")
print(accuracy, f1, mean_iou)
