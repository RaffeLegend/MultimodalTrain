import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from model import load_model
from load_eval import load_data
from load import process_vision_info

import re
import json

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
# 配置路径
json_path = "training_with_caption_path.json"  # 含参考描述字段 reference
root_path = "/root/autodl-tmp/"      # 图像文件根路径

# 加载模型和处理器
#model, processor = load_model()
#model.eval()  # 设为评估模式

import torch
from peft import PeftModel

# Load Model with PEFT adapter
model = AutoModelForImageTextToText.from_pretrained(
  "google/gemma-3-4b-it",
  device_map="auto",
  torch_dtype=torch.bfloat16,
  attn_implementation="eager",
).eval()
# model = PeftModel.from_pretrained(base_model, "/root/autodl-tmp/BiMi/").eval()
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/BiMi/")


# 加载数据
dataset = load_data(json_path, root_path)

# 生成单条样本描述
def generate(sample):
    messages = sample['messages']
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

# result 
def extract(text):
    # 正则表达式提取 classification 和 bbox（包括 null 情况）
    pattern = r'<answer>\{\{"classification":\s*"(.+?)",\s*"region":\s*\{"bbox":\s*(null|\[[^\]]*\])\}\}\}</answer>'

    matches = re.findall(pattern, text)

    results = []
    for classification, bbox_str in matches:
        if bbox_str == "null":
            bbox = None
        else:
            # 安全解析 bbox 字符串为 float 列表
            try:
                bbox = [float(x.strip()) for x in bbox_str.strip('[]').split(',')]
            except ValueError:
                bbox = None  # fallback if malformed

        return classification, bbox

def iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

# 评估函数（输入：数据集子集）
def evaluate_model(samples):
    predictions = []
    references = []
    
    for sample in tqdm(samples):
        print("-------------------------sample------------------")
        print(sample)
        print("-------------------------------------------------")
        pred = generate(sample)
        print("-------------------------prediction--------------")
        print(pred)
        print("--------------------------------------------------")
        predictions.append(pred)
        classification, bbox = extract(text)
        predictions.append(classification)
        detections.append(bbox)
        prediction_ref.append(sample["label"])
        detections_ref.append(sample["bbox"])

    # 适用于分类任务
    accuracy = accuracy_score(predictions, prediction_ref)
    f1 = f1_score(predictions, prediction_ref, average="macro")  # 或 micro, weighted 等
    # 计算平均IoU
    ious = [iou(p, r) for p, r in zip(detections, detections_ref)]
    mean_iou = sum(ious) / len(ious)

    return accuracy, f1, mean_iou

# 从数据集中选择前N个样本作为评估集
eval_samples = dataset[2500:]  # 或使用你自己的 eval_dataset 切分方式

# 执行评估
accuracy, f1, mean_iou = evaluate_model(eval_samples)

print("Evaluation Results:")
print(accuracy, f1, mean_iou)
