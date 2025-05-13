import torch
from sklearn.metrics import accuracy_score, f1_score

from load_eval import load_data
from load import process_vision_info

from utils import extract, iou
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
  "OpenGVLab/InternVL3-8B",
  device_map="auto",
  torch_dtype=torch.bfloat16,
  attn_implementation="eager",
).eval()
# model = PeftModel.from_pretrained(base_model, "/root/autodl-tmp/BiMi/").eval()
processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-8B")

# 加载数据
dataset, dataset_json = load_data(json_path, root_path)

# 生成单条样本描述
def generate(sample):
    messages = sample['messages']
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

# 评估函数（输入：数据集子集）
def evaluate_model(samples, ref_samples):
    predictions = []
    detections = []
    prediction_ref = []
    detections_ref = []
    
    for sample, sample_json in zip(samples, ref_samples):
        print("-------------------------sample------------------")
        print(sample)
        print("-------------------------------------------------")
        pred = generate(sample)
        print("-------------------------prediction--------------")
        print(pred)
        print("--------------------------------------------------")
        predictions.append(pred)
        image_width, image_height = sample["messages"][1]["content"][1]["image"].size
        classification, bbox = extract(pred, image_height, image_width)
        print(classification, bbox)
        predictions.append(classification)
        detections.append(bbox)
        prediction_ref.append(sample_json["label"])
        detections_ref.append(sample_json["bbox"])

    # 适用于分类任务
    accuracy = accuracy_score(predictions, prediction_ref)
    f1 = f1_score(predictions, prediction_ref, average="macro")  # 或 micro, weighted 等
    # 计算平均IoU
    ious = [iou(p, r) for p, r in zip(detections, detections_ref)]
    mean_iou = sum(ious) / len(ious)

    return accuracy, f1, mean_iou

# 从数据集中选择前N个样本作为评估集
eval_samples = dataset[2500:]  # 或使用你自己的 eval_dataset 切分方式
ref_samples = dataset_json[2500:]

# 执行评估
accuracy, f1, mean_iou = evaluate_model(eval_samples, ref_samples)

print("Evaluation Results:")
print(accuracy, f1, mean_iou)
