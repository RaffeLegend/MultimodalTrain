import torch
from evaluate import load as load_metric
from tqdm import tqdm

from model import load_model
from load import load_data
from load import process_vision_info

# 配置路径
json_path = "processed_data_2.json"  # 含参考描述字段 reference
root_path = "/root/autodl-tmp/"      # 图像文件根路径

# 加载模型和处理器
model, processor = load_model()
model.eval()  # 设为评估模式

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

# 评估函数（输入：数据集子集）
def evaluate_model(samples):
    predictions = []
    references = []
    
    for sample in tqdm(samples):
        pred = generate(sample)
        predictions.append(pred)
        references.append(sample["reference"])

    rouge = load_metric("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results

# 从数据集中选择前N个样本作为评估集
eval_samples = dataset[:50]  # 或使用你自己的 eval_dataset 切分方式

# 执行评估
results = evaluate_model(eval_samples)

print("Evaluation Results:")
print(results)
