import json
import re

# 加载原始数据
data_path = '../data/washington_response_3.json'
with open(data_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 处理后的新数据列表
processed_data = []

count = 0
num = 0
for item in raw_data:
    if 'response' in item:
        # 用正则提取 **"..."** 之间的内容
        match = re.search(r'\*\*(.*?)\*\*', item['response'])
        if match:
            count += 1
            headline = match.group(1).strip()
            headline = re.sub(r'^[\"“”「」『』]+|[\"“”「」『』]+$', '', headline).strip()
            processed_data.append({
                "text": item["text"],
                "image_path": item["image_path"],
                "headline": headline
            })
            num += len(item["text"]) + len(headline)
        else:
            print(item['response'])

# 保存处理后的数据
with open('processed_data.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print("处理完成，输出保存为 processed_data.json", count, num)