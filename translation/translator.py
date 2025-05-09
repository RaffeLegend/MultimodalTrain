import json
import requests

# 你的 Google Cloud API Key
API_KEY = "AIzaSyBE3BsCBKTDE14liAQzlvwDKyREoZEjHA4"

# 统计变量
total_chars = 0

def translate_text(text, target='zh-CN'):
    """使用 API Key 调用 Google Translate 翻译文本"""
    global total_chars
    total_chars += len(text)  # 累计字符数

    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        'q': text,
        'target': target,
        'format': 'text',
        'key': API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['data']['translations'][0]['translatedText']
    else:
        print(f"翻译失败，状态码：{response.status_code}，错误信息：{response.text}")
        return None

def translate_json_file(file_path):
    """读取 JSON 文件，翻译 text 和 headline 字段，保存回文件"""
    global total_chars

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    count = 0
    for item in data:
        if 'text' in item:
            try:
                translated = translate_text(item['text'])
                item['translation'] = translated
            except Exception as e:
                print(f"翻译 text 字段失败，错误信息：{e}")
                item['translation'] = None
        if 'headline' in item:
            try:
                translated = translate_text(item['headline'])
                item['headline_translation'] = translated
            except Exception as e:
                print(f"翻译 headline 字段失败，错误信息：{e}")
                item['headline_translation'] = None
        count += 1
        print(f"翻译进度：{count}/{len(data)}", end='\r')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print("\n🎯 翻译完成，已写回原 JSON 文件。")
    print(f"📊 总字符数：{total_chars}")
    print(f"💰 预计费用：${total_chars / 1_000_000 * 20:.2f}（按 $20/百万字符计）")

# 使用示例
translate_json_file("../tools/processed_data.json")