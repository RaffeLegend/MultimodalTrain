from PIL import Image
import json
import os

# System message for the assistant
system_message = "You are an expert in misinformation detection area."

# User prompt that combines the user query and the schema
user_prompt = """Please analyze the given image <start_of_image> containing both Chinese and English subtitles and complete the following three tasks:Classification Task: classify the alignment between the image and the subtitles into one of the following six categories:"All Consistent","Image Manipulated","Both Misaligned","Chinese Misaligned", "English Misaligned","All Inconsistent".Manipulation Detection: if the image has been manipulated, return one or more bounding boxes for the manipulated regions in the format: {"bbox":[x\_min, y\_min, x\_max, y\_max]}. If no manipulation is found, return an empty list.Decision Explanation: briefly explain your thinking before the classification and any detected regions.Return your output using the following format, wrapped in tags:<think>EXPLANATION</think><answer>{"classification": RESULT, "region": {"bbox":[X_MIN,Y_MIN,X_MAX,Y_MAX]}}</answer>"""

response_format = """<think>your think step</think><answer>{{"classification": {RESULT}, "region": {{"bbox":[{BBOX}]}}}}</answer>"""

def format(sample, response=response_format):
    if sample["bbox"] is not None:
        return response_format.format(RESULT=json.dumps(sample["label"]),BBOX=", ".join(str(x) for x in sample["bbox"]))
    else:
        return response_format.format(RESULT=json.dumps(sample["label"]),BBOX=None)

# Convert dataset to OAI messages
def format_data(sample, root_path):
    print(sample.keys())
    print(sample["bbox"])
    with Image.open(os.path.join(root_path, sample["captioned_path"])) as image:
        image = image.convert("RGB")
    return {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_message
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image",
                            "image": image
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text", 
                            "text": format(sample)
                        }
                    ]
                },
                ]
            }

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

def load_data(json_path,root_path):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    # Convert dataset to OAI messages
    # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
    dataset = [format_data(sample, root_path) for sample in dataset]
    dataset = dataset[:100]  # Limit to the first 100 samples

    return dataset
