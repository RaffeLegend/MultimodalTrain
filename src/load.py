from datasets import load_dataset
from PIL import Image
import json
import os

# System message for the assistant
system_message = "You are an expert product description writer for Amazon."

# User prompt that combines the user query and the schema
user_prompt = """English: <English>{english}</English>, Chinese: <Chinese>{chinese}</Chinese>"""

# Convert dataset to OAI messages
def format_data(sample, root_path):
    with Image.open(os.path.join(root_path, sample["image_path"])) as image:
        image = image.convert("RGB")
    return {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant that understands English, Chinese and images."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe the <start_of_image> user's in English and Chinese by a sentence."
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
                            "text": user_prompt.format(english=sample["text"],chinese=sample["translation"])
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

    return dataset
