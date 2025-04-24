from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.utils import preprocess_image

model = AutoModelForCausalLM.from_pretrained("output", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

image_description = "A man holding a torch during a protest."
prompt = f"Image description: {image_description}\nIs this a real news image? Answer yes or no."

output = pipe(prompt, max_new_tokens=10)[0]['generated_text']
print(output)