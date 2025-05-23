import torch
from trl import SFTTrainer
from model import load_model
from load_stage_2 import load_data
from utils import load_config, load_args
from utils import collate_fn

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from peft import PeftModel

json_path = "training_with_caption_path.json"  # Path to the JSONL file
root_path = "/root/autodl-tmp/"  # Path to the directory containing the images
model, processor = load_model()
dataset = load_data(json_path, root_path)
peft_config = load_config()
args = load_args()

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=lambda examples: collate_fn(examples, processor),
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

# Load Model base model
model_id = "google/gemma-3-4b-it" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
model = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
# peft_model = PeftModel.from_pretrained(model, args.output_dir)
# merged_model = peft_model.merge_and_unload()
# merged_model.save_pretrained("/root/autodl-tmp/merged_model", safe_serialization=True, max_shard_size="2GB")

# processor = AutoProcessor.from_pretrained(args.output_dir)
# processor.save_pretrained("/root/autodl-tmp/merged_model")


# free the memory again
# del model
# del trainer
# torch.cuda.empty_cache()
