import torch
from trl import SFTTrainer
from .model import load_model
from .load import load_data
from .utils import load_config, load_args
from .utils import collate_fn

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from peft import PeftModel

model, processor = load_model()
dataset = load_data()
peft_config = load_config()
args = load_args()

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

# free the memory again
# del model
# del trainer
# torch.cuda.empty_cache()

# Load Model base model
model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
model = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoProcessor.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")