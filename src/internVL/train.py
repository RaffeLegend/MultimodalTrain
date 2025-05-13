# import some libraries
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial

logger = get_logger()
seed_everything(42)

model_id = "Qwen/Qwen2.5-3B-Instruct"  # 模型ID
output_dir = "/root/autodl-tmp/biqwen/"  # 输出目录
data_path = "converted_dialog.jsonl"  # 数据集路径

# 获取模型和template，并加入可训练的LoRA模块
model, tokenizer = get_model_tokenizer(model_id, device_map="auto", torch_dtype="bfloat16")
logger.info(f'model_info: {model.model_info}')
template = get_template(model.model_meta.template, tokenizer, max_length=1024)
template.set_mode('train')

lora_rank = 8
lora_alpha = 32
target_modules = find_all_linears(model)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# Print model structure and trainable parameters.
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

# Download and load the dataset, split it into a training set and a validation set,
# and encode the text data into tokens.
train_dataset, val_dataset = load_dataset(data_path, split_dataset_ratio=0.1, num_proc=4,
        model_name="BIMIBI", model_author="RAFFE", seed=42, use_hf=False)

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=4)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=4)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')


# Print a sample
template.print_inputs(train_dataset[0])
# lora
lora_rank = 8
lora_alpha = 32

# training_args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=42,
)

# Get the trainer and start the training.
model.enable_input_require_grads()  # Compatible with gradient checkpointing
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
)
trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

# Visualize the training loss.
# You can also use the TensorBoard visualization interface during training by entering
# `tensorboard --logdir '{output_dir}/runs'` at the command line.
images_dir = os.path.join(output_dir, 'images')
logger.info(f'images_dir: {images_dir}')
plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # save images
