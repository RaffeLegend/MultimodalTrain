from peft import LoraConfig
from trl import SFTConfig
import re
import json

from load import process_vision_info

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

args = SFTConfig(
    output_dir="/root/autodl-tmp/BiMi",     # directory to save and repository id
    num_train_epochs=1,                         # number of training epochs
    per_device_train_batch_size=1,              # batch size per device during training
    gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="epoch",                      # save checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    push_to_hub=True,                           # push model to hub
    report_to="tensorboard",                    # report metrics to tensorboard
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
args.remove_unused_columns = False # important for collator

def load_config():
    return peft_config

def load_args():
    return args

# Create a data collator to encode text and image pairs
def collate_fn(examples, processor):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

# result
def try_fix_json_string(s):
    # 尝试快速修复常见括号问题
    stack = []
    fixed = ""

    for char in s:
        if char in '{[':
            stack.append(char)
            fixed += char
        elif char in '}]':
            if stack:
                last = stack[-1]
                if (last == '{' and char == '}') or (last == '[' and char == ']'):
                    stack.pop()
                    fixed += char
                else:
                    # 错误匹配：跳过这个 char
                    continue
            else:
                # 没有开启括号却出现关闭，跳过
                continue
        else:
            fixed += char

    # 结束后补齐括号
    while stack:
        last = stack.pop()
        if last == '{':
            fixed += '}'
        elif last == '[':
            fixed += ']'

    return fixed

def extract(text, image_width, image_height):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not match:
        return None, None

    raw_json = match.group(1).strip()

    try:
        # 尝试第一次直接解析
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # 如果失败就尝试修复括号后再解析
        try:
            fixed_json = try_fix_json_string(raw_json)
            data = json.loads(fixed_json)
        except json.JSONDecodeError as e:
            print(f"[ERROR] 修复 JSON 失败: {e}")
            return None, None

    classification = data.get("classification", "")
    regions = data.get("region", [])

    if isinstance(regions, list) and len(regions) > 0:
        first_region = regions[0]
        if isinstance(first_region, dict):
            bbox = first_region.get("bbox", [])
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                norm_bbox = [
                    x1 / image_width,
                    y1 / image_height,
                    x2 / image_width,
                    y2 / image_height
                ]
                return classification, norm_bbox

    return classification, None

def iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0
