# MultimodalTrain
The repo for misinformation model training
# Gemma3 LoRA Finetune: Real News Detection

This project fine-tunes Google's Gemma3 model with LoRA to classify whether an image with a description is real news or not.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
bash train.sh
```

## Inference
```bash
python inference/generate.py