# Textual Inversion with Diffusers

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/huggingface-diffusers-orange)](https://github.com/huggingface/diffusers)

This repository demonstrates **textual inversion** training using the [Hugging Face Diffusers](https://github.com/huggingface/diffusers) library with Stable Diffusion 1.5 and Stable Diffusion XL (SDXL).

---

## Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
---

## Features

- Train textual inversion embeddings on custom datasets  
- Uses SDXL pretrained models from Hugging Face  
- Supports training on single or multiple scenes  
- Flexible learning rate and scheduler options  

---

## Requirements

- Python 3.8 or higher  
- `accelerate` package for distributed training  
- `diffusers` repository (cloned locally)  
- CUDA-enabled GPU recommended for training  

---

## Installation

### 1. Clone This Repository and prepare the data

```bash
git clone https://github.com/BaristaBandits/Textual-Inversion.git
cd Textual-Inversion
mkdir data
unzip bajirao_scene1.zip -d data
unzip bajirao_scene2.zip -d data
cd Textual-Inversion/diffusers/examples/textual_inversion
pip install -r requirements.txt
```

### 2. Import the data and directory and the model, for example
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATA_DIR="Textual-Inversion/data/bajirao_scene1"
```

### 3. Run the following command to train the textual inversion model (alter hyperparameters if required)
```bash

accelerate launch --num_processes=1 --main_process_port=0  /Textual-Inversion/diffusers/examples/textual_inversion/textual_inversion_sdxl.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--learnable_property="" \
--placeholder_token="<bajirao_face>" \
--initializer_token="person" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--max_train_steps=5000 \
--learning_rate=1.0e-04 \
--scale_lr \
--lr_scheduler="constant" \
--lr_warmup_steps=500 \
--output_dir= <output_path> \
```
### 4. For Inferencing run the following command
```bash
python Inference.py \
  --checkpoint <checkpoint_path>
  --token "<bajirao_face>" \
  --output_dir <output_directory_path>
```


