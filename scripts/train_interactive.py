#!/usr/bin/env python
"""Interactive wrapper around ``train.py``.

Prompts the user for dataset location and a name for the resulting LoRA
adapters before executing the training procedure.
"""
import os
import subprocess
import sys
from pathlib import Path
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, load_from_disk
import torchaudio.transforms as T
from snac import SNAC
import torch
from transformers import TrainingArguments, Trainer

# Load model and tokenizer
MODEL_NAME = os.environ.get('MODEL_NAME', 'unsloth/orpheus-3b-0.1-ft')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
    cache_dir=CACHE_DIR,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# dataset loading with interactive prompt
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')
default_dataset = "MrDragonFox/Elise"
print("Select dataset source:")
print("1. Hugging Face link")
print("2. Local Whisper dataset")
choice = input("Choice [1]: ").strip() or "1"
dataset = None
if choice == "2":
    dataset_root = DATA_DIR
    if not os.path.isdir(dataset_root):
        print(f"Directory {dataset_root} does not exist.")
        sys.exit(1)
    dataset_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    if not dataset_dirs:
        print(f"No datasets found in {dataset_root}.")
        sys.exit(1)
    print("Select the dataset:")
    for idx, name in enumerate(dataset_dirs, 1):
        print(f"{idx}. {name}")
    sub_choice = input("Choice [1]: ").strip() or "1"
    if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(dataset_dirs):
        selected = dataset_dirs[int(sub_choice) - 1]
    else:
        selected = dataset_dirs[0]
    ds_path = os.path.join(dataset_root, selected)

    dataset = load_from_disk(ds_path)
else:
    dataset_link = input(f"Dataset to load [{default_dataset}]: ").strip() or default_dataset
    dataset = load_dataset(dataset_link, split="train", cache_dir=DATA_DIR)

default_lora_name = "run1"
lora_name = input(f"Name for this LoRA [{default_lora_name}]: ").strip() or default_lora_name
save_dir = os.path.join("lora_models", lora_name, "lora_model")
os.makedirs(save_dir, exist_ok=True)

# Tokenization functions
ds_sample_rate = dataset[0]["audio"]["sampling_rate"]
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir=CACHE_DIR)
snac_model = snac_model.to("cuda")

import locale
locale.getpreferredencoding = lambda: "UTF-8"

def tokenise_audio(waveform):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))
    return all_codes

def add_codes(example):
    codes_list = None
    try:
        answer_audio = example.get("audio")
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
    example["codes_list"] = codes_list
    return example

dataset = dataset.map(add_codes, remove_columns=["audio"])

# Special tokens
TOKENISER_LENGTH = 128256
start_of_text = 128000
end_of_text = 128009
start_of_speech = TOKENISER_LENGTH + 1
end_of_speech = TOKENISER_LENGTH + 2
start_of_human = TOKENISER_LENGTH + 3
end_of_human = TOKENISER_LENGTH + 4
start_of_ai = TOKENISER_LENGTH + 5
end_of_ai = TOKENISER_LENGTH + 6
pad_token = TOKENISER_LENGTH + 7

audio_tokens_start = TOKENISER_LENGTH + 10

dataset = dataset.filter(lambda x: x["codes_list"] is not None)
dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)

def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")
    result = vals[:7]
    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(vals[i:i + 7])
    example["codes_list"] = result
    return example

dataset = dataset.map(remove_duplicate_frames)

tok_info = """*** HERE you can modify the text prompt
If you are training a multi-speaker model (e.g., canopylabs/orpheus-3b-0.1-ft),
ensure that the dataset includes a 'source' field and format the input accordingly:
- Single-speaker: f"{example['text']}"
- Multi-speaker: f"{example['source']}: {example['text']}"
"""
print(tok_info)

def create_input_ids(example):
    text_prompt = f"{example['source']}: {example['text']}" if 'source' in example else example['text']
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    text_ids.append(end_of_text)
    example['text_tokens'] = text_ids
    input_ids = (
        [start_of_human]
        + example['text_tokens']
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example['codes_list']
        + [end_of_speech]
        + [end_of_ai]
    )
    example['input_ids'] = input_ids
    example['labels'] = input_ids
    example['attention_mask'] = [1] * len(input_ids)
    return example

dataset = dataset.map(create_input_ids, remove_columns=['text', 'codes_list'])
columns_to_keep = ['input_ids', 'labels', 'attention_mask']
columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
dataset = dataset.remove_columns(columns_to_remove)

# Training
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# Memory stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"LoRA adapters saved under {save_dir}")
