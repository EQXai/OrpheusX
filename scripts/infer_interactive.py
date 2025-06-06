#!/usr/bin/env python
"""Interactive wrapper around ``infer.py``.

This script prompts the user for a LoRA model to load and a list of
prompts, then generates audio for each prompt, saving the results under
the ``audio_output`` directory without overwriting existing files.
"""
import os
import torch
import torchaudio
from unsloth import FastLanguageModel
from snac import SNAC
from peft import PeftModel

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_model(model_name, lora_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        cache_dir=CACHE_DIR,
    )
    if lora_path and os.path.isdir(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def get_output_path(base_name: str = "audio", ext: str = ".wav") -> str:
    """Return a unique file path under ``audio_output`` without overwriting."""
    os.makedirs("audio_output", exist_ok=True)
    idx = 1
    while True:
        suffix = f"_{idx}" if idx > 1 else ""
        path = os.path.join("audio_output", f"{base_name}{suffix}{ext}")
        if not os.path.exists(path):
            return path
        idx += 1


def main():
    model_name = "unsloth/orpheus-3b-0.1-ft"
    lora_root = "lora_models"
    lora_dirs = []
    if os.path.isdir(lora_root):
        lora_dirs = [d for d in os.listdir(lora_root) if os.path.isdir(os.path.join(lora_root, d))]
    lora_choice = None
    if lora_dirs:
        print("Available LoRA models:")
        for idx, name in enumerate(lora_dirs, 1):
            print(f"{idx}. {name}")
        choice = input("Select LoRA by number [1]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(lora_dirs):
            lora_choice = lora_dirs[int(choice) - 1]
        else:
            lora_choice = lora_dirs[0]
    else:
        print("No LoRA models found. Running base model only.")
    lora_path = os.path.join(lora_root, lora_choice, "lora_model") if lora_choice else None
    model, tokenizer = load_model(model_name, lora_path)

    snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz', cache_dir=CACHE_DIR)
    snac_model = snac_model.to('cpu')

    num_gens = input("How many prompts to generate [1]: ").strip()
    num_gens = int(num_gens or "1")
    prompts = []
    for i in range(num_gens):
        p = input(f"Prompt {i+1}: ").strip()
        prompts.append(p)

    for text in prompts:
        input_ids = tokenizer(text, return_tensors='pt').input_ids
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        modified_input = torch.cat([start_token, input_ids, end_tokens], dim=1)
        padding = 0
        attention_mask = torch.ones_like(modified_input)
        input_ids_cuda = modified_input.to('cuda')
        attn_cuda = attention_mask.to('cuda')
        generated = model.generate(
            input_ids=input_ids_cuda,
            attention_mask=attn_cuda,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
            use_cache=True,
        )
        token_to_find = 128257
        token_to_remove = 128258
        token_indices = (generated == token_to_find).nonzero(as_tuple=True)
        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated
        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)
        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)
        def redistribute_codes(code_list):
            layer_1 = []
            layer_2 = []
            layer_3 = []
            for i in range((len(code_list)+1)//7):
                layer_1.append(code_list[7*i])
                layer_2.append(code_list[7*i+1]-4096)
                layer_3.append(code_list[7*i+2]-(2*4096))
                layer_3.append(code_list[7*i+3]-(3*4096))
                layer_2.append(code_list[7*i+4]-(4*4096))
                layer_3.append(code_list[7*i+5]-(5*4096))
                layer_3.append(code_list[7*i+6]-(6*4096))
            codes = [torch.tensor(layer_1).unsqueeze(0),
                     torch.tensor(layer_2).unsqueeze(0),
                     torch.tensor(layer_3).unsqueeze(0)]
            audio_hat = snac_model.decode(codes)
            return audio_hat
        samples = [redistribute_codes(c) for c in code_lists]
        for sample in samples:
            path = get_output_path()
            audio_2d = sample.squeeze(0)
            torchaudio.save(path, audio_2d.detach().cpu(), 24000)
            print(f'Audio written to {path}')

if __name__ == '__main__':
    main()
