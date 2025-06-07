#!/usr/bin/env python
import argparse
import os
import json
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

def main():
    parser = argparse.ArgumentParser(description='Interactive Orpheus TTS inference')
    parser.add_argument('--model', default='unsloth/orpheus-3b-0.1-ft', help='Model name or path')
    parser.add_argument('--lora', default='lora_model', help='Path to trained LoRA adapters')
    args = parser.parse_args()
    model, tokenizer = load_model(args.model, args.lora)

    snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz', cache_dir=CACHE_DIR)
    snac_model = snac_model.to('cpu')

    print(
        "You can use these tokens for preconfigured expressions: "
        "<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
    )

    prompt_root = "prompt_list"
    prompt_files = []
    if os.path.isdir(prompt_root):
        prompt_files = [f for f in os.listdir(prompt_root) if f.endswith(".json")]

    prompt_list = None
    if prompt_files:
        print("Available prompt lists:")
        loaded_lists = []
        for idx, fname in enumerate(prompt_files, 1):
            path = os.path.join(prompt_root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
            if not isinstance(data, list):
                data = []
            loaded_lists.append(data)
            print(f"{idx}. {fname} ({len(data)} prompts)")
        choice = input("Select list by number or 0 to skip [0]: ").strip() or "0"
        if choice.isdigit() and 1 <= int(choice) <= len(loaded_lists):
            idx = int(choice) - 1
            prompt_list = loaded_lists[idx]
            print("Preview of selected list:")
            for p in prompt_list[:3]:
                print(f"- {p}")

    while True:
        if prompt_list is not None:
            if not prompt_list:
                break
            text = prompt_list.pop(0)
            print(f"Prompt: {text}")
        else:
            text = input('Enter text (or blank to quit): ').strip()
        if not text:
            break
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
            path = 'output.wav'
            audio_2d = sample.squeeze(0)
            torchaudio.save(path, audio_2d.detach().cpu(), 24000)
            print(f'Audio written to {path}')

if __name__ == '__main__':
    main()
