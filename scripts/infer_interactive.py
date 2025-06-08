#!/usr/bin/env python
"""Interactive wrapper around ``infer.py``.

This script prompts the user for one or more LoRA models and a list of
prompts. Audio for each prompt is generated sequentially and written to
``audio_output/<lora_name>/`` using incrementing file names so existing
files are never overwritten.
"""
import os
import json
import argparse
import torch
import torchaudio
import re
from unsloth import FastLanguageModel
from snac import SNAC
from peft import PeftModel

# Ensure repo root is on the path when executed from ``scripts``
import sys
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from audio_utils import concat_with_fade

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

def get_output_path(lora_name: str, ext: str = ".wav") -> str:
    """Return a unique file path for a LoRA under ``audio_output``."""
    base_dir = os.path.join("audio_output", lora_name)
    os.makedirs(base_dir, exist_ok=True)

    idx = 1
    while True:
        path = os.path.join(base_dir, f"{lora_name}_{idx}{ext}")
        if not os.path.exists(path):
            return path
        idx += 1


def split_prompt_by_tokens(text: str, tokenizer, chunk_size: int = 50) -> list[torch.Tensor]:
    """Split text into token chunks without breaking words."""
    words = text.split()
    segments: list[str] = []
    current: list[str] = []
    token_len = 0
    for w in words:
        n_tokens = len(tokenizer(w, add_special_tokens=False).input_ids)
        if token_len + n_tokens > chunk_size and current:
            segments.append(" ".join(current))
            current = [w]
            token_len = n_tokens
        else:
            current.append(w)
            token_len += n_tokens
    if current:
        segments.append(" ".join(current))
    return [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments]


def split_prompt_by_sentences(
    text: str, tokenizer, chunk_size: int = 50
) -> list[torch.Tensor]:
    """Split text into sentence groups up to ``chunk_size`` tokens."""
    raw_parts = [s.strip() for s in re.split(r"(?<=[.!?,])\s+", text.strip()) if s.strip()]
    sentences: list[str] = []
    for part in raw_parts:
        if sentences:
            prev = sentences[-1]
            if prev.endswith(",") and (part.endswith(",") or len(part.split()) < 3):
                sentences[-1] = prev + " " + part
                continue
        sentences.append(part)
    segments: list[str] = []
    current: list[str] = []
    for sent in sentences:
        candidate = " ".join(current + [sent])
        token_len = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if token_len > chunk_size and current:
            segments.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    if current:
        segments.append(" ".join(current))
    return [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments]


def generate_audio_segment(
    tokens: torch.Tensor,
    model,
    snac_model,
    max_new_tokens: int = 1200,
) -> torch.Tensor:
    """Generate audio for a single token chunk and return as 1D tensor."""
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
    modified_input = torch.cat([start_token, tokens.unsqueeze(0), end_tokens], dim=1)
    attention_mask = torch.ones_like(modified_input)
    input_ids_cuda = modified_input.to("cuda")
    attn_cuda = attention_mask.to("cuda")
    generated = model.generate(
        input_ids=input_ids_cuda,
        attention_mask=attn_cuda,
        max_new_tokens=max_new_tokens,
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
        cropped_tensor = generated[:, last_occurrence_idx + 1 :]
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
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
        ]
        audio_hat = snac_model.decode(codes)
        return audio_hat

    samples = [redistribute_codes(c) for c in code_lists]
    return samples[0].squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Interactive inference")
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Segment prompts",
    )
    parser.add_argument(
        "--segment-by",
        choices=["tokens", "sentence"],
        default="tokens",
        help="Segmentation method when using --segment",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1200,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    model_name = "unsloth/orpheus-3b-0.1-ft"
    lora_root = "lora_models"
    lora_dirs = []
    if os.path.isdir(lora_root):
        lora_dirs = [d for d in os.listdir(lora_root) if os.path.isdir(os.path.join(lora_root, d))]

    if lora_dirs:
        print("Available LoRA models:")
        for idx, name in enumerate(lora_dirs, 1):
            print(f"{idx}. {name}")
        choice = input("Select LoRA by number(s) [1]: ").strip() or "1"
        indices = []
        for part in choice.split(','):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(lora_dirs):
                    indices.append(idx)
        if not indices:
            indices = [1]
        selected_loras = [lora_dirs[i - 1] for i in indices]
    else:
        print("No LoRA models found. Running base model only.")
        selected_loras = [None]

    snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz', cache_dir=CACHE_DIR)
    snac_model = snac_model.to('cpu')

    script_dir = os.path.dirname(__file__)
    prompt_root = os.path.join(script_dir, "..", "prompt_list")

    prompt_files = []
    if os.path.isdir(prompt_root):
        prompt_files = [f for f in os.listdir(prompt_root) if f.endswith(".json")]

    prompts = []
    if prompt_files:
        mode = (
            input(
                "Enter '1' to type prompts manually or '2' to load a prompt list [1]: "
            ).strip()
            or "1"
        )
    else:
        mode = "1"

    if mode == "2" and prompt_files:
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
        choice = input("Select list by number [1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(loaded_lists):
            idx = int(choice) - 1
        else:
            idx = 0
        prompts = loaded_lists[idx]
        print("Preview of selected list:")
        for p in prompts[:3]:
            print(f"- {p}")
    else:
        num_gens = input("How many prompts to generate [1]: ").strip()
        num_gens = int(num_gens or "1")
        print(
            "You can use these tokens for preconfigured expressions: "
            "<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
        )
        for i in range(num_gens):
            p = input(f"Prompt {i+1}: ").strip()
            prompts.append(p)

    if args.segment:
        segment_choice = True
    else:
        segment_choice = (
            input("Segment prompts? (y/N): ").strip().lower() == "y"
        )

    for lora_choice in selected_loras:
        lora_path = os.path.join(lora_root, lora_choice, "lora_model") if lora_choice else None
        model, tokenizer = load_model(model_name, lora_path)

        for text in prompts:
            if segment_choice:
                if args.segment_by == "sentence":
                    segments = split_prompt_by_sentences(text, tokenizer)
                else:
                    segments = split_prompt_by_tokens(text, tokenizer)
            else:
                segments = [tokenizer(text, return_tensors="pt").input_ids.squeeze(0)]
            audio_parts = [
                generate_audio_segment(
                    ids, model, snac_model, max_new_tokens=args.max_tokens
                )
                for ids in segments
            ]
            final_audio = concat_with_fade(audio_parts)
            path = get_output_path(lora_choice or "base_model")
            torchaudio.save(path, final_audio.detach().cpu(), 24000)
            print(f"Audio written to {path}")

if __name__ == '__main__':
    main()
