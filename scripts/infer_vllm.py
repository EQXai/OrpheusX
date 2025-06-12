#!/usr/bin/env python
import argparse
import os
import json
import torch
import torchaudio
import re
import gc
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from snac import SNAC
from orpheusx.utils.segment_utils import (
    split_prompt_by_tokens as _split_prompt_by_tokens,
    split_prompt_by_sentences as _split_prompt_by_sentences,
    print_segment_log as _print_segment_log,
)

# Root of repository to load helper modules when run from ``scripts`` directory
import sys
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from audio_utils import concat_with_fade

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_model(model_name, lora_path=None):
    """Load a model with vLLM and optional LoRA weights."""
    llm = LLM(model=model_name)
    lora_request = None
    if lora_path and os.path.isdir(lora_path):
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=lora_path,
            base_model_name=model_name,
        )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer, lora_request


def split_prompt_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int = 50,
    return_text: bool = False,
) -> list[torch.Tensor] | tuple[list[str], list[torch.Tensor]]:
    """Split text into token chunks without breaking words (shared wrapper)."""
    return _split_prompt_by_tokens(text, tokenizer, chunk_size, return_text)


def print_segment_log(prompt: str, segments: list[str]) -> None:
    """Print segment boundaries for a prompt (shared wrapper)."""
    _print_segment_log(prompt, segments)


def split_prompt_by_sentences(
    text: str,
    tokenizer,
    chunk_size: int = 50,
    return_text: bool = False,
) -> list[torch.Tensor] | tuple[list[str], list[torch.Tensor]]:
    """Split text into sentence groups not exceeding *chunk_size* tokens (shared wrapper)."""
    return _split_prompt_by_sentences(text, tokenizer, chunk_size, return_text)


def generate_audio_segment(
    tokens: torch.Tensor,
    llm: LLM,
    snac_model,
    lora_request: LoRARequest | None,
    max_new_tokens: int = 1200,
) -> torch.Tensor:
    """Generate audio for given token IDs and return as 1D tensor using vLLM."""
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
    modified_input = torch.cat([start_token, tokens.unsqueeze(0), end_tokens], dim=1)
    prompt_ids = modified_input.squeeze(0).tolist()
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        stop_token_ids=[128258],
    )
    outputs = llm.generate(
        prompts=None,
        sampling_params=sampling,
        prompt_token_ids=[prompt_ids],
        lora_request=lora_request,
    )
    generated_ids = outputs[0].outputs[0].token_ids
    generated = torch.tensor(generated_ids, dtype=torch.int64)
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
    parser = argparse.ArgumentParser(description='Interactive Orpheus TTS inference')
    parser.add_argument('--model', default='unsloth/orpheus-3b-0.1-ft', help='Model name or path')
    parser.add_argument('--lora', default='lora_model', help='Path to trained LoRA adapters')
    parser.add_argument('--segment', action='store_true', help='Segment prompts')
    parser.add_argument(
        '--segment-by',
        choices=['tokens', 'sentence'],
        default='tokens',
        help='Segmentation method when using --segment',
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1200,
        help='Maximum number of tokens to generate',
    )
    parser.add_argument(
        '--fade_ms',
        type=int,
        default=60,
        help='Crossfade duration in milliseconds',
    )
    args = parser.parse_args()
    llm, tokenizer, lora_request = load_model(args.model, args.lora)

    snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz', cache_dir=CACHE_DIR)
    snac_model = snac_model.to('cpu')

    print(
        "You can use these tokens for preconfigured expressions: "
        "<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
    )

    script_dir = os.path.dirname(__file__)
    prompt_root = os.path.join(script_dir, "..", "prompt_list")

    prompt_files = []
    if os.path.isdir(prompt_root):
        prompt_files = [f for f in os.listdir(prompt_root) if f.endswith(".json")]

    prompt_list = None
    if prompt_files:
        mode = (
            input(
                "Enter '1' to type prompts manually or '2' to load a prompt list [1]: "
            ).strip()
            or "1"
        )
        if mode == "2":
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
        if args.segment:
            if args.segment_by == 'sentence':
                seg_text, segments = split_prompt_by_sentences(text, tokenizer, return_text=True)
            else:
                seg_text, segments = split_prompt_by_tokens(text, tokenizer, return_text=True)
            print_segment_log(text, seg_text)
        else:
            segments = [tokenizer(text, return_tensors='pt').input_ids.squeeze(0)]
        start_time = time.perf_counter()
        final_audio = None
        for ids in segments:
            part = generate_audio_segment(
                ids,
                llm,
                snac_model,
                lora_request,
                max_new_tokens=args.max_tokens,
            )
            if final_audio is None:
                final_audio = part
            else:
                final_audio = concat_with_fade([final_audio, part], fade_ms=args.fade_ms)
            torch.cuda.empty_cache()
            gc.collect()
        if final_audio is None:
            continue
        elapsed = time.perf_counter() - start_time
        duration = final_audio.shape[-1] / 24000
        if duration:
            rate = elapsed / duration
        else:
            rate = 0.0
        print(f"Inference time: {elapsed:.2f}s ({rate:.2f}s per generated second)")
        path = 'output.wav'
        torchaudio.save(path, final_audio.detach().cpu(), 24000)
        print(f'Audio written to {path}')
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
