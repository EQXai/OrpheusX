from __future__ import annotations

import re
import time
import gc
from pathlib import Path
from typing import List, Tuple

import torch
import gradio as gr
from transformers import AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel
from snac import SNAC

from tools.logger_utils import get_logger
from audio_utils import concat_with_fade
from orpheusx.constants import (
    CACHE_DIR,
    MODEL_NAME,
    LORA_DIR,
    REPO_ROOT,
    STOP_FLAG as _STOP_FLAG,
)

logger = get_logger("pipeline.infer")

# Cached objects to avoid reloading for every request
_LOADED_MODEL_NAME: str | None = None
_LOADED_LORA_PATH: str | None = None
_LOADED_MODEL = None
_LOADED_TOKENIZER = None
_SNAC_MODEL = None
_PIPELINE_TOKENIZER = None

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

def load_model(base_model: str, lora_path: str | None):
    """Load *base_model* with optional LoRA and cache the pair for reuse."""
    global _LOADED_MODEL_NAME, _LOADED_LORA_PATH, _LOADED_MODEL, _LOADED_TOKENIZER
    if (
        _LOADED_MODEL is not None
        and _LOADED_MODEL_NAME == base_model
        and _LOADED_LORA_PATH == lora_path
    ):
        return _LOADED_MODEL, _LOADED_TOKENIZER

    if _LOADED_MODEL is not None:
        del _LOADED_MODEL  # free GPU RAM
        torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        cache_dir=str(CACHE_DIR),
    )
    if lora_path and Path(lora_path).is_dir():
        model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)

    _LOADED_MODEL_NAME = base_model
    _LOADED_LORA_PATH = lora_path
    _LOADED_MODEL = model
    _LOADED_TOKENIZER = tokenizer
    return model, tokenizer


def get_snac_model():
    """Singleton loader for the SNAC codec."""
    global _SNAC_MODEL
    if _SNAC_MODEL is None:
        _SNAC_MODEL = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir=str(CACHE_DIR))
        _SNAC_MODEL = _SNAC_MODEL.to("cpu")
    return _SNAC_MODEL


def get_pipeline_tokenizer():
    """Tokenizer used solely for prompt-length checks in the UI."""
    global _PIPELINE_TOKENIZER
    if _PIPELINE_TOKENIZER is None:
        _PIPELINE_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
    return _PIPELINE_TOKENIZER

# -----------------------------------------------------------------------------
# Text segmentation utilities (truncated version of originals)
# -----------------------------------------------------------------------------

def _merge_short_segments(
    texts: List[str],
    tokens: List[torch.Tensor],
    min_tokens: int,
) -> Tuple[List[str], List[torch.Tensor]]:
    if min_tokens <= 0:
        return texts, tokens
    merged_texts: List[str] = []
    merged_tokens: List[torch.Tensor] = []
    for t_text, t_tokens in zip(texts, tokens):
        if merged_tokens and t_tokens.numel() < min_tokens:
            merged_texts[-1] = merged_texts[-1] + " " + t_text
            merged_tokens[-1] = torch.cat([merged_tokens[-1], t_tokens])
        else:
            merged_texts.append(t_text)
            merged_tokens.append(t_tokens)
    return merged_texts, merged_tokens


def split_prompt_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 50,
    min_tokens: int = 0,
    return_text: bool = False,
):
    words = text.split()
    segments: List[str] = []
    current: List[str] = []
    token_len = 0
    for w in words:
        n_tokens = len(tokenizer(w, add_special_tokens=False).input_ids)
        if token_len + n_tokens > max_tokens and current:
            segments.append(" ".join(current))
            current = [w]
            token_len = n_tokens
        else:
            current.append(w)
            token_len += n_tokens
    if current:
        segments.append(" ".join(current))
    token_segments = [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments]
    segments, token_segments = _merge_short_segments(segments, token_segments, min_tokens)
    return (segments, token_segments) if return_text else token_segments


def print_segment_log(prompt: str, segments: List[str]) -> None:
    offset = 0
    for idx, seg in enumerate(segments, 1):
        start = prompt.find(seg, offset)
        end = start + len(seg)
        logger.info("%d: chars %d-%d: %s", idx, start, end, seg)
        offset = end

# ---- A more heuristic split based on punctuation (condensed) ----

def split_prompt_full(text: str, tokenizer, chars=None, return_text: bool = False):
    if not chars:
        chars = [",", ".", "?", "!"]
    char_class = "".join(re.escape(c) for c in chars)
    pattern = rf"[^{char_class}]+(?:[{char_class}]+|$)"
    segments = [p.strip() for p in re.findall(pattern, text.strip()) if p.strip()]
    token_segments = [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments]
    return (segments, token_segments) if return_text else token_segments

# -----------------------------------------------------------------------------
# Audio generation helpers
# -----------------------------------------------------------------------------

def _redistribute_codes(code_list):
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
    return codes


def generate_audio_segment(tokens: torch.Tensor, model, snac_model, max_new_tokens: int = 1200):
    logger.info("Generating segment with %d tokens", tokens.numel())
    t0 = time.perf_counter()
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
    logger.info("Model.generate finished in %.2fs", time.perf_counter() - t0)

    token_to_find = 128257
    token_to_remove = 128258
    token_indices = (generated == token_to_find).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        last_idx = token_indices[1][-1].item()
        cropped = generated[:, last_idx + 1 :]
    else:
        cropped = generated
    processed_rows = [(row[row != token_to_remove]) for row in cropped]

    samples = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed = row[:new_length] - 128266
        codes = _redistribute_codes(trimmed.tolist())
        audio_hat = snac_model.decode(codes)
        samples.append(audio_hat.squeeze(0))
    return samples[0]


def _get_output_path(lora_name: str, ext: str = ".wav") -> Path:
    base_dir = REPO_ROOT / "audio_output" / lora_name
    base_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        path = base_dir / f"{lora_name}_{idx}{ext}"
        if not path.exists():
            return path
        idx += 1


def generate_audio(
    text: str,
    lora_name: str | None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 1200,
    segment: bool = False,
    segment_by: str = "tokens",
    seg_chars: list[str] | None = None,
    seg_min_tokens: int = 0,
    seg_max_tokens: int = 50,
    seg_gap: float = 0.0,
    fade_ms: int = 60,
) -> str:
    from orpheusx import constants as _c

    model_name = MODEL_NAME
    lora_path = None
    if lora_name:
        lora_path = LORA_DIR / lora_name / "lora_model"
    model, tokenizer = load_model(model_name, str(lora_path) if lora_path else None)

    snac_model = get_snac_model()

    token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
    logger.info("Generating audio (%d tokens) using %s", token_count, lora_name or "base_model")
    start_time = time.perf_counter()

    if segment:
        if segment_by == "full_segment":
            seg_text, segments = split_prompt_full(
                text, tokenizer, chars=seg_chars, return_text=True
            )
        else:
            seg_text, segments = split_prompt_by_tokens(
                text,
                tokenizer,
                max_tokens=seg_max_tokens,
                min_tokens=seg_min_tokens,
                return_text=True,
            )
        print_segment_log(text, seg_text)
    else:
        single = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        segments = [single]

    final_audio = None
    for s in segments:
        part = generate_audio_segment(s, model, snac_model, max_new_tokens=max_new_tokens)
        if final_audio is None:
            final_audio = part
        else:
            final_audio = concat_with_fade(
                [final_audio, part], sample_rate=24000, fade_ms=fade_ms, gap_ms=int(seg_gap * 1000)
            )
        torch.cuda.empty_cache()
        gc.collect()

    if final_audio is None:
        return ""

    elapsed = time.perf_counter() - start_time
    duration = final_audio.shape[-1] / 24000
    rate = elapsed / duration if duration else 0.0
    logger.info("Inference time: %.2fs (%.2fs per generated second)", elapsed, rate)

    lora_name = lora_name or "base_model"
    path = _get_output_path(lora_name)
    import torchaudio  # local import to avoid the heavy dependency unless needed

    torchaudio.save(str(path), final_audio.detach().cpu(), 24000)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Saved audio to %s", path)
    return str(path)


# -----------------------------------------------------------------------------
# Batch helper (multiple prompts/LORAs)
# -----------------------------------------------------------------------------

def generate_batch(
    prompts: list[str],
    loras: list[str],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    segment: bool,
    segment_by: str,
    seg_chars: list[str] | None,
    seg_min_tokens: int,
    seg_max_tokens: int,
    seg_gap: float = 0.0,
    fade_ms: int = 60,
) -> tuple[str, str]:
    from orpheusx import constants as _c

    if not prompts:
        return "", ""
    loras = loras or [None]

    results: list[tuple[str, str]] = []
    last_path = ""
    total = len(prompts) * len(loras)
    step = 0
    progress = gr.Progress()
    for lora in loras:
        for text in prompts:
            if _c.STOP_FLAG:
                _c.STOP_FLAG = False
                return "", ""
            progress(step / total, desc=f"Generating {lora or 'base'}...")
            logger.info("Generating prompt '%s' with lora '%s'", text, lora or 'base_model')
            path = generate_audio(
                text,
                None if lora == "<base>" else lora,
                temperature,
                top_p,
                repetition_penalty,
                max_new_tokens,
                segment,
                segment_by,
                seg_chars,
                seg_min_tokens,
                seg_max_tokens,
                seg_gap,
                fade_ms,
            )
            torch.cuda.empty_cache()
            gc.collect()
            caption = f"{lora or 'base'}: {text}"[:60]
            results.append((path, caption))
            last_path = path
            step += 1
    progress(1)

    html_items = [
        f"<div style='margin-bottom:1em'><p>{caption}</p><audio controls src='file={path}'></audio></div>"
        for path, caption in results
    ]
    return "\n".join(html_items), last_path

__all__ = [
    "load_model",
    "get_snac_model",
    "generate_audio_segment",
    "generate_audio",
    "generate_batch",
    "get_pipeline_tokenizer",
] 