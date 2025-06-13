from __future__ import annotations

import time
from pathlib import Path


from tools.logger_utils import get_logger
from scripts.prepare_dataset import prepare_dataset

from orpheusx.constants import (
    DATASETS_DIR,
    LORA_DIR,
    SOURCE_AUDIO_DIR,
    STOP_FLAG as _STOP_FLAG,
)
from orpheusx.pipeline.data import load_prompts
from orpheusx.pipeline.train import train_lora_single
from orpheusx.pipeline.infer import (
    generate_audio,
    generate_batch,
    get_pipeline_tokenizer,
)

logger = get_logger("pipeline.full")

# -----------------------------------------------------------------------------
# Dataset status helpers (for UI)
# -----------------------------------------------------------------------------

def dataset_status(name: str) -> str:
    ds_name = Path(name).stem
    lora_path = LORA_DIR / ds_name / "lora_model"
    return "Model already created" if lora_path.is_dir() else ""


def dataset_status_multi(names: list[str]) -> str:
    msgs: list[str] = []
    for name in names:
        ds_name = Path(name).stem
        lora_path = LORA_DIR / ds_name / "lora_model"
        status = "Model already created" if lora_path.is_dir() else ""
        msgs.append(f"{ds_name}: {status}")
    return "\n".join(msgs)

# -----------------------------------------------------------------------------
# Full end-to-end pipeline for a single dataset
# -----------------------------------------------------------------------------

def run_full_pipeline(dataset_file: str, prompt: str, fade_ms: int = 60):
    from orpheusx import constants as _c

    if not dataset_file:
        return "No dataset selected", ""
    if not prompt:
        return "Prompt is empty", ""

    ds_name = Path(dataset_file).stem
    audio_path = SOURCE_AUDIO_DIR / dataset_file
    dataset_dir = DATASETS_DIR / ds_name
    lora_dir = LORA_DIR / ds_name / "lora_model"

    msgs: list[str] = []

    if _c.STOP_FLAG:
        _c.STOP_FLAG = False
        return "Stopped", ""

    if not dataset_dir.is_dir():
        prepare_dataset(str(audio_path), str(dataset_dir))
        msgs.append("Dataset prepared")
    else:
        msgs.append("Dataset already prepared")

    if _c.STOP_FLAG:
        _c.STOP_FLAG = False
        return "Stopped", ""

    if not lora_dir.is_dir():
        train_lora_single(str(dataset_dir), ds_name, True)
        msgs.append("LoRA trained")
    else:
        msgs.append("LoRA already trained")

    tokenizer = get_pipeline_tokenizer()
    token_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    use_segmentation = token_len > 50

    if _c.STOP_FLAG:
        _c.STOP_FLAG = False
        return "Stopped", ""

    if use_segmentation:
        out_path = generate_audio(
            prompt,
            ds_name,
            max_new_tokens=2400,
            segment=True,
            segment_by="tokens",
            seg_min_tokens=0,
            seg_max_tokens=50,
            seg_gap=0.0,
            fade_ms=fade_ms,
        )
    else:
        out_path = generate_audio(prompt, ds_name, fade_ms=fade_ms)

    msgs.append(f"Audio saved to {out_path}")
    return "\n".join(msgs), out_path

# -----------------------------------------------------------------------------
# Batch version operating on multiple datasets/prompts
# -----------------------------------------------------------------------------

def run_full_pipeline_batch(
    dataset_files: list[str],
    prompt: str,
    prompt_file: str | None,
    batch: int,
    fade_ms: int = 60,
):
    from orpheusx import constants as _c

    if not dataset_files:
        return "No dataset selected", ""

    # Determine prompts list
    if prompt_file:
        prompts = load_prompts(prompt_file)
        if not prompts:
            return "Prompt list is empty", ""
    else:
        if not prompt:
            return "Prompt is empty", ""
        prompts = [prompt] * max(1, int(batch or 1))

    msgs: list[str] = []
    html_blocks: list[str] = []
    total = len(dataset_files)

    tokenizer = get_pipeline_tokenizer()
    seg_needed = any(len(tokenizer(p, add_special_tokens=False).input_ids) > 50 for p in prompts)
    max_tokens = 2400 if seg_needed else 1200

    for idx, dataset_file in enumerate(dataset_files, 1):
        if _c.STOP_FLAG:
            _c.STOP_FLAG = False
            return "Stopped", ""

        ds_name = Path(dataset_file).stem
        audio_path = SOURCE_AUDIO_DIR / dataset_file
        dataset_dir = DATASETS_DIR / ds_name
        lora_dir = LORA_DIR / ds_name / "lora_model"

        base_progress = (idx - 1) / total

        if not dataset_dir.is_dir():
            prepare_dataset(str(audio_path), str(dataset_dir))
            msgs.append(f"{ds_name}: dataset prepared")
        else:
            msgs.append(f"{ds_name}: dataset already prepared")

        if _c.STOP_FLAG:
            _c.STOP_FLAG = False
            return "Stopped", ""

        if not lora_dir.is_dir():
            train_lora_single(str(dataset_dir), ds_name, True)
            msgs.append(f"{ds_name}: LoRA trained")
        else:
            msgs.append(f"{ds_name}: LoRA already trained")

        if _c.STOP_FLAG:
            _c.STOP_FLAG = False
            return "Stopped", ""

        html, _ = generate_batch(
            prompts,
            [ds_name],
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            max_new_tokens=max_tokens,
            segment=seg_needed,
            segment_by="tokens",
            seg_chars=None,
            seg_min_tokens=0,
            seg_max_tokens=50,
            seg_gap=0.0,
            fade_ms=fade_ms,
        )
        html_blocks.append(html)

    return "\n".join(msgs), "<hr/>".join(html_blocks)

__all__ = [
    "dataset_status",
    "dataset_status_multi",
    "run_full_pipeline",
    "run_full_pipeline_batch",
] 