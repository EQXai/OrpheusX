from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr

from tools.logger_utils import get_logger
from scripts.prepare_dataset import prepare_dataset

from orpheusx.constants import (
    DATASETS_DIR,
    LORA_DIR,
    PROMPT_LIST_DIR,
    SOURCE_AUDIO_DIR,
    STOP_FLAG as _STOP_FLAG,
)

logger = get_logger("pipeline.data")

# -----------------------------------------------------------------------------
# Directory listing helpers
# -----------------------------------------------------------------------------

def list_datasets() -> list[str]:
    """Return the list of prepared dataset directories."""
    if not DATASETS_DIR.is_dir():
        return []
    return sorted([d.name for d in DATASETS_DIR.iterdir() if d.is_dir()])


def list_loras() -> list[str]:
    """Return the list of available LoRA sub-directories."""
    if not LORA_DIR.is_dir():
        return []
    return sorted([d.name for d in LORA_DIR.iterdir() if d.is_dir()])


def list_prompt_files() -> list[str]:
    """JSON prompt list files located under *PROMPT_LIST_DIR*."""
    if not PROMPT_LIST_DIR.is_dir():
        return []
    return sorted([f.name for f in PROMPT_LIST_DIR.glob("*.json")])


def load_prompts(file_name: str) -> list[str]:
    """Load prompts from a JSON list stored in *PROMPT_LIST_DIR*."""
    path = PROMPT_LIST_DIR / file_name
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        logger.exception("Failed to load prompts from %s", path)
        return []
    if not isinstance(data, list):
        return []
    return [str(p) for p in data if isinstance(p, str)]


def list_source_audio() -> list[str]:
    """WAV/MP3 audio available for dataset creation."""
    if not SOURCE_AUDIO_DIR.is_dir():
        return []
    return sorted(
        [f.name for f in SOURCE_AUDIO_DIR.iterdir() if f.suffix.lower() in (".wav", ".mp3")]
    )

# -----------------------------------------------------------------------------
# Dataset preparation utility designed for the Gradio UI
# -----------------------------------------------------------------------------

def prepare_datasets_ui(
    upload_file: str,
    name: str,
    existing: list[str] | None,
    min_tokens: int = 0,
    max_tokens: int | None = None,
) -> str:
    """Prepare one or more datasets from uploaded *upload_file* or *existing* audio.

    This mirrors the original implementation from *gradio_app.py* but relies on the
    central *STOP_FLAG* from :pymod:`orpheusx.constants`.
    """
    from orpheusx import constants as _c  # local import to avoid circular deps

    tasks: list[tuple[str, str]] = []
    if upload_file:
        if not name:
            return "Please provide a dataset name for the uploaded audio."
        tasks.append((upload_file, name))
    for fname in existing or []:
        audio_path = SOURCE_AUDIO_DIR / fname
        tasks.append((str(audio_path), Path(fname).stem))
    if not tasks:
        return "No audio selected."

    msgs: list[str] = []
    logger.info("Preparing %d dataset(s)", len(tasks))
    total = len(tasks)
    progress = gr.Progress()
    for idx, (audio_path, ds_name) in enumerate(tasks, start=1):
        if _c.STOP_FLAG:
            _c.STOP_FLAG = False
            return "Stopped"
        start = time.perf_counter()
        progress((idx - 1) / total, desc=f"Preparing {ds_name}...")
        out_dir = DATASETS_DIR / ds_name
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            prepare_dataset(
                audio_path,
                str(out_dir),
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
            msgs.append(f"{ds_name}: success")
            elapsed = time.perf_counter() - start
            logger.info("%s prepared in %.2fs", ds_name, elapsed)
        except Exception as e:  # pragma: no cover - best effort
            elapsed = time.perf_counter() - start
            logger.exception("Error preparing %s after %.2fs", ds_name, elapsed)
            msgs.append(f"{ds_name}: failed ({e})")
        progress(idx / total)
    return "\n".join(msgs)

# -----------------------------------------------------------------------------
# Convenience helper for the Gradio UI to refresh choices
# -----------------------------------------------------------------------------

def refresh_lists() -> tuple[gr.components.Dropdown, gr.components.Dropdown]:
    """Reload dataset and prompt list choices from disk."""
    return (
        gr.update(choices=list_source_audio()),
        gr.update(choices=[""] + list_prompt_files()),
    )

__all__ = [
    "list_datasets",
    "list_loras",
    "list_prompt_files",
    "load_prompts",
    "list_source_audio",
    "prepare_datasets_ui",
    "refresh_lists",
] 