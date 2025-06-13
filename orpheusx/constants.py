from __future__ import annotations

import os
from pathlib import Path

# Project root is the parent of this package directory (orpheusx/..)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Common top-level directories used throughout the codebase
DATASETS_DIR: Path = REPO_ROOT / "datasets"
LORA_DIR: Path = REPO_ROOT / "lora_models"
PROMPT_LIST_DIR: Path = REPO_ROOT / "prompt_list"
SOURCE_AUDIO_DIR: Path = REPO_ROOT / "source_audio"
CACHE_DIR: Path = REPO_ROOT / "models"

# Default base model name. Can be overridden with env var MODEL_NAME
MODEL_NAME: str = os.environ.get("MODEL_NAME", "unsloth/orpheus-3b-0.1-ft")

# Maximum number of prompts to display in the UI at once (kept from original script)
MAX_PROMPTS: int = 5

# Global flag that long-running tasks should periodically check
STOP_FLAG: bool = False

__all__ = [
    "REPO_ROOT",
    "DATASETS_DIR",
    "LORA_DIR",
    "PROMPT_LIST_DIR",
    "SOURCE_AUDIO_DIR",
    "CACHE_DIR",
    "MODEL_NAME",
    "MAX_PROMPTS",
    "STOP_FLAG",
] 