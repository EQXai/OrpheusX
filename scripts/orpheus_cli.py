#!/usr/bin/env python
"""Simple interactive CLI to manage OrpheusX workflows."""
import os
import subprocess
import time
from pathlib import Path
from tools.logger_utils import get_logger
import json
import gc
import torch
import sys

# Resolve repository root based on script location
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Ensure top-level directory is on PYTHONPATH for direct imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = get_logger("orpheus_cli")

# Directory constants
DATASETS_DIR = REPO_ROOT / "datasets"
LORA_DIR = REPO_ROOT / "lora_models"
SOURCE_AUDIO_DIR = REPO_ROOT / "source_audio"

# Common yes/no helper
def _yes(prompt: str, default: bool = False) -> bool:
    reply = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not reply:
        return default
    return reply.startswith("y")

def run_script(command, shell=False):
    """Run a command as subprocess and stream its output with logging."""
    cmd_str = command if isinstance(command, str) else " ".join(command)
    logger.info("Running: %s", cmd_str)
    start = time.perf_counter()
    try:
        subprocess.run(command, check=True, cwd=str(SCRIPTS_DIR), shell=shell)
        elapsed = time.perf_counter() - start
        logger.info("Finished in %.2fs", elapsed)
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - start
        logger.error("Command failed after %.2fs: %s", elapsed, exc)


def install():
    run_script(["bash", "install.sh"])


def create_dataset():
    multi = input("Create datasets in batch? (y/N): ").strip().lower() == "y"
    cmd = ["python", "prepare_dataset_interactive.py"]
    while True:
        run_script(cmd)
        if not multi:
            break
        again = input("Create another dataset? (y/N): ").strip().lower()
        if again != "y":
            break


def train():
    multi = input("Train multiple models? (y/N): ").strip().lower() == "y"
    while True:
        run_script(["python", "train_interactive.py"])
        if not multi:
            break
        again = input("Train another model? (y/N): ").strip().lower()
        if again != "y":
            break


def infer():
    multi = input("Run inference on multiple models? (y/N): ").strip().lower() == "y"
    segment = input("Segment prompts every 50 tokens? (y/N): ").strip().lower() == "y"
    cmd = ["python", "infer_interactive.py"]
    if segment:
        cmd.append("--segment")
    while True:
        run_script(cmd)
        if not multi:
            break
        again = input("Run inference with another model? (y/N): ").strip().lower()
        if again != "y":
            break


def check_environment():
    """Run the environment verification script."""
    run_script(["python", "check_env.py"])


def prepare_dataset_cli():
    """Interactive dataset preparation with token limits."""
    from scripts.prepare_dataset import prepare_dataset  # heavy deps, local import

    # List available audio files
    audio_files = [f for f in SOURCE_AUDIO_DIR.iterdir() if f.suffix.lower() in (".mp3", ".wav")]
    if not audio_files:
        print("No audio files found in 'source_audio'. Add files and retry.")
        return

    print("Select the audio file(s) to process (comma separated numbers):")
    for idx, f in enumerate(audio_files, 1):
        print(f"{idx}. {f.name}")
    choice = input("Choice(s) [1]: ").strip() or "1"
    indices = []
    for part in choice.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(audio_files):
                indices.append(idx)
    if not indices:
        indices = [1]

    # Token limits
    min_tokens = input("Minimum tokens per segment [0]: ").strip()
    min_tokens = int(min_tokens) if min_tokens else 0
    max_tokens = input("Maximum tokens per segment (blank = no limit): ").strip()
    max_tokens_val = int(max_tokens) if max_tokens else None

    for idx in indices:
        audio_path = audio_files[idx - 1]
        out_dir = DATASETS_DIR / audio_path.stem
        print(f"\nPreparing dataset '{audio_path.name}' → {out_dir} …")
        try:
            prepare_dataset(str(audio_path), str(out_dir), min_tokens=min_tokens, max_tokens=max_tokens_val)
            print("✓ Success")
        except Exception as exc:  # pragma: no cover
            print(f"✗ Failed: {exc}")


def train_advanced():
    """Train one or more LoRAs with advanced hyper-parameters."""
    # Deferred import because it pulls many ML deps
    from gradio_app import train_lora_single

    # Select datasets (local only for simplicity)
    if not DATASETS_DIR.is_dir():
        print("No datasets directory found. Prepare a dataset first.")
        return
    dataset_dirs = [d.name for d in DATASETS_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print("No datasets available.")
        return
    print("Select dataset(s) to train (comma separated numbers):")
    for idx, name in enumerate(dataset_dirs, 1):
        print(f"{idx}. {name}")
    choice = input("Choice(s) [1]: ").strip() or "1"
    sel = []
    for part in choice.split(','):
        if part.strip().isdigit():
            i = int(part.strip())
            if 1 <= i <= len(dataset_dirs):
                sel.append(dataset_dirs[i - 1])
    if not sel:
        sel = [dataset_dirs[0]]

    # Hyper-parameters
    def _num(prompt, default, cast=float):
        val = input(f"{prompt} [{default}]: ").strip()
        return cast(val) if val else default

    batch_size = _num("Batch size", 1, int)
    grad_steps = _num("Gradient accumulation steps", 4, int)
    warm_steps = _num("Warmup steps", 5, int)
    max_steps = _num("Max steps", 60, int)
    epochs = _num("Epochs", 1, int)
    lr = _num("Learning rate", 2e-4, float)
    log_steps = _num("Logging steps", 1, int)
    weight_decay = _num("Weight decay", 0.01, float)
    optim = input("Optimizer [adamw_8bit]: ").strip() or "adamw_8bit"
    scheduler = input("LR scheduler type [linear]: ").strip() or "linear"

    for ds in sel:
        src_path = DATASETS_DIR / ds
        print(f"\nTraining LoRA '{ds}' from {src_path} …")
        try:
            train_lora_single(
                str(src_path),
                ds,
                True,
                batch_size,
                grad_steps,
                warm_steps,
                max_steps,
                epochs,
                lr,
                log_steps,
                weight_decay,
                optim,
                scheduler,
            )
            print("✓ Training finished")
        except Exception as exc:
            print(f"✗ Training failed: {exc}")


def inference_advanced():
    """Generate audio with all advanced options."""
    from gradio_app import generate_audio

    # LoRA selection
    loras = ["<base>"]
    if LORA_DIR.is_dir():
        loras += [d.name for d in LORA_DIR.iterdir() if d.is_dir()]
    print("Available LoRA models:")
    for idx, name in enumerate(loras, 1):
        print(f"{idx}. {name}")
    sel = input("Select LoRA by number(s) [1]: ").strip() or "1"
    indices = [int(p) for p in sel.split(',') if p.strip().isdigit()]
    indices = [i for i in indices if 1 <= i <= len(loras)] or [1]
    chosen_loras = [loras[i - 1] for i in indices]

    # Prompt(s)
    prompts: list[str] = []
    while True:
        prompt = input("Enter prompt (blank to finish): ").strip()
        if not prompt:
            if prompts:
                break
            print("Please enter at least one prompt.")
            continue
        prompts.append(prompt)

    # Generation parameters
    def _f(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        return type(default)(val) if val else default

    temperature = _f("Temperature", 0.6)
    top_p = _f("Top-p", 0.95)
    rep_penalty = _f("Repetition penalty", 1.1)
    max_tokens = _f("Max new tokens", 1200)

    segment = _yes("Segment text?", False)
    if segment:
        seg_method = input("Method (tokens / sentence / full_segment) [tokens]: ").strip() or "tokens"
        if seg_method not in {"tokens", "sentence", "full_segment"}:
            seg_method = "tokens"
        seg_min = _f("Min tokens per segment", 0)
        seg_max = _f("Max tokens per segment", 50)
        seg_chars_raw = input("Characters to split on (e.g. ,.?!) [, . ? !]: ").strip()
        seg_chars = [c.strip() for c in seg_chars_raw.split() if c.strip()] or [",", ".", "?", "!"]
        seg_gap = _f("Gap between segments (seconds)", 0.0)
    else:
        seg_method = "tokens"
        seg_min = 0
        seg_max = 50
        seg_chars = []
        seg_gap = 0.0

    for lora in chosen_loras:
        for text in prompts:
            print(f"\nGenerating with LoRA '{lora}'…")
            try:
                path = generate_audio(
                    text,
                    None if lora == "<base>" else lora,
                    temperature,
                    top_p,
                    rep_penalty,
                    max_tokens,
                    segment,
                    seg_method,
                    seg_chars,
                    seg_min,
                    seg_max,
                    seg_gap,
                )
                print(f"✓ Saved to {path}")
            except Exception as exc:
                print(f"✗ Generation failed: {exc}")
            # Aggressive cleanup to free VRAM on small GPUs
            torch.cuda.empty_cache()
            gc.collect()


def auto_pipeline():
    """One-click: prepare dataset → train LoRA → generate audio."""
    from scripts.prepare_dataset import prepare_dataset
    from gradio_app import train_lora_single, generate_audio

    # Audio select
    audio_files = [f for f in SOURCE_AUDIO_DIR.iterdir() if f.suffix.lower() in (".mp3", ".wav")]
    if not audio_files:
        print("No audio files found in 'source_audio'.")
        return
    print("Select audio file [1]:")
    for idx, f in enumerate(audio_files, 1):
        print(f"{idx}. {f.name}")
    idx = input("Choice: ").strip() or "1"
    if not idx.isdigit() or not (1 <= int(idx) <= len(audio_files)):
        idx = 1
    audio_path = audio_files[int(idx) - 1]

    prompt = input("Prompt to synthesise: ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return

    ds_dir = DATASETS_DIR / audio_path.stem
    try:
        if not ds_dir.is_dir():
            print("Preparing dataset…")
            prepare_dataset(str(audio_path), str(ds_dir))
        else:
            print("Dataset already exists → skipping.")

        lora_name = audio_path.stem
        lora_path = LORA_DIR / lora_name / "lora_model"
        if not lora_path.is_dir():
            print("Training LoRA…")
            train_lora_single(str(ds_dir), lora_name, True)
        else:
            print("LoRA already trained → skipping.")

        print("Generating audio…")
        out_path = generate_audio(prompt, lora_name)
        print(f"✓ Done. File: {out_path}")
    except Exception as exc:
        print(f"✗ Pipeline failed: {exc}")


def launch_gradio():
    """Start the Gradio web UI on the default port (7860)."""
    run_script(["python", str(REPO_ROOT / "gradio_app.py")])


MENU_OPTIONS = {
    "1": ("Check environment", check_environment),
    "2": ("Install dependencies", install),
    "3": ("Prepare dataset", prepare_dataset_cli),
    "4": ("Train LoRA", train_advanced),
    "5": ("Inference", inference_advanced),
    "6": ("Auto Pipeline", auto_pipeline),
    "7": ("Launch Gradio UI", launch_gradio),
    "8": ("Exit", None),
}


def main() -> None:
    while True:
        print("\nOrpheusX CLI")
        for key, (label, _) in MENU_OPTIONS.items():
            print(f"{key}. {label}")
        choice = input("Choose an option [8]: ").strip() or "8"
        if choice == "8":
            print("Bye!")
            break
        action = MENU_OPTIONS.get(choice)
        if not action:
            print("Invalid option")
            continue
        label, func = action
        logger.info("Selected option: %s", label)
        func()


if __name__ == "__main__":
    main()
