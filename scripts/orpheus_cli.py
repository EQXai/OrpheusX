#!/usr/bin/env python
"""Simple interactive CLI to manage OrpheusX workflows."""
import os
import subprocess
import time
from pathlib import Path
from tools.logger_utils import get_logger

# Resolve repository root based on script location
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

logger = get_logger("orpheus_cli")


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


MENU_OPTIONS = {
    "1": ("Installation", install),
    "2": ("Create dataset", create_dataset),
    "3": ("Training", train),
    "4": ("Inference", infer),
    "5": ("Exit", None),
}


def main() -> None:
    while True:
        print("\nOrpheusX CLI")
        for key, (label, _) in MENU_OPTIONS.items():
            print(f"{key}. {label}")
        choice = input("Choose an option [5]: ").strip() or "5"
        if choice == "5":
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
