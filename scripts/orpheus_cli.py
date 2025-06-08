#!/usr/bin/env python
"""Simple interactive CLI to manage OrpheusX workflows."""
import os
import subprocess
from pathlib import Path

# Resolve repository root based on script location
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def run_script(command, shell=False):
    """Run a command as subprocess and stream its output."""
    try:
        subprocess.run(command, check=True, cwd=str(SCRIPTS_DIR), shell=shell)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {exc}")


def install():
    run_script(["bash", "install.sh"])


def create_dataset():
    multi = input("Create datasets in batch? (y/N): ").strip().lower() == "y"
    while True:
        run_script(["python", "prepare_dataset_interactive.py"])
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
    segment = input("Segment prompts every 30 tokens? (y/N): ").strip().lower() == "y"
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
        _, func = action
        func()


if __name__ == "__main__":
    main()
