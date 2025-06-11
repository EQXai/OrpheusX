#!/usr/bin/env python
"""Train a LoRA model using the GRPO algorithm."""
import argparse
from pathlib import Path

from gradio_app import train_grpo_single, DATASETS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO LoRA")
    parser.add_argument("dataset", help="HF dataset link or local path")
    parser.add_argument("name", help="Name of the LoRA output directory")
    parser.add_argument("--local", action="store_true", help="Dataset is a local path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--scheduler", default="linear")
    args = parser.parse_args()

    msg = train_grpo_single(
        args.dataset,
        args.name,
        args.local,
        args.batch_size,
        args.grad_steps,
        args.epochs,
        args.lr,
        args.log_steps,
        args.weight_decay,
        args.optim,
        args.scheduler,
    )
    print(msg)


if __name__ == "__main__":
    main()
