#!/usr/bin/env python
"""Verify that the required environment is properly configured.

This script checks for active virtual environments, required packages,
CUDA availability and prints the package versions. It exits with status 1
if any required package is missing.
"""
import importlib
import os
import sys

REQUIRED_PACKAGES = [
    "unsloth",
    "bitsandbytes",
    "accelerate",
    "xformers",
    "peft",
    "trl",
    "triton",
    "cut_cross_entropy",
    "unsloth_zoo",
    "snac",
    "datasets",
    "transformers",
    "torchaudio",
    "whisperx",
    "soundfile",
    "librosa",
    "vllm",
    "evaluate",
    "jiwer",
    "wespeaker",
    "cloudpickle",
    "openai",
    "cpuinfo",
    "llguidance",
    "xgrammar",
    "torchmetrics",
    "audiobox_aesthetics",
]

def check_virtual_env():
    in_venv = (
        hasattr(sys, "base_prefix")
        and sys.base_prefix != sys.prefix
        or os.environ.get("VIRTUAL_ENV")
    )
    if in_venv:
        print(f"Using virtualenv: {sys.prefix}")
    else:
        print("WARNING: no virtualenv detected")


def check_package(pkg_name):
    try:
        module = importlib.import_module(pkg_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{pkg_name} {version} ... OK")
        return True
    except Exception as exc:
        print(f"MISSING: {pkg_name} ({exc})")
        return False


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            cuda_ver = torch.version.cuda or "unknown"
            if cuda_ver != "unknown":
                try:
                    major, minor = map(int, cuda_ver.split("."))
                    if major > 12 or (major == 12 and minor > 4):
                        print("WARNING: Detected CUDA version {}. OrpheusX only supports up to 12.4".format(cuda_ver))
                except ValueError:
                    pass
        else:
            print("WARNING: CUDA not available")
    except ImportError as exc:
        print(f"torch not installed: {exc}")

def check_ffmpeg():
    import shutil
    if shutil.which("ffmpeg"):
        print("ffmpeg ... OK")
        return True
    else:
        print("MISSING: ffmpeg")
        return False


def main():
    check_virtual_env()
    print("Checking required packages...")
    success = True
    for pkg in REQUIRED_PACKAGES:
        if not check_package(pkg):
            success = False
    check_cuda()
    if not check_ffmpeg():
        success = False
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
