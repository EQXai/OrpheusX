#!/usr/bin/env python
"""Prepare a local dataset using Whisper transcription.

This script uses the functions from ``tools/Whisper/run.py`` to transcribe an audio
file and segment it. The resulting ``.wav`` and ``.txt`` pairs are then
converted into a ``datasets`` Dataset and saved to disk (and as a ``.parquet``
file) so that it can be loaded with ``load_from_disk`` or other tools.
"""
import os
import sys
from pathlib import Path
from datasets import Audio, Dataset

# Make sure the repository root is on the Python path so ``tools`` can be found
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import helper functions from the Whisper package
from tools.Whisper import run as whisper_run
from tools.Whisper.upload import load_dataset_from_folder


def prepare_dataset(
    audio_path: str,
    output_dir: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> None:
    """Transcribe ``audio_path`` and save the dataset under ``output_dir``.

    Audio is segmented into fragments around 20 seconds long (typically
    between 15 and 25 seconds) without cutting words so that each clip matches
    its transcript exactly. ``min_tokens`` and ``max_tokens`` can be used to
    control the token length of each segment.
    """
    audio_path = Path(audio_path).resolve()
    base = audio_path.stem
    temp_out = Path("whisperx_out")
    segment_out = Path("segments") / base

    # Run WhisperX transcription and segmentation
    json_path = whisper_run.run_whisperx(audio_path, temp_out)

    # Always split into 15–25 second fragments without cutting words.
    # When token limits are provided, they are forwarded to the segmentation
    # helper to optionally enforce a token range.
    whisper_run.segment_audio(
        audio_path,
        json_path,
        segment_out,
        max_tokens=max_tokens,
        min_tokens=min_tokens or 0,
    )

    # Build Dataset and store it on disk
    dataset = load_dataset_from_folder(segment_out)
    dataset = dataset.cast_column("audio", Audio())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)

    # Build a DataFrame with the raw audio bytes so the Parquet file is
    # completely self‑contained (otherwise only the file paths are stored).
    import pandas as pd

    rows = []
    for item in dataset:
        audio_path = item["audio"]["path"]
        sr = item["audio"]["sampling_rate"]
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        rows.append({"audio": audio_bytes, "sampling_rate": sr, "text": item["text"]})

    df = pd.DataFrame(rows)

    # Also store the dataset in Parquet format for easy sharing
    parquet_path = output_dir / "dataset.parquet"
    df.to_parquet(parquet_path)

    print(f"Dataset saved under {output_dir.resolve()}")
    print(f"Parquet file written to {parquet_path.resolve()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio and save as dataset")
    parser.add_argument("audio", help="Path to audio file (.mp3 or .wav)")
    parser.add_argument("output", help="Directory to save the dataset")
    parser.add_argument("--min-tokens", type=int, default=0, help="Minimum tokens per segment")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens per segment")
    args = parser.parse_args()

    prepare_dataset(args.audio, args.output, args.min_tokens, args.max_tokens)


if __name__ == "__main__":
    main()
