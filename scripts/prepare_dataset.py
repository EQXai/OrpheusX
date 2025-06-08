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
    max_tokens: int = 50,
    min_duration: float | None = None,
    model_max_len: int = 2048,
) -> None:
    """Transcribe ``audio_path`` and save the dataset under ``output_dir``.

    Parameters
    ----------
    audio_path : str
        Path to the source audio file.
    output_dir : str
        Directory where the dataset will be stored.
    max_tokens : int, optional
        Maximum number of tokens per audio segment. Defaults to 50.
    min_duration : float, optional
        Minimum duration in seconds for each segment. ``max_len`` will be set to
        ``min_duration + 5`` to provide a small buffer.
    """
    audio_path = Path(audio_path).resolve()
    base = audio_path.stem
    temp_out = Path("whisperx_out")
    segment_out = Path("segments") / base

    # Run WhisperX transcription and segmentation
    json_path = whisper_run.run_whisperx(audio_path, temp_out)

    # Compute segment length limits
    if min_duration is not None:
        min_len = float(min_duration)
        max_len = min_len + 5.0
    else:
        min_len = 10.0
        max_len = 15.0

    # Estimate maximum allowed length based on the model context
    TEXT_TOKEN_BUFFER = 256
    SNAC_FRAME_RATE = 12  # Hz
    TOKENS_PER_SECOND = SNAC_FRAME_RATE * 7
    allowed_tokens = model_max_len - TEXT_TOKEN_BUFFER
    max_audio_seconds = allowed_tokens / TOKENS_PER_SECOND
    max_len = min(max_len, max_audio_seconds)

    whisper_run.segment_audio(
        audio_path,
        json_path,
        segment_out,
        min_len=min_len,
        max_len=max_len,
        max_tokens=max_tokens,
        target_samples=None,
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
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens per audio segment",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        help="Minimum duration in seconds per segment",
    )
    parser.add_argument(
        "--model_max_len",
        type=int,
        default=2048,
        help="Model max length used to estimate max audio duration",
    )
    args = parser.parse_args()

    prepare_dataset(
        args.audio,
        args.output,
        max_tokens=args.max_tokens,
        min_duration=args.min_duration,
        model_max_len=args.model_max_len,
    )


if __name__ == "__main__":
    main()
