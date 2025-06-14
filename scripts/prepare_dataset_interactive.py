#!/usr/bin/env python
"""Interactive wrapper for ``prepare_dataset.py`` using WhisperX.

This script lists audio files found in ``source_audio`` and lets the user
select one to transcribe. The resulting dataset is saved under ``datasets``
with a folder name matching the audio file stem.
"""
import os
from pathlib import Path

# Import the helper from the existing script
from prepare_dataset import prepare_dataset


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    audio_dir = repo_root / "source_audio"
    dataset_root = repo_root / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    audio_files = [f for f in os.listdir(audio_dir)
                   if f.lower().endswith((".mp3", ".wav"))]
    if not audio_files:
        print("No audio files found in 'source_audio'.")
        return

    print("Select the audio file(s) to process (comma separated numbers):")
    for idx, name in enumerate(audio_files, 1):
        print(f"{idx}. {name}")

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

    for idx in indices:
        selected = audio_files[idx - 1]
        audio_path = audio_dir / selected
        output_dir = dataset_root / Path(selected).stem
        prepare_dataset(str(audio_path), str(output_dir))

        print(f"Dataset directory: {output_dir.resolve()}")
        print(f"Parquet file: {(output_dir / 'dataset.parquet').resolve()}")


if __name__ == "__main__":
    main()
