# OrpheusX TTS Scripts

## Installation

Run the installation script to fetch all dependencies. It creates a virtual environment named `venv` in the repository root and installs PyTorch 2.6.0 built for CUDA 12.4 automatically. The script also warns if your CUDA runtime is newer.  
It installs WhisperX and its requirements (`librosa`, `soundfile` and the `ffmpeg` binary) so you can generate datasets from audio files directly.

```bash
bash scripts/install.sh
```

After installation activate the environment:

```bash
source venv/bin/activate
```

**Important:** OrpheusX only supports CUDA 12.4 or lower. Using a newer CUDA runtime may cause installation failures.

## Verify your environment

Use the environment check script to confirm that all required packages are available and that CUDA is detected.

```bash
python scripts/check_env.py
```

## CLI menu

You can manage all tasks using the interactive command line menu:

```bash
python scripts/orpheus_cli.py
```

The CLI lets you run the installer, create WhisperX datasets, train new LoRA models and launch inference. Each option can loop over multiple datasets or models if desired.

## Training

Execute the training script to download the dataset, preprocess it and start training. Models and datasets are cached under `models/` and `datasets/` in the repository root. The script lets you choose one or more datasets and trains a separate LoRA adapter for each one.

```bash
python scripts/train_interactive.py
```

Training settings mirror those found in the original notebook (60 steps, LoRA adapters etc.). The resulting LoRA weights will be written under `lora_models/<name>/lora_model/`.

## Inference

Run interactive inference to generate audio from custom text.

```bash
python scripts/infer_interactive.py
```

The script prompts for text and saves the resulting audio under `audio_output/<lora_name>/` using incrementing file names so previous results are kept.

## Interactive scripts

For convenience, `train_interactive.py` and `infer_interactive.py` provide an interactive workflow. Both support selecting multiple datasets or LoRA models which are processed sequentially. Generated audio is saved under `audio_output/<lora_name>` without overwriting existing files. To create datasets interactively you can use `prepare_dataset_interactive.py`, which lists audio files from the `source_audio` folder and builds a WhisperX dataset for each selection.

## Preparing datasets with Whisper

The `Whisper` directory contains tools to convert long audio recordings into a Hugging Face style dataset. Use `prepare_dataset.py` to create a local dataset from an audio file:

```bash
python scripts/prepare_dataset.py path/to/audio.mp3 datasets/my_dataset
```

The script runs WhisperX to transcribe and segment the audio, then saves a dataset under `datasets/my_dataset`. A copy of the dataset is also written to `datasets/my_dataset/dataset.parquet`. This Parquet file embeds the raw audio so it can be shared or loaded directly without the accompanying `.wav` files. You can load this dataset in the interactive training script by choosing the *Local Whisper dataset* option and providing the saved folder path.
