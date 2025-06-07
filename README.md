# OrpheusX Scripts

## Installation

Run the installation script to fetch all dependencies. It creates a virtual environment named `venv` in the repository root and installs PyTorch 2.6.0 built for CUDA 12.4 automatically. The script also warns if your CUDA runtime is newer.  
It installs WhisperX and its requirements (`librosa`, `soundfile` and the `ffmpeg` binary) so you can generate datasets from audio files directly.

**Important:** OrpheusX only supports CUDA 12.4 or lower. Using a newer CUDA runtime may cause installation failures.

## CLI menu

You can manage all tasks using the interactive command line menu:

```bash
python scripts/orpheus_cli.py
```

The CLI lets you run the installer, create WhisperX datasets, train new LoRA models and launch inference. Each option can loop over multiple datasets or models if desired.
