# 🚀 OrpheusX – Speech-to-Speech Fine-Tuning Toolkit

OrpheusX is a powerful tool for creating, training, and running custom speech-to-speech (STS) or text-to-speech (TTS) models. It includes an interactive CLI to guide you through every step.

---

## 🛠️ System Requirements

- **Python** ≥ 3.10    
- **CUDA Toolkit**: **version 12.4 only**  
  ⚠️ *Using a newer CUDA version may cause installation issues.*

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/EQXai/OrpheusX.git
cd OrpheusX
```

Installation script:

```bash
bash scripts/install.sh
```

Activate venv::

```bash
source venv/bin/activate
```

Everything else (installation, dataset creation, training, inference) is handled interactively through the CLI.

To start:

## 🌐 Gradio Interface

If you prefer a simple web UI instead of the CLI, run:

```bash
python gradio_app.py
```

## 🌐 Terminal CLI

```bash
python scripts/orpheus_cli.py
```

# 📁 Where to place the audio dataset:

##  Audio Organization

- Input audio files should be placed in the folder: `source_audio/`

#  Output Directory:

- The output of the processed audio files can be found in: `scripts/audio_output/`
  
---

## 🧩 Features

- Install dependencies
- Create WhisperX datasets with segments between 15 and 25 seconds long
- Train LoRA models
- Run inference
- Each audio clip aligns exactly with its transcript while staying in the 15–25 second range.
- Datasets are saved under `datasets/<name>` and include a standalone `dataset.parquet` file
- Batch options let you process multiple datasets or LoRAs in one go
- Training can load a local dataset or a Hugging Face link
- Inference results are written to `audio_output/<lora_name>/` with incrementing filenames
- Prompt lists in `prompt_list/` can be loaded for batch generation
- Use `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>` for preset expressions
- When prompts are segmented, pieces are joined with a short crossfade for smooth audio
- The crossfade defaults to **60 ms** and can be adjusted via the `--fade_ms` argument or the UI slider
- Verify your environment with `python scripts/check_env.py` (checks packages, CUDA and ffmpeg)
- The dataset helper lists audio files from `source_audio/` so you can pick them interactively
- Segmentation mode prints the start and end index of each chunk so you know where text was split
- Trained LoRA adapters are stored under `lora_models/<dataset>/lora_model`
- Detailed logs for the CLI and Gradio interface are saved under `logs/orpheus.log`
- The web UI includes **Cancel Task** and **Exit UI** buttons for quick control

All features are available via an interactive command-line menu.

---

The script launches on port 18188 by default.
The web UI lets you prepare datasets, train LoRAs and run inference.
Training and inference tabs include dropdowns listing local datasets or
available LoRA models and can also load prompt lists from `prompt_list/`.

The "Max New Tokens" setting defaults to 1200. The model has a 2048 token
context limit, so the sum of prompt tokens and new tokens should not exceed
this value.

The interface offers two presets under "Advanced Settings": **Short Audio**
and **Long Audio**. The short preset keeps segmentation disabled and limits
new tokens to 1200, while the long preset enables sentence segmentation and
allows up to 2400 new tokens.

### Prompt Segmentation

Long prompts can be split automatically into sentence-based chunks so that very
long texts can be processed in stages.  The `--segment` option in the CLI (and
the checkbox in the web UI) enables this behavior.  Text is divided at sentence
boundaries and grouped into pieces of roughly a configurable number of
characters (300 by default).  Each chunk ends with punctuation, which keeps the
generated speech natural when the parts are reassembled.  You can adjust the
maximum chunk length to trade off quality for speed.


While generating audio, the CLI prints a segmentation log showing where each
chunk starts and ends. After all segments are generated they are automatically
crossfaded to hide the cuts. The default fade length is 60 ms.

### Parallel Generation

Long-form synthesis can be accelerated by processing segments concurrently.
The `scripts/infer.py` tool accepts `--parallel` to enable this mode and
`--batch_size` to control how many chunks run at once. The Gradio web UI
exposes the same setting in Advanced Settings via a checkbox and batch size
input. Parallel mode reduces overall latency on capable GPUs when working with
very long prompts.

