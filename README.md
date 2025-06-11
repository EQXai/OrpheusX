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
python scripts/install.sh
```

Activate venv::

```bash
source venv/bin/activate
```

Everything else (installation, dataset creation, training, inference) is handled interactively through the CLI.

To start:

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
- Train GRPO LoRA models using `scripts/train_grpo.py`
- GRPO dependencies are included in `requirements.txt`
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

All features are available via an interactive command-line menu.

---

## 🌐 Gradio Interface (UNDER DEVELOPMENT)

If you prefer a simple web UI instead of the CLI, run:

```bash
python gradio_app.py
```
The script will ask which port you want to use before launching.
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

Long prompts can be split automatically during inference to avoid hitting the
token limit. Use `--segment` along with `--segment-by tokens` (default),
`--segment-by sentence`, or `--segment-by full_segment` to control how text is
chunked. Sentence segmentation usually produces smoother audio because pauses
occur at natural boundaries. The `full_segment` mode splits on every comma,
period, question mark or exclamation point, producing the smallest possible
chunks and ignoring the token count settings. When splitting by sentences,
commas are also considered separators. The
algorithm ignores consecutive commas and merges pieces shorter than three words
with their neighbors so lists or short phrases aren't broken awkwardly. Enable
sentence segmentation for long prompts with natural pause points.

The Gradio interface lets you choose which punctuation characters trigger
segmentation (comma, period, question mark and exclamation point) and specify a
minimum and maximum token count for each segment. The limits are ignored when
`full_segment` is selected.

There is also a **Full Segment Test** tab in the web UI. This simplified
interface exposes only character-based segmentation so you can verify how the
`full_segment` mode splits prompts without any token settings.

While generating audio, the CLI prints a segmentation log showing where each
chunk starts and ends. After all segments are generated they are automatically
crossfaded to hide the cuts. The default fade length is 60 ms.

