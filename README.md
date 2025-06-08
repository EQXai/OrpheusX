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
- Run inference
- Each audio clip aligns exactly with its transcript while staying in the 15–25 second range.

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

### Prompt Segmentation

Long prompts can be split automatically during inference to avoid hitting the
token limit. Use `--segment` along with `--segment-by tokens` (default) or
`--segment-by sentence` to control how text is chunked. Sentence segmentation
usually produces smoother audio because pauses occur at natural boundaries.
When splitting by sentences, commas are also considered separators. The
algorithm ignores consecutive commas and merges pieces shorter than three words
with their neighbors so lists or short phrases aren't broken awkwardly. Enable
sentence segmentation for long prompts with natural pause points.

