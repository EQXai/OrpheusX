# 🚀 OrpheusX – Speech-to-Speech Fine-Tuning Toolkit

OrpheusX is a powerful tool for creating, training, and running custom text-to-speech (TTS). It includes an interactive Gradio to guide you through every step.

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

To start:

## 🌐 Gradio Interface


```bash
python gradio_app.py
```


# 📁 Where to place the audio dataset:

##  Audio Organization

- Input audio files should be placed in the folder: `source_audio/`

#  Output Directory:

- The output of the processed audio files can be found in: `scripts/audio_output/`
  
---
