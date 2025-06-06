# 🧠 Audio + Text Dataset Generation for Hugging Face

This repository contains two scripts to automate:

1. **Transcribing audio using WhisperX and splitting it into short segments.**
2. **Uploading the resulting `.wav` + `.txt` pairs as a dataset to the Hugging Face Hub.**

---

## 📁 File Structure

- `run.py`: Transcribes an audio file and splits it into 10–15 second segments with corresponding text.
- `upload.py`: Loads the audio/text pairs as a Hugging Face `Dataset` and uploads it to the Hub.

---

## ✅ Requirements

Install necessary dependencies:

```bash
pip install transformers librosa soundfile datasets
sudo apt install ffmpeg
```

Also make sure you have `whisperx` installed and available in the terminal.

---

## 🔧 Usage

### 1. Transcribe and segment audio (`run.py`)

```bash
python run.py path/to/audio.mp3
```

This will produce:

- A `.json` transcription file via WhisperX.
- `.wav` clips inside `recortes/<base_name>/`
- `.txt` transcripts matching each audio clip.

**Example output:**
```
recortes/
└── audio.mp3/
    ├── audio_001.wav
    ├── audio_001.txt
    ├── audio_002.wav
    ├── audio_002.txt
    └── ...
```

---

### 2. Upload to Hugging Face (`upload.py`)

```bash
python upload.py recortes/audio.mp3 --repo_name your_user/dataset_name --token hf_xxx
```

- If `--repo_name` or `--token` are not provided, you'll be prompted to enter them interactively.
- The result is uploaded as a Hugging Face `Dataset` in audio-text format.

---

## 📝 Technical Notes

- Segments are capped at **50 tokens**, with lengths between **10 and 15 seconds**.
- Final sample rate is **24 kHz**, suitable for models like CSM / Orpheus.

---

## 🔒 Security

- The token is optional if you've already run `huggingface-cli login`.
- If entered manually, it will be hidden using `getpass`.

---

## 🧪 Debugging

Both scripts print `DEBUG:` messages throughout to help trace execution flow.

