# OrpheusTTS-WebUI

This is a fork of the [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) project, adding a Gradio WebUI that runs smoothly on WSL and CUDA.

![image](https://github.com/user-attachments/assets/4b738f1d-23ed-477b-ac84-db0d5b04c76c)

https://github.com/user-attachments/assets/5e441285-b10f-4149-b691-df061c5ddcbb

## ✅ Latest Updates (20/03/2025)

### Long-Form Text Processing
- **Tabbed Interface**: The UI now features a dedicated "Long Form Content" tab for processing larger text inputs
- **Smart Text Chunking**: Automatically splits long text into smaller chunks at sentence boundaries
- **Parallel Processing**: Processes multiple chunks simultaneously for faster generation
- **Seamless Audio Stitching**: Combines multiple audio segments into one cohesive output file
- **Progress Tracking**: Real-time progress indicators during the generation process

### Technical Improvements
- **Enhanced Logging**: Better error handling and diagnostic information
- **Memory Optimization**: Improved cleanup of temporary files
- **Expanded Parameter Ranges**: Maximum tokens extended to 16384 for longer audio generation
- **Batch Size Control**: Adjust the number of chunks processed in parallel to balance speed and resource usage

## Features

- **Easy-to-use Web Interface**: Simple Gradio UI for text-to-speech generation
- **WSL & CUDA Compatible**: Optimized for Windows Subsystem for Linux with CUDA support
- **Memory Optimized**: Addresses common memory issues on consumer GPUs
- **Voice Selection**: Access to all 8 voices from the original model
- **Emotive Tags Support**: Full support for all emotion tags

## Quick Start (WSL/Linux)

```bash
# Clone the repository
git clone https://github.com/Saganaki22/OrpheusTTS-WebUI.git
cd OrpheusTTS-WebUI

# Run the setup script
chmod +x setup_orpheus.sh
./setup_orpheus.sh

# Launch the app
./launch_orpheus.sh
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 3090 / 4090)
- WSL2 or Linux
- PyTorch 2.6.0 with CUDA
- Hugging Face account with access to the Orpheus TTS models

## Available Voices

The WebUI provides access to all 8 voices in order of conversational realism:
- tara
- jess
- leo
- leah
- dan
- mia
- zac
- zoe

## Emotive Tags

Add emotion to your speech with tags:
- `<laugh>`
- `<chuckle>`
- `<sigh>`
- `<cough>`
- `<sniffle>`
- `<groan>`
- `<yawn>`
- `<gasp>`

## Long Form Text Processing

The new Long Form feature lets you generate speech for larger text inputs:

1. **Text Chunking**: Text is automatically split into manageable chunks at sentence boundaries
2. **Parallel Processing**: Process multiple chunks simultaneously based on the batch size setting
3. **Parameter Optimization**: The Long Form tab offers optimized default settings for extended content
4. **Simple Assembly**: All audio chunks are automatically combined into a single cohesive output file

This is ideal for:
- Articles and blog posts
- Scripts and dialogues
- Books and stories
- Any text content that exceeds a few paragraphs

## Troubleshooting

If you encounter "KV cache" errors, the setup script should address these automatically. If problems persist, try:
- Reducing `max_model_len` in the `orpheus_wrapper.py` file
- Ensuring your GPU has enough VRAM (recommended 12GB+)
- Setting `gpu_memory_utilization` to a lower value (0.7-0.8)
- For Long Form processing, try reducing the batch size to limit memory usage

---

# Official Orpheus TTS Documentation

## Overview
Orpheus TTS is an open-source text-to-speech system built on the Llama-3b backbone. Orpheus demonstrates the emergent capabilities of using LLMs for speech synthesis. We offer comparisons of the models below to leading closed models like Eleven Labs and PlayHT in our blog post.

[Check out our blog post](https://canopylabs.ai/model-releases)


https://github.com/user-attachments/assets/ce17dd3a-f866-4e67-86e4-0025e6e87b8a


## Abilities

- **Human-Like Speech**: Natural intonation, emotion, and rhythm that is superior to SOTA closed source models
- **Zero-Shot Voice Cloning**: Clone voices without prior fine-tuning
- **Guided Emotion and Intonation**: Control speech and emotion characteristics with simple tags
- **Low Latency**: ~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming

## Models

We provide three models in this release, and additionally we offer the data processing scripts and sample datasets to make it very straightforward to create your own finetune.

1. [**Finetuned Prod**](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod) – A finetuned model for everyday TTS applications

2. [**Pretrained**](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) – Our base model trained on 100k+ hours of English speech data


### Inference
#### Simple setup on colab
1. [Colab For Tuned Model](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing) (not streaming, see below for realtime streaming) – A finetuned model for everyday TTS applications.
2. [Colab For Pretrained Model](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing) – This notebook is set up for conditioned generation but can be extended to a range of tasks.

#### Prompting

1. The `finetune-prod` models: for the primary model, your text prompt is formatted as `{name}: I went to the ...`. The options for name in order of conversational realism (subjective benchmarks) are "tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe". Our python package does this formatting for you, and the notebook also prepends the appropriate string. You can additionally add the following emotive tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`.

2. The pretrained model: you can either generate speech just conditioned on text, or generate speech conditioned on one or more existing text-speech pairs in the prompt. Since this model hasn't been explicitly trained on the zero-shot voice cloning objective, the more text-speech pairs you pass in the prompt, the more reliably it will generate in the correct voice.

Additionally, use regular LLM generation args like `temperature`, `top_p`, etc. as you expect for a regular LLM. `repetition_penalty>=1.1`is required for stable generations. Increasing `repetition_penalty` and `temperature` makes the model speak faster.


## Finetune Model

Here is an overview of how to finetune your model on any text and speech.
This is a very simple process analogous to tuning an LLM using Trainer and Transformers.

You should start to see high quality results after ~50 examples but for best results, aim for 300 examples/speaker.

1. Your dataset should be a huggingface dataset in [this format](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
2. We prepare the data using this [this notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing). This pushes an intermediate dataset to your Hugging Face account which you can can feed to the training script in finetune/train.py. Preprocessing should take less than 1 minute/thousand rows.
3. Modify the `finetune/config.yaml` file to include your dataset and training properties, and run the training script. You can additionally run any kind of huggingface compatible process like Lora to tune the model.
   ```bash
    pip install transformers datasets wandb trl flash_attn torch
    huggingface-cli login <enter your HF token>
    wandb login <wandb token>
    accelerate launch train.py
   ```

# Checklist

- [x] Release 3b pretrained model and finetuned models
- [ ] Release pretrained and finetuned models in sizes: 1b, 400m, 150m parameters
- [ ] Fix glitch in realtime streaming package that occasionally skips frames.
- [ ] Fix voice cloning Colab notebook implementation
