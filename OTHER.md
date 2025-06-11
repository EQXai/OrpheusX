# Comprehensive Repository Documentation

This document provides a complete overview of the repository, including the directory structure, dependencies, and explanations for each code file. The full source code for every script is included with additional comments explaining functionality.

---

## Directory Structure

```
.git/FETCH_HEAD
.git/HEAD
.git/config
.git/description
.git/index
.git/packed-refs
.gitignore
FULL_REPO_DOCUMENTATION.md
README.md
__pycache__/orpheus.cpython-312.pyc
__pycache__/orpheus_wrapper.cpython-312.pyc
emotions.txt
finetune/config.yaml
finetune/train.py
orpheus.py
orpheus_wrapper.py
pretrain/config.yaml
pretrain/readme.md
pretrain/train.py
realtime_streaming_example/client.html
realtime_streaming_example/main.py
setup_orpheus.sh
```

---

## Dependencies

Key dependencies used in this repository include:

- **gradio** for the web UI
- **orpheus-speech** containing the core TTS model
- **vllm** for optimized model execution
- **torch** for tensor operations
- **flask** for the streaming example
- **huggingface_hub** for model downloads
- **wandb** for experiment tracking

Ensure these packages are installed when running the scripts.

---

## Source Code Overview
Each code file from the repository is included below. All functions contain explanatory docstrings and inline comments that describe their behavior.

### orpheus.py
This script defines the Gradio WebUI for the Orpheus text-to-speech model.
```python
import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import time
import os
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
model = None
MODEL_SAMPLE_RATE = 24000

model_path = None # Set this to your local model path if needed
model_name = model_path if model_path else "canopylabs/orpheus-tts-0.1-finetune-prod"

def load_model(model_name=model_name):
    """Load the Orpheus TTS model."""
    global model
    try:
        logger.info(f"Loading model from: {model_name}")
        model = OrpheusModel(model_name=model_name)
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def generate_speech(prompt, voice, temperature, top_p, repetition_penalty, max_tokens):
    """Generate speech for a single text input."""
    if model is None:
        load_model()
    
    # Start timing
    start_time = time.monotonic()
    
    # Generate speech from the provided text
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens
    )
    
    # Create a unique output filename to avoid overwriting previous generations
    output_filename = f"output_{int(time.time())}.wav"
    
    # Write the audio to a WAV file
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MODEL_SAMPLE_RATE)
        
        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        
        duration = total_frames / wf.getframerate()
    
    processing_time = time.monotonic() - start_time
    result_message = f"Generated {duration:.2f} seconds of audio in {processing_time:.2f} seconds"
    logger.info(result_message)

    return output_filename, result_message

def chunk_text(text, max_chunk_size=300):
    """Split text into smaller chunks at sentence boundaries."""
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Split on sentence delimiters while preserving the delimiter
    delimiter_pattern = r'(?<=[.!?])\s+'
    segments = re.split(delimiter_pattern, text)

    # Process segments to ensure each has appropriate ending punctuation
    sentences = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Check if segment already ends with a delimiter
        if not segment[-1] in ['.', '!', '?']:
            segment += '.'

        sentences.append(segment)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would make the chunk too long, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(current_chunk)

    logger.info(f"Text chunked into {len(chunks)} segments")
    return chunks

async def process_chunk(chunk, voice, temperature, top_p, repetition_penalty, max_tokens, temp_dir, current_idx, total_chunks):
    """Process a single chunk asynchronously."""
    # Run the model inference in a separate thread since it's blocking
    loop = asyncio.get_event_loop()

    def generate_for_chunk():
        return model.generate_speech(
            prompt=chunk,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens
        )

    # Execute the model inference (this runs in a thread)
    syn_tokens = await loop.run_in_executor(None, generate_for_chunk)

    # Create a filename for this chunk
    chunk_filename = os.path.join(temp_dir, f"chunk_{current_idx}.wav")

    # Write the audio to a WAV file
    with wave.open(chunk_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MODEL_SAMPLE_RATE)

        chunk_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            chunk_frames += frame_count
            wf.writeframes(audio_chunk)

        chunk_duration = chunk_frames / wf.getframerate()

    return chunk_filename, chunk_duration

async def generate_long_form_speech_async(long_text, voice, temperature, top_p, repetition_penalty,
                                         batch_size=4, max_tokens=4096, progress=None):
    """Async version of generate_long_form_speech."""
    start_time = time.monotonic()
    if progress is not None:
        progress(0, desc="Preparing text chunks")

    # Chunk the text
    chunks = chunk_text(long_text)
    if progress is not None:
        progress(0.1, desc=f"Text split into {len(chunks)} chunks")

    # Create a directory for batch files
    temp_dir = f"longform_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Created temp directory: {temp_dir}")

    # Use a semaphore to limit concurrent processing to batch_size
    semaphore = asyncio.Semaphore(batch_size)
    total_chunks = len(chunks)
    all_audio_files = []
    total_duration = 0
    processed_chunks = 0

    async def process_chunk_with_semaphore(chunk, idx):
        nonlocal processed_chunks
        async with semaphore:
            try:
                filename, duration = await process_chunk(
                    chunk, voice, temperature, top_p, repetition_penalty,
                    max_tokens, temp_dir, idx, total_chunks
                )
                processed_chunks += 1
                if progress is not None:
                    progress(processed_chunks / total_chunks,
                        desc=f"Processed chunk {processed_chunks}/{total_chunks}")
                return filename, duration
            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {str(e)}")
                raise  # Re-raise to be caught by gather

    # Create tasks for ALL chunks and process them concurrently with semaphore limiting parallelism
    tasks = [process_chunk_with_semaphore(chunk, idx) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    # Process results
    for filename, duration in results:
        all_audio_files.append(filename)
        total_duration += duration
    # Combine all audio files
    if progress is not None:
        progress(0.9, desc="Combining audio files")

    combined_filename = f"longform_output_{int(time.time())}.wav"
    logger.info(f"Combining {len(all_audio_files)} audio chunks into {combined_filename}")

    # Use a simple concatenation approach
    data = []
    for file in sorted(all_audio_files, key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])):
        with wave.open(file, 'rb') as w:
            data.append([w.getparams(), w.readframes(w.getnframes())])

    with wave.open(combined_filename, 'wb') as output:
        if data:
            output.setparams(data[0][0])
            for i in range(len(data)):
                output.writeframes(data[i][1])

    # Clean up temporary files
    for file in all_audio_files:
        try:
            os.remove(file)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {file}: {e}")

    try:
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to delete temp directory {temp_dir}: {e}")

    # Calculate processing time
    processing_time = time.monotonic() - start_time
    result_message = f"Generated {total_duration:.2f} seconds of audio from {total_chunks} chunks in {processing_time:.2f} seconds"
    logger.info(result_message)
    
    if progress is not None:
        progress(1.0, desc="Complete")

    return combined_filename, result_message

def generate_long_form_speech(long_text, voice, temperature, top_p, repetition_penalty, batch_size=4, max_tokens=4096, progress=gr.Progress()):
    """Generate speech for long-form text by chunking and processing in parallel batches."""
    if model is None:
        load_model()

    # Use asyncio to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        print("Running long form speech generation")
        return loop.run_until_complete(
            generate_long_form_speech_async(
                long_text, voice, temperature, top_p,
                repetition_penalty, batch_size, max_tokens, progress
            )
        )
    finally:
        loop.close()

def cleanup_files():
    """Clean up generated audio files."""
    count = 0
    for file in os.listdir():
        if (file.startswith("output_") or file.startswith("longform_output_")) and file.endswith(".wav"):
            try:
                os.remove(file)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {file}: {e}")

    # Also clean up any leftover temporary directories
    for dir_name in os.listdir():
        if dir_name.startswith("longform_") and os.path.isdir(dir_name):
            try:
                # Remove any files inside
                for file in os.listdir(dir_name):
                    os.remove(os.path.join(dir_name, file))
                os.rmdir(dir_name)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete directory {dir_name}: {e}")

    logger.info(f"Cleanup completed. Removed {count} files/directories.")

# Create the Gradio interface
def create_ui():
    """Create the Gradio user interface."""
    with gr.Blocks(title="OrpheusTTS-WebUI", theme=gr.themes.Default()) as demo:
        # Title and description
        gr.Markdown("<div align='center'><h1>OrpheusTTS-WebUI</h1></div>")

        gr.Markdown("""<div align='center'>Generate realistic speech from text using the OrpheusTTS model.
**Available voices:** tara, jess, leo, leah, dan, mia, zac, zoe (in order of conversational realism)

**Available emotive tags:** `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

**Note:** Increasing repetition_penalty and temperature makes the model speak faster. Increasing Max Tokens extends the maximum duration of genrated audio.
</div>
        """)

        # Create tabs container
        with gr.Tabs(selected=0) as tabs:
            # Tab 1: Single Text Generation
            with gr.Tab("Single Text"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Text input area
                        prompt = gr.Textbox(
                            label="Text Input",
                            placeholder="Enter text to convert to speech...",
                            lines=3
                        )

                        with gr.Row():
                            voice = gr.Dropdown(
                                choices=["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"],
                                label="Voice",
                                value="tara"
                            )

                        with gr.Row():
                            max_tokens = gr.Slider(
                                label="Max Tokens",
                                value=2048,
                                minimum=128,
                                maximum=16384,
                                step=128
                            )

                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Temperature"
                            )
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top P"
                            )
                            rep_penalty = gr.Slider(
                                minimum=1.1,
                                maximum=2.0,
                                value=1.2,
                                step=0.1,
                                label="Repetition Penalty"
                            )
                        max_tokens = gr.Slider(
                    minimum=1200,
                    maximum=3600,
                    value=1200,
                    step=100,
                    label="Max Tokens"
                )

                        submit_btn = gr.Button("Generate Speech", variant="primary")
                        gr.Examples(
                            examples=[
                                "Man, the way social media has, um, completely changed how we interact is just wild, right?",
                                "I just got back from my vacation <sigh> and I'm already feeling stressed about work.",
                                "Did you hear what happened at the party last night? <laugh> It was absolutely ridiculous!",
                                "I've been working on this project for hours <yawn> and I still have so much to do.",
                                "The concert was amazing! <gasp> You should have seen the light show!"
                            ],
                            inputs=prompt,
                            label="Example Prompts"
                        )

                    with gr.Column(scale=1):
                        audio_output = gr.Audio(label="Generated Speech")
                        result_text = gr.Textbox(label="Generation Stats", interactive=False)

                # Connect the generate_speech function to the interface
                submit_btn.click(
                    fn=generate_speech,
                    inputs=[prompt, voice, temperature, top_p, rep_penalty, max_tokens],
                    outputs=[audio_output, result_text]
                )

            # Tab 2: Long Form Content
            with gr.Tab("Long Form Content"):
                with gr.Row():
                    with gr.Column(scale=2):
                        long_form_prompt = gr.Textbox(
                            label="Long Form Text Input",
                            placeholder="Enter long text to convert to speech...",
                            lines=15
                        )

                        with gr.Row():
                            lf_voice = gr.Dropdown(
                                choices=["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"],
                                label="Voice",
                                value="tara"
                            )

                        with gr.Row():
                            lf_max_tokens = gr.Slider(
                                label="Max Tokens",
                                value=4096,
                                minimum=128,
                                maximum=16384,
                                step=128
                            )

                            lf_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.6,
                                step=0.1,
                                label="Temperature"
                            )

                            lf_top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.8,
                                step=0.05,
                                label="Top P"
                            )

                            lf_rep_penalty = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.1,
                                step=0.1,
                                label="Repetition Penalty"
                            )

                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=1,
                                label="Batch Size (chunks processed in parallel)"
                            )

                        lf_submit_btn = gr.Button("Generate Long Form Speech", variant="primary")

                        gr.Examples(
                            examples=[
                                """How Long Form Processing Works:
Text is automatically split into chunks at sentence boundaries.
Chunks are processed in batches based on the batch size.
Higher batch sizes may be faster but require more memory.
Finally, all chunks are combined into a single audio file.
""",
                                """I just got back from my vacation <sigh> and I'm already feeling stressed about work.
Did you hear what happened at the party last night? <laugh> It was absolutely ridiculous!
I've been working on this project for hours <yawn> and I still have so much to do.
The concert was amazing! You should have seen the light show!
"""
                            ],
                            inputs=long_form_prompt,
                            label="Example Prompts"
                        )

                        
                    with gr.Column(scale=1):
                        lf_audio_output = gr.Audio(label="Generated Long Form Speech")
                        lf_result_text = gr.Textbox(label="Generation Stats", interactive=False)

                # Connect the long form generation function
                lf_submit_btn.click(
                    fn=generate_long_form_speech,
                    inputs=[long_form_prompt, lf_voice, lf_temperature, lf_top_p, lf_rep_penalty, batch_size, lf_max_tokens],
                    outputs=[lf_audio_output, lf_result_text]
                )

        # Add footer with links
        gr.Markdown("""<div align='center' style='margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;'>
    <a href="https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained" target="_blank">Hugging Face</a> |
    <a href="https://github.com/Saganaki22/OrpheusTTS-WebUI" target="_blank">WebUI GitHub</a> |
    <a href="https://github.com/canopyai/Orpheus-TTS" target="_blank">Official GitHub</a>
    </div>""")

        # Register cleanup for when the interface closes
        demo.load(cleanup_files)

    return demo

# Main function to run the app
if __name__ == "__main__":
    # Initialize the app
    logger.info("Starting OrpheusTTS-WebUI")

    # Create and launch the UI
    demo = create_ui()
    demo.launch(share=False)  # Set share=False to disable public URL
```

### orpheus_wrapper.py
Wrapper that sets environment variables and launches the main UI.
```python
#!/usr/bin/env python3
"""
Wrapper script for Orpheus TTS to enforce vLLM configuration.
"""
import os
import sys
import logging

# Set environment variables to control vLLM
os.environ["VLLM_MAX_MODEL_LEN"] = "100000"
os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.9"
os.environ["VLLM_DISABLE_LOGGING"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"

try:
    # Import the necessary modules
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from orpheus_tts.engine_class import OrpheusModel
    
    # Store the original from_engine_args method
    original_from_engine_args = AsyncLLMEngine.from_engine_args
    
    # Define a patched version that doesn't use disable_log_requests
    def patched_from_engine_args(engine_args, **kwargs):
        # Override the max_model_len in engine_args
        engine_args.max_model_len = 100000
        engine_args.gpu_memory_utilization = 0.9
        
        print(f"Patched from_engine_args called with max_model_len={engine_args.max_model_len}")
        
        # Call the original without any extra kwargs
        return original_from_engine_args(engine_args)
    
    # Replace the class method
    AsyncLLMEngine.from_engine_args = staticmethod(patched_from_engine_args)
    print("Successfully patched AsyncLLMEngine.from_engine_args")
    
except Exception as e:
    print(f"Warning: Failed to patch AsyncLLMEngine: {e}")

# Now import and run the Orpheus app
print("Starting Orpheus TTS...")

# Import the Gradio app
import orpheus

# Actually run the Gradio app
if __name__ == "__main__":
    demo = orpheus.create_ui()
    demo.launch(share=False)
```

### emotions.txt
List of available emotion tags supported by the model.
```text
happy
normal
digust
disgust
longer
sad
frustrated
slow
excited
whisper
panicky
curious
surprise
fast
crying
deep
sleepy
angry
high
shout
```

### finetune/train.py
Training script for fine-tuning the TTS model on your dataset.
```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")


ds = load_dataset(dsn, split="train") 

wandb.init(project=project_name, name = run_name)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    learning_rate=learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()

```

### pretrain/train.py
Multinode pretraining script mixing text and speech data.
```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml
import wandb
from huggingface_hub import HfApi

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn1 = config["text_QA_dataset"]
dsn2 = config["TTS_dataset"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
config_ratio = config["ratio"]




class BatchedRatioDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total, ratio=config_ratio):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.ratio = ratio  

        num_cycles_ds1 = len(dataset1) // (batch_total * ratio)
        num_cycles_ds2 = len(dataset2) // batch_total
        self.num_cycles = min(num_cycles_ds1, num_cycles_ds2)

        self.length = self.num_cycles * (ratio + 1) * batch_total

    def __len__(self):
        print("accessing length", self.length)
        return int(self.length)

    def __getitem__(self, index):
        # Compute the cycle length in terms of samples.
        cycle_length = (self.ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos_in_cycle = index % cycle_length

        if pos_in_cycle < self.ratio * self.batch_total:
            batch_in_cycle = pos_in_cycle // self.batch_total
            sample_in_batch = pos_in_cycle % self.batch_total
            ds1_index = cycle * self.ratio * self.batch_total + batch_in_cycle * self.batch_total + sample_in_batch
            return self.dataset1[ds1_index]
        else:
            # We are in the dataset2 batch for this cycle.
            sample_in_batch = pos_in_cycle - self.ratio * self.batch_total
            ds2_index = cycle * self.batch_total + sample_in_batch
            return self.dataset2[ds2_index]



class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class FSDPTrainer(Trainer):
    def __init__(self, *args, log_ratio=config_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_id = base_repo_id
        self.api = HfApi()

        self.log_ratio = log_ratio
        self.text_step  = 0
        self.audio_step = 0

    def get_train_dataloader(self):
        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        if self.is_world_process_zero():
            global_step = self.state.global_step
            # Each cycle is (log_ratio + 1) steps: first log_ratio steps for text_loss, then one for audio_loss.
            cycle_length = self.log_ratio + 1
            if (global_step % cycle_length) + self.log_ratio - 1 < self.log_ratio:
                wandb.log({"audio_loss": logs["loss"], "audio_step": self.audio_step})
                self.audio_step += 1
            else:
                wandb.log({"text_loss": logs["loss"], "text_step": self.text_step})
                self.text_step += 1

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.save_and_push_model(output_dir)

    def save_and_push_model(self, output_dir):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)


def data_collator(features):
    # max_length = 2656 # set a crop based on vram - ideally you have stacked all sequences to the same length
    # from 3b on 8 h100s fsdp, at bf16, 8192 works well.
    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


wandb.init(project=project_name, name=run_name)


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, attn_implementation="flash_attention_2")


number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))


ds1 = load_dataset(dsn1, split="train")
ds2 = load_dataset(dsn2, split="train")


batch_total = batch_size * number_processes
train_dataset = BatchedRatioDataset(ds1, ds2, batch_total, ratio=config_ratio)


training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine", 
)


trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    log_ratio=config_ratio
)

trainer.train()
```

### pretrain/readme.md
Notes describing the pretraining approach for Orpheus.
```markdown
# Pretraining
## Overview
We find that trying to keep good semantic understanding of text boosts the models ability when speaking naturally and empathetically. We propose training the model on batches of speech and text. If you want the model to retain a large part of its text ability - i.e. function as an end-to-end speech model you should keep the ratio of text batch :speech batch as 2:1 to start and then gradually decrease to 1:1 throughout training. If your model is just trained for TTS start with 1:1 and gradually decrease to 0:1.


### Disclaimer

This code was copy and pasted into this repo quickly so there maybe bugs. The general outline should be pretty straightforward. It's also set up for multinode training.

Depending on how good the models reasoning abilities to be (and what specifically you want to retain), you can choose with text-based dataset you use. Using simple datasets with QA pairs (for finetuning like ) works pretty well. You can also try using wikipedia - to boost the 
```

### realtime_streaming_example/main.py
Simple Flask server demonstrating streaming synthesis.
```python
from flask import Flask, Response, request
import struct
from orpheus_tts import OrpheusModel

app = Flask(__name__)
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,             
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')

    def generate_audio_stream():
        yield create_wav_header()

        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=2000,
            temperature=0.4,
            top_p=0.9
        )
        for chunk in syn_tokens:
            yield chunk

    return Response(generate_audio_stream(), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
```

### realtime_streaming_example/client.html
Browser client for the streaming example.
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Streaming Audio Playback</title>
</head>
<body>
  <h1>Streaming Audio Playback</h1>
  <form id="promptForm">
    <label for="promptInput">Enter Prompt:</label><br>
    <textarea id="promptInput" rows="4" cols="50" placeholder="Type your prompt here" required></textarea><br>
    <button type="submit">Play Audio</button>
  </form>
  <audio id="audioPlayer" controls autoplay></audio>
  <script>
    const base_url = `https://<enter-hostname-here>`;

    document.getElementById("promptForm").addEventListener("submit", function(event) {
      event.preventDefault();
      const prompt = document.getElementById("promptInput").value;
      const encodedPrompt = encodeURIComponent(prompt);
      const audioUrl = `${base_url}/tts?prompt=` + encodedPrompt;
      
      // Set the audio element's src to your endpoint to stream and play the audio data
      const audioPlayer = document.getElementById("audioPlayer");
      audioPlayer.src = audioUrl;
      audioPlayer.load();
      audioPlayer.play().catch(err => console.error("Playback error:", err));
    });
  </script>
</body>
</html>
```

### setup_orpheus.sh
Setup script for creating a virtual environment and launching the app.
```bash
#!/bin/bash
set -e  # Exit on error

echo "======================================="
echo "OrpheusTTS-WebUI Setup Script"
echo "======================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip

# Install other required packages
echo "Installing other dependencies..."
pip install orpheus-speech gradio vllm torch huggingface_hub

# Create launch script
echo "Creating launch script..."
cat > launch_orpheus.sh << 'EOF'
#!/bin/bash
set -e

# Activate virtual environment
source venv/bin/activate

# Execute the wrapper script
python orpheus_wrapper.py
EOF

# Make the launch script executable
chmod +x launch_orpheus.sh

# Log in to Hugging Face
echo "======================================="
echo "You need to log in to Hugging Face to access the model."
echo "If you don't have an account, create one at https://huggingface.co/join"
echo "======================================="
read -p "Press Enter to continue to Hugging Face login..."
huggingface-cli login

# Remind about model access
echo "======================================="
echo "IMPORTANT: The OrpheusTTS model is a gated model."
echo "You need to request access at:"
echo "https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod"
echo "https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained"
echo "======================================="
echo "Once approved, you'll be able to use the model."
echo "======================================="

# Make the wrapper executable
echo "Making the wrapper script executable..."
chmod +x orpheus_wrapper.py

echo "Setup complete! Run ./launch_orpheus.sh to start the application."
echo "=======================================" 
```

### README.md
Primary project README describing features and usage.
```markdown
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
```

### finetune/config.yaml
Example configuration for the fine-tuning script.
Example configuration for the fine-tuning script.
```yaml
# CHANGE THIS TO YOUR OWN DATASET
TTS_dataset: <PATH_TO_YOUR_DATASET>

model_name: "canopylabs/orpheus-tts-0.1-pretrained"

# Training Args
epochs: 1
batch_size: 1
number_processes: 1
pad_token: 128263
save_steps: 5000
learning_rate: 5.0e-5

# Naming and paths
save_folder: "checkpoints"
project_name: "tuning-orpheus"
run_name: "5e5-0"
```

### pretrain/config.yaml
Example configuration for the pretraining script.
```yaml
# Model
model_name: "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your base model must be compatible with the tokenizer and transformers library
tokenizer_name: "meta-llama/Llama-3.2-3B-Instruct"

# Training Args
epochs: 1
batch_size: 1
number_processes: 8
pad_token: 128263
save_steps: 12000
learning_rate: 5.0e-5
ratio: <see read me to choose value>

# Datasets
text_QA_dataset: <speech input-ids>
TTS_dataset: <text-input-ids>

# Naming and paths
save_folder: "checkpoints"
project_name: "pretrain-orpheus"
run_name: "pretrain-orpheus"
```
