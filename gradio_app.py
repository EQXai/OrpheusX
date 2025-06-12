import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import time
import os
import logging
import re
import asyncio


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
model = None
MODEL_SAMPLE_RATE = 24000

model_path = None  # Set this to your local model path if needed
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

    start_time = time.monotonic()
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )
    output_filename = f"output_{int(time.time())}.wav"
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
    text = re.sub(r"\s+", " ", text)
    delimiter_pattern = r'(?<=[.!?])\s+'
    segments = re.split(delimiter_pattern, text)
    sentences = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if segment[-1] not in ['.', '!', '?']:
            segment += '.'
        sentences.append(segment)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    if current_chunk:
        chunks.append(current_chunk)
    logger.info(f"Text chunked into {len(chunks)} segments")
    return chunks


async def process_chunk(chunk, voice, temperature, top_p, repetition_penalty, max_tokens, temp_dir, current_idx, total_chunks):
    """Process a single chunk asynchronously."""
    loop = asyncio.get_event_loop()

    def generate_for_chunk():
        return model.generate_speech(
            prompt=chunk,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

    syn_tokens = await loop.run_in_executor(None, generate_for_chunk)
    chunk_filename = os.path.join(temp_dir, f"chunk_{current_idx}.wav")
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

    chunks = chunk_text(long_text)
    if progress is not None:
        progress(0.1, desc=f"Text split into {len(chunks)} chunks")

    temp_dir = f"longform_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Created temp directory: {temp_dir}")

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
                raise

    tasks = [process_chunk_with_semaphore(chunk, idx) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    for filename, duration in results:
        all_audio_files.append(filename)
        total_duration += duration
    if progress is not None:
        progress(0.9, desc="Combining audio files")

    combined_filename = f"longform_output_{int(time.time())}.wav"
    logger.info(f"Combining {len(all_audio_files)} audio chunks into {combined_filename}")

    data = []
    for file in sorted(all_audio_files, key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])):
        with wave.open(file, 'rb') as w:
            data.append([w.getparams(), w.readframes(w.getnframes())])

    with wave.open(combined_filename, 'wb') as output:
        if data:
            output.setparams(data[0][0])
            for i in range(len(data)):
                output.writeframes(data[i][1])

    for file in all_audio_files:

        try:
            os.remove(file)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {file}: {e}")

    try:
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to delete temp directory {temp_dir}: {e}")

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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            generate_long_form_speech_async(
                long_text, voice, temperature, top_p,
                repetition_penalty, batch_size, max_tokens, progress

            )
        )
    finally:
        loop.close()


def cleanup_files():
    count = 0
    for file in os.listdir():
        if (file.startswith("output_") or file.startswith("longform_output_")) and file.endswith(".wav"):
            try:
                os.remove(file)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {file}: {e}")

    for dir_name in os.listdir():
        if dir_name.startswith("longform_") and os.path.isdir(dir_name):
            try:
                for file in os.listdir(dir_name):
                    os.remove(os.path.join(dir_name, file))
                os.rmdir(dir_name)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete directory {dir_name}: {e}")

    logger.info(f"Cleanup completed. Removed {count} files/directories.")


def create_ui():
    """Create the Gradio user interface."""
    with gr.Blocks(title="OrpheusTTS-WebUI", theme=gr.themes.Default()) as demo:
        gr.Markdown("<div align='center'><h1>OrpheusTTS-WebUI</h1></div>")
        gr.Markdown("""<div align='center'>Generate realistic speech from text using the OrpheusTTS model.
**Available voices:** tara, jess, leo, leah, dan, mia, zac, zoe (in order of conversational realism)

**Available emotive tags:** `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

**Note:** Increasing repetition_penalty and temperature makes the model speak faster. Increasing Max Tokens extends the maximum duration of genrated audio.
</div>
        """)

        with gr.Tabs(selected=0) as tabs:
            with gr.Tab("Single Text"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(label="Text Input", placeholder="Enter text to convert to speech...", lines=3)
                        with gr.Row():
                            voice = gr.Dropdown(choices=["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"], label="Voice", value="tara")
                        with gr.Row():
                            max_tokens = gr.Slider(label="Max Tokens", value=2048, minimum=128, maximum=16384, step=128)
                            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P")
                            rep_penalty = gr.Slider(minimum=1.1, maximum=2.0, value=1.2, step=0.1, label="Repetition Penalty")
                        max_tokens = gr.Slider(minimum=1200, maximum=3600, value=1200, step=100, label="Max Tokens")
                        submit_btn = gr.Button("Generate Speech", variant="primary")
                        gr.Examples(
                            examples=[
                                "Man, the way social media has, um, completely changed how we interact is just wild, right?",
                                "I just got back from my vacation <sigh> and I'm already feeling stressed about work.",
                                "Did you hear what happened at the party last night? <laugh> It was absolutely ridiculous!",
                                "I've been working on this project for hours <yawn> and I still have so much to do.",
                                "The concert was amazing! <gasp> You should have seen the light show!",
                            ],
                            inputs=prompt,
                            label="Example Prompts",
                        )
                    with gr.Column(scale=1):
                        audio_output = gr.Audio(label="Generated Speech")
                        result_text = gr.Textbox(label="Generation Stats", interactive=False)
                submit_btn.click(
                    fn=generate_speech,
                    inputs=[prompt, voice, temperature, top_p, rep_penalty, max_tokens],
                    outputs=[audio_output, result_text],
                )

            with gr.Tab("Long Form Content"):
                with gr.Row():
                    with gr.Column(scale=2):
                        long_form_prompt = gr.Textbox(label="Long Form Text Input", placeholder="Enter long text to convert to speech...", lines=15)
                        with gr.Row():
                            lf_voice = gr.Dropdown(choices=["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"], label="Voice", value="tara")
                        with gr.Row():
                            lf_max_tokens = gr.Slider(label="Max Tokens", value=4096, minimum=128, maximum=16384, step=128)
                            lf_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.6, step=0.1, label="Temperature")
                            lf_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.05, label="Top P")
                            lf_rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition Penalty")
                            batch_size = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Batch Size (chunks processed in parallel)")
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
""",
                            ],
                            inputs=long_form_prompt,
                            label="Example Prompts",
                        )
                    with gr.Column(scale=1):
                        lf_audio_output = gr.Audio(label="Generated Long Form Speech")
                        lf_result_text = gr.Textbox(label="Generation Stats", interactive=False)
                lf_submit_btn.click(
                    fn=generate_long_form_speech,
                    inputs=[long_form_prompt, lf_voice, lf_temperature, lf_top_p, lf_rep_penalty, batch_size, lf_max_tokens],
                    outputs=[lf_audio_output, lf_result_text],
                )

        gr.Markdown("""<div align='center' style='margin-top: 20px; padding: 10px; border-top: 1px solid #ccc;'>
    <a href='https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained' target='_blank'>Hugging Face</a> |
    <a href='https://github.com/Saganaki22/OrpheusTTS-WebUI' target='_blank'>WebUI GitHub</a> |
    <a href='https://github.com/canopyai/Orpheus-TTS' target='_blank'>Official GitHub</a>
    </div>""")

        demo.load(cleanup_files)
    return demo

if __name__ == "__main__":
    logger.info("Starting OrpheusTTS-WebUI")
    demo = create_ui()
    demo.launch(server_port=18188, share=False)
