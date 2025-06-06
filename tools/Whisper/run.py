import os
import json
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from transformers import AutoTokenizer
import numpy as np
import librosa
import soundfile as sf

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/csm-1b")

def run_whisperx(audio_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f">> Transcribing {audio_path.name} with WhisperX...")
    subprocess.run([
        "whisperx",
        str(audio_path),
        "--output_dir", str(output_dir),
        "--output_format", "json"
    ])

    return output_dir / f"{audio_path.stem}.json"

def ffmpeg_cut(input_file, start_time, end_time, output_wav_file, target_samples=335752, sampling_rate=24000):
    # Calculate the duration in seconds for the target number of samples
    target_duration = target_samples / sampling_rate
    duration = end_time - start_time

    # Create a temporary filename for the initial cut in MP3
    temp_mp3_for_cut = output_wav_file.with_suffix('.temp_cut.mp3')

    # Perform the initial cut to a temporary MP3 file
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", input_file,
        "-t", f"{min(duration, target_duration):.3f}",
        "-ar", str(sampling_rate),
        "-c:a", "mp3",
        str(temp_mp3_for_cut)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load the audio from the temporary MP3
    audio, sr = librosa.load(temp_mp3_for_cut, sr=sampling_rate)
    temp_mp3_for_cut.unlink()  # Delete the temporary MP3 file

    # Verify and standardize the audio length
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    
    # Save the standardized audio as the final WAV file
    sf.write(output_wav_file, audio, sr)

def split_text_to_tokens(text, max_tokens=50):
    """Splits a text into subsegments of less than max_tokens, including special tokens."""
    tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(tokens) <= max_tokens:
        return [(text, len(tokens))]

    words = text.split()
    subsegments = []
    current_subsegment = []
    current_tokens = []

    for word in words:
        temp_subsegment = current_subsegment + [word]
        temp_text = " ".join(temp_subsegment)
        temp_tokens = tokenizer(temp_text, add_special_tokens=True)["input_ids"]
        if len(temp_tokens) <= max_tokens:
            current_subsegment = temp_subsegment
            current_tokens = temp_tokens
        else:
            if current_subsegment:
                subsegments.append((" ".join(current_subsegment), len(current_tokens)))
                current_subsegment = [word]
                current_tokens = tokenizer(word, add_special_tokens=True)["input_ids"]
            else:
                truncated_word = word[:50]
                subsegments.append((truncated_word, len(tokenizer(truncated_word, add_special_tokens=True)["input_ids"])))
                current_subsegment = []
                current_tokens = []

    if current_subsegment:
        subsegments.append((" ".join(current_subsegment), len(current_tokens)))

    return subsegments

def segment_audio(audio_path, json_file, output_dir, min_len=10.0, max_len=15.0, max_tokens=50, target_samples=335752):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk = []
    chunk_token_count = 0
    chunk_start = None
    chunk_end = None
    accumulated = 0.0
    part = 0
    base_name = audio_path.stem

    for seg_idx, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        sub_texts = split_text_to_tokens(text, max_tokens=max_tokens)

        for sub_text, sub_token_count in sub_texts:
            if chunk_start is None:
                chunk_start = start

            temp_chunk = chunk + [sub_text]
            temp_chunk_text = " ".join(temp_chunk)
            temp_tokens = tokenizer(temp_chunk_text, add_special_tokens=True)["input_ids"]
            temp_token_count = len(temp_tokens)

            if temp_token_count <= max_tokens:
                chunk.append(sub_text)
                chunk_token_count = temp_token_count
                chunk_end = end
                accumulated = chunk_end - chunk_start
            else:
                if chunk and accumulated >= min_len:
                    part += 1
                    suffix = f"{part:03d}"
                    out_audio = output_dir / f"{base_name}_{suffix}.wav"
                    out_text = output_dir / f"{base_name}_{suffix}.txt"

                    ffmpeg_cut(audio_path, chunk_start, chunk_end, out_audio, target_samples=target_samples)
                    with open(out_text, "w", encoding="utf-8") as f:
                        f.write(" ".join(chunk))
                    audio, sr = librosa.load(out_audio, sr=24000)
                    print(f"Segment {suffix}: {chunk_token_count} tokens, {accumulated:.2f} seconds, {len(audio)} samples")

                chunk = [sub_text]
                chunk_token_count = sub_token_count
                chunk_start = start
                chunk_end = end
                accumulated = end - chunk_start

            if accumulated >= max_len or (seg_idx == len(segments) - 1 and sub_text == sub_texts[-1]):
                if chunk and accumulated >= min_len:
                    part += 1
                    suffix = f"{part:03d}"
                    out_audio = output_dir / f"{base_name}_{suffix}.wav"
                    out_text = output_dir / f"{base_name}_{suffix}.txt"

                    ffmpeg_cut(audio_path, chunk_start, chunk_end, out_audio, target_samples=target_samples)
                    with open(out_text, "w", encoding="utf-8") as f:
                        f.write(" ".join(chunk))
                    audio, sr = librosa.load(out_audio, sr=24000)
                    print(f"Segment {suffix}: {chunk_token_count} tokens, {accumulated:.2f} seconds, {len(audio)} samples")

                chunk = []
                chunk_token_count = 0
                chunk_start = None
                chunk_end = None
                accumulated = 0.0

    if chunk and accumulated >= min_len:
        part += 1
        suffix = f"{part:03d}"
        out_audio = output_dir / f"{base_name}_{suffix}.wav"
        out_text = output_dir / f"{base_name}_{suffix}.txt"

        ffmpeg_cut(audio_path, chunk_start, chunk_end, out_audio, target_samples=target_samples)
        with open(out_text, "w", encoding="utf-8") as f:
            f.write(" ".join(chunk))
        audio, sr = librosa.load(out_audio, sr=24000)
        print(f"Segment {suffix}: {chunk_token_count} tokens, {accumulated:.2f} seconds, {len(audio)} samples")

def main(audio_path):
    audio_path = Path(audio_path).resolve()
    base_name = audio_path.stem
    temp_out = Path("whisperx_out")
    segment_out = Path("segments") / base_name

    json_path = run_whisperx(audio_path, temp_out)
    segment_audio(audio_path, json_path, segment_out)

    print(f"\n✅ Segments saved in: {segment_out.resolve()}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("audio", help="Path to the audio file (.mp3 or .wav)")
    args = parser.parse_args()
    main(args.audio)
