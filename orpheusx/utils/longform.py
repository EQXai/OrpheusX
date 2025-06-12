"""Helpers for asynchronous long-form speech generation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Iterable
import uuid
import shutil
import array

import torch

from audio_utils import concat_with_fade

__all__ = [
    "chunk_text",
    "process_chunk",
    "generate_long_form_speech_async",
    "generate_long_form_speech",
]


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split *text* into roughly ``max_chars`` sized chunks on sentence boundaries."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            # Flush current chunk if any
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i : i + max_chars])
            continue
        candidate = f"{current} {sent}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


async def process_chunk(
    text: str,
    index: int,
    out_dir: Path,
    generator: Callable[[str], str],
) -> Path:
    """Generate speech for ``text`` using ``generator`` and move to *out_dir*."""
    tmp_path = await asyncio.to_thread(generator, text)
    src = Path(tmp_path)
    dest = out_dir / f"chunk_{index:04d}{src.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), dest)
    return dest


async def generate_long_form_speech_async(
    text: str,
    generator: Callable[[str], str],
    out_dir: Path | str,
    *,
    max_chars: int = 400,
    fade_ms: int = 60,
) -> str:
    """Generate long-form speech by processing segments concurrently."""
    out_dir = Path(out_dir)
    chunks = chunk_text(text, max_chars=max_chars)
    tasks = [process_chunk(c, i, out_dir, generator) for i, c in enumerate(chunks)]
    paths = await asyncio.gather(*tasks)

    import wave
    import array

    tensors: list[torch.Tensor] = []
    sr = 24000
    for p in paths:
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        arr = array.array("h")
        arr.frombytes(frames)
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0) / 32768.0
        tensors.append(t)
    final_audio = concat_with_fade(tensors, sample_rate=sr, fade_ms=fade_ms)
    final_path = out_dir / f"long_{uuid.uuid4().hex}.wav"
    import wave
    samples = (final_audio.squeeze(0).clamp(-1, 1) * 32767).to(torch.int16).tolist()
    with wave.open(str(final_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(array.array("h", samples).tobytes())

    cleanup_files(paths)
    return str(final_path)


def cleanup_files(paths: Iterable[Path]) -> None:
    """Delete temporary files if they exist."""
    for p in paths:
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass


def generate_long_form_speech(*args, **kwargs) -> str:
    """Synchronous wrapper for :func:`generate_long_form_speech_async`."""
    return asyncio.run(generate_long_form_speech_async(*args, **kwargs))

