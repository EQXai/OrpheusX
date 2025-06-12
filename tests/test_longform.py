from pathlib import Path
import uuid
import os
import sys
import asyncio

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orpheusx.utils.longform import (
    chunk_text,
    generate_long_form_speech_async,
    generate_long_form_speech,
)


def dummy_generator_factory(base: Path):
    def _gen(_: str) -> str:
        path = base / f"{uuid.uuid4().hex}.wav"
        import wave
        import struct

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * 240)
        return str(path)
    return _gen


def test_chunk_text_simple():
    text = "Hello world. This is a test of the chunking logic."
    chunks = chunk_text(text, max_chars=20)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_generate_long_form_async(tmp_path):
    generator = dummy_generator_factory(tmp_path)
    result = asyncio.run(
        generate_long_form_speech_async(
            "One. Two. Three.", generator, tmp_path, max_chars=5, fade_ms=0
        )
    )
    assert Path(result).is_file()


def test_generate_long_form_sync(tmp_path):
    generator = dummy_generator_factory(tmp_path)
    result = generate_long_form_speech(
        "One. Two.", generator, tmp_path, max_chars=5, fade_ms=0
    )
    assert Path(result).is_file()
