"""Utility sub-package for OrpheusX.

Re-exports commonly used helpers to minimise import paths.
"""

from .segment_utils import (
    split_prompt_by_tokens,
    split_prompt_by_sentences,
    print_segment_log,
)
from .longform import (
    chunk_text,
    process_chunk,
    generate_long_form_speech_async,
    generate_long_form_speech,
)

__all__ = [
    "split_prompt_by_tokens",
    "split_prompt_by_sentences",
    "print_segment_log",
    "chunk_text",
    "process_chunk",
    "generate_long_form_speech_async",
    "generate_long_form_speech",
]
