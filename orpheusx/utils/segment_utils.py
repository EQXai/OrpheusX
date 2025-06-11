from __future__ import annotations

import re
from typing import Tuple, List

import torch

__all__ = [
    "split_prompt_by_tokens",
    "split_prompt_by_sentences",
    "print_segment_log",
]


def split_prompt_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int = 50,
    return_text: bool = False,
) -> list[torch.Tensor] | tuple[list[str], list[torch.Tensor]]:
    """Split *text* into chunks of at most *chunk_size* tokens.

    The algorithm keeps whole-word boundaries so that the tokenizer
    result for each segment never exceeds *chunk_size* tokens.
    """
    words = text.split()
    segments: list[str] = []
    current_words: list[str] = []
    token_len = 0

    for word in words:
        word_tokens = tokenizer(word, add_special_tokens=False).input_ids
        # If appending *word* would exceed the token budget, finalise current segment.
        if token_len + len(word_tokens) > chunk_size and current_words:
            segments.append(" ".join(current_words))
            current_words = [word]
            token_len = len(word_tokens)
        else:
            current_words.append(word)
            token_len += len(word_tokens)

    if current_words:
        segments.append(" ".join(current_words))

    token_segments = [
        tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments
    ]
    return (segments, token_segments) if return_text else token_segments


def split_prompt_by_sentences(
    text: str,
    tokenizer,
    chunk_size: int = 50,
    return_text: bool = False,
) -> list[torch.Tensor] | tuple[list[str], list[torch.Tensor]]:
    """Split *text* on sentence boundaries without exceeding *chunk_size* tokens.

    A light heuristic groups comma-terminated clauses that are shorter than
    three words with the previous sentence to avoid excessive fragmentation.
    """
    raw_parts = [s.strip() for s in re.split(r"(?<=[.!?,])\s+", text.strip()) if s.strip()]

    sentences: list[str] = []
    for part in raw_parts:
        if sentences:
            prev = sentences[-1]
            # Merge very short clauses following a comma.
            if prev.endswith(",") and (part.endswith(",") or len(part.split()) < 3):
                sentences[-1] = prev + " " + part
                continue
        sentences.append(part)

    segments: list[str] = []
    current: list[str] = []
    for sent in sentences:
        candidate = " ".join(current + [sent])
        token_len = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if token_len > chunk_size and current:
            segments.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)

    if current:
        segments.append(" ".join(current))

    token_segments = [
        tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments
    ]
    return (segments, token_segments) if return_text else token_segments


def print_segment_log(prompt: str, segments: list[str]) -> None:
    """Pretty-print segment boundaries for debugging purposes."""
    offset = 0
    for idx, seg in enumerate(segments, 1):
        start = prompt.find(seg, offset)
        end = start + len(seg)
        print(f"{idx}: chars {start}-{end}: {seg}")
        offset = end 