from __future__ import annotations

import asyncio
from typing import List

import torch

import re
from .segment_utils import print_segment_log
from audio_utils import concat_with_fade

__all__ = [
    "generate_segments_parallel",
    "generate_long_form_speech_async",
    "chunk_text",
]


def chunk_text(text: str, max_chunk_size: int = 300) -> List[str]:
    """Split ``text`` into chunks of roughly ``max_chunk_size`` characters."""
    text = re.sub(r"\s+", " ", text)
    delimiter_pattern = r'(?<=[.!?])\s+'
    segments = re.split(delimiter_pattern, text)
    sentences: list[str] = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if segment[-1] not in '.!?':
            segment += '.'
        sentences.append(segment)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chunk_size and current:
            chunks.append(current)
            current = sentence
        else:
            current += (" " + sentence) if current else sentence
    if current:
        chunks.append(current)
    return chunks

def _generate_segment(
    tokens: torch.Tensor,
    model,
    snac_model,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate audio for given token IDs and return as 1D tensor."""
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
    modified_input = torch.cat([start_token, tokens.unsqueeze(0), end_tokens], dim=1)
    attention_mask = torch.ones_like(modified_input)
    input_ids_cuda = modified_input.to("cuda")
    attn_cuda = attention_mask.to("cuda")
    generated = model.generate(
        input_ids=input_ids_cuda,
        attention_mask=attn_cuda,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
        use_cache=True,
    )
    token_to_find = 128257
    token_to_remove = 128258
    token_indices = (generated == token_to_find).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated[:, last_occurrence_idx + 1 :]
    else:
        cropped_tensor = generated
    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)
    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    def redistribute_codes(code_list):
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
        ]
        audio_hat = snac_model.decode(codes)
        return audio_hat

    samples = [redistribute_codes(c) for c in code_lists]
    return samples[0].squeeze(0)


async def generate_segments_parallel(
    segments: List[torch.Tensor],
    model,
    snac_model,
    *,
    max_new_tokens: int = 1200,
    batch_size: int = 4,
    fade_ms: int = 60,
    gap_ms: int = 0,
) -> torch.Tensor:
    """Generate audio for ``segments`` concurrently and concatenate the results."""

    semaphore = asyncio.Semaphore(batch_size)
    loop = asyncio.get_event_loop()

    async def process(tokens: torch.Tensor, idx: int):
        async with semaphore:
            audio = await loop.run_in_executor(
                None, _generate_segment, tokens, model, snac_model, max_new_tokens
            )
            return idx, audio

    tasks = [asyncio.create_task(process(t, i)) for i, t in enumerate(segments)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    audios = [a for _, a in results]
    if not audios:
        return torch.tensor([])
    final = audios[0]
    for part in audios[1:]:
        final = concat_with_fade([final, part], fade_ms=fade_ms, gap_ms=gap_ms)
    return final


async def generate_long_form_speech_async(
    text: str,
    model,
    tokenizer,
    snac_model,
    *,
    segment: bool = True,
    chunk_size: int = 300,
    batch_size: int = 4,
    max_new_tokens: int = 1200,
    fade_ms: int = 60,
    gap_ms: int = 0,
) -> torch.Tensor:
    """Split ``text`` into chunks, generate them in parallel and concatenate."""

    if segment:
        seg_text = chunk_text(text, max_chunk_size=chunk_size)
        print_segment_log(text, seg_text)
        segments = [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in seg_text]
    else:
        segments = [tokenizer(text, return_tensors="pt").input_ids.squeeze(0)]

    return await generate_segments_parallel(
        segments,
        model,
        snac_model,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        fade_ms=fade_ms,
        gap_ms=gap_ms,
    )
