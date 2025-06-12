# -*- coding: utf-8 -*-
"""Simple Gradio interface for OrpheusX workflows.

This script exposes dataset preparation, model training and inference
through a web UI. It reuses the existing command line utilities without
modifying them.
"""
from __future__ import annotations

import os
import sys
import signal
from pathlib import Path
import json
import re
import unsloth  # must be imported before transformers
from transformers import AutoTokenizer
import gradio as gr
import gc
import time
from tools.logger_utils import get_logger

# Helper for audio concatenation with crossfade
from audio_utils import concat_with_fade

# The prepare_dataset helper can be imported safely
from scripts.prepare_dataset import prepare_dataset

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
# Match the CLI scripts which store LoRAs under ``lora_models``
LORA_DIR = REPO_ROOT / "lora_models"
PROMPT_LIST_DIR = REPO_ROOT / "prompt_list"
SOURCE_AUDIO_DIR = REPO_ROOT / "source_audio"
MAX_PROMPTS = 5

logger = get_logger("gradio_app")
STOP_FLAG = False


def stop_current() -> str:
    """Signal long-running tasks to stop."""
    global STOP_FLAG
    STOP_FLAG = True
    return "Stop signal sent"


def exit_app() -> None:
    """Terminate the Gradio process."""
    os._exit(0)


def list_datasets() -> list[str]:
    if not DATASETS_DIR.is_dir():
        return []
    return sorted([d.name for d in DATASETS_DIR.iterdir() if d.is_dir()])


def list_loras() -> list[str]:
    if not LORA_DIR.is_dir():
        return []
    return sorted([d.name for d in LORA_DIR.iterdir() if d.is_dir()])


def list_prompt_files() -> list[str]:
    if not PROMPT_LIST_DIR.is_dir():
        return []
    return sorted([f.name for f in PROMPT_LIST_DIR.glob("*.json")])


def load_prompts(file_name: str) -> list[str]:
    """Return prompts from a JSON file under ``PROMPT_LIST_DIR``."""
    path = PROMPT_LIST_DIR / file_name
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to load prompts from %s", path)
        return []
    if not isinstance(data, list):
        return []
    return [str(p) for p in data if isinstance(p, str)]


def list_source_audio() -> list[str]:
    if not SOURCE_AUDIO_DIR.is_dir():
        return []
    return sorted(
        [f.name for f in SOURCE_AUDIO_DIR.iterdir() if f.suffix.lower() in (".wav", ".mp3")]
    )


def prepare_datasets_ui(
    upload_file: str,
    name: str,
    existing: list[str] | None,
    min_tokens: int = 0,
    max_tokens: int | None = None,
) -> str:
    """Prepare one or more datasets from uploaded or existing audio files."""
    global STOP_FLAG
    tasks: list[tuple[str, str]] = []
    if upload_file:
        if not name:
            return "Please provide a dataset name for the uploaded audio."
        tasks.append((upload_file, name))
    for fname in existing or []:
        audio_path = SOURCE_AUDIO_DIR / fname
        tasks.append((str(audio_path), Path(fname).stem))
    if not tasks:
        return "No audio selected."

    msgs = []
    logger.info("Preparing %d dataset(s)", len(tasks))
    total = len(tasks)
    progress = gr.Progress()
    for idx, (audio_path, ds_name) in enumerate(tasks, start=1):
        if STOP_FLAG:
            STOP_FLAG = False
            return "Stopped"
        start = time.perf_counter()
        progress((idx - 1) / total, desc=f"Preparing {ds_name}...")
        out_dir = DATASETS_DIR / ds_name
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            prepare_dataset(
                audio_path,
                str(out_dir),
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
            msgs.append(f"{ds_name}: success")
            elapsed = time.perf_counter() - start
            logger.info("%s prepared in %.2fs", ds_name, elapsed)
        except Exception as e:  # pragma: no cover - best effort
            elapsed = time.perf_counter() - start
            logger.exception("Error preparing %s after %.2fs", ds_name, elapsed)
            msgs.append(f"{ds_name}: failed ({e})")
        progress(idx / total)
    return "\n".join(msgs)


# ---- Training ----
# Copy of train_dataset() from scripts/train_interactive.py with minimal changes
from datasets import load_dataset, load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
import torchaudio.transforms as T
from snac import SNAC
import torch
from transformers import TrainingArguments, Trainer

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/orpheus-3b-0.1-ft")
CACHE_DIR = REPO_ROOT / "models"


def train_lora_single(
    dataset_source: str,
    lora_name: str,
    is_local: bool,
    batch_size: int = 1,
    grad_steps: int = 4,
    warm_steps: int = 5,
    max_steps: int = 60,
    epochs: int = 1,
    lr: float = 2e-4,
    log_steps: int = 1,
    weight_decay: float = 0.01,
    optim: str = "adamw_8bit",
    scheduler: str = "linear",
) -> str:
    """Train a single LoRA on a dataset."""
    logger.info("Training LoRA %s from %s", lora_name, dataset_source)
    start_time = time.perf_counter()
    if is_local:
        dataset = load_from_disk(dataset_source)
    else:
        dataset = load_dataset(dataset_source, split="train", cache_dir=str(DATASETS_DIR))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        cache_dir=str(CACHE_DIR),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    save_dir = LORA_DIR / lora_name.replace("/", "_") / "lora_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    ds_sample_rate = dataset[0]["audio"]["sampling_rate"]
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir=str(CACHE_DIR))
    snac_model = snac_model.to("cuda")

    import locale
    locale.getpreferredencoding = lambda: "UTF-8"

    def tokenise_audio(waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)
        waveform = waveform.unsqueeze(0).to("cuda")
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
            all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))
        return all_codes

    def add_codes(example):
        codes_list = None
        try:
            answer_audio = example.get("audio")
            if answer_audio and "array" in answer_audio:
                audio_array = answer_audio["array"]
                codes_list = tokenise_audio(audio_array)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
        example["codes_list"] = codes_list
        return example

    dataset = dataset.map(add_codes, remove_columns=["audio"])

    TOKENISER_LENGTH = 128256
    start_of_text = 128000
    end_of_text = 128009
    start_of_speech = TOKENISER_LENGTH + 1
    end_of_speech = TOKENISER_LENGTH + 2
    start_of_human = TOKENISER_LENGTH + 3
    end_of_human = TOKENISER_LENGTH + 4
    start_of_ai = TOKENISER_LENGTH + 5
    end_of_ai = TOKENISER_LENGTH + 6

    dataset = dataset.filter(lambda x: x["codes_list"] is not None)
    dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)

    def remove_duplicate_frames(example):
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")
        result = vals[:7]
        for i in range(7, len(vals), 7):
            current_first = vals[i]
            previous_first = result[-7]
            if current_first != previous_first:
                result.extend(vals[i:i + 7])
        example["codes_list"] = result
        return example

    dataset = dataset.map(remove_duplicate_frames)

    def create_input_ids(example):
        text_prompt = f"{example.get('source', '')}: {example['text']}" if 'source' in example else example['text']
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)
        example['text_tokens'] = text_ids
        input_ids = (
            [start_of_human]
            + example['text_tokens']
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + example['codes_list']
            + [end_of_speech]
            + [end_of_ai]
        )
        example['input_ids'] = input_ids
        example['labels'] = input_ids
        example['attention_mask'] = [1] * len(input_ids)
        return example

    dataset = dataset.map(create_input_ids, remove_columns=['text', 'codes_list'])

    before_len = len(dataset)
    dataset = dataset.filter(lambda x: len(x['input_ids']) <= 2048)
    skipped = before_len - len(dataset)
    if skipped:
        print(f"Skipped {skipped} sample(s) exceeding 2048 tokens.")

    columns_to_keep = ['input_ids', 'labels', 'attention_mask']
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_steps,
            warmup_steps=warm_steps,
            max_steps=max_steps,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=log_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=scheduler,
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    elapsed = time.perf_counter() - start_time
    logger.info("LoRA %s trained in %.2fs", lora_name, elapsed)
    return f"LoRA saved under {save_dir.resolve()}"


def train_loras(
    hf_links: str,
    local_datasets: list[str],
    batch_size: int,
    grad_steps: int,
    warm_steps: int,
    max_steps: int,
    epochs: int,
    lr: float,
    log_steps: int,
    weight_decay: float,
    optim: str,
    scheduler: str,
) -> str:
    """Train one or more LoRAs based on the provided sources."""
    global STOP_FLAG
    dataset_info: list[tuple[str, str, bool]] = []
    links = [l.strip() for l in hf_links.splitlines() if l.strip()]
    for link in links:
        name = link.split("/")[-1]
        dataset_info.append((link, name, False))
    for ds in local_datasets:
        dataset_info.append((str(DATASETS_DIR / ds), ds, True))
    if not dataset_info:
        return "No datasets selected."
    msgs = []
    total = len(dataset_info)
    logger.info("Training %d LoRA(s)", total)
    progress = gr.Progress()
    for idx, (src, name, is_local) in enumerate(dataset_info, start=1):
        if STOP_FLAG:
            STOP_FLAG = False
            return "Stopped"
        progress((idx - 1) / total, desc=f"Training {name}...")
        start = time.perf_counter()
        try:
            msg = train_lora_single(
                src,
                name,
                is_local,
                batch_size,
                grad_steps,
                warm_steps,
                max_steps,
                epochs,
                lr,
                log_steps,
                weight_decay,
                optim,
                scheduler,
            )
            msgs.append(f"{name}: success")
            elapsed = time.perf_counter() - start
            logger.info("%s trained in %.2fs", name, elapsed)
        except Exception as e:  # pragma: no cover - best effort
            elapsed = time.perf_counter() - start
            logger.exception("Training failed for %s after %.2fs", name, elapsed)
            msgs.append(f"{name}: failed ({e})")
    progress(1)
    return "\n".join(msgs)

# ---- Inference ----
from peft import PeftModel

# Cache for loaded models to avoid reloading on every prompt
_LOADED_MODEL_NAME: str | None = None
_LOADED_LORA_PATH: str | None = None
_LOADED_MODEL = None
_LOADED_TOKENIZER = None
_SNAC_MODEL = None
_PIPELINE_TOKENIZER = None


def load_model(base_model: str, lora_path: str | None):
    """Load a model/Lora pair reusing any already loaded model."""
    global _LOADED_MODEL_NAME, _LOADED_LORA_PATH, _LOADED_MODEL, _LOADED_TOKENIZER
    if (
        _LOADED_MODEL is not None
        and _LOADED_MODEL_NAME == base_model
        and _LOADED_LORA_PATH == lora_path
    ):
        return _LOADED_MODEL, _LOADED_TOKENIZER

    if _LOADED_MODEL is not None:
        del _LOADED_MODEL
        torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        cache_dir=str(CACHE_DIR),
    )
    if lora_path and os.path.isdir(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)

    _LOADED_MODEL_NAME = base_model
    _LOADED_LORA_PATH = lora_path
    _LOADED_MODEL = model
    _LOADED_TOKENIZER = tokenizer
    return model, tokenizer


def get_output_path(lora_name: str, ext: str = ".wav") -> Path:
    base_dir = REPO_ROOT / "audio_output" / lora_name
    base_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        path = base_dir / f"{lora_name}_{idx}{ext}"
        if not path.exists():
            return path
        idx += 1


def get_snac_model():
    """Load SNAC model once and cache it."""
    global _SNAC_MODEL
    if _SNAC_MODEL is None:
        _SNAC_MODEL = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir=str(CACHE_DIR))
        _SNAC_MODEL = _SNAC_MODEL.to("cpu")
    return _SNAC_MODEL


def get_pipeline_tokenizer():
    """Return a cached tokenizer for length checks."""
    global _PIPELINE_TOKENIZER
    if _PIPELINE_TOKENIZER is None:
        _PIPELINE_TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(CACHE_DIR),
        )
    return _PIPELINE_TOKENIZER


def _merge_short_segments(
    texts: list[str],
    tokens: list[torch.Tensor],
    min_tokens: int,
) -> tuple[list[str], list[torch.Tensor]]:
    """Merge segments shorter than ``min_tokens`` with the previous one."""
    if min_tokens <= 0:
        return texts, tokens
    merged_texts: list[str] = []
    merged_tokens: list[torch.Tensor] = []
    for t_text, t_tokens in zip(texts, tokens):
        if merged_tokens and t_tokens.numel() < min_tokens:
            merged_texts[-1] = merged_texts[-1] + " " + t_text
            merged_tokens[-1] = torch.cat([merged_tokens[-1], t_tokens])
        else:
            merged_texts.append(t_text)
            merged_tokens.append(t_tokens)
    return merged_texts, merged_tokens


def split_prompt_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 50,
    min_tokens: int = 0,
    return_text: bool = False,
) -> list[torch.Tensor] | tuple[list[str], list[torch.Tensor]]:
    """Split text into token chunks without cutting words."""
    words = text.split()
    segments: list[str] = []
    current: list[str] = []
    token_len = 0
    for w in words:
        n_tokens = len(tokenizer(w, add_special_tokens=False).input_ids)
        if token_len + n_tokens > max_tokens and current:
            segments.append(" ".join(current))
            current = [w]
            token_len = n_tokens
        else:
            current.append(w)
            token_len += n_tokens
    if current:
        segments.append(" ".join(current))
    token_segments = [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments]
    segments, token_segments = _merge_short_segments(segments, token_segments, min_tokens)
    return (segments, token_segments) if return_text else token_segments


def print_segment_log(prompt: str, segments: list[str]) -> None:
    """Print segment boundaries for a prompt."""
    logger.info("Segmentation log:")
    offset = 0
    for idx, seg in enumerate(segments, 1):
        start = prompt.find(seg, offset)
        end = start + len(seg)
        logger.info("%d: chars %d-%d: %s", idx, start, end, seg)
        offset = end


def split_prompt_full(
    text: str,
    tokenizer,
    chars: list[str] | None = None,
    return_text: bool = False,
) -> list[torch.Tensor] | tuple[list[str], list[torch.Tensor]]:
    """Split ``text`` at selected punctuation marks ignoring token limits."""
    if not chars:
        chars = [",", ".", "?", "!"]
    char_class = "".join(re.escape(c) for c in chars)
    pattern = rf"[^{char_class}]+(?:[{char_class}]+|$)"
    segments = [p.strip() for p in re.findall(pattern, text.strip()) if p.strip()]
    token_segments = [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments]
    return (segments, token_segments) if return_text else token_segments


def split_prompt_by_sentences(
    text: str,
    tokenizer,
    chars: list[str] | None = None,
    max_tokens: int = 50,
    min_tokens: int = 0,
    return_text: bool = False,
) -> list[torch.Tensor] | tuple[list[str], list[torch.Tensor]]:
    """Split text into sentence groups within a token range."""
    if not chars:
        chars = [",", ".", "?", "!"]
    char_class = "".join(re.escape(c) for c in chars)
    pattern = rf"(?<=[{char_class}])\s+"
    raw_parts = [s.strip() for s in re.split(pattern, text.strip()) if s.strip()]
    sentences: list[str] = []
    for part in raw_parts:
        if sentences:
            prev = sentences[-1]
            if prev.endswith(",") and (part.endswith(",") or len(part.split()) < 3):
                sentences[-1] = prev + " " + part
                continue
        sentences.append(part)
    segments: list[str] = []
    current: list[str] = []
    for sent in sentences:
        candidate = " ".join(current + [sent])
        token_len = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if token_len > max_tokens and current:
            segments.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    if current:
        segments.append(" ".join(current))
    token_segments = [tokenizer(s, return_tensors="pt").input_ids.squeeze(0) for s in segments]
    segments, token_segments = _merge_short_segments(segments, token_segments, min_tokens)
    return (segments, token_segments) if return_text else token_segments


def generate_audio_segment(
    tokens: torch.Tensor,
    model,
    snac_model,
    max_new_tokens: int = 1200,
) -> torch.Tensor:
    logger.info("Generating segment with %d tokens", tokens.numel())
    t0 = time.perf_counter()
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
    logger.info("Model.generate finished in %.2fs", time.perf_counter() - t0)
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


def generate_audio(
    text: str,
    lora_name: str | None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 1200,
    segment: bool = False,
    segment_by: str = "tokens",
    seg_chars: list[str] | None = None,
    seg_min_tokens: int = 0,
    seg_max_tokens: int = 50,
    seg_gap: float = 0.0,
    fade_ms: int = 60,
) -> str:
    model_name = MODEL_NAME
    lora_path = None
    if lora_name:
        lora_path = LORA_DIR / lora_name / "lora_model"
    model, tokenizer = load_model(model_name, str(lora_path) if lora_path else None)

    snac_model = get_snac_model()

    token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
    logger.info("Generating audio (%d tokens) using %s", token_count, lora_name or "base_model")
    start_time = time.perf_counter()
    if segment:
        if segment_by == "sentence":
            seg_text, segments = split_prompt_by_sentences(
                text,
                tokenizer,
                chars=seg_chars,
                max_tokens=seg_max_tokens,
                min_tokens=seg_min_tokens,
                return_text=True,
            )
        elif segment_by == "full_segment":
            seg_text, segments = split_prompt_full(
                text,
                tokenizer,
                chars=seg_chars,
                return_text=True,
            )
        else:
            seg_text, segments = split_prompt_by_tokens(
                text,
                tokenizer,
                max_tokens=seg_max_tokens,
                min_tokens=seg_min_tokens,
                return_text=True,
            )
        print_segment_log(text, seg_text)
        for i, seg in enumerate(segments, 1):
            logger.info("Segment %d tokens: %d", i, seg.numel())
    else:
        single = tokenizer(text, return_tensors='pt').input_ids.squeeze(0)
        logger.info("Single segment tokens: %d", single.numel())
        segments = [single]
    final_audio = None
    for s in segments:
        part = generate_audio_segment(
            s, model, snac_model, max_new_tokens=max_new_tokens
        )
        if final_audio is None:
            final_audio = part
        else:
            final_audio = concat_with_fade(
                [final_audio, part], sample_rate=24000, fade_ms=fade_ms, gap_ms=int(seg_gap * 1000)
            )
        torch.cuda.empty_cache()
        gc.collect()
    if final_audio is None:
        return ""
    elapsed = time.perf_counter() - start_time
    duration = final_audio.shape[-1] / 24000
    rate = elapsed / duration if duration else 0.0
    logger.info("Inference time: %.2fs (%.2fs per generated second)", elapsed, rate)
    lora_name = lora_name or "base_model"
    path = get_output_path(lora_name)
    import torchaudio
    torchaudio.save(str(path), final_audio.detach().cpu(), 24000)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Saved audio to %s", path)
    return str(path)


def generate_batch(
    prompts: list[str],
    loras: list[str],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    segment: bool,
    segment_by: str,
    seg_chars: list[str] | None,
    seg_min_tokens: int,
    seg_max_tokens: int,
    seg_gap: float = 0.0,
    fade_ms: int = 60,
) -> tuple[str, str]:
    """Generate audio for multiple prompts/LORAs."""
    global STOP_FLAG
    if not prompts:
        return "", ""
    loras = loras or [None]
    results: list[tuple[str, str]] = []
    last_path = ""
    total = len(prompts) * len(loras)
    step = 0
    progress = gr.Progress()
    for lora in loras:
        for text in prompts:
            if STOP_FLAG:
                STOP_FLAG = False
                return "", ""
            progress(step / total, desc=f"Generating {lora or 'base'}...")
            logger.info("Generating prompt '%s' with lora '%s'", text, lora or 'base_model')
            path = generate_audio(
                text,
                None if lora == "<base>" else lora,
                temperature,
                top_p,
                repetition_penalty,
                max_new_tokens,
                segment,
                segment_by,
                seg_chars,
                seg_min_tokens,
                seg_max_tokens,
                seg_gap,
                fade_ms,
            )
            torch.cuda.empty_cache()
            gc.collect()
            caption = f"{lora or 'base'}: {text}"[:60]
            results.append((path, caption))
            last_path = path
            step += 1
    progress(1)
    html_items = []
    for path, caption in results:
        html_items.append(
            f"<div style='margin-bottom:1em'>"
            f"<p>{caption}</p>"
            f"<audio controls src='file={path}'></audio>"
            f"</div>"
        )
    return "\n".join(html_items), last_path


def dataset_status(name: str) -> str:
    """Return model status message for a dataset."""
    ds_name = Path(name).stem
    lora_path = LORA_DIR / ds_name / "lora_model"
    return "Model already created" if lora_path.is_dir() else ""


def run_full_pipeline(dataset_file: str, prompt: str, fade_ms: int = 60) -> tuple[str, str]:
    """Prepare dataset, train LoRA and run inference."""
    global STOP_FLAG
    if not dataset_file:
        return "No dataset selected", ""
    if not prompt:
        return "Prompt is empty", ""
    ds_name = Path(dataset_file).stem
    audio_path = SOURCE_AUDIO_DIR / dataset_file
    dataset_dir = DATASETS_DIR / ds_name
    lora_dir = LORA_DIR / ds_name / "lora_model"
    progress = gr.Progress()
    msgs = []
    if STOP_FLAG:
        STOP_FLAG = False
        return "Stopped", ""
    if not dataset_dir.is_dir():
        progress(0.0, desc="Preparing dataset")
        prepare_dataset(str(audio_path), str(dataset_dir))
        msgs.append("Dataset prepared")
    else:
        msgs.append("Dataset already prepared")
    if STOP_FLAG:
        STOP_FLAG = False
        return "Stopped", ""
    if not lora_dir.is_dir():
        progress(0.33, desc="Training LoRA")
        train_lora_single(str(dataset_dir), ds_name, True)
        msgs.append("LoRA trained")
    else:
        msgs.append("LoRA already trained")
    tokenizer = get_pipeline_tokenizer()
    token_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    use_segmentation = token_len > 50
    progress(0.66, desc="Generating audio")
    if STOP_FLAG:
        STOP_FLAG = False
        return "Stopped", ""
    if use_segmentation:
        out_path = generate_audio(
            prompt,
            ds_name,
            max_new_tokens=2400,
            segment=True,
            segment_by="sentence",
            seg_chars=[",", ".", "?", "!"],
            seg_min_tokens=0,
            seg_max_tokens=50,
            seg_gap=0.0,
            fade_ms=fade_ms,
        )
    else:
        out_path = generate_audio(prompt, ds_name, fade_ms=fade_ms)
    progress(1, desc="Done")
    msgs.append(f"Audio saved to {out_path}")
    return "\n".join(msgs), out_path


# ---- Gradio Interface ----
dataset_choices = list_datasets()
lora_choices = list_loras()
prompt_files = list_prompt_files()


def refresh_lists() -> tuple:

    """Reload datasets, LoRAs and prompt lists from disk."""
    return (
        gr.update(choices=list_datasets()),
        gr.update(choices=["<base>"] + list_loras()),
        gr.update(choices=list_prompt_files()),
        gr.update(choices=list_source_audio()),
        gr.update(choices=list_source_audio()),
        gr.update(choices=list_source_audio()),
    )

with gr.Blocks() as demo:
    gr.Markdown("# OrpheusX Gradio Interface")

    refresh_btn = gr.Button("Refresh directories")
    stop_btn = gr.Button("Stop Task")
    exit_btn = gr.Button("Exit")

    with gr.Tabs():
        with gr.Tab("Unified"):
            with gr.Tabs():
                with gr.Tab("Auto Pipeline"):
                    auto_dataset = gr.Dropdown(choices=list_source_audio(), label="Dataset")
                    auto_status = gr.Markdown()
                    auto_prompt = gr.Textbox(label="Prompt")
                    auto_btn = gr.Button("Run Pipeline")
                    auto_log = gr.Textbox()
                    auto_audio = gr.Audio(label="Output")

                    auto_dataset.change(dataset_status, auto_dataset, auto_status)
                    auto_btn.click(run_full_pipeline, [auto_dataset, auto_prompt], [auto_log, auto_audio])

        with gr.Tab("Main"):
            with gr.Tabs():
                with gr.Tab("Prepare Dataset"):
                    audio_input = gr.Audio(type="filepath", label="Upload audio")
                    local_audio = gr.Dropdown(choices=list_source_audio(), multiselect=True, label="Existing audio file(s)")
                    dataset_name = gr.Textbox(label="Dataset Name (for upload)")
                    segment_tokens = gr.Number(value=50, precision=0, label="Max tokens per segment", visible=False)
                    segment_duration = gr.Number(value=0, precision=1, label="Min seconds per segment", visible=False)
                    model_max_len = gr.Number(value=2048, precision=0, label="Model max length", visible=False)
                    prepare_btn = gr.Button("Prepare")
                    prepare_output = gr.Textbox()
                    prepare_btn.click(
                        prepare_datasets_ui,
                        [
                            audio_input,
                            dataset_name,
                            local_audio,
                        ],
                        prepare_output,
                    )

                with gr.Tab("Train LoRA"):
                    hf_input = gr.Textbox(label="HF dataset link (one per line)")
                    local_ds = gr.Dropdown(choices=dataset_choices, multiselect=True, label="Local dataset(s)")
                    model_max_len_train = gr.Number(value=2048, precision=0, label="Model max length", visible=False)
                    with gr.Accordion("Ajustes avanzados", open=False):
                        batch_size = gr.Number(value=1, precision=0, label="Batch size")
                        grad_steps = gr.Number(value=4, precision=0, label="Gradient accumulation")
                        warmup_steps = gr.Number(value=5, precision=0, label="Warmup steps")
                        max_steps = gr.Number(value=60, precision=0, label="Max steps")
                        epochs = gr.Number(value=1, precision=0, label="Epochs")
                        lr = gr.Number(value=2e-4, label="Learning rate")
                        log_steps = gr.Number(value=1, precision=0, label="Logging steps")
                        weight_decay = gr.Number(value=0.01, label="Weight decay")
                        optim = gr.Textbox(value="adamw_8bit", label="Optimizer")
                        scheduler = gr.Textbox(value="linear", label="LR scheduler type")
                    train_btn = gr.Button("Train")
                    train_output = gr.Textbox()
                    train_btn.click(
                        train_loras,
                        [
                            hf_input,
                            local_ds,
                            batch_size,
                            grad_steps,
                            warmup_steps,
                            max_steps,
                            epochs,
                            lr,
                            log_steps,
                            weight_decay,
                            optim,
                            scheduler,
                        ],
                        train_output,
                    )

                with gr.Tab("Inference"):
                    mode = gr.Radio(["Manual", "Prompt List"], value="Manual", label="Prompt source")
                    num_prompts = gr.Number(value=1, precision=0, label="Number of prompts")
                    prompt_boxes = [gr.Textbox(label=f"Prompt {i+1}", visible=(i == 0)) for i in range(MAX_PROMPTS)]
                    prompt_list_dd = gr.Dropdown(choices=prompt_files, label="Prompt list", visible=False)
                    lora_used = gr.Dropdown(choices=["<base>"] + lora_choices, multiselect=True, label="LoRA(s)")
                    with gr.Accordion("Advanced Settings", open=False):
                        profile_sel = gr.Radio(
                            ["Short Audio", "Long Audio"],
                            value="Short Audio",
                            label="Preset",
                        )
                        temperature = gr.Slider(0.1, 1.5, value=0.6, label="Temperature")
                        top_p = gr.Slider(0.5, 1.0, value=0.95, label="Top P")
                        rep_penalty = gr.Slider(1.0, 2.0, value=1.1, label="Repetition Penalty")
                        max_tokens = gr.Number(value=1200, precision=0, label="Max New Tokens")
                        segment_chk = gr.Checkbox(label="Segment text")
                        segment_method = gr.Radio([
                            "tokens",
                            "sentence",
                            "full_segment",
                        ], value="tokens", label="Segment by")
                        segment_chars = gr.CheckboxGroup(
                            [",", ".", "?", "!"],
                            value=[",", ".", "?", "!"],
                            label="Segment characters",
                        )
                        seg_min_tokens = gr.Number(value=0, precision=0, label="Min tokens per segment")
                        seg_max_tokens = gr.Number(value=50, precision=0, label="Max tokens per segment")
                        seg_gap = gr.Number(value=0.0, precision=1, label="Gap between segments (s)")
                        fade_ms_inp = gr.Number(value=60, precision=0, label="Crossfade (ms)")

                    def _apply_preset(preset: str):
                        if preset == "Long Audio":
                            return (
                                gr.update(value=2400),
                                gr.update(value=True),
                                gr.update(value="sentence"),
                            )
                        return (
                            gr.update(value=1200),
                            gr.update(value=False),
                            gr.update(value="tokens"),
                        )

                    profile_sel.change(
                        _apply_preset,
                        profile_sel,
                        [max_tokens, segment_chk, segment_method],
                    )
                    infer_btn = gr.Button("Generate")
                    clear_btn = gr.Button("Clear Gallery")
                    gallery = gr.HTML(label="Outputs")
                    last_audio = gr.Audio(label="Last Audio")

                    def run_infer(*args):  # type: ignore
                        prompts: list[str] = []
                        base_idx = 2 + MAX_PROMPTS
                        pfile = args[base_idx]
                        loras = args[base_idx + 1]
                        profile = args[base_idx + 2]
                        temperature = float(args[base_idx + 3])
                        top_p = float(args[base_idx + 4])
                        rep_penalty = float(args[base_idx + 5])
                        max_tokens = int(args[base_idx + 6])
                        segment = bool(args[base_idx + 7])
                        seg_method = args[base_idx + 8]
                        seg_chars = args[base_idx + 9] or []
                        seg_min = int(args[base_idx + 10] or 0)
                        seg_max = int(args[base_idx + 11] or 50)
                        seg_gap = float(args[base_idx + 12] or 0)
                        fade_ms = int(args[base_idx + 13] or 60)

                        if profile == "Long Audio":
                            segment = True
                            seg_method = "sentence"
                            max_tokens = max(max_tokens, 2400)
                        if args[0] == "Manual":
                            num = int(args[1])
                            for box in args[2 : 2 + MAX_PROMPTS][:num]:
                                if box:
                                    prompts.append(box)
                        else:
                            prompts = load_prompts(pfile)
                        return generate_batch(
                            prompts,
                            loras,
                            temperature,
                            top_p,
                            rep_penalty,
                            max_tokens,
                            segment,
                            seg_method,
                            seg_chars,
                            seg_min,
                            seg_max,
                            seg_gap,
                            fade_ms,
                        )

                    infer_btn.click(
                        run_infer,
                        [
                            mode,
                            num_prompts,
                            *prompt_boxes,
                            prompt_list_dd,
                            lora_used,
                            profile_sel,
                            temperature,
                            top_p,
                            rep_penalty,
                            max_tokens,
                            segment_chk,
                            segment_method,
                            segment_chars,
                            seg_min_tokens,
                            seg_max_tokens,
                            seg_gap,
                            fade_ms_inp,
                        ],
                        [gallery, last_audio],
                    )

                    clear_btn.click(lambda: ("", None), None, [gallery, last_audio], queue=False)



    refresh_btn.click(
        refresh_lists,
        None,
        [local_ds, lora_used, prompt_list_dd, local_audio, local_audio_tok, auto_dataset],
    )
    stop_btn.click(stop_current, None, None)
    exit_btn.click(exit_app, None, None)
if __name__ == "__main__":
    demo.launch(server_port=18188)
