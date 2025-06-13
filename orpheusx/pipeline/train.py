from __future__ import annotations

import time
import os
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
import torchaudio.transforms as T
from unsloth import FastLanguageModel, is_bfloat16_supported
from snac import SNAC

from tools.logger_utils import get_logger
from orpheusx.constants import (
    DATASETS_DIR,
    LORA_DIR,
    CACHE_DIR,
    MODEL_NAME,
    STOP_FLAG as _STOP_FLAG,
)

logger = get_logger("pipeline.train")

# -----------------------------------------------------------------------------
# Single-dataset LoRA training
# -----------------------------------------------------------------------------

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
    """Train a single LoRA adapter on *dataset_source*.

    The function is mostly lifted from the original *gradio_app.py* script.
    """
    logger.info("Training LoRA %s from %s", lora_name, dataset_source)
    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if is_local:
        dataset = load_from_disk(dataset_source)
    else:
        dataset = load_dataset(dataset_source, split="train", cache_dir=str(DATASETS_DIR))

    # ------------------------------------------------------------------
    # Load base model & prepare for LoRA fine-tuning
    # ------------------------------------------------------------------
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
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

    # ------------------------------------------------------------------
    # Audio to discrete token conversion via SNAC
    # ------------------------------------------------------------------
    ds_sample_rate = dataset[0]["audio"]["sampling_rate"]
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir=str(CACHE_DIR))
    snac_model = snac_model.to("cuda")

    import locale  # noqa: WPS433  (same workaround as original code)

    locale.getpreferredencoding = lambda: "UTF-8"  # type: ignore  # pragma: no cover

    # -------------------------- helpers -------------------------------
    def _tokenise_audio(waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
        resample = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
        waveform = resample(waveform).unsqueeze(0).to("cuda")
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        all_codes: list[int] = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
            all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))
        return all_codes

    def _add_codes(example):
        codes_list = None
        try:
            answer_audio = example.get("audio")
            if answer_audio and "array" in answer_audio:
                codes_list = _tokenise_audio(answer_audio["array"])
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping row due to error: %s", exc)
        example["codes_list"] = codes_list
        return example

    dataset = dataset.map(_add_codes, remove_columns=["audio"])

    TOKENISER_LENGTH = 128256
    start_of_text = 128000
    end_of_text = 128009
    start_of_speech = TOKENISER_LENGTH + 1
    end_of_speech = TOKENISER_LENGTH + 2
    start_of_human = TOKENISER_LENGTH + 3
    end_of_human = TOKENISER_LENGTH + 4
    start_of_ai = TOKENISER_LENGTH + 5
    end_of_ai = TOKENISER_LENGTH + 6

    # Filter out rows where codes could not be generated
    dataset = dataset.filter(lambda x: x["codes_list"] is not None and len(x["codes_list"]) > 0)

    # Remove duplicate frames helper (unchanged from original)
    def _remove_duplicate_frames(example):
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")
        result = vals[:7]
        for i in range(7, len(vals), 7):
            current_first = vals[i]
            previous_first = result[-7]
            if current_first != previous_first:
                result.extend(vals[i : i + 7])
        example["codes_list"] = result
        return example

    dataset = dataset.map(_remove_duplicate_frames)

    # Build text+audio input IDs
    def _create_input_ids(example):
        text_prompt = (
            f"{example.get('source', '')}: {example['text']}" if "source" in example else example["text"]
        )
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)
        example["text_tokens"] = text_ids
        input_ids = (
            [start_of_human]
            + example["text_tokens"]
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + example["codes_list"]
            + [end_of_speech]
            + [end_of_ai]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        return example

    dataset = dataset.map(_create_input_ids, remove_columns=["text", "codes_list"])

    before_len = len(dataset)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= 2048)
    skipped = before_len - len(dataset)
    if skipped:
        logger.info("Skipped %d sample(s) >2048 tokens", skipped)

    cols = ["input_ids", "labels", "attention_mask"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in cols])

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

# -----------------------------------------------------------------------------
# Multiple-dataset convenience wrapper (loops over train_lora_single)
# -----------------------------------------------------------------------------

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
    """Train several LoRAs based on *hf_links* and/or local *datasets*."""
    from orpheusx import constants as _c  # avoid circular import

    dataset_info: list[tuple[str, str, bool]] = []
    links = [l.strip() for l in hf_links.splitlines() if l.strip()]
    for link in links:
        name = link.split("/")[-1]
        dataset_info.append((link, name, False))
    for ds in local_datasets:
        dataset_info.append((str(DATASETS_DIR / ds), ds, True))
    if not dataset_info:
        return "No datasets selected."

    msgs: list[str] = []
    total = len(dataset_info)
    logger.info("Training %d LoRA(s)", total)
    for idx, (src, name, is_local) in enumerate(dataset_info, start=1):
        if _c.STOP_FLAG:
            _c.STOP_FLAG = False
            return "Stopped"
        start = time.perf_counter()
        try:
            _ = train_lora_single(
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
        except Exception as exc:  # pragma: no cover
            elapsed = time.perf_counter() - start
            logger.exception("Training failed for %s after %.2fs", name, elapsed)
            msgs.append(f"{name}: failed ({exc})")
    return "\n".join(msgs)

__all__ = [
    "train_lora_single",
    "train_loras",
] 