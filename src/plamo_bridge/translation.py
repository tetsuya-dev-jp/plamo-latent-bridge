"""Actual PLaMo English translations used by the translation-pivot baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config, translation_max_new_tokens
from .features import load_source_model, source_rows
from .prompts import plamo_translation_prompt
from .utils import (
    Stopwatch,
    environment_info,
    peak_gpu_memory_gib,
    read_jsonl,
    reset_peak_gpu_memory,
    resolve_model_revision,
    sync_cuda,
    write_json,
    write_jsonl,
)


def translation_path(config: Config, split: str) -> Path:
    return Path(config.paths.data_dir) / f"translations_{split}.jsonl"


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    configured = model.generation_config.eos_token_id
    if configured is None:
        configured = tokenizer.eos_token_id
    if isinstance(configured, (list, tuple)):
        return {int(token_id) for token_id in configured}
    return {int(configured)}


def _translation_cache_problems(
    config: Config, rows: list[dict[str, Any]], existing: list[dict[str, Any]], expected_cap: int
) -> list[str]:
    required = ("id", "ja_question", "translation", "source_model_id", "source_revision", "max_new_tokens")
    problems = []
    for field in required:
        if any(field not in row for row in existing):
            problems.append(f"missing {field}")
    if problems:
        return problems
    if [row["id"] for row in existing] != [row["id"] for row in rows]:
        problems.append("id mismatch")
    if [row["ja_question"] for row in existing] != [row["ja_question"] for row in rows]:
        problems.append("ja_question mismatch")
    if any(row["source_model_id"] != config.source.model_id for row in existing):
        problems.append("source_model_id mismatch")
    if any(row["source_revision"] != config.source.revision for row in existing):
        problems.append("source_revision mismatch")
    if any(row["max_new_tokens"] != expected_cap for row in existing):
        problems.append("max_new_tokens mismatch")
    return problems


def _decode_translation(
    tokenizer: Any, model: Any, token_ids: torch.Tensor, max_new_tokens: int
) -> tuple[str, int, bool, bool]:
    pad_id = tokenizer.pad_token_id
    eos_ids = _eos_ids(model, tokenizer)
    emitted = 0
    terminated = False
    for token_id in token_ids.tolist():
        if token_id == pad_id:
            break
        emitted += 1
        if token_id in eos_ids:
            terminated = True
            break
    text = tokenizer.decode(token_ids[:emitted], skip_special_tokens=True).strip()
    hit_limit = not terminated and emitted >= max_new_tokens
    return text, emitted, terminated, hit_limit


def translate_split(config: Config, split: str, batch_size: int = 8, overwrite: bool = False) -> dict[str, Any]:
    output_path = translation_path(config, split)
    expected_cap = translation_max_new_tokens(config, split)
    if output_path.exists() and not overwrite:
        existing = read_jsonl(output_path)
        rows = source_rows(config, split)
        problems = _translation_cache_problems(config, rows, existing, expected_cap)
        if problems:
            raise RuntimeError(
                f"translation cache for split {split} is stale ({'; '.join(problems)}); "
                "rebuild it with `translate --overwrite`"
            )
        return {"split": split, "status": "skipped_matching_cache", "path": str(output_path)}

    reset_peak_gpu_memory()
    load_stopwatch = Stopwatch()
    tokenizer, model = load_source_model(config)
    load_seconds = load_stopwatch.elapsed()
    revision = resolve_model_revision(config.source.model_id, config.source.revision)
    rows = source_rows(config, split)
    prompts = [plamo_translation_prompt(tokenizer, row["ja_question"]) for row in rows]
    translations: list[dict[str, Any]] = []
    sync_cuda()
    translation_stopwatch = Stopwatch()

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        prompt_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=expected_cap,
            )
        generated = output[:, prompt_width:]
        for index, (row, generated_ids) in enumerate(zip(batch_rows, generated)):
            text, emitted, terminated, hit_limit = _decode_translation(
                tokenizer, model, generated_ids, expected_cap
            )
            translations.append(
                {
                    "id": row["id"],
                    "split": split,
                    "ja_question": row["ja_question"],
                    "translation": text,
                    "prompt_tokens": int(inputs["attention_mask"][index].sum()),
                    "generated_tokens": emitted,
                    "terminated": terminated,
                    "hit_translation_limit": hit_limit,
                    "max_new_tokens": expected_cap,
                    "source_model_id": config.source.model_id,
                    "source_revision": revision,
                }
            )
    sync_cuda()
    translation_seconds = translation_stopwatch.elapsed()

    write_jsonl(output_path, translations)
    meta = {
        "split": split,
        "status": "translated",
        "count": len(translations),
        "source_model_id": config.source.model_id,
        "source_revision": revision,
        "prompt_format": "official PFML3 translation request, target_lang=en, source_lang=ja",
        "max_new_tokens": expected_cap,
        "decoding": "greedy",
        "load_seconds": load_seconds,
        "translation_seconds": translation_seconds,
        "hit_limit_count": sum(row["hit_translation_limit"] for row in translations),
        "terminated_count": sum(row["terminated"] for row in translations),
        "peak_gpu_memory_gib": peak_gpu_memory_gib(),
        "environment": environment_info(),
    }
    write_json(Path(config.paths.data_dir) / f"translations_{split}_meta.json", meta)
    return meta
