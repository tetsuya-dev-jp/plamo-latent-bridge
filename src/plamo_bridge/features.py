"""Cached, detached source hidden-state extraction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config
from .data import split_path
from .pooling import pool_valid_tokens
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
)

POOLING_RULE = (
    "Ordered contiguous bins over valid non-padding source tokens spanning the full "
    "valid sequence, including its final state. Prompt-format tokens are included. "
    "If the requested prefix is longer than the valid sequence, bins repeat the "
    "nearest valid token."
)
PROMPT_FORMAT = "PFML3 translation request via the official source chat template, Japanese question only"
MEASUREMENT_FIELDS = (
    "created_at",
    "prompt_example",
    "token_length_min",
    "token_length_max",
    "extraction_seconds",
    "load_seconds",
    "peak_gpu_memory_gib",
    "environment",
)


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_source_model(config: Config) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.source.model_id,
        revision=config.source.revision,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.source.model_id,
        revision=config.source.revision,
        trust_remote_code=True,
        dtype=_torch_dtype(config.source.dtype),
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return tokenizer, model


def source_rows(config: Config, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(split_path(config.paths.data_dir, split))
    limit = config.limits.for_split(split)
    return rows if limit is None else rows[:limit]


def feature_path(config: Config, split: str, layer: int, prefix_length: int) -> Path:
    return Path(config.paths.features_dir) / f"{split}_layer{layer}_k{prefix_length}.npy"


def manifest_path(config: Config, split: str) -> Path:
    return Path(config.paths.features_dir) / f"{split}_manifest.json"


def _read_manifest(config: Config, split: str) -> dict[str, Any] | None:
    path = manifest_path(config, split)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_problems(
    config: Config,
    split: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    revision: str | None = None,
) -> list[str]:
    required = {
        "item_ids": [row["id"] for row in rows],
        "question_texts": [row["ja_question"] for row in rows],
        "count": len(rows),
        "layers": config.source.layers,
        "prefix_lengths": config.source.prefix_lengths,
        "source_model_id": config.source.model_id,
        "source_revision_requested": config.source.revision,
        "source_dtype": config.source.dtype,
        "pooling_rule": POOLING_RULE,
        "prompt_format": PROMPT_FORMAT,
    }
    if revision is not None:
        required["source_revision_resolved"] = revision
    problems = []
    for field, expected in required.items():
        if field not in manifest:
            problems.append(f"missing {field}")
        elif manifest[field] != expected:
            problems.append(f"{field} mismatch")
    if revision is None and "source_revision_resolved" not in manifest:
        problems.append("missing source_revision_resolved")
    measurement = manifest.get("measurement")
    if not isinstance(measurement, dict):
        problems.append("missing measurement")
    else:
        for field in MEASUREMENT_FIELDS:
            if field not in measurement:
                problems.append(f"missing measurement.{field}")
    return problems


def _split_metadata(manifest: dict[str, Any], status: str) -> dict[str, Any]:
    measurement = dict(manifest["measurement"])
    metadata = {
        "source_model_id": manifest["source_model_id"],
        "source_revision_requested": manifest["source_revision_requested"],
        "source_revision_resolved": manifest["source_revision_resolved"],
        "source_dtype": manifest["source_dtype"],
        "pooling_rule": manifest["pooling_rule"],
        "prompt_format": manifest["prompt_format"],
        "count": manifest["count"],
        "status": status,
    }
    metadata.update(measurement)
    return metadata


def _skipped_split_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    return _split_metadata(manifest, "skipped_matching_cache")


def _stale_cache_error(split: str, problems: list[str]) -> RuntimeError:
    detail = "; ".join(problems) if problems else "cache contents are incomplete"
    return RuntimeError(
        f"feature cache for split {split} is stale ({detail}); "
        "rebuild it with `features --overwrite`"
    )


def extract_source_features(
    config: Config,
    splits: list[str],
    batch_size: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    reset_peak_gpu_memory()
    load_timer = Stopwatch()
    tokenizer, model = load_source_model(config)
    load_seconds = load_timer.elapsed()
    revision = resolve_model_revision(config.source.model_id, config.source.revision)
    feature_dir = Path(config.paths.features_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    run_environment = environment_info()
    meta: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_id": config.source.model_id,
        "source_revision_requested": config.source.revision,
        "source_revision_resolved": revision,
        "layers": config.source.layers,
        "prefix_lengths": config.source.prefix_lengths,
        "dtype": config.source.dtype,
        "splits": {},
        "prompt_format": PROMPT_FORMAT,
        "pooling_rule": POOLING_RULE,
        "load_seconds": load_seconds,
        "environment": run_environment,
    }

    for split in splits:
        rows = source_rows(config, split)
        manifest = _read_manifest(config, split)
        problems = (
            ["missing manifest"]
            if manifest is None
            else _manifest_problems(config, split, rows, manifest, revision)
        )
        if not overwrite and not problems:
            meta["splits"][split] = _skipped_split_metadata(manifest)
            continue
        existing = [
            feature_path(config, split, layer, prefix_length)
            for layer in config.source.layers
            for prefix_length in config.source.prefix_lengths
        ]
        if not overwrite and any(path.exists() for path in existing):
            raise _stale_cache_error(split, problems)

        prompts = [plamo_translation_prompt(tokenizer, row["ja_question"]) for row in rows]
        cached: dict[tuple[int, int], list[np.ndarray]] = {
            (layer, prefix_length): []
            for layer in config.source.layers
            for prefix_length in config.source.prefix_lengths
        }
        token_lengths: list[int] = []
        sync_cuda()
        extraction_timer = Stopwatch()
        for start in range(0, len(rows), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
            token_lengths.extend(inputs["attention_mask"].sum(dim=1).tolist())
            for layer in config.source.layers:
                hidden = outputs.hidden_states[layer]
                for prefix_length in config.source.prefix_lengths:
                    pooled = pool_valid_tokens(hidden, inputs["attention_mask"], prefix_length)
                    cached[(layer, prefix_length)].append(pooled.float().cpu().numpy())
        sync_cuda()
        extraction_seconds = extraction_timer.elapsed()

        for (layer, prefix_length), chunks in cached.items():
            array = np.concatenate(chunks, axis=0)
            np.save(feature_path(config, split, layer, prefix_length), array)
        np.save(feature_dir / f"{split}_token_lengths.npy", np.asarray(token_lengths, dtype=np.int32))
        split_manifest = {
            "item_ids": [row["id"] for row in rows],
            "question_texts": [row["ja_question"] for row in rows],
            "count": len(rows),
            "layers": config.source.layers,
            "prefix_lengths": config.source.prefix_lengths,
            "source_model_id": config.source.model_id,
            "source_revision_requested": config.source.revision,
            "source_revision_resolved": revision,
            "source_dtype": config.source.dtype,
            "pooling_rule": POOLING_RULE,
            "prompt_format": PROMPT_FORMAT,
            "measurement": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "prompt_example": prompts[0] if prompts else None,
                "token_length_min": int(min(token_lengths)) if token_lengths else None,
                "token_length_max": int(max(token_lengths)) if token_lengths else None,
                "extraction_seconds": extraction_seconds,
                "load_seconds": load_seconds,
                "peak_gpu_memory_gib": peak_gpu_memory_gib(),
                "environment": run_environment,
            },
        }
        write_json(manifest_path(config, split), split_manifest)
        meta["splits"][split] = _split_metadata(split_manifest, "extracted")

    meta["peak_gpu_memory_gib"] = peak_gpu_memory_gib()
    write_json(feature_dir / "extraction_meta.json", meta)
    return meta


def load_feature_array(
    config: Config,
    split: str,
    layer: int,
    prefix_length: int,
    expected_rows: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    if expected_rows is not None:
        manifest = _read_manifest(config, split)
        if manifest is None:
            raise ValueError(
                f"feature cache for split {split} is stale (missing manifest); "
                "rebuild it with `features --overwrite`"
            )
        problems = _manifest_problems(config, split, expected_rows, manifest)
        if problems:
            raise ValueError(
                f"feature cache for split {split} is stale ({'; '.join(problems)}); "
                "rebuild it with `features --overwrite`"
            )
    return np.load(feature_path(config, split, layer, prefix_length))


def load_token_lengths(config: Config, split: str) -> np.ndarray:
    return np.load(Path(config.paths.features_dir) / f"{split}_token_lengths.npy")
