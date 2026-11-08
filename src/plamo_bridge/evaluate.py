"""Textual baselines and bridge evaluation with per-item predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bridge import Bridge, SoftPrefix, load_checkpoint, parameter_count
from .config import Config, answer_max_new_tokens
from .data import split_path
from .features import load_feature_array
from .prompts import INSTRUCTION, bridge_parts, soft_prefix_parts
from .target import generate_embedding_answers, generate_text_answers, load_target_model
from .train import run_id_for
from .utils import (
    Stopwatch,
    environment_info,
    exact_match,
    extract_last_number,
    last_number_exact_match,
    normalize_answer,
    peak_gpu_memory_gib,
    read_jsonl,
    reset_peak_gpu_memory,
    resolve_model_revision,
    set_seed,
    sync_cuda,
    write_json,
    write_jsonl,
)


def _rows(config: Config, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(split_path(config.paths.data_dir, split))
    limit = config.limits.for_split(split)
    return rows if limit is None else rows[:limit]


def _prediction_path(config: Config, condition: str, split: str) -> Path:
    return Path(config.paths.results_dir) / config.experiment_id / "predictions" / f"{condition}_{split}.jsonl"


def _write_meta(config: Config, condition: str, split: str, meta: dict[str, Any]) -> None:
    path = Path(config.paths.results_dir) / config.experiment_id / "meta" / f"{condition}_{split}.json"
    write_json(path, meta)


def _prediction_row(
    row: dict[str, Any],
    condition: str,
    prediction: str,
    generated_tokens: int,
    terminated: bool,
    hit_limit: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    last_number = extract_last_number(prediction)
    return {
        "id": row["id"],
        "split": row["split"],
        "operation": row["operation"],
        "template_id": row["template_id"],
        "condition": condition,
        "gold": row["answer"],
        "prediction_raw": prediction,
        "prediction_normalized": normalize_answer(prediction),
        "correct": exact_match(prediction, row["answer"]),
        "last_number": str(last_number) if last_number is not None else None,
        "last_number_correct": last_number_exact_match(prediction, row["answer"]),
        "generated_tokens": generated_tokens,
        "terminated": terminated,
        "hit_limit": hit_limit,
        "max_new_tokens": max_new_tokens,
    }


def evaluate_text(config: Config, conditions: list[str], splits: list[str], batch_size: int = 8) -> list[dict[str, Any]]:
    reset_peak_gpu_memory()
    load_stopwatch = Stopwatch()
    tokenizer, model = load_target_model(config)
    load_seconds = load_stopwatch.elapsed()
    revision = resolve_model_revision(config.target.model_id, config.target.revision)
    results = []
    for condition in conditions:
        if condition not in {"text_ja", "text_en", "text_plamo_en"}:
            raise ValueError(f"unknown text condition: {condition}")
        for split in splits:
            rows = _rows(config, split)
            questions = _text_questions(config, condition, split, rows)
            max_new_tokens = answer_max_new_tokens(config, split)
            predictions = []
            sync_cuda()
            evaluation_stopwatch = Stopwatch()
            for start in range(0, len(rows), batch_size):
                batch_questions = questions[start : start + batch_size]
                answers = generate_text_answers(model, tokenizer, batch_questions, max_new_tokens)
                for row, (prediction, generated_tokens, terminated, hit_limit) in zip(
                    rows[start : start + batch_size], answers
                ):
                    predictions.append(
                        _prediction_row(
                            row,
                            condition,
                            prediction,
                            generated_tokens,
                            terminated,
                            hit_limit,
                            max_new_tokens,
                        )
                    )
            sync_cuda()
            evaluation_seconds = evaluation_stopwatch.elapsed()
            write_jsonl(_prediction_path(config, condition, split), predictions)
            meta = {
                "condition": condition,
                "split": split,
                "count": len(rows),
                "target_model_id": config.target.model_id,
                "target_revision_requested": config.target.revision,
                "target_revision_resolved": revision,
                "instruction": INSTRUCTION,
                "max_new_tokens": max_new_tokens,
                "load_seconds": load_seconds,
                "evaluation_seconds": evaluation_seconds,
                "peak_gpu_memory_gib": peak_gpu_memory_gib(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_meta(config, condition, split, meta)
            results.append(meta)
    write_environment(config)
    return results


def _text_questions(
    config: Config, condition: str, split: str, rows: list[dict[str, Any]]
) -> list[str]:
    if condition == "text_ja":
        return [row["ja_question"] for row in rows]
    if condition == "text_en":
        if any(row.get("en_question") is None for row in rows):
            raise ValueError("English questions are unavailable for this split")
        return [row["en_question"] for row in rows]
    translation_path = Path(config.paths.data_dir) / f"translations_{split}.jsonl"
    if not translation_path.exists():
        raise FileNotFoundError(f"missing translations: run translate --split {split} first")
    translations = {row["id"]: row["translation"] for row in read_jsonl(translation_path)}
    missing = [row["id"] for row in rows if row["id"] not in translations]
    if missing:
        raise ValueError(f"missing {len(missing)} translations for split {split}")
    return [translations[row["id"]] for row in rows]


def _checkpoint_path(config: Config, condition: str, layer: int | None, prefix_length: int, seed: int) -> Path:
    run_id = run_id_for(condition, layer, prefix_length, seed)
    return Path(config.paths.runs_dir) / config.experiment_id / run_id / "checkpoints" / "best.pt"


def evaluate_bridge(
    config: Config,
    condition: str,
    split: str,
    layer: int | None,
    prefix_length: int,
    seed: int,
    untrained: bool = False,
    shuffle: bool = False,
    batch_size: int = 8,
) -> dict[str, Any]:
    if condition not in {"bridge", "soft_prefix"}:
        raise ValueError("condition must be 'bridge' or 'soft_prefix'")
    if condition == "bridge" and layer is None:
        raise ValueError("bridge evaluation requires a source layer")
    if untrained and shuffle:
        raise ValueError("untrained and shuffled controls are separate")

    set_seed(seed)
    reset_peak_gpu_memory()
    total_stopwatch = Stopwatch()
    load_stopwatch = Stopwatch()
    tokenizer, model = load_target_model(config)
    load_seconds = load_stopwatch.elapsed()
    target_revision = resolve_model_revision(config.target.model_id, config.target.revision)
    rows = _rows(config, split)
    features = None
    mapping_path = None
    shuffled_source_ids = None
    if condition == "bridge":
        features = load_feature_array(
            config, split, int(layer), prefix_length, expected_rows=rows
        )
        if len(features) != len(rows):
            raise ValueError("feature count does not match rows")
        module: torch.nn.Module = Bridge(features.shape[-1], model.config.hidden_size, prefix_length)
        if shuffle:
            mapping_path, shuffled_source_ids = _shuffle_features(
                config, rows, features, seed, split, int(layer), prefix_length
            )
            mapping = np.load(mapping_path.with_suffix(".npy"))
            features = features[mapping]
    else:
        module = SoftPrefix(model.config.hidden_size, prefix_length)

    checkpoint_path = _checkpoint_path(config, condition, layer, prefix_length, seed)
    if not untrained:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        load_checkpoint(checkpoint_path, module)
    module = module.to(model.device)
    module.eval()

    suffix = "_untrained" if untrained else "_shuffled" if shuffle else ""
    condition_name = f"{run_id_for(condition, layer, prefix_length, seed)}{suffix}"
    parts = _evaluation_parts(tokenizer, condition, rows)
    max_new_tokens = answer_max_new_tokens(config, split)
    predictions = []
    sync_cuda()
    evaluation_stopwatch = Stopwatch()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_parts = parts[start : start + batch_size]
        indices = list(range(start, start + len(batch_rows)))
        if condition == "bridge":
            source = torch.tensor(features[indices], dtype=torch.float32, device=model.device)
            prefixes = module(source)
        else:
            prefixes = module(len(batch_rows))
        answers = generate_embedding_answers(
            model,
            tokenizer,
            batch_parts,
            list(prefixes),
            max_new_tokens,
        )
        for offset, (row, (prediction, generated_tokens, terminated, hit_limit)) in enumerate(
            zip(batch_rows, answers)
        ):
            prediction_row = _prediction_row(
                row,
                condition_name,
                prediction,
                generated_tokens,
                terminated,
                hit_limit,
                max_new_tokens,
            )
            if shuffled_source_ids is not None:
                prediction_row["shuffled_source_id"] = shuffled_source_ids[start + offset]
            predictions.append(prediction_row)
    sync_cuda()
    evaluation_seconds = evaluation_stopwatch.elapsed()

    if mapping_path is not None:
        _validate_shuffle_mapping(mapping_path, predictions)

    write_jsonl(_prediction_path(config, condition_name, split), predictions)
    meta = {
        "condition": condition_name,
        "base_condition": condition,
        "split": split,
        "count": len(rows),
        "layer": layer,
        "prefix_length": prefix_length,
        "seed": seed,
        "untrained": untrained,
        "shuffled": shuffle,
        "shuffle_mapping": str(mapping_path) if mapping_path else None,
        "checkpoint": None if untrained else str(checkpoint_path),
        "trainable_parameters": parameter_count(module),
        "target_model_id": config.target.model_id,
        "target_revision_requested": config.target.revision,
        "target_revision_resolved": target_revision,
        "instruction": INSTRUCTION,
        "max_new_tokens": max_new_tokens,
        "load_seconds": load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "setup_seconds": round(total_stopwatch.elapsed() - load_seconds - evaluation_seconds, 3),
        "elapsed_seconds": total_stopwatch.elapsed(),
        "peak_gpu_memory_gib": peak_gpu_memory_gib(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_meta(config, condition_name, split, meta)
    write_environment(config)
    return meta


def _validate_shuffle_mapping(mapping_path: Path, predictions: list[dict[str, Any]]) -> None:
    with mapping_path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    expected = {entry["target_item_id"]: entry["source_item_id"] for entry in mapping}
    for prediction in predictions:
        item_id = prediction["id"]
        if item_id not in expected:
            raise ValueError(f"shuffle mapping is missing item {item_id}")
        if prediction["shuffled_source_id"] != expected[item_id]:
            raise ValueError(f"shuffle mapping mismatch for item {item_id}")


def _evaluation_parts(tokenizer: Any, condition: str, rows: list[dict[str, Any]]) -> list[Any]:
    if condition == "bridge":
        return [bridge_parts(tokenizer) for _ in rows]
    return [soft_prefix_parts(tokenizer, row["ja_question"]) for row in rows]


def _shuffle_features(
    config: Config,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    seed: int,
    split: str,
    layer: int,
    prefix_length: int,
) -> tuple[Path, list[str]]:
    rng = np.random.default_rng(seed + 9001)
    permutation = rng.permutation(len(rows))
    if len(rows) > 1:
        while np.all(permutation == np.arange(len(rows))):
            permutation = rng.permutation(len(rows))
    mapping = []
    for target_index, source_index in enumerate(permutation.tolist()):
        mapping.append(
            {
                "target_index": target_index,
                "source_index": source_index,
                "target_item_id": rows[target_index]["id"],
                "source_item_id": rows[source_index]["id"],
            }
        )
    directory = Path(config.paths.results_dir) / config.experiment_id / "shuffle"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shuffle_{split}_l{layer}_k{prefix_length}_s{seed}.json"
    write_json(path, mapping)
    np.save(path.with_suffix(".npy"), permutation)
    return path, [entry["source_item_id"] for entry in mapping]


def write_environment(config: Config) -> None:
    path = Path(config.paths.results_dir) / config.experiment_id / "environment.json"
    write_json(
        path,
        {
            "config": config.to_dict(),
            "environment": environment_info(),
            "source_model_id": config.source.model_id,
            "source_revision_requested": config.source.revision,
            "target_model_id": config.target.model_id,
            "target_revision_requested": config.target.revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
