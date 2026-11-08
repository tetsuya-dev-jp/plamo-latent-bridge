"""Bridge and soft-prefix training with a frozen target backbone."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bridge import Bridge, SoftPrefix, parameter_count, save_checkpoint
from .config import Config
from .data import split_path
from .features import load_feature_array
from .prompts import INSTRUCTION, bridge_parts, soft_prefix_parts
from .target import assemble_teacher_forced_batch, load_target_model
from .utils import (
    Stopwatch,
    environment_info,
    peak_gpu_memory_gib,
    read_jsonl,
    reset_peak_gpu_memory,
    resolve_model_revision,
    set_seed,
    write_json,
)


def _rows(config: Config, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(split_path(config.paths.data_dir, split))
    limit = config.limits.for_split(split)
    return rows if limit is None else rows[:limit]


def run_id_for(condition: str, layer: int | None, prefix_length: int, seed: int) -> str:
    if condition == "bridge":
        return f"bridge_l{layer}_k{prefix_length}_s{seed}"
    return f"softprefix_k{prefix_length}_s{seed}"


def _make_prefixes(
    module: torch.nn.Module,
    condition: str,
    features: np.ndarray | None,
    indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    if condition == "bridge":
        assert features is not None
        source = torch.tensor(features[indices], dtype=torch.float32, device=device)
        return module(source)
    return module(len(indices))


def _teacher_forced_loss(
    model: Any,
    tokenizer: Any,
    module: torch.nn.Module,
    condition: str,
    parts: list[Any],
    features: np.ndarray | None,
    indices: list[int],
    with_grad: bool,
) -> tuple[torch.Tensor, int]:
    device = model.device
    prefixes = _make_prefixes(module, condition, features, indices, device)
    batch_parts = [parts[index] for index in indices]
    batch = assemble_teacher_forced_batch(
        model.get_input_embeddings(),
        [part.before for part in batch_parts],
        list(prefixes),
        [part.after for part in batch_parts],
        [part.answer for part in batch_parts],
    )
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        output = model(**batch)
    answer_tokens = int((batch["labels"] != -100).sum().item())
    return output.loss, answer_tokens


def _validation_loss(
    model: Any,
    tokenizer: Any,
    module: torch.nn.Module,
    condition: str,
    parts: list[Any],
    features: np.ndarray | None,
    batch_size: int,
) -> float:
    module.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(parts), batch_size):
        indices = list(range(start, min(start + batch_size, len(parts))))
        loss, tokens = _teacher_forced_loss(
            model, tokenizer, module, condition, parts, features, indices, with_grad=False
        )
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    module.train()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def train(
    config: Config,
    condition: str,
    seed: int,
    layer: int | None = None,
    prefix_length: int = 16,
) -> dict[str, Any]:
    if condition not in {"bridge", "soft_prefix"}:
        raise ValueError("condition must be 'bridge' or 'soft_prefix'")
    if condition == "bridge" and layer is None:
        raise ValueError("bridge training requires a source layer")
    set_seed(seed)
    reset_peak_gpu_memory()
    total_stopwatch = Stopwatch()
    load_stopwatch = Stopwatch()

    tokenizer, model = load_target_model(config)
    load_seconds = load_stopwatch.elapsed()
    train_rows = _rows(config, "train")
    val_rows = _rows(config, "val")
    features = None
    if condition == "bridge":
        feature_array = load_feature_array(
            config, "train", int(layer), prefix_length, expected_rows=train_rows
        )
        if len(feature_array) != len(train_rows):
            raise ValueError("train feature count does not match train rows")
        features = feature_array
        module: torch.nn.Module = Bridge(feature_array.shape[-1], model.config.hidden_size, prefix_length)
    else:
        module = SoftPrefix(model.config.hidden_size, prefix_length)
    module = module.to(model.device)
    module.train()

    if condition == "bridge":
        train_parts = [bridge_parts(tokenizer, str(row["answer"])) for row in train_rows]
        val_parts = [bridge_parts(tokenizer, str(row["answer"])) for row in val_rows]
    else:
        train_parts = [soft_prefix_parts(tokenizer, row["ja_question"], str(row["answer"])) for row in train_rows]
        val_parts = [soft_prefix_parts(tokenizer, row["ja_question"], str(row["answer"])) for row in val_rows]

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    run_id = run_id_for(condition, layer, prefix_length, seed)
    run_dir = Path(config.paths.runs_dir) / config.experiment_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    rng = random.Random(seed)

    for epoch in range(1, config.training.epochs + 1):
        epoch_start = Stopwatch()
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        step_losses = []
        step_tokens = 0
        for start in range(0, len(order), config.training.batch_size):
            indices = order[start : start + config.training.batch_size]
            loss, tokens = _teacher_forced_loss(
                model, tokenizer, module, condition, train_parts, features, indices, with_grad=True
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), config.training.grad_clip)
            optimizer.step()
            step_losses.append(float(loss.item()))
            step_tokens += tokens

        train_loss = sum(step_losses) / len(step_losses)
        val_features = None
        if condition == "bridge":
            val_features = load_feature_array(
                config, "val", int(layer), prefix_length, expected_rows=val_rows
            )
            if len(val_features) != len(val_parts):
                raise ValueError("val feature count does not match val rows")
        val_loss = _validation_loss(
            model, tokenizer, module, condition, val_parts, val_features, config.training.batch_size
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "steps": len(step_losses),
                "answer_tokens": step_tokens,
                "elapsed_seconds": epoch_start.elapsed(),
                "peak_gpu_memory_gib": peak_gpu_memory_gib(),
                "learning_rate": config.training.learning_rate,
            }
        )
        write_json(run_dir / "history.json", history)

    assert best_state is not None
    module.load_state_dict(best_state)
    metadata = {
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "condition": condition,
        "layer": layer,
        "prefix_length": prefix_length,
        "seed": seed,
        "train_items": len(train_rows),
        "val_items": len(val_rows),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "trainable_parameters": parameter_count(module),
        "target_model_id": config.target.model_id,
        "target_revision_requested": config.target.revision,
        "target_revision_resolved": resolve_model_revision(config.target.model_id, config.target.revision),
        "source_model_id": config.source.model_id if condition == "bridge" else None,
        "source_revision_requested": config.source.revision if condition == "bridge" else None,
        "source_revision_resolved": (
            resolve_model_revision(config.source.model_id, config.source.revision)
            if condition == "bridge"
            else None
        ),
        "instruction": INSTRUCTION,
        "training": {
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "grad_clip": config.training.grad_clip,
        },
        "load_seconds": load_seconds,
        "training_seconds": total_stopwatch.elapsed() - load_seconds,
        "elapsed_seconds": total_stopwatch.elapsed(),
        "peak_gpu_memory_gib": peak_gpu_memory_gib(),
        "environment": environment_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_checkpoint(run_dir / "checkpoints" / "best.pt", module, metadata)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "config.json", config.to_dict())
    return {"run_dir": str(run_dir), "metadata": metadata, "history": history}

