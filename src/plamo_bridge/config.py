"""Configuration loading from small JSON files with one-level inheritance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Paths:
    data_dir: Path
    features_dir: Path
    runs_dir: Path
    results_dir: Path


@dataclass
class SourceConfig:
    model_id: str
    revision: str
    layers: list[int]
    prefix_lengths: list[int]
    dtype: str


@dataclass
class TargetConfig:
    model_id: str
    revision: str
    dtype: str
    attn_implementation: str


@dataclass
class DataConfig:
    train: int
    val: int
    test: int
    heldout: int
    seed: int


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float


@dataclass
class GenerationConfig:
    max_new_tokens: int
    mgsm_max_new_tokens: int
    translation_max_new_tokens: int
    mgsm_translation_max_new_tokens: int


@dataclass
class EvaluationConfig:
    bootstrap_resamples: int
    bootstrap_seed: int


@dataclass
class Limits:
    train: int | None = None
    val: int | None = None
    test: int | None = None
    heldout: int | None = None
    mgsm: int | None = None

    def for_split(self, split: str) -> int | None:
        return getattr(self, split, None)


@dataclass
class Config:
    experiment_id: str
    seed: int
    paths: Paths
    source: SourceConfig
    target: TargetConfig
    data: DataConfig
    training: TrainingConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    limits: Limits = field(default_factory=Limits)
    cpu_threads: int = 6

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("data_dir", "features_dir", "runs_dir", "results_dir"):
            result["paths"][name] = str(result["paths"][name])
        return result


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _load_raw(path.parent / parent)
    return _deep_merge(base, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    raw = _load_raw(Path(path))
    paths = Paths(**raw["paths"])
    limits_raw = raw.get("limits") or {}
    return Config(
        experiment_id=raw["experiment_id"],
        seed=raw["seed"],
        paths=paths,
        source=SourceConfig(**raw["source"]),
        target=TargetConfig(**raw["target"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        generation=GenerationConfig(**raw["generation"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        limits=Limits(**limits_raw),
        cpu_threads=raw.get("cpu_threads", 6),
    )


def answer_max_new_tokens(config: Config, split: str) -> int:
    return config.generation.mgsm_max_new_tokens if split == "mgsm" else config.generation.max_new_tokens


def translation_max_new_tokens(config: Config, split: str) -> int:
    return (
        config.generation.mgsm_translation_max_new_tokens
        if split == "mgsm"
        else config.generation.translation_max_new_tokens
    )
