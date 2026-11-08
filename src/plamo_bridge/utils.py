"""Shared IO, determinism, environment and metric helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_cpu_threads(count: int) -> None:
    try:
        import torch

        torch.set_num_threads(count)
    except ImportError:
        pass


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "accelerate", "numpy", "scipy", "matplotlib", "numba")
        },
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["torch"]["device_name"] = props.name
            info["torch"]["total_memory_gib"] = round(props.total_memory / 1024**3, 3)
    except ImportError:
        info["torch"] = None
    return info


def resolve_model_revision(model_id: str, fallback: str) -> str:
    try:
        from huggingface_hub import model_info

        return model_info(model_id, revision=fallback).sha
    except Exception:
        if re.fullmatch(r"[0-9a-f]{40}", fallback):
            return fallback
        raise


def peak_gpu_memory_gib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except ImportError:
        return None


def reset_peak_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Stopwatch:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return round(time.perf_counter() - self.started, 3)


_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|plamo:tag|>", "<|plamo:bos|>")


def normalize_answer(text: str) -> str:
    """Strip documented formatting and canonicalize purely numeric answers."""
    normalized = text.strip()
    for token in _SPECIAL_TOKENS:
        normalized = normalized.replace(token, "")
    normalized = normalized.translate(_FULLWIDTH_DIGITS)
    normalized = normalized.replace(",", "").replace("，", "").strip()
    normalized = normalized.strip("。.")
    normalized = normalized.strip()
    if re.fullmatch(r"[+-]?\d+", normalized):
        return str(int(normalized))
    return normalized


def exact_match(prediction: str | None, reference: int | str) -> bool:
    if prediction is None:
        return False
    return normalize_answer(prediction) == str(int(reference))


_NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?")


def extract_last_number(text: str | None) -> Decimal | None:
    """Return the last numeric literal in an answer using a conservative rule."""
    if text is None:
        return None
    normalized = normalize_answer(text)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    matches = _NUMBER_PATTERN.findall(normalized)
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def last_number_exact_match(prediction: str | None, reference: int | str) -> bool:
    value = extract_last_number(prediction)
    return value is not None and value == Decimal(int(reference))
