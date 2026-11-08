"""Trainable projection and learned soft-prefix modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class Bridge(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, prefix_length: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.projection = nn.Linear(source_dim, target_dim)

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        return self.projection(source_features)


class SoftPrefix(nn.Module):
    def __init__(self, target_dim: int, prefix_length: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.embedding = nn.Parameter(torch.randn(prefix_length, target_dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def save_checkpoint(path: str | Path, module: nn.Module, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, module: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"])
    return payload["metadata"]
