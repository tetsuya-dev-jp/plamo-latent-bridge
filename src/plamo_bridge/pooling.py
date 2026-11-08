"""Ordered mean pooling over valid, non-padding source tokens."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def pool_valid_tokens(hidden_states: Tensor, attention_mask: Tensor, prefix_length: int) -> Tensor:
    """Pool each sequence's valid tokens into a fixed ordered prefix.

    Tokens are split into ``prefix_length`` contiguous chunks in order. Each
    chunk is mean-pooled. When a sequence is shorter than the requested prefix,
    chunks may repeat the nearest valid token so that the prefix length and
    order are fixed without consulting padding states.
    """
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must match the first two hidden-state dimensions")
    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")

    pooled = []
    for batch_index in range(hidden_states.shape[0]):
        valid = hidden_states[batch_index][attention_mask[batch_index].bool()]
        length = valid.shape[0]
        if length == 0:
            raise ValueError("a sequence has no valid tokens")
        chunks = []
        for chunk_index in range(prefix_length):
            start = chunk_index * length // prefix_length
            end = (chunk_index + 1) * length // prefix_length
            if end <= start:
                chunks.append(valid[start])
            else:
                chunks.append(valid[start:end].float().mean(dim=0))
        pooled.append(torch.stack(chunks))
    return torch.stack(pooled)
