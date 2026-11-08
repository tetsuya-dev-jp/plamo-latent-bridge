from __future__ import annotations

import torch

from plamo_bridge.pooling import pool_valid_tokens


def test_pooling_uses_only_valid_tokens_and_preserves_order():
    hidden = torch.zeros(2, 6, 1)
    hidden[0, :, 0] = torch.tensor([999.0, 999.0, 1.0, 2.0, 3.0, 4.0])
    hidden[1, :, 0] = torch.tensor([5.0, 6.0, 7.0, 999.0, 999.0, 999.0])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])

    pooled = pool_valid_tokens(hidden, mask, prefix_length=2)

    assert torch.allclose(pooled[0, :, 0], torch.tensor([1.5, 3.5]))
    assert torch.allclose(pooled[1, :, 0], torch.tensor([5.0, 6.5]))


def test_pooling_repeats_when_prefix_is_longer_than_valid_sequence():
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.ones(1, 3, dtype=torch.long)

    pooled = pool_valid_tokens(hidden, mask, prefix_length=5)

    assert pooled.shape == (1, 5, 1)
    assert torch.allclose(pooled[0, :, 0], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0]))


def test_pooling_rejects_empty_sequence():
    hidden = torch.zeros(1, 2, 1)
    mask = torch.zeros(1, 2, dtype=torch.long)
    try:
        pool_valid_tokens(hidden, mask, prefix_length=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
