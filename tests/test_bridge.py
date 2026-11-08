from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from plamo_bridge.bridge import Bridge, SoftPrefix, parameter_count
from plamo_bridge.target import assemble_teacher_forced_batch


def _tiny_model() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _batch(model, prefixes):
    return assemble_teacher_forced_batch(
        model.get_input_embeddings(),
        before_ids=[[1, 2], [1, 2]],
        prefix_embeds=list(prefixes),
        after_ids=[[3, 4, 5], [3, 4, 5]],
        answer_ids=[[6], [6, 7]],
    )


def test_only_bridge_receives_gradients_and_updates():
    model = _tiny_model()
    bridge = Bridge(source_dim=8, target_dim=32, prefix_length=3)
    source = torch.randn(2, 3, 8)
    before = bridge.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=0.01)

    loss = model(**_batch(model, bridge(source))).loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert bridge.projection.weight.grad is not None
    assert torch.isfinite(bridge.projection.weight.grad).all()
    assert not torch.equal(before, bridge.projection.weight.detach())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert parameter_count(bridge) == 8 * 32 + 32


def test_soft_prefix_receives_gradients_and_updates():
    model = _tiny_model()
    prefix = SoftPrefix(target_dim=32, prefix_length=3)
    before = prefix.embedding.detach().clone()
    optimizer = torch.optim.AdamW(prefix.parameters(), lr=0.01)

    loss = model(**_batch(model, prefix(2))).loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert prefix.embedding.grad is not None
    assert not torch.equal(before, prefix.embedding.detach())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert parameter_count(prefix) == 3 * 32
