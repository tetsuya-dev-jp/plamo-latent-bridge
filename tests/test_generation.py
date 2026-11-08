from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from plamo_bridge.target import assemble_generation_batch


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
    return Qwen2ForCausalLM(config).eval()


def _trim(row: torch.Tensor, pad_id: int = 0) -> list[int]:
    values = row.tolist()
    while values and values[-1] == pad_id:
        values.pop()
    return values


def test_left_padded_batched_generation_matches_single_examples():
    model = _tiny_model()
    embedding = model.get_input_embeddings()
    before = [[1, 2, 3], [1, 2]]
    after = [[4, 5], [4]]
    batch = assemble_generation_batch(embedding, before, None, after)
    assert "position_ids" not in batch
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]
    assert torch.equal(batch["inputs_embeds"][1, 2:], embedding(torch.tensor([1, 2, 4])))

    batched = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=6,
        eos_token_id=63,
        pad_token_id=0,
        bos_token_id=1,
    )
    singles = []
    for before_ids, after_ids in zip(before, after):
        sequence = torch.cat(
            [embedding(torch.tensor(before_ids)), embedding(torch.tensor(after_ids))], dim=0
        )
        single = model.generate(
            inputs_embeds=sequence.unsqueeze(0),
            attention_mask=torch.ones(1, sequence.shape[0], dtype=torch.long),
            do_sample=False,
            max_new_tokens=6,
            eos_token_id=63,
            pad_token_id=0,
            bos_token_id=1,
        )
        singles.append(single)
    for index, single in enumerate(singles):
        assert _trim(batched[index]) == _trim(single[0])
