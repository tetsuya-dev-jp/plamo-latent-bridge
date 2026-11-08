from __future__ import annotations

import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

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
    return Qwen2ForCausalLM(config).eval()


def test_labels_mask_prefix_and_instruction_and_loss_aligns_causally():
    model = _tiny_model()
    embedding = model.get_input_embeddings()
    prefix = torch.randn(2, 3, 32)
    batch = assemble_teacher_forced_batch(
        embedding,
        before_ids=[[1, 2], [1, 2]],
        prefix_embeds=[prefix[0], prefix[1]],
        after_ids=[[3, 4, 5], [3, 4, 5]],
        answer_ids=[[6], [6, 7]],
    )

    answer_start = 2 + 3 + 3
    assert torch.all(batch["labels"][:, :answer_start] == -100)
    assert batch["labels"][0, answer_start].item() == 6
    assert batch["labels"][1, answer_start].item() == 6
    assert batch["labels"][1, answer_start + 1].item() == 7
    assert batch["labels"][0, answer_start + 1].item() == -100

    output = model(**batch)
    manual = F.cross_entropy(
        output.logits[:, :-1, :].reshape(-1, output.logits.shape[-1]),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(output.loss, manual, atol=1e-6)


def test_answer_length_does_not_shift_labels():
    model = _tiny_model()
    batch = assemble_teacher_forced_batch(
        model.get_input_embeddings(),
        before_ids=[[1], [1]],
        prefix_embeds=None,
        after_ids=[[2, 3], [2, 3]],
        answer_ids=[[4], [4, 5, 6]],
    )
    assert batch["labels"][0, 3].item() == 4
    assert batch["labels"][0, 4].item() == -100
    assert batch["labels"][1, 3].item() == 4
    assert batch["labels"][1, 5].item() == 6
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
