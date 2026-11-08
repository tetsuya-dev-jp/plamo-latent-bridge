"""Frozen target model loading, embedding batches and free-running decoding."""

from __future__ import annotations

from typing import Any

import torch

from .config import Config
from .prompts import ensure_source_marker, text_prompt


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_target_model(config: Config) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.target.model_id,
        revision=config.target.revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.target.model_id,
        revision=config.target.revision,
        dtype=_torch_dtype(config.target.dtype),
        attn_implementation=config.target.attn_implementation,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    ensure_source_marker(tokenizer)
    return tokenizer, model


def assemble_teacher_forced_batch(
    embedding: torch.nn.Module,
    before_ids: list[list[int]],
    prefix_embeds: list[torch.Tensor] | None,
    after_ids: list[list[int]],
    answer_ids: list[list[int]],
) -> dict[str, torch.Tensor]:
    """Build inputs_embeds and labels with prompt positions masked.

    The labels are placed on answer tokens only. Hugging Face shifts labels
    internally, so the final instruction position predicts the first answer
    token while every prefix and instruction position remains ignored.
    """
    sequences: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    dtype = embedding.weight.dtype
    device = embedding.weight.device
    for index, (before, after, answer) in enumerate(zip(before_ids, after_ids, answer_ids)):
        parts = [embedding(torch.tensor(before, device=device))]
        prefix_length = 0
        if prefix_embeds is not None:
            prefix = prefix_embeds[index].to(device=device, dtype=dtype)
            parts.append(prefix)
            prefix_length = prefix.shape[0]
        parts.append(embedding(torch.tensor(after, device=device)))
        answer_tensor = embedding(torch.tensor(answer, device=device))
        parts.append(answer_tensor)
        sequence = torch.cat(parts, dim=0)
        labels = torch.full((sequence.shape[0],), -100, dtype=torch.long, device=device)
        answer_start = len(before) + prefix_length + len(after)
        labels[answer_start : answer_start + len(answer)] = torch.tensor(
            answer, dtype=torch.long, device=device
        )
        sequences.append(sequence)
        label_rows.append(labels)

    max_length = max(sequence.shape[0] for sequence in sequences)
    batch = len(sequences)
    inputs_embeds = torch.zeros((batch, max_length, embedding.weight.shape[1]), dtype=dtype, device=device)
    labels = torch.full((batch, max_length), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_length), dtype=torch.long, device=device)
    for index, (sequence, label_row) in enumerate(zip(sequences, label_rows)):
        length = sequence.shape[0]
        inputs_embeds[index, :length] = sequence
        labels[index, :length] = label_row
        attention_mask[index, :length] = 1
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
    }


def assemble_generation_batch(
    embedding: torch.nn.Module,
    before_ids: list[list[int]],
    prefix_embeds: list[torch.Tensor] | None,
    after_ids: list[list[int]],
) -> dict[str, torch.Tensor]:
    sequences: list[torch.Tensor] = []
    dtype = embedding.weight.dtype
    device = embedding.weight.device
    for index, (before, after) in enumerate(zip(before_ids, after_ids)):
        parts = [embedding(torch.tensor(before, device=device))]
        if prefix_embeds is not None:
            parts.append(prefix_embeds[index].to(device=device, dtype=dtype))
        parts.append(embedding(torch.tensor(after, device=device)))
        sequences.append(torch.cat(parts, dim=0))
    max_length = max(sequence.shape[0] for sequence in sequences)
    batch = len(sequences)
    inputs_embeds = torch.zeros((batch, max_length, embedding.weight.shape[1]), dtype=dtype, device=device)
    attention_mask = torch.zeros((batch, max_length), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        length = sequence.shape[0]
        # Decoder-only batched generation requires left padding. Let generation
        # derive position ids from this attention mask on every step.
        inputs_embeds[index, max_length - length :] = sequence
        attention_mask[index, max_length - length :] = 1
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }


def _decode_generated(
    tokenizer: Any, generated: torch.Tensor, max_new_tokens: int
) -> list[tuple[str, int, bool, bool]]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    results = []
    for row in generated:
        token_ids = row.tolist()
        emitted = 0
        terminated = False
        for token_id in token_ids:
            if token_id == pad_id:
                break
            emitted += 1
            if token_id == eos_id:
                terminated = True
                break
        text = tokenizer.decode(token_ids[:emitted], skip_special_tokens=True).strip()
        hit_limit = not terminated and emitted >= max_new_tokens
        results.append((text, emitted, terminated, hit_limit))
    return results


def generate_embedding_answers(
    model: Any,
    tokenizer: Any,
    parts: list[Any],
    prefix_embeds: list[torch.Tensor] | None,
    max_new_tokens: int,
) -> list[tuple[str, int, bool, bool]]:
    embedding = model.get_input_embeddings()
    batch = assemble_generation_batch(
        embedding,
        [part.before for part in parts],
        prefix_embeds,
        [part.after for part in parts],
    )
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return _decode_generated(tokenizer, generated, max_new_tokens)


def generate_text_answers(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    max_new_tokens: int,
) -> list[tuple[str, int, bool, bool]]:
    prompts = [text_prompt(tokenizer, question) for question in questions]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return _decode_generated(tokenizer, generated[:, prompt_width:], max_new_tokens)
