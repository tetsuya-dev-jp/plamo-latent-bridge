"""Target chat prompts, bridge insertion, and PLaMo translation requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INSTRUCTION = "Solve the following arithmetic problem. Give only the final number."
SOURCE_MARKER = "<|source_prefix|>"


@dataclass
class PromptParts:
    before: list[int]
    after: list[int]
    answer: list[int]


def ensure_source_marker(tokenizer: Any) -> int:
    marker_id = tokenizer.convert_tokens_to_ids(SOURCE_MARKER)
    if marker_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [SOURCE_MARKER]})
        marker_id = tokenizer.convert_tokens_to_ids(SOURCE_MARKER)
    return marker_id


def _render_user_prompt(tokenizer: Any, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _answer_ids(tokenizer: Any, answer: str | None) -> list[int]:
    if answer is None:
        return []
    ids = _tokenize(tokenizer, answer)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id == tokenizer.unk_token_id:
        end_id = tokenizer.eos_token_id
    return ids + [end_id]


def _split_on_marker(tokenizer: Any, text: str, marker_id: int) -> tuple[list[int], list[int]]:
    ids = _tokenize(tokenizer, text)
    positions = [index for index, token_id in enumerate(ids) if token_id == marker_id]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one source marker, found {len(positions)}")
    position = positions[0]
    return ids[:position], ids[position + 1 :]


def bridge_parts(tokenizer: Any, answer: str | None = None) -> PromptParts:
    marker_id = ensure_source_marker(tokenizer)
    text = _render_user_prompt(tokenizer, f"{SOURCE_MARKER}\n{INSTRUCTION}")
    before, after = _split_on_marker(tokenizer, text, marker_id)
    return PromptParts(before=before, after=after, answer=_answer_ids(tokenizer, answer))


def soft_prefix_parts(tokenizer: Any, question: str, answer: str | None = None) -> PromptParts:
    marker_id = ensure_source_marker(tokenizer)
    content = f"{SOURCE_MARKER}{question}\n\n{INSTRUCTION}"
    text = _render_user_prompt(tokenizer, content)
    before, after = _split_on_marker(tokenizer, text, marker_id)
    return PromptParts(before=before, after=after, answer=_answer_ids(tokenizer, answer))


def text_parts(tokenizer: Any, question: str, answer: str | None = None) -> PromptParts:
    text = _render_user_prompt(tokenizer, f"{INSTRUCTION}\n\n{question}")
    return PromptParts(before=_tokenize(tokenizer, text), after=[], answer=_answer_ids(tokenizer, answer))


def text_prompt(tokenizer: Any, question: str) -> str:
    return _render_user_prompt(tokenizer, f"{INSTRUCTION}\n\n{question}")


def plamo_translation_prompt(tokenizer: Any, question: str) -> str:
    """Official PFML3 translation request. Only the question is supplied."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        target_lang="en",
        source_lang="ja",
        tokenize=False,
        add_generation_prompt=True,
    )
