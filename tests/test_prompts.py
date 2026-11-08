from __future__ import annotations

from plamo_bridge.prompts import SOURCE_MARKER, bridge_parts, soft_prefix_parts, text_parts


class FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {SOURCE_MARKER: 900, "<|im_end|>": 901}
        self.unk_token_id = 0
        self.eos_token_id = 901

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, 0)

    def add_special_tokens(self, mapping):
        for token in mapping["additional_special_tokens"]:
            self._vocab[token] = 9000 + len(self._vocab)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<s>" + messages[0]["content"] + "</s>"

    def __call__(self, text, add_special_tokens=False):
        ids = []
        index = 0
        while index < len(text):
            for token, token_id in self._vocab.items():
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return {"input_ids": ids}


def test_bridge_training_and_generation_share_prompt_boundaries():
    tokenizer = FakeTokenizer()
    training = bridge_parts(tokenizer, "42")
    generation = bridge_parts(tokenizer, None)
    assert training.before == generation.before
    assert training.after == generation.after
    assert training.answer[-1] == tokenizer.eos_token_id


def test_soft_prefix_training_and_generation_share_prompt_boundaries():
    tokenizer = FakeTokenizer()
    training = soft_prefix_parts(tokenizer, "問題です。", "42")
    generation = soft_prefix_parts(tokenizer, "問題です。", None)
    assert training.before == generation.before
    assert training.after == generation.after
    assert training.answer[-1] == tokenizer.eos_token_id


def test_text_prompt_supervises_answer_and_termination_only():
    tokenizer = FakeTokenizer()
    parts = text_parts(tokenizer, "問題です。", "42")
    assert parts.after == []
    assert parts.answer[-1] == tokenizer.eos_token_id
    assert parts.answer[:-1] == tokenizer("42")["input_ids"]
