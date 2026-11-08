from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plamo_bridge.translation import translate_split
from plamo_bridge.utils import write_jsonl


def _config(tmp_path: Path, revision: str = "a" * 40) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(data_dir=tmp_path),
        source=SimpleNamespace(model_id="source", revision=revision),
        generation=SimpleNamespace(translation_max_new_tokens=128, mgsm_translation_max_new_tokens=512),
        limits=SimpleNamespace(for_split=lambda split: None),
    )


def _write_current_rows(tmp_path: Path, question: str = "new question") -> None:
    write_jsonl(
        tmp_path / "test.jsonl",
        [{"id": "a", "split": "test", "operation": "addition", "ja_question": question, "answer": 1}],
    )


def _translation_row(question: str = "new question", **overrides) -> dict:
    row = {
        "id": "a",
        "split": "test",
        "ja_question": question,
        "translation": "translated",
        "max_new_tokens": 128,
        "source_model_id": "source",
        "source_revision": "a" * 40,
    }
    row.update(overrides)
    return row


def _write_translations(tmp_path: Path, row: dict) -> None:
    write_jsonl(tmp_path / "translations_test.jsonl", [row])


def test_translation_cache_rejects_same_ids_with_changed_question(tmp_path):
    config = _config(tmp_path)
    _write_current_rows(tmp_path, question="new question")
    _write_translations(tmp_path, _translation_row(question="old question"))

    with pytest.raises(RuntimeError, match="ja_question mismatch"):
        translate_split(config, "test", overwrite=False)


def test_translation_cache_rejects_model_revision_and_cap_mismatch(tmp_path):
    config = _config(tmp_path)
    _write_current_rows(tmp_path)
    cases = (
        (_translation_row(id="b"), "id mismatch"),
        (_translation_row(source_model_id="other"), "source_model_id mismatch"),
        (_translation_row(source_revision="b" * 40), "source_revision mismatch"),
        (_translation_row(max_new_tokens=64), "max_new_tokens mismatch"),
    )
    for row, message in cases:
        _write_translations(tmp_path, row)
        with pytest.raises(RuntimeError, match=message):
            translate_split(config, "test", overwrite=False)


def test_translation_cache_requires_stored_question_and_provenance(tmp_path):
    config = _config(tmp_path)
    _write_current_rows(tmp_path)
    row = _translation_row()
    del row["ja_question"]
    _write_translations(tmp_path, row)

    with pytest.raises(RuntimeError, match="missing ja_question"):
        translate_split(config, "test", overwrite=False)


def test_matching_translation_cache_is_reused_without_loading_source(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_current_rows(tmp_path)
    _write_translations(tmp_path, _translation_row())

    def fail_load(_config):
        raise AssertionError("matching translation cache must not load PLaMo")

    monkeypatch.setattr("plamo_bridge.translation.load_source_model", fail_load)
    result = translate_split(config, "test", overwrite=False)
    assert result["status"] == "skipped_matching_cache"
