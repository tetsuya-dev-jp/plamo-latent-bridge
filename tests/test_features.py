from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plamo_bridge.config import load_config
from plamo_bridge.features import (
    POOLING_RULE,
    PROMPT_FORMAT,
    _manifest_problems,
    _skipped_split_metadata,
    load_feature_array,
)


def _config(tmp_path: Path, dtype: str = "bfloat16") -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(features_dir=tmp_path),
        source=SimpleNamespace(
            layers=[8],
            prefix_lengths=[16],
            model_id="source",
            revision="a" * 40,
            dtype=dtype,
        ),
    )


def _manifest(
    question: str = "question",
    dtype: str = "bfloat16",
    pooling_rule: str = POOLING_RULE,
    prompt_format: str = PROMPT_FORMAT,
    include_question_texts: bool = True,
    include_measurement: bool = True,
) -> dict:
    manifest = {
        "item_ids": ["a"],
        "count": 1,
        "layers": [8],
        "prefix_lengths": [16],
        "source_model_id": "source",
        "source_revision_requested": "a" * 40,
        "source_revision_resolved": "a" * 40,
        "source_dtype": dtype,
        "pooling_rule": pooling_rule,
        "prompt_format": prompt_format,
    }
    if include_question_texts:
        manifest["question_texts"] = [question]
    if include_measurement:
        manifest["measurement"] = {
            "created_at": "2026-01-01T00:00:00+00:00",
            "prompt_example": "prompt",
            "token_length_min": 10,
            "token_length_max": 12,
            "extraction_seconds": 1.5,
            "load_seconds": 0.5,
            "peak_gpu_memory_gib": 1.0,
            "environment": {"python": "3.12"},
        }
    return manifest


def _write_cache(tmp_path: Path, manifest: dict) -> None:
    np.save(tmp_path / "train_layer8_k16.npy", np.zeros((1, 16, 4), dtype=np.float32))
    (tmp_path / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _rows(question: str = "question") -> list[dict]:
    return [{"id": "a", "ja_question": question}]


def test_feature_cache_accepts_matching_manifest(tmp_path):
    config = _config(tmp_path)
    _write_cache(tmp_path, _manifest())
    array = load_feature_array(config, "train", 8, 16, expected_rows=_rows())
    assert array.shape == (1, 16, 4)


def test_feature_cache_rejects_same_ids_with_changed_question(tmp_path):
    config = _config(tmp_path)
    _write_cache(tmp_path, _manifest(question="old question"))
    try:
        load_feature_array(config, "train", 8, 16, expected_rows=_rows("new question"))
    except ValueError as error:
        assert "question_texts mismatch" in str(error)
        return
    raise AssertionError("expected stale-cache ValueError")


def test_feature_cache_rejects_dtype_pooling_and_prompt_mismatch(tmp_path):
    config = _config(tmp_path)
    cases = (
        (_manifest(dtype="float32"), "source_dtype mismatch"),
        (_manifest(pooling_rule="different rule"), "pooling_rule mismatch"),
        (_manifest(prompt_format="different prompt"), "prompt_format mismatch"),
    )
    for manifest, message in cases:
        _write_cache(tmp_path, manifest)
        try:
            load_feature_array(config, "train", 8, 16, expected_rows=_rows())
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"expected ValueError for {message}")


def test_feature_cache_requires_new_question_and_measurement_fields(tmp_path):
    config = _config(tmp_path)
    cases = (
        (_manifest(include_question_texts=False), "missing question_texts"),
        (_manifest(include_measurement=False), "missing measurement"),
    )
    for manifest, message in cases:
        _write_cache(tmp_path, manifest)
        try:
            load_feature_array(config, "train", 8, 16, expected_rows=_rows())
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"expected ValueError for {message}")


def test_matching_skip_preserves_original_measurement_and_provenance(tmp_path):
    config = _config(tmp_path)
    manifest = _manifest(question="question")
    _write_cache(tmp_path, manifest)
    assert _manifest_problems(config, "train", _rows(), manifest) == []

    metadata = _skipped_split_metadata(manifest)

    assert metadata["status"] == "skipped_matching_cache"
    assert metadata["extraction_seconds"] == 1.5
    assert metadata["created_at"] == "2026-01-01T00:00:00+00:00"
    assert metadata["environment"] == {"python": "3.12"}
    assert metadata["source_model_id"] == "source"
    assert metadata["source_revision_resolved"] == "a" * 40
    assert metadata["source_dtype"] == "bfloat16"


def test_smoke_paths_are_isolated():
    config = load_config(Path("configs/smoke.json"))
    for path in (
        config.paths.data_dir,
        config.paths.features_dir,
        config.paths.runs_dir,
        config.paths.results_dir,
    ):
        assert str(path).startswith("local/smoke/")


def test_shared_answer_and_translation_caps():
    from plamo_bridge.config import answer_max_new_tokens, translation_max_new_tokens

    config = load_config(Path("configs/default.json"))
    assert answer_max_new_tokens(config, "test") == 128
    assert answer_max_new_tokens(config, "mgsm") == 256
    assert translation_max_new_tokens(config, "test") == 128
    assert translation_max_new_tokens(config, "mgsm") == 512
