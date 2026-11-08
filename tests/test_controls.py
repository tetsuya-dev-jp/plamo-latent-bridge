from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plamo_bridge.evaluate import _shuffle_features, _validate_shuffle_mapping


def test_shuffle_mapping_is_explicit_and_not_identity(tmp_path):
    rows = [{"id": f"item-{index}"} for index in range(6)]
    features = np.arange(6)[:, None, None]
    config = SimpleNamespace(
        paths=SimpleNamespace(results_dir=tmp_path),
        experiment_id="test",
    )

    path, source_ids = _shuffle_features(config, rows, features, seed=11, split="test", layer=8, prefix_length=16)

    mapping = json.loads(Path(path).read_text())
    permutation = np.load(Path(path).with_suffix(".npy"))
    assert len(mapping) == 6
    assert source_ids == [entry["source_item_id"] for entry in mapping]
    assert not np.array_equal(permutation, np.arange(6))
    assert sorted(permutation.tolist()) == list(range(6))
    assert mapping[0]["target_item_id"] == "item-0"


def test_shuffle_predictions_must_match_saved_mapping(tmp_path):
    rows = [{"id": f"item-{index}"} for index in range(4)]
    features = np.arange(4)[:, None, None]
    config = SimpleNamespace(paths=SimpleNamespace(results_dir=tmp_path), experiment_id="test")
    path, source_ids = _shuffle_features(config, rows, features, 5, "test", 8, 16)
    predictions = [
        {"id": row["id"], "shuffled_source_id": source_id}
        for row, source_id in zip(rows, source_ids)
    ]
    _validate_shuffle_mapping(Path(path), predictions)
    predictions[0]["shuffled_source_id"] = "wrong"
    try:
        _validate_shuffle_mapping(Path(path), predictions)
    except ValueError:
        return
    raise AssertionError("expected mapping mismatch")
