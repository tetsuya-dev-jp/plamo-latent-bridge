from __future__ import annotations

from plamo_bridge.data import OPERATIONS, SPLITS, generate_dataset, split_path
from plamo_bridge.utils import read_jsonl


def test_counts_balance_answers_and_disjoint_semantics(tmp_path):
    counts = {"train": 32, "val": 8, "test": 16, "heldout": 8}
    generate_dataset(counts, seed=7, data_dir=tmp_path)
    rows = []
    for split in SPLITS:
        split_rows = read_jsonl(tmp_path / f"{split}.jsonl")
        assert len(split_rows) == counts[split]
        assert len({row["id"] for row in split_rows}) == len(split_rows)
        per_operation = {}
        for row in split_rows:
            per_operation[row["operation"]] = per_operation.get(row["operation"], 0) + 1
            assert _expected_answer(row["semantic_key"]) == row["answer"]
            assert row["ja_question"] and row["en_question"]
            if row["operation"] == "two_step":
                parts = row["semantic_key"].split(":")
                assert int(parts[2]) <= int(parts[3])
        assert per_operation == {operation: counts[split] // len(OPERATIONS) for operation in OPERATIONS}
        rows.extend(split_rows)

    keys = [row["semantic_key"] for row in rows]
    assert len(keys) == len(set(keys))

    main_templates = {
        row["template_id"] for row in rows if row["split"] != "heldout"
    }
    heldout_templates = {row["template_id"] for row in rows if row["split"] == "heldout"}
    assert heldout_templates
    assert main_templates.isdisjoint(heldout_templates)


def test_generation_is_deterministic(tmp_path):
    counts = {"train": 8, "val": 4, "test": 8, "heldout": 4}
    generate_dataset(counts, seed=13, data_dir=tmp_path / "a")
    generate_dataset(counts, seed=13, data_dir=tmp_path / "b")
    for split in SPLITS:
        assert read_jsonl(tmp_path / "a" / f"{split}.jsonl") == read_jsonl(tmp_path / "b" / f"{split}.jsonl")


def test_split_path_for_mgsm(tmp_path):
    assert split_path(tmp_path, "train").name == "train.jsonl"
    assert split_path(tmp_path, "mgsm").name == "mgsm_ja_test.jsonl"


def _expected_answer(key: str) -> int:
    parts = key.split(":")
    if parts[0] == "add":
        return int(parts[1]) + int(parts[2])
    if parts[0] == "sub":
        return int(parts[1]) - int(parts[2])
    if parts[0] == "mul":
        return int(parts[1]) * int(parts[2])
    if parts[1] == "add_sub":
        return int(parts[2]) + int(parts[3]) - int(parts[4])
    return int(parts[2]) * int(parts[3]) - int(parts[4])
