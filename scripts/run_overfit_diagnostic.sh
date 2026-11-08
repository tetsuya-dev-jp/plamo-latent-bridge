#!/usr/bin/env bash
# Post hoc capacity diagnostic for the latent bridge.
# This is NOT a generalization experiment: validation reuses the same items as
# training (resubstitution). The lead runs this after the primary matrix is
# terminal. All outputs live under the overfit_diagnostic experiment.
set -euo pipefail

PYTHON=.venv/bin/python
CONFIG=configs/overfit.json
DATA_DIR=diagnostics/overfit/data

# 1. Generate the isolated diagnostic splits with the existing data CLI.
$PYTHON -m plamo_bridge --config "$CONFIG" data

# 2. Replace the disjoint validation split with a copy of the 16 training items.
#    Validation becomes a resubstitution split: same item IDs, questions and
#    answers, only the recorded split label changes. dataset_meta.json is
#    rewritten so it truthfully describes the intentional overlap instead of
#    leaving the ordinary "all splits disjoint" note in place.
$PYTHON - "$DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


train_rows = read_jsonl(data_dir / "train.jsonl")
val_rows = [dict(row, split="val") for row in train_rows]


def content(row):
    return (row["ja_question"], row["en_question"], row["answer"])


# Self-check: validation must be an exact resubstitution copy of all 16 items.
assert len(train_rows) == 16, f"expected 16 train items, got {len(train_rows)}"
assert len(val_rows) == 16, f"expected 16 validation items, got {len(val_rows)}"
assert [row["id"] for row in val_rows] == [row["id"] for row in train_rows]
assert [content(row) for row in val_rows] == [content(row) for row in train_rows]

with (data_dir / "val.jsonl").open("w", encoding="utf-8") as handle:
    for row in val_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

meta_path = data_dir / "dataset_meta.json"
with meta_path.open(encoding="utf-8") as handle:
    meta = json.load(handle)

# The copied val split uses every template in train, so describe it from the
# actual rows rather than the original disjoint val metadata (which omitted
# main_mul_bags).
actual_val_template_ids = sorted({row["template_id"] for row in val_rows})
meta["template_ids"]["val"] = actual_val_template_ids
assert meta["template_ids"]["val"] == actual_val_template_ids, (
    "val template_ids do not match the actual validation rows"
)

meta["validation_policy"] = "resubstitution_same_items_as_train"
meta["diagnostic_only"] = True
meta["splits_disjoint"] = False
meta["held_out_evaluation"] = False
meta["note"] = (
    "Diagnostic-only resubstitution data: the 16 validation items are an exact "
    "copy of the 16 training items, so train and validation are intentionally "
    "not disjoint. This split cannot measure held-out generalization."
)
meta["interpretation"] = (
    "Post hoc capacity check only. Success means the bridge can memorize a tiny "
    "supervised mapping; it is not evidence of generalization."
)

# Self-check: the metadata must explicitly permit the intended train/val overlap.
assert meta["validation_policy"] == "resubstitution_same_items_as_train"
assert meta["diagnostic_only"] is True
assert meta["splits_disjoint"] is False
assert meta["held_out_evaluation"] is False
assert "not disjoint" in meta["note"].lower()

with meta_path.open("w", encoding="utf-8") as handle:
    json.dump(meta, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

print(f"resubstitution validation written: {len(val_rows)} items")
PY

# 3. Extract source features for the training split and its resubstitution copy.
$PYTHON -m plamo_bridge --config "$CONFIG" features --splits train,val --overwrite

# 4. Train the L8/K16 bridge for 100 epochs on the 16-item training set.
$PYTHON -m plamo_bridge --config "$CONFIG" train \
  --condition bridge --layer 8 --prefix-length 16 --seed 11

# 5. Evaluate the trained checkpoint on the same 16 items (resubstitution).
$PYTHON -m plamo_bridge --config "$CONFIG" evaluate-bridge \
  --layer 8 --prefix-length 16 --seed 11 --splits train

# 6. Shuffle source features across items to test whether memorization depends
#    on the correct source-to-answer correspondence.
$PYTHON -m plamo_bridge --config "$CONFIG" evaluate-bridge \
  --layer 8 --prefix-length 16 --seed 11 --shuffle --splits train
