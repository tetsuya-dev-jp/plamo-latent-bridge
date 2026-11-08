"""Procedural paired arithmetic data and the external MGSM Japanese loader."""

from __future__ import annotations

import random
import re
import urllib.request
from pathlib import Path
from typing import Any

from .utils import write_json, write_jsonl

OPERATIONS = ("addition", "subtraction", "multiplication", "two_step")
SPLITS = ("train", "val", "test", "heldout")

NAMES = (
    {"ja": "アキ", "en": "Aki"},
    {"ja": "ミナ", "en": "Mina"},
    {"ja": "ハル", "en": "Haru"},
    {"ja": "ソラ", "en": "Sora"},
    {"ja": "ユウ", "en": "Yuu"},
    {"ja": "レン", "en": "Ren"},
)

OBJECTS = (
    {"ja": "りんご", "en": "apples"},
    {"ja": "クッキー", "en": "cookies"},
    {"ja": "ボール", "en": "balls"},
    {"ja": "キャンディー", "en": "candies"},
    {"ja": "カード", "en": "cards"},
    {"ja": "シール", "en": "stickers"},
)


def split_path(data_dir: str | Path, split: str) -> Path:
    filename = "mgsm_ja_test.jsonl" if split == "mgsm" else f"{split}.jsonl"
    return Path(data_dir) / filename


def _template(template_id: str, ja: str, en: str, subtype: str | None = None) -> dict[str, Any]:
    return {"id": template_id, "ja": ja, "en": en, "subtype": subtype}


MAIN_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "addition": [
        _template(
            "main_add_receive",
            "{name}は{obj}を{a}個持っています。さらに{b}個もらいました。{obj}は全部で何個ですか？",
            "{name} has {a} {en}. They receive {b} more. How many {en} do they have in total?",
        ),
        _template(
            "main_add_shop",
            "店に{obj}が{a}個あります。そこへ{b}個届きました。店の{obj}は全部で何個ですか？",
            "A shop has {a} {en}. Another {b} arrive. How many {en} are in the shop in total?",
        ),
    ],
    "subtraction": [
        _template(
            "main_sub_give",
            "{name}は{obj}を{a}個持っています。そのうち{b}個を友達にあげました。残りは何個ですか？",
            "{name} has {a} {en}. They give {b} to a friend. How many are left?",
        ),
        _template(
            "main_sub_use",
            "箱に{obj}が{a}個入っています。{name}は{b}個を使いました。箱には何個残っていますか？",
            "A box contains {a} {en}. {name} uses {b} of them. How many remain in the box?",
        ),
    ],
    "multiplication": [
        _template(
            "main_mul_boxes",
            "{name}は{obj}が入った箱を{a}箱持っています。1箱に{b}個ずつ入っています。{obj}は全部で何個ですか？",
            "{name} has {a} boxes of {en}. Each box contains {b}. How many {en} are there in total?",
        ),
        _template(
            "main_mul_bags",
            "{name}は1袋に{obj}を{b}個ずつ入れた袋を{a}袋作りました。{obj}は全部で何個ですか？",
            "{name} made {a} bags with {b} {en} in each bag. How many {en} are there in total?",
        ),
    ],
    "two_step": [
        _template(
            "main_two_buy_give",
            "{name}は{obj}を{a}個持っています。{b}個買い足したあと、{c}個を友達にあげました。残りは何個ですか？",
            "{name} has {a} {en}. They buy {b} more, then give {c} to a friend. How many are left?",
            subtype="add_sub",
        ),
        _template(
            "main_two_box_use",
            "{name}は1箱に{obj}が{b}個ずつ入った箱を{a}箱持っています。そのうち{c}個を使いました。残りは何個ですか？",
            "{name} has {a} boxes with {b} {en} in each box. They use {c} of them. How many are left?",
            subtype="mul_sub",
        ),
    ],
}

HELDOUT_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "addition": [
        _template(
            "heldout_add_home",
            "{name}の家には{obj}が{a}個あります。となりの家から{b}個持ってきました。{obj}は何個になりましたか？",
            "{name}'s home has {a} {en}. They bring {b} more from next door. How many {en} are there now?",
        )
    ],
    "subtraction": [
        _template(
            "heldout_sub_lost",
            "{name}は{obj}を{a}個持っていました。{b}個をなくしてしまいました。今は何個残っていますか？",
            "{name} had {a} {en}. They lost {b} of them. How many remain now?",
        )
    ],
    "multiplication": [
        _template(
            "heldout_mul_prepare",
            "{name}は{obj}を{b}個ずつ入れた袋を{a}袋用意しました。{obj}は全部で何個ありますか？",
            "{name} prepared {a} bags with {b} {en} in each bag. What is the total number of {en}?",
        )
    ],
    "two_step": [
        _template(
            "heldout_two_order_use",
            "はじめに{obj}が{a}個ありました。{b}個を追加で注文し、その後{c}個を使いました。手元には何個ありますか？",
            "There are {a} {en} at first. {b} more are ordered, and then {c} are used. How many are left?",
            subtype="add_sub",
        ),
        _template(
            "heldout_two_return",
            "{name}は1箱に{obj}が{b}個ずつ入った箱を{a}箱注文しました。届いたうち{c}個を返品しました。手元には何個残りますか？",
            "{name} ordered {a} boxes with {b} {en} in each box. They return {c} of them. How many remain?",
            subtype="mul_sub",
        ),
    ],
}


def _sample_semantic(operation: str, rng: random.Random, seen: set[str]) -> dict[str, Any]:
    while True:
        if operation == "addition":
            a, b = rng.randint(10, 99), rng.randint(2, 99)
            key = f"add:{min(a, b)}:{max(a, b)}"
            params = {"a": a, "b": b, "answer": a + b}
        elif operation == "subtraction":
            a = rng.randint(10, 99)
            b = rng.randint(1, a - 1)
            key = f"sub:{a}:{b}"
            params = {"a": a, "b": b, "answer": a - b}
        elif operation == "multiplication":
            a, b = rng.randint(2, 30), rng.randint(2, 30)
            key = f"mul:{min(a, b)}:{max(a, b)}"
            params = {"a": a, "b": b, "answer": a * b}
        elif operation == "two_step":
            if rng.random() < 0.5:
                a, b = rng.randint(20, 99), rng.randint(2, 50)
                c = rng.randint(2, a + b - 1)
                key = f"two:add_sub:{min(a, b)}:{max(a, b)}:{c}"
                params = {"a": a, "b": b, "c": c, "subtype": "add_sub", "answer": a + b - c}
            else:
                a, b = rng.randint(2, 15), rng.randint(2, 12)
                c = rng.randint(1, a * b - 1)
                key = f"two:mul_sub:{min(a, b)}:{max(a, b)}:{c}"
                params = {"a": a, "b": b, "c": c, "subtype": "mul_sub", "answer": a * b - c}
        else:
            raise ValueError(f"Unknown operation: {operation}")
        if key in seen:
            continue
        seen.add(key)
        params["semantic_key"] = key
        return params


def generate_dataset(
    counts: dict[str, int],
    seed: int,
    data_dir: str | Path,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    for split in SPLITS:
        if counts[split] % len(OPERATIONS) != 0:
            raise ValueError(f"{split} count must be divisible by {len(OPERATIONS)}")
    rng = random.Random(seed)
    seen: set[str] = set()
    contents: dict[str, list[dict[str, Any]]] = {}
    template_ids: dict[str, set[str]] = {}

    for split in SPLITS:
        rows: list[dict[str, Any]] = []
        used_templates: set[str] = set()
        heldout = split == "heldout"
        per_operation = counts[split] // len(OPERATIONS)
        for operation in OPERATIONS:
            for index in range(per_operation):
                params = _sample_semantic(operation, rng, seen)
                candidates = (HELDOUT_TEMPLATES if heldout else MAIN_TEMPLATES)[operation]
                if operation == "two_step":
                    candidates = [t for t in candidates if t["subtype"] == params["subtype"]]
                template = rng.choice(candidates)
                name = rng.choice(NAMES)
                obj = rng.choice(OBJECTS)
                values = {
                    "name": name["ja"],
                    "obj": obj["ja"],
                    "en": obj["en"],
                    **params,
                }
                en_values = {
                    "name": name["en"],
                    "obj": obj["en"],
                    "en": obj["en"],
                    **params,
                }
                row = {
                    "id": f"{split}_{operation}_{index:04d}",
                    "split": split,
                    "operation": operation,
                    "template_id": template["id"],
                    "semantic_key": params["semantic_key"],
                    "ja_question": template["ja"].format(**values),
                    "en_question": template["en"].format(**en_values),
                    "answer": params["answer"],
                }
                rows.append(row)
                used_templates.add(template["id"])
        rng.shuffle(rows)
        contents[split] = rows
        template_ids[split] = used_templates

    for split, rows in contents.items():
        write_jsonl(data_dir / f"{split}.jsonl", rows)

    meta = {
        "seed": seed,
        "counts": counts,
        "operations": list(OPERATIONS),
        "template_ids": {split: sorted(ids) for split, ids in template_ids.items()},
        "semantic_key_rule": (
            "Commutative operand pairs are sorted before key construction, including "
            "the first two operands of two-step add/sub and mul/sub problems."
        ),
        "note": "Synthetic paired arithmetic data. Semantic tuples are disjoint across splits.",
    }
    write_json(data_dir / "dataset_meta.json", meta)
    return meta


MGSM_COMMIT = "452a21ad3dae5668c06ceeac21ff073e1e40f9be"
MGSM_URL = f"https://raw.githubusercontent.com/google-research/url-nlp/{MGSM_COMMIT}/mgsm/mgsm_ja.tsv"


def fetch_mgsm(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    raw = urllib.request.urlopen(MGSM_URL, timeout=60).read().decode("utf-8")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        question, answer = line.split("\t")
        rows.append(
            {
                "id": f"mgsm_{index:04d}",
                "split": "mgsm",
                "operation": "mgsm",
                "template_id": "external_mgsm_ja",
                "semantic_key": None,
                "ja_question": question,
                "en_question": None,
                "answer": int(re.sub(r"[,\s]", "", answer)),
            }
        )
    write_jsonl(data_dir / "mgsm_ja_test.jsonl", rows)
    meta = {
        "source_url": MGSM_URL,
        "source_commit": MGSM_COMMIT,
        "source_project": "google-research/url-nlp MGSM",
        "count": len(rows),
        "note": "Japanese MGSM test split. Reference answers are evaluation-only.",
    }
    write_json(data_dir / "mgsm_meta.json", meta)
    return meta
