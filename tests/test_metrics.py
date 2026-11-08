from __future__ import annotations

import numpy as np
import pytest

from plamo_bridge.analysis import paired_bootstrap
from plamo_bridge.utils import exact_match, extract_last_number, last_number_exact_match, normalize_answer


def test_normalize_answer_documented_formatting():
    assert normalize_answer(" 42 ") == "42"
    assert normalize_answer("42.") == "42"
    assert normalize_answer("4,200") == "4200"
    assert normalize_answer("４２") == "42"
    assert normalize_answer("<|im_end|>42") == "42"
    assert normalize_answer("The answer is 42") == "The answer is 42"


def test_exact_match_requires_numeric_equivalence():
    assert exact_match("42.", 42)
    assert exact_match("４２", 42)
    assert not exact_match("The answer is 42", 42)
    assert not exact_match(None, 42)


def test_paired_bootstrap_is_itemwise_and_deterministic():
    rows_a = [{"id": f"i{index}", "correct": index < 8} for index in range(10)]
    rows_b = [{"id": f"i{index}", "correct": index < 5} for index in range(10)]
    result = paired_bootstrap(rows_a, rows_b, resamples=2000, seed=3)
    assert result["n_items"] == 10
    assert result["delta"] == 0.3
    assert result["ci95"][0] <= result["delta"] <= result["ci95"][1]
    assert result["resampling_unit"].startswith("test item")
    again = paired_bootstrap(rows_a, rows_b, resamples=2000, seed=3)
    assert again == result
    assert np.isfinite(result["ci95"]).all()


def test_paired_bootstrap_preserves_fractional_seed_averages():
    rows_a = [
        {"id": "i0", "correct": 0.0},
        {"id": "i1", "correct": 1 / 3},
        {"id": "i2", "correct": 2 / 3},
    ]
    rows_b = [{"id": f"i{index}", "correct": 0.0} for index in range(3)]
    result = paired_bootstrap(rows_a, rows_b, resamples=500, seed=7)
    assert result["n_items"] == 3
    assert result["delta"] == pytest.approx(1 / 3)


def test_last_number_diagnostic_extracts_final_numeric_literal():
    assert extract_last_number("30箱 × 11個 = 330個") == 330
    assert extract_last_number("There are 27 boxes and 5 cookies each; total 135.") == 135
    assert extract_last_number("3.0") == 3
    assert last_number_exact_match("3.0", 3)
    assert last_number_exact_match("The answer is 42", 42)
    assert not last_number_exact_match("The answer is 42", 41)
    assert not last_number_exact_match("no number here", 1)
    assert extract_last_number("1. 2") == 2
