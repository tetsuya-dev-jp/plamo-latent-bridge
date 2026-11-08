"""Summaries, paired item bootstrap intervals and result-driven plots."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import Config
from .utils import read_jsonl, write_json

PREFERRED_CONDITIONS = (
    "text_ja",
    "text_en",
    "text_plamo_en",
    "bridge_l8_k16_s11",
    "bridge_l8_k16_s23",
    "bridge_l8_k16_s37",
    "bridge_l4_k16_s11",
    "bridge_l16_k16_s11",
    "bridge_l8_k4_s11",
    "bridge_l8_k16_s11_untrained",
    "bridge_l8_k16_s11_shuffled",
    "softprefix_k16_s11",
)

PAIR_DEFINITIONS = (
    ("text_ja", "text_en"),
    ("text_ja", "text_plamo_en"),
    ("text_ja", "bridge_l8_k16_s11"),
    ("text_en", "bridge_l8_k16_s11"),
    ("text_plamo_en", "bridge_l8_k16_s11"),
    ("bridge_l8_k16_s11_untrained", "bridge_l8_k16_s11"),
    ("bridge_l8_k16_s11_shuffled", "bridge_l8_k16_s11"),
    ("bridge_l8_k16_s11", "softprefix_k16_s11"),
    ("bridge_l4_k16_s11", "bridge_l8_k16_s11"),
    ("bridge_l8_k4_s11", "bridge_l8_k16_s11"),
    ("bridge_l16_k16_s11", "bridge_l8_k16_s11"),
)

SEED_CONDITIONS = {
    "bridge_l8_k16": ("bridge_l8_k16_s11", "bridge_l8_k16_s23", "bridge_l8_k16_s37"),
}


def _load_predictions(config: Config) -> dict[tuple[str, str], list[dict[str, Any]]]:
    directory = Path(config.paths.results_dir) / config.experiment_id / "predictions"
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(directory.glob("*.jsonl")):
        for row in read_jsonl(path):
            grouped[(row["condition"], row["split"])].append(row)
    return grouped


def _summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    by_operation: dict[str, dict[str, Any]] = {}
    for operation in sorted({row["operation"] for row in rows}):
        subset = [row for row in rows if row["operation"] == operation]
        correct = sum(bool(row["correct"]) for row in subset)
        last_number_correct = sum(bool(row.get("last_number_correct")) for row in subset)
        by_operation[operation] = {
            "n": len(subset),
            "correct": correct,
            "exact_match": correct / len(subset) if subset else None,
            "last_number_correct": last_number_correct,
            "last_number_exact_match": last_number_correct / len(subset) if subset else None,
        }
    correct = sum(bool(row["correct"]) for row in rows)
    last_number_correct = sum(bool(row.get("last_number_correct")) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "exact_match": correct / len(rows) if rows else None,
        "last_number_correct": last_number_correct,
        "last_number_exact_match": last_number_correct / len(rows) if rows else None,
        "hit_limit_count": sum(bool(row.get("hit_limit")) for row in rows),
        "terminated_count": sum(bool(row.get("terminated")) for row in rows),
        "by_operation": by_operation,
        "mean_generated_tokens": (
            float(np.mean([row["generated_tokens"] for row in rows])) if rows else None
        ),
    }


def paired_bootstrap(
    condition_a: list[dict[str, Any]],
    condition_b: list[dict[str, Any]],
    resamples: int,
    seed: int,
    flag_key: str = "correct",
) -> dict[str, Any]:
    values_a = {row["id"]: float(row[flag_key]) for row in condition_a}
    values_b = {row["id"]: float(row[flag_key]) for row in condition_b}
    common = sorted(set(values_a) & set(values_b))
    differences = np.asarray([values_a[item] - values_b[item] for item in common], dtype=float)
    observed = float(differences.mean()) if len(common) else None
    if len(common) < 2:
        return {
            "n_items": len(common),
            "delta": observed,
            "ci95": None,
            "resamples": resamples,
            "resampling_unit": "test item (same item across conditions)",
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(common), size=(resamples, len(common)))
    means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5]).tolist()
    return {
        "n_items": len(common),
        "delta": observed,
        "ci95": [float(lower), float(upper)],
        "resamples": resamples,
        "resampling_unit": "test item (same item across conditions)",
    }


def _seed_analysis(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    split: str,
    config: Config,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name, conditions in SEED_CONDITIONS.items():
        available = {
            condition: grouped[(condition, split)]
            for condition in conditions
            if (condition, split) in grouped
        }
        if not available:
            continue
        per_seed = {
            condition.rsplit("_s", 1)[-1]: _summarize_rows(rows)["exact_match"]
            for condition, rows in available.items()
        }
        scores = [score for score in per_seed.values() if score is not None]
        seed_mean = float(np.mean(scores)) if scores else None
        seed_sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
        result[group_name] = {
            "per_seed": per_seed,
            "mean": seed_mean,
            "sd_across_seeds": seed_sd,
            "note": "Seeds are not treated as independent test items.",
        }
        if ("text_ja", split) in grouped:
            by_id: dict[str, list[float]] = defaultdict(list)
            for rows in available.values():
                local = {row["id"]: float(row["correct"]) for row in rows}
                for item_id, value in local.items():
                    by_id[item_id].append(value)
            averaged = [{"id": key, "correct": float(np.mean(values))} for key, values in by_id.items()]
            result[f"{group_name}_vs_text_ja"] = paired_bootstrap(
                averaged,
                grouped[("text_ja", split)],
                config.evaluation.bootstrap_resamples,
                config.evaluation.bootstrap_seed,
            )
    return result


def _ordered_conditions(conditions: Iterable[str]) -> list[str]:
    conditions = set(conditions)
    ordered = [condition for condition in PREFERRED_CONDITIONS if condition in conditions]
    ordered.extend(sorted(conditions - set(ordered)))
    return ordered


def analyze(config: Config) -> dict[str, Any]:
    grouped = _load_predictions(config)
    if not grouped:
        raise FileNotFoundError("no prediction files found")
    splits = sorted({split for _, split in grouped})
    summary: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "bootstrap": {
            "resamples": config.evaluation.bootstrap_resamples,
            "seed": config.evaluation.bootstrap_seed,
            "resampling_unit": "test item (same item across conditions)",
        },
        "splits": {},
        "paired_bootstrap": [],
        "seed_variation": {},
    }
    for split in splits:
        conditions = _ordered_conditions(
            condition for (condition, row_split) in grouped if row_split == split
        )
        summary["splits"][split] = {
            "conditions": {
                condition: _summarize_rows(grouped[(condition, split)]) for condition in conditions
            }
        }
        for condition_a, condition_b in PAIR_DEFINITIONS:
            if (condition_a, split) in grouped and (condition_b, split) in grouped:
                strict = paired_bootstrap(
                    grouped[(condition_a, split)],
                    grouped[(condition_b, split)],
                    config.evaluation.bootstrap_resamples,
                    config.evaluation.bootstrap_seed,
                )
                diagnostic = paired_bootstrap(
                    grouped[(condition_a, split)],
                    grouped[(condition_b, split)],
                    config.evaluation.bootstrap_resamples,
                    config.evaluation.bootstrap_seed,
                    flag_key="last_number_correct",
                )
                strict.update(
                    {
                        "split": split,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "last_number": diagnostic,
                    }
                )
                summary["paired_bootstrap"].append(strict)
        summary["seed_variation"][split] = _seed_analysis(grouped, split, config)

    plot_dir = Path(config.paths.results_dir) / config.experiment_id / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    learning_plot = plot_dir / "learning_curves.png"
    comparison_plot = plot_dir / "accuracy_comparison.png"
    summary["plots"] = {
        "learning_curves_png": str(learning_plot),
        "learning_curves_pdf": str(learning_plot.with_suffix(".pdf")),
        "accuracy_comparison_png": str(comparison_plot),
        "accuracy_comparison_pdf": str(comparison_plot.with_suffix(".pdf")),
    }
    _plot_learning(config, learning_plot)
    _plot_comparison(grouped, summary, comparison_plot, config)
    write_json(Path(config.paths.results_dir) / config.experiment_id / "summary.json", summary)
    return summary


def _load_histories(config: Config) -> list[tuple[str, list[dict[str, Any]]]]:
    directory = Path(config.paths.runs_dir) / config.experiment_id
    histories = []
    for path in sorted(directory.glob("*/history.json")):
        with path.open(encoding="utf-8") as handle:
            histories.append((path.parent.name, json.load(handle)))
    return histories


def _plot_learning(config: Config, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = _load_histories(config)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run_name, history in histories:
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label=run_name)
        axes[1].plot(epochs, [row["val_loss"] for row in history], marker="o", label=run_name)
    for axis, title in zip(axes, ("Training loss", "Validation loss")):
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    if histories:
        axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _human_label(condition: str) -> str:
    fixed = {
        "text_ja": "Text JA",
        "text_en": "Text EN",
        "text_plamo_en": "PLaMo translation",
        "softprefix_k16_s11": "Soft prefix K16 s11",
    }
    if condition in fixed:
        return fixed[condition]
    parts = condition.split("_")
    if condition.startswith("bridge_l"):
        layer = parts[1][1:]
        prefix = parts[2][1:]
        seed = parts[3][1:]
        label = f"Bridge L{layer} K{prefix} s{seed}"
        if condition.endswith("_untrained"):
            label += " untrained"
        elif condition.endswith("_shuffled"):
            label += " shuffled"
        return label
    return condition


def _plot_comparison(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    summary: dict[str, Any],
    output: Path,
    config: Config,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [split for split in ("test", "heldout") if split in summary["splits"]]
    if not splits:
        splits = sorted(summary["splits"])
    conditions = _ordered_conditions(
        condition for split in splits for condition in summary["splits"][split]["conditions"]
    )
    labels = [_human_label(condition) for condition in conditions]
    height = max(5.5, len(conditions) * 0.42)
    figure, axes = plt.subplots(1, 2, figsize=(14, height), sharey=True)
    bar_height = 0.8 / max(1, len(splits))
    metrics = (
        ("exact_match", "correct", "Strict exact match"),
        ("last_number_exact_match", "last_number_correct", "Last-number diagnostic"),
    )
    for axis, (metric, flag_key, title) in zip(axes, metrics):
        for split_index, split in enumerate(splits):
            values = []
            errors = []
            positions = []
            for condition_index, condition in enumerate(conditions):
                scores = summary["splits"][split]["conditions"].get(condition)
                if scores is None:
                    continue
                positions.append(condition_index + (split_index - (len(splits) - 1) / 2) * bar_height)
                value = scores[metric]
                lower, upper = _accuracy_interval(
                    grouped.get((condition, split), []),
                    config.evaluation.bootstrap_resamples,
                    config.evaluation.bootstrap_seed,
                    flag_key,
                )
                values.append(value)
                errors.append((value - lower, upper - value))
            axis.barh(
                positions,
                values,
                height=bar_height,
                label=split,
                xerr=np.asarray(errors).T,
                capsize=2,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
            )
        axis.set_xlim(0, 1)
        axis.set_xlabel("Accuracy")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.legend(loc="lower right")
    axes[0].set_yticks(np.arange(len(conditions)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle("Accuracy by condition (bars: 95% item-bootstrap interval)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _accuracy_interval(
    rows: list[dict[str, Any]], resamples: int, seed: int, flag_key: str = "correct"
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    values = np.asarray([float(bool(row.get(flag_key))) for row in rows])
    if len(values) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)
