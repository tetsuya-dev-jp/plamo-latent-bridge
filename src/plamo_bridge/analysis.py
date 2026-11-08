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

SPLIT_TITLES = (
    ("test", "Synthetic test"),
    ("heldout", "Template holdout"),
    ("mgsm", "MGSM"),
)


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
        "single_condition_interval": (
            "Wilson score 95% confidence interval on per-item binary correctness "
            "(not a paired effect interval; paired differences are in paired_bootstrap)."
        ),
    }
    _plot_learning(config, learning_plot)
    _plot_comparison(grouped, summary, comparison_plot)
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

    available = dict(_load_histories(config))
    histories = [
        (run_name, available[run_name])
        for run_name in SEED_CONDITIONS["bridge_l8_k16"]
        if run_name in available
    ]
    styles = {
        "bridge_l8_k16_s11": ("#000000", "o", "-"),
        "bridge_l8_k16_s23": ("#555555", "s", "--"),
        "bridge_l8_k16_s37": ("#999999", "^", "-."),
    }
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for run_name, history in histories:
        epochs = [row["epoch"] for row in history]
        color, marker, linestyle = styles.get(run_name, ("#000000", "o", "-"))
        label = f"seed {run_name.rsplit('_s', 1)[-1]}"
        axes[0].plot(
            epochs,
            [row["train_loss"] for row in history],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=3.5,
            linewidth=1.2,
            label=label,
        )
        axes[1].plot(
            epochs,
            [row["val_loss"] for row in history],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=3.5,
            linewidth=1.2,
            label=label,
        )
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    axes[0].set_ylabel("Mean minibatch CE")
    axes[1].set_ylabel("Answer-token-weighted CE")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("Main L8/K16 setting (three seeds)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _human_label(condition: str) -> str:
    fixed = {
        "text_ja": "Text prompt (JA)",
        "text_en": "Text prompt (EN)",
        "text_plamo_en": "PLaMo translation (EN)",
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
            label += " (untrained)"
        elif condition.endswith("_shuffled"):
            label += " (shuffled)"
        return label
    return condition


def _split_sample_size(summary: dict[str, Any], split: str) -> int | None:
    sizes = {scores["n"] for scores in summary["splits"][split]["conditions"].values()}
    return sizes.pop() if len(sizes) == 1 else None


def _plot_comparison(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    split_titles = dict(SPLIT_TITLES)
    splits = [split for split, _ in SPLIT_TITLES if split in summary["splits"]]
    if not splits:
        splits = sorted(summary["splits"])
    conditions = _ordered_conditions(
        condition for split in splits for condition in summary["splits"][split]["conditions"]
    )
    labels = [_human_label(condition) for condition in conditions]
    metrics = (
        ("exact_match", "correct", "o", "black"),
        ("last_number_exact_match", "last_number_correct", "s", "white"),
    )
    offset = 0.17

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8})
    figure, axes = plt.subplots(
        1,
        len(splits),
        figsize=(7.6, 6.2),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axes = axes[0]
    for axis, split in zip(axes, splits):
        for condition_index, condition in enumerate(conditions):
            scores = summary["splits"][split]["conditions"].get(condition)
            if scores is None:
                continue
            rows = grouped.get((condition, split), [])
            for metric_index, (metric, flag_key, marker, face) in enumerate(metrics):
                value = scores[metric]
                if value is None:
                    continue
                lower, upper = _accuracy_interval(rows, flag_key)
                point = 100.0 * value
                y_position = condition_index + (metric_index - 0.5) * 2 * offset
                axis.errorbar(
                    point,
                    y_position,
                    xerr=[[point - 100.0 * lower], [100.0 * upper - point]],
                    fmt=marker,
                    color="black",
                    markerfacecolor=face,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    markersize=4.0,
                    elinewidth=0.8,
                    capsize=1.5,
                    capthick=0.8,
                    linestyle="none",
                    zorder=3,
                )
        sample_size = _split_sample_size(summary, split)
        title = split_titles.get(split, split)
        if sample_size is not None:
            title = f"{title} (n={sample_size})"
        axis.set_title(title)
        axis.set_xlim(0, 100)
        axis.set_xlabel("Accuracy")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].set_yticks(np.arange(len(conditions)))
    axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[0].set_ylabel("Study condition")
    axes[0].invert_yaxis()

    handles = [
        Line2D([], [], marker="o", color="black", linestyle="none", markersize=4.0,
               label="Strict exact match"),
        Line2D([], [], marker="s", color="black", markerfacecolor="white", linestyle="none",
               markersize=4.0, label="Last-number diagnostic"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        fontsize=7.5,
        frameon=False,
    )
    figure.text(
        0.5,
        0.008,
        "Points: per-condition accuracy. Bars: Wilson score 95% CI from per-item binary "
        "correctness (not paired differences). Blank rows were not run on that split.",
        ha="center",
        fontsize=6.5,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _accuracy_interval(
    rows: list[dict[str, Any]], flag_key: str = "correct"
) -> tuple[float, float]:
    if not rows:
        return 0.0, 1.0
    from scipy.stats import binomtest

    correct = sum(bool(row.get(flag_key)) for row in rows)
    interval = binomtest(correct, len(rows)).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    return float(interval.low), float(interval.high)
