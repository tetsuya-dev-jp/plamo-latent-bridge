"""Command line entry points for data, features, training, evaluation and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .config import Config, load_config
from .data import fetch_mgsm, generate_dataset
from .evaluate import evaluate_bridge, evaluate_text, write_environment
from .features import extract_source_features
from .train import train
from .translation import translate_split
from .utils import configure_cpu_threads


def _splits(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _data(config: Config, args: argparse.Namespace) -> None:
    meta = generate_dataset(
        {
            "train": config.data.train,
            "val": config.data.val,
            "test": config.data.test,
            "heldout": config.data.heldout,
        },
        config.data.seed,
        config.paths.data_dir,
    )
    _print(meta)
    if args.mgsm:
        _print(fetch_mgsm(config.paths.data_dir))


def _features(config: Config, args: argparse.Namespace) -> None:
    _print(
        extract_source_features(
            config,
            splits=_splits(args.splits),
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    )


def _translate(config: Config, args: argparse.Namespace) -> None:
    for split in _splits(args.splits):
        _print(translate_split(config, split, batch_size=args.batch_size, overwrite=args.overwrite))


def _train(config: Config, args: argparse.Namespace) -> None:
    _print(
        train(
            config,
            condition=args.condition,
            seed=args.seed,
            layer=args.layer,
            prefix_length=args.prefix_length,
        )
    )


def _evaluate_text(config: Config, args: argparse.Namespace) -> None:
    _print(evaluate_text(config, _splits(args.conditions), _splits(args.splits), args.batch_size))


def _evaluate_bridge(config: Config, args: argparse.Namespace) -> None:
    for split in _splits(args.splits):
        _print(
            evaluate_bridge(
                config,
                condition=args.condition,
                split=split,
                layer=args.layer,
                prefix_length=args.prefix_length,
                seed=args.seed,
                untrained=args.untrained,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
            )
        )


def _analyze(config: Config, args: argparse.Namespace) -> None:
    _print(analyze(config))


def _smoke(config: Config, args: argparse.Namespace) -> None:
    write_environment(config)
    if args.regenerate_data or not (Path(config.paths.data_dir) / "train.jsonl").exists():
        _print(
            generate_dataset(
                {
                    "train": config.data.train,
                    "val": config.data.val,
                    "test": config.data.test,
                    "heldout": config.data.heldout,
                },
                config.data.seed,
                config.paths.data_dir,
            )
        )
    _print(
        extract_source_features(
            config,
            splits=["train", "val", "test", "heldout"],
            batch_size=min(8, config.training.batch_size),
            overwrite=args.overwrite,
        )
    )
    for split in ("test", "heldout"):
        _print(translate_split(config, split, batch_size=8, overwrite=args.overwrite))
    _print(train(config, condition="bridge", seed=11, layer=8, prefix_length=16))
    for split in ("test", "heldout"):
        _print(
            evaluate_bridge(
                config,
                condition="bridge",
                split=split,
                layer=8,
                prefix_length=16,
                seed=11,
                batch_size=4,
            )
        )
    _print(evaluate_text(config, ["text_ja", "text_en", "text_plamo_en"], ["test", "heldout"], batch_size=4))
    _print(analyze(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="generate paired synthetic arithmetic splits")
    data.add_argument("--mgsm", action="store_true", help="also fetch the external Japanese MGSM test split")
    data.set_defaults(func=_data)

    features = subparsers.add_parser("features", help="extract and cache detached source features")
    features.add_argument("--splits", default="train,val,test,heldout")
    features.add_argument("--batch-size", type=int, default=8)
    features.add_argument("--overwrite", action="store_true")
    features.set_defaults(func=_features)

    translate = subparsers.add_parser("translate", help="generate English translations with PLaMo")
    translate.add_argument("--splits", default="test,heldout")
    translate.add_argument("--batch-size", type=int, default=8)
    translate.add_argument("--overwrite", action="store_true")
    translate.set_defaults(func=_translate)

    training = subparsers.add_parser("train", help="train a bridge or a learned soft prefix")
    training.add_argument("--condition", choices=["bridge", "soft_prefix"], required=True)
    training.add_argument("--layer", type=int, default=None)
    training.add_argument("--prefix-length", type=int, default=16)
    training.add_argument("--seed", type=int, required=True)
    training.set_defaults(func=_train)

    evaluate_text_parser = subparsers.add_parser("evaluate-text", help="evaluate frozen target textual baselines")
    evaluate_text_parser.add_argument("--conditions", default="text_ja,text_en,text_plamo_en")
    evaluate_text_parser.add_argument("--splits", default="test,heldout")
    evaluate_text_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_text_parser.set_defaults(func=_evaluate_text)

    evaluate_bridge_parser = subparsers.add_parser("evaluate-bridge", help="evaluate a trained bridge or control")
    evaluate_bridge_parser.add_argument("--condition", choices=["bridge", "soft_prefix"], default="bridge")
    evaluate_bridge_parser.add_argument("--layer", type=int, default=8)
    evaluate_bridge_parser.add_argument("--prefix-length", type=int, default=16)
    evaluate_bridge_parser.add_argument("--seed", type=int, default=11)
    evaluate_bridge_parser.add_argument("--splits", default="test,heldout")
    evaluate_bridge_parser.add_argument("--untrained", action="store_true")
    evaluate_bridge_parser.add_argument("--shuffle", action="store_true")
    evaluate_bridge_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_bridge_parser.set_defaults(func=_evaluate_bridge)

    analyze_parser = subparsers.add_parser("analyze", help="write summaries, bootstrap intervals and plots")
    analyze_parser.set_defaults(func=_analyze)

    smoke = subparsers.add_parser("smoke", help="run a small end-to-end GPU smoke from configs/smoke.json")
    smoke.add_argument("--regenerate-data", action="store_true")
    smoke.add_argument("--overwrite", action="store_true")
    smoke.set_defaults(func=_smoke)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    configure_cpu_threads(config.cpu_threads)
    args.func(config, args)


if __name__ == "__main__":
    main()
