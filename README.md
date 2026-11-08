# PLaMo latent bridge

Small exploratory experiment connecting frozen hidden states from
`pfnet/plamo-3-translate-0.8b` to the frozen causal language model
`Qwen/Qwen2.5-1.5B-Instruct` through a trainable projection.

This is a controlled synthetic arithmetic study. It is not evidence of broad
multilingual reasoning. The latent bridge receives task supervision that the
three frozen textual baselines do not; that asymmetry is part of the design
and must be reported with every comparison.

## What is implemented

- Procedurally generated Japanese/English paired arithmetic problems with
  addition, subtraction, multiplication and two-step composition.
  Semantic tuples are disjoint across train, validation, test and a separate
  held-out-template test. The default counts are 1,024 / 128 / 256 / 128.
- Cached, detached PLaMo source features at layers 4, 8 and 16, pooled over
  valid non-padding tokens into ordered prefixes of length 16 (main) and 4
  (ablation). Bins are contiguous and span the full valid sequence, including
  its final state. Prompt-format tokens are included because the PFML3
  translation request is encoded as a whole. The source sees only a PFML3 translation request containing the
  Japanese question; it never sees an answer or the paired English reference.
- Actual PLaMo English translations for the translation-pivot baseline.
- A trainable projection (`Linear(source_dim -> target_dim)`) injected into
  the target chat prompt. Teacher-forced labels touch only the answer tokens,
  and gradients flow through the frozen target into the bridge.
- A learned soft-prefix baseline conditioned on the Japanese question supplied
  as ordinary target text. Its parameter count is recorded separately and it
  is not described as parameter matched.
- Text baselines: Japanese question, paired English question, and
  PLaMo-generated English translation.
- Untrained and shuffled-source-feature controls. The shuffle mapping is saved
  explicitly item by item.
- Per-item predictions, per-operation summaries, seed variation, paired item
  bootstrap intervals, learning curves and an accuracy comparison plot.
- An optional loader for the external Japanese MGSM test split.

## Environment

Use Python 3.12 or later and install a PyTorch build compatible with your
accelerator in an isolated environment. The [PyTorch installation guide](https://pytorch.org/get-started/locally/)
provides commands for supported compute platforms. This study uses an AMD Radeon
RX 9070 XT with 16 GiB VRAM and PyTorch 2.12.0 with ROCm 7.14; GPU execution on
other hardware has not been measured here.

```bash
uv venv .venv --python 3.12
# Install the appropriate PyTorch build into .venv before the next command.
uv pip install --python .venv/bin/python -e . accelerate==1.15.0
```

The project fixes `transformers==4.57.1` and both upstream model revisions.
Other resolved package versions are recorded with the experiment environment.
PLaMo's tokenizer requires `numba`. An existing accelerator environment can also
be reused through a Python path file; when doing so, install `accelerate` with
`--no-deps` and ensure its dependencies are present, so a package resolver does
not replace the inherited accelerator-specific PyTorch build.

Upstream model weights are not included in this repository and remain under
their own licenses. The code here is MIT licensed.

## Commands

Run from the repository root. The default configuration pins model revisions
and uses six CPU threads.

```bash
# 1. Synthetic data, optionally with the external Japanese MGSM test split
.venv/bin/python -m plamo_bridge data --mgsm

# 2. Cached detached source features for all splits
.venv/bin/python -m plamo_bridge features --splits train,val,test,heldout

# 3. Actual PLaMo translations for the pivot baseline
.venv/bin/python -m plamo_bridge translate --splits test,heldout

# 4. Frozen target textual baselines
.venv/bin/python -m plamo_bridge evaluate-text \
  --conditions text_ja,text_en,text_plamo_en --splits test,heldout

# 5. Main bridge, three seeds
.venv/bin/python -m plamo_bridge train --condition bridge --layer 8 --prefix-length 16 --seed 11
.venv/bin/python -m plamo_bridge train --condition bridge --layer 8 --prefix-length 16 --seed 23
.venv/bin/python -m plamo_bridge train --condition bridge --layer 8 --prefix-length 16 --seed 37

# 6. Layer and prefix ablations
.venv/bin/python -m plamo_bridge train --condition bridge --layer 4 --prefix-length 16 --seed 11
.venv/bin/python -m plamo_bridge train --condition bridge --layer 16 --prefix-length 16 --seed 11
.venv/bin/python -m plamo_bridge train --condition bridge --layer 8 --prefix-length 4 --seed 11

# 7. Learned soft-prefix baseline
.venv/bin/python -m plamo_bridge train --condition soft_prefix --prefix-length 16 --seed 11

# 8. Evaluate every bridge/control on test and held-out templates
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed 11 --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed 23 --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed 37 --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 4 --prefix-length 16 --seed 11 --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 16 --prefix-length 16 --seed 11 --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 4 --seed 11 --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed 11 --untrained --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed 11 --shuffle --splits test,heldout
.venv/bin/python -m plamo_bridge evaluate-bridge --condition soft_prefix --prefix-length 16 --seed 11 --splits test,heldout

# 9. Summaries, paired item bootstrap intervals and plots
.venv/bin/python -m plamo_bridge analyze
```

The same full matrix is in `scripts/run_full_experiment.sh`. For the external
Japanese MGSM split, run `data --mgsm`, then `features --splits mgsm`,
`translate --splits mgsm`, `evaluate-text --splits mgsm` and
`evaluate-bridge ... --splits mgsm`; `analyze` reports that split separately.

A small end-to-end GPU smoke uses `configs/smoke.json` (8 train items, 4
validation items, one epoch, one optimizer step, four test and four held-out
items):

```bash
.venv/bin/python -m plamo_bridge --config configs/smoke.json smoke --regenerate-data --overwrite
```

Tests are offline and use tiny random Qwen configurations:

```bash
.venv/bin/python -m pytest
```

## Method notes

- Source prompt: the official PLaMo chat template with `target_lang="en"` and
  `source_lang="ja"`, containing only the Japanese question. This is the
  official PFML3 translation-request format.
- Target prompt: the target tokenizer's own chat template. The projected
  prefix replaces a dedicated source marker inside the user message, followed
  by a fixed English instruction: "Solve the following arithmetic problem.
  Give only the final number."
- Teacher forcing: labels are set only on answer tokens plus the target
  end-of-message token. Prefix, instruction and padding positions are `-100`.
- Evaluation: greedy free-running answers are the endpoint. The primary
  strict score strips surrounding whitespace, special-token strings, commas,
  full-width digits and a trailing Japanese/ASCII period; a purely numeric
  result is then compared as an integer. Prose answers are not parsed. A
  separately labeled last-number diagnostic takes the last numeric literal in
  the raw answer and compares its exact decimal value with the reference.
  Raw answers, both scores, generation lengths, termination flags and token
  caps are saved.
- Decoding caps: 128 answer tokens for synthetic conditions and 256 for the
  external MGSM split. Translation caps are 128 tokens for synthetic splits
  and 512 for MGSM.
- Training defaults: AdamW, learning rate 1e-3, up to 10 epochs, full-run
  batch size 16 and gradient clipping 1.0. The smoke configuration keeps batch
  8 and one epoch for wiring only. Checkpoints are selected by validation
  loss. Test results are written per item, so exploratory test exposure can be
  inspected later.
- Bootstrap: resampling is over test items, with the same item used in both
  conditions. Training seeds are reported separately and are not treated as
  independent items.

## Outputs

```
data/                      generated dataset and optional MGSM split
features/                  cached detached source features (ignored by git)
runs/<experiment>/<run>/   config, environment, history and best checkpoint
results/<experiment>/      predictions, metadata, shuffle mapping, summary, plots
```

Per-item predictions are JSONL. Summary metrics are JSON. The two required
plots are saved as PNG and vector PDF under
`results/<experiment>/plots/learning_curves.*` and
`results/<experiment>/plots/accuracy_comparison.*`. The smoke configuration
uses isolated paths under `local/smoke/` so its shortened caches can never be
consumed by a full run. Feature caches carry an item-ID manifest and are
rebuilt or rejected when rows, layers, prefixes or the source revision change.

## Limitations

- The training data are small, controlled synthetic templates with exact
  arithmetic answers.
- The bridge is trained with answer supervision while the textual baselines
  are frozen zero-shot systems.
- Synthetic-only training cannot establish broad multilingual reasoning, and
  pretraining contamination cannot be excluded.
- The external MGSM split is reported separately if it can be downloaded and
  executed. Its reference answers are evaluation-only.
