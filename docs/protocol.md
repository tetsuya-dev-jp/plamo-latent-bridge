# PLaMo latent bridge: exploratory experiment protocol

## Objective and scope

Test whether frozen PLaMo 3 Translate hidden states can provide useful inputs to a separate frozen causal language model through a trainable bridge. Compare this interface with ordinary text inputs and controls, retaining negative results and per-item evidence. Experimental implementation was produced with DeepSeek V4.1 Flash.

The inputs are generated arithmetic questions and public Japanese MGSM questions. Public artifacts include code, generated data, measured results and the report; upstream model weights and feature caches are obtained locally.

## Models and compute

- Source: `pfnet/plamo-3-translate-0.8b`, frozen, BF16 where supported; capture hidden states at layers 4, 8 and 16. Record the exact resolved upstream revisions.
- Target: `Qwen/Qwen2.5-1.5B-Instruct`, frozen, BF16, eager attention as necessary for ROCm compatibility. Use its own tokenizer and chat template.
- GPU: AMD Radeon RX 9070 XT, approximately 16 GiB; a GPU matrix multiplication succeeded in the installed PyTorch 2.12 ROCm runtime on 2026-11-08.
- Preserve the existing research environment. Use a project virtual environment that can reuse installed ROCm PyTorch without replacing it. Record versions and actual peak allocated GPU memory.

## Architecture and supervision

Encode only the source question, with the official translation request prefix where appropriate; no reference answer or target translation enters source feature extraction. Pool valid, non-padding source token states into an ordered prefix (main: 16 tokens; ablation: 4). Use ordered contiguous bins spanning the full valid sequence, including its final state; that final causal state must be computed after the complete source question. Record the exact pooling rule and whether prompt-format tokens are included. A small projection maps these states into target embedding dimensions. Concatenate the projected prefix with a fixed English answer instruction represented using the target tokenizer. Train only the projection with teacher-forced answer cross-entropy. Do not detach the target forward pass: gradients must pass through the frozen target into the bridge.

Training and free generation must share exactly the same template-complete prompt, prefix position and assistant-answer boundary. Supervise the first answer token through the answer termination token, masking all prompt and padding positions; start free generation at that same boundary. Neither target reasoning traces nor target answers are available during evaluation. Free-running greedy answers are the primary endpoint; teacher-forced losses are diagnostics only. The originally planned strict exact-match score strips documented formatting and normalizes numeric strings. Also report a separately labeled last-number exact-match diagnostic, using a documented numeric extraction rule, because strict numeric-only compliance is not equivalent to arithmetic correctness. Publish raw answers, both scores, generation lengths, termination status and token caps. Use 128 answer tokens for all synthetic conditions and 256 for all external MGSM conditions; use up to 512 tokens for MGSM translation.

## Data and partitions

Use reproducible procedurally generated Japanese/English paired short arithmetic word problems spanning addition, subtraction, multiplication and a two-step composition. Answers are computed exactly. Keep semantic tuples disjoint between train, validation and test, independently of surface text. Fix the generator seed and save the generated dataset before model runs. Start with 1,024 train, 128 validation and 256 test examples, balanced by operation. Add a separate 128-example test using held-out surface templates. These are controlled synthetic tasks, not evidence of broad multilingual reasoning.

Use an external natural-language evaluation if feasible: the public Japanese MGSM test split, with its reference answers strictly evaluation-only. Report this separately; synthetic-only training and pretraining contamination limitations must be explicit. Do not manufacture any benchmark result when download or execution fails.

## Comparisons

Use identical test items and target answer instructions:

1. Frozen target, direct Japanese question.
2. Frozen target, paired English question (controlled reference-language diagnostic).
3. PLaMo-generated English translation, then frozen target.
4. Trained latent bridge (main layer 8, prefix 16), three seeds: 11, 23, 37.
5. Layer ablations 4 and 16 (seed 11).
6. Prefix length 4 (layer 8, seed 11).
7. Untrained bridge and shuffled-source-feature test controls, with explicit shuffle mapping.
8. A text-conditioned learned soft-prefix baseline, supervised on the same training examples, with source Japanese still provided as ordinary target text. Record its smaller parameter count; do not call it parameter matched.

The learned bridge has task supervision that the three frozen textual baselines do not. State this asymmetry prominently. Use validation loss to choose epoch/checkpoint. Fix all choices before final test evaluation and retain any exploratory test exposure in the record.

## Training, measurements and checks

Benchmark a small batch before full runs; choose a feasible batch size without changing the research question. Initial candidates: AdamW, learning rate 1e-3, up to 10 epochs, batch 8, gradient clipping 1.0. Record actual settings and changes; select checkpoints using validation data only. Save per-epoch loss, validation loss, elapsed times, peak memory, selected checkpoint, every per-item prediction and metrics.

An independent memory probe used 32 training and eight validation examples generated with seed 20261109, one epoch, and the longer text-conditioned soft-prefix path. Batch 16 used 9.149 GiB of peak allocated GPU memory and completed two optimizer steps. Select batch 16 for all full training conditions based on this resource check; no test answers were evaluated in that probe.

Check split disjointness, arithmetic answers, masking/pooling, causal answer loss alignment, that only bridge parameters receive gradients, and that bridge parameters update while both backbones remain frozen. Include tiny tests that do not need downloading full model weights. Avoid artifact hashes.

Report per-condition exact match, per-operation and held-out-template results, seed variation, learning curves and paired item bootstrap confidence intervals for selected differences. Explain the resampling unit; do not treat correlated training seeds as independent test items. Distinguish measured throughput from expected performance, and source extraction/translation costs from target decoding costs.

## Pre-full-run pilot observations and amendments

A wiring smoke used eight training items, four validation items and four items from each preliminary evaluation split, with a 16-token answer cap. It verified real source extraction, one optimizer step and generation, but is not quantitative evidence for model comparison. Inspection of those pilot outputs found mathematically correct answers accompanied by explanations that failed strict exact match. The secondary last-number diagnostic and larger, shared decoding caps above were specified before full runs to expose this formatting confound. Preserve the strict score as originally planned; do not relabel formatting improvements as reasoning improvements.

A separate generator audit found two train/test duplicates after canonicalizing the two multiplicative operands in two-step arithmetic. Correct the key construction and regenerate canonical data before full runs. The pilot and revised full dataset are exploratory, not a preregistered confirmatory evaluation; document their separation and any reused semantic examples.

## Sources

- PLaMo model and implementation: https://huggingface.co/pfnet/plamo-3-translate-0.8b
- Qwen target: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- LangBridge, ACL 2024: https://aclanthology.org/2024.acl-long.405/

## Execution records

The primary matrix and the later resubstitution diagnostic have separate configurations, predictions, run histories and provenance records. The English report presents their results and distinguishes implementation checks, memorization capacity and held-out task performance.
