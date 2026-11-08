# Source evidence and interpretation boundaries

Retrieved 2026-11-08. These notes distinguish source claims from this project's hypotheses and measured results. Experimental result files, rather than this literature note, are the authority for reported numbers.

## PLaMo 3 Translate

Source: https://huggingface.co/pfnet/plamo-3-translate-0.8b and its linked `config.json`, `modeling_plamo.py`, and usage examples.

The model card describes a translation-specialized causal LM, available in four sizes, with 53 supported languages and optional context, glossary and Japanese style controls. Use the official translation chat template, including its language tags. The 0.8B configuration has hidden size 1,536 and 16 decoder layers. Its implementation exposes `output_hidden_states` and `inputs_embeds`. These interfaces make the proposed experiment implementable; they do not establish that hidden states transfer useful information to another LM.

Do not equate manufacturer translation benchmarks with downstream reasoning performance. No source claims that this particular model is a trained general-purpose multilingual encoder. Its causal source representations and the ordered pooling bottleneck differ from a bidirectional encoder. Do not infer the training-time context limit from `max_position_embeddings` alone.

The weights use the PLaMo Community License, not this project's code license. Personal and non-commercial academic use is described as available without prior registration. The public repository will link to upstream weights and will not redistribute them. License source: https://plamo.preferredai.jp/info/plamo-community-license-en

## LangBridge

Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat and Minjoon Seo. *LangBridge: Multilingual Reasoning Without Multilingual Supervision*. ACL 2024, pp. 7502–7522. DOI: 10.18653/v1/2024.acl-long.405.

Primary source: https://aclanthology.org/2024.acl-long.405/ ; full paper: https://aclanthology.org/2024.acl-long.405.pdf

Sections 3.2–3.3 and Figure 2 describe connecting a multilingual encoder to a frozen LM through a learned linear map and a trainable end token. The multilingual encoder is frozen for adapted finetuned LMs and trainable for adapted pretrained LMs. English-only alignment and cross-lingual transfer are central to their design. The paper reports improvements on its tested model/task combinations; those findings are not evidence of a benefit for PLaMo with Qwen.

This project instead tests a frozen translation-specialized causal decoder, pooled intermediate states, a small instruction-tuned target, and Japanese task-supervised alignment. It is an exploratory architectural adaptation, not a reproduction of LangBridge and not a claim to have invented hidden-state bridging. Do not describe Japanese supervised training as zero-shot cross-lingual adaptation.

## Qwen target

Source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

Use the model's own tokenizer and chat template. The card lists the target under Apache 2.0. Record the actual configuration and revision when loaded. A small instruction-tuned target is a practical test bed, not a surrogate for all reasoning LMs.

## MGSM

Freda Shi and colleagues. *Language Models are Multilingual Chain-of-Thought Reasoners*. arXiv:2210.03057 (2022). https://arxiv.org/abs/2210.03057

The abstract describes manually translating 250 GSM8K problems into ten languages. This project uses the official Japanese TSV from https://github.com/google-research/url-nlp/tree/452a21ad3dae5668c06ceeac21ff073e1e40f9be/mgsm , with source provenance and per-item evaluation IDs retained. Its CC BY 4.0 license is in the MGSM subdirectory. Reference answers are used only for scoring. The numeric-only zero-shot decoding differs from the original chain-of-thought prompting study. Benchmark pretraining contamination cannot be ruled out.

## Planned interpretations

- Learned vs random bridge: evidence that task-supervised adaptation affects this interface; not a causal isolation of translation pretraining.
- Correct vs shuffled source prefix: evidence of source dependence conditional on this trained model and shuffle; not proof that the target performs every arithmetic operation.
- Bridge vs frozen text: a practical pipeline comparison with unequal task supervision. The matched-data soft-prefix textual baseline helps contextualize that inequality but is not parameter matched.
- Layer or prefix differences: empirical effects for the measured choices; do not generalize to a universal best layer or language-neutral representations.
- Synthetic accuracy: evidence on a controlled generator. Template holdout and MGSM, when actually executed, are distinct generalization endpoints.
