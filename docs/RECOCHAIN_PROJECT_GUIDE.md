# RecoChain Project Guide

## Positioning

This repository is no longer a plain upstream GRID checkout. The current working branch is a modified research codebase centered on:

**RecoChain: Cache-Aware Unified Generative Retrieval and Ranking**

The original `README.md` still describes baseline GRID. This document is the branch-specific reference for future sessions.

Cross-branch working memory lives outside this repository at `/home/MMReco2021/liuyu/RECOCHAIN_NOTES.md`, so it remains visible after switching git branches.

## Research Positioning

The current research direction should be understood as a unified generate-and-rank framework, not as a fully differentiable end-to-end loss paper.

Core claim:

- RecoChain extends TIGER-style semantic ID generation into a cache-aware autoregressive ranking workflow.
- The same decoder first generates beam candidates, then continues scoring those candidates by reusing the generation-stage KV cache.
- The reranking head should be interpreted as a residual ranking scorer over a beam list, not as an independent second prediction of the ground-truth item probability.
- The listwise objective is a ranking-consistency fix: given a generated beam list, learn a normalized ranking distribution over candidates.

Important non-claims:

- Do not claim that the ranking loss is fully differentiable through top-k beam selection.
- Do not frame the paper as primarily a pointwise-vs-listwise loss paper.
- Do not rely on "joint loss is end-to-end" as the main innovation, because the beam construction step remains discrete.

The intended paper story is:

```text
TIGER-style semantic ID generation
        -> beam candidates
        -> same autoregressive decoder continues candidate scoring
        -> KV-cache-aware listwise reranking
```

## Baseline and Environment

- Baseline project: `GRID` from Snap Research.
- Current branch: `taobao-dec-only-hash-similar-itemkey`.
- Runtime environment used for this project:
  - conda env name: `GRID`
  - python path: `/home/MMReco2021/.conda/envs/GRID/bin/python`
  - Python version expected by the lockfile: `3.10`
- Environment: keep the same Python/CUDA/runtime setup as upstream GRID unless this branch explicitly documents a change.
- Dependency source of truth: [requirements.txt](/home/MMReco2021/liuyu/GRID/requirements.txt:1)
- `requirements.txt` was generated with a CUDA 12.4 PyTorch index; see the header comments in [requirements.txt](/home/MMReco2021/liuyu/GRID/requirements.txt:1).
- Entry points:
  - training: [src/train.py](/home/MMReco2021/liuyu/GRID/src/train.py:1)
  - inference: [src/inference.py](/home/MMReco2021/liuyu/GRID/src/inference.py:1)

## What Changed Relative to Upstream GRID

The current branch has substantially diverged from the original TIGER-style setup. The main changes visible from code and git history are:

- The recommendation model is now decoder-only, using a LLaMA backbone instead of the upstream encoder-decoder path.
- Training is organized as explicit tasks via `training_task`, instead of implicit phase switching.
- A click prediction head is added for reranking.
- Beam generation results and reranked results are both evaluated.
- Similar-item context is introduced through hashed item embeddings.
- Sequence handling was changed around semantic ID formatting and padding behavior.
- Logging and metadata persistence are much richer, including git state and Slurm artifact tracking.
- Auto-restart / resume metadata is built into the training launcher.

Key files:

- model: [src/models/modules/semantic_id/tiger_generation_model.py](/home/MMReco2021/liuyu/GRID/src/models/modules/semantic_id/tiger_generation_model.py:1)
- base logging/eval behavior: [src/models/modules/base_module.py](/home/MMReco2021/liuyu/GRID/src/models/modules/base_module.py:1)
- retrieval metrics: [src/components/eval_metrics.py](/home/MMReco2021/liuyu/GRID/src/components/eval_metrics.py:1)
- run metadata capture: [src/utils/logging_utils.py](/home/MMReco2021/liuyu/GRID/src/utils/logging_utils.py:1)
- restart logic: [src/utils/restart_job.py](/home/MMReco2021/liuyu/GRID/src/utils/restart_job.py:1)

## Model Summary

Current primary training model:

- class: `SemanticIDEncoderDecoder`
- file: [src/models/modules/semantic_id/tiger_generation_model.py](/home/MMReco2021/liuyu/GRID/src/models/modules/semantic_id/tiger_generation_model.py:346)
- actual backbone in training config: `transformers.models.llama.modeling_llama.LlamaModel`
- config source: [configs/experiment/tiger_train_flat.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_flat.yaml:246)

High-level behavior:

1. Autoregressively generate semantic IDs with beam search.
2. Use beam candidates as retrieval outputs.
3. Reuse the beam-search KV cache and continue the same decoder path to score candidates.
4. Use a rerank/click head as a residual ranking scorer over the beam list.
5. Optionally train SID generation only, or jointly train SID generation and click reranking.

Additional model components:

- `sid_head`: predicts hierarchical SID tokens.
- `click_head`: produces a residual rerank logit for each beam candidate.
- `item_embedding_table`: dual-hash item embedding used in similar-item context construction.
- `powers` buffer: converts hierarchical SID digits into flattened item keys.

Recommended scoring interpretation:

```text
score(i | u, B) = beam_log_prob_weight * log P_gen(i | u)
                + rerank_logit_weight * r_click(u, i, B)
```

Here `B` is the generated beam list. `r_click` is not treated as a standalone calibrated click probability in the main formulation; it is a residual utility used to reorder candidates within `B`.

## Supported Experimental Modes

### 1. SID-only autoregressive training

- config switch: `training_task=sid`
- active override: [configs/experiment/tiger_train_sid.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_sid.yaml:1)
- behavior:
  - train `sid_head`
  - freeze `click_head`
  - validation/test primary metrics come from beam/base retrieval

### 2. Joint SID + rerank multitask training

- config switch: `training_task=click`
- active override: [configs/experiment/tiger_train_click.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_click.yaml:1)
- behavior:
  - train `sid_head` and `click_head`
  - beam search produces hard negatives
  - click head reranks beam candidates with a listwise softmax objective over the beam list
  - the main rerank loss is applied on natural beam lists that already contain the ground-truth item
  - no-hit batches can optionally add a light GT-injected auxiliary listwise loss
  - final rerank score is `beam_log_prob_weight * beam_log_prob + rerank_logit_weight * rerank_logit`
  - rerank fusion weight sweeps are available via `tiger_train_click_rerank02` and `tiger_train_click_rerank03`
  - validation/test primary metrics come from rerank results
  - gain metrics are logged as `rerank - beam`

### 3. SID model plus a separate small rerank model

This mode is part of the experiment landscape, and is intended as a strong standalone baseline rather than as the main proposed model.

Research role:

- this line represents the modular two-stage reranking family
- it should serve as a serious baseline against the unified RecoChain path
- it should not be turned into another shared autoregressive scorer, otherwise the method boundary becomes unclear

Design guidance for the small-reranker line:

- it does not need to inherit the exact rerank formulation from the joint branch
- it is acceptable, and encouraged, to reference stronger rerank patterns from papers or industry practice
- candidate directions include richer user-candidate interaction features, lightweight transformer rerankers, two-tower-plus-cross features, and other classic ranking architectures, as long as they remain meaningfully smaller and more modular than the joint model

What is already available in code:

- evaluator/logging structure supports beam and rerank outputs separately
- metrics can expose both base and rerank views

What is not obviously standardized in config:

- a standalone rerank model training/inference pipeline with its own experiment yaml and launcher

If this mode becomes primary, add a dedicated experiment config rather than overloading the joint-training path.

## Metric Semantics

Current metric logging behavior is task-aware:

- for `training_task=sid`
  - `val/recall@5`, `val/ndcg@5`, etc. map to beam/base metrics
  - no `*_gain` metrics are logged
- for `training_task=click`
  - `val/recall@5`, `val/ndcg@5`, etc. map to rerank metrics
  - `val/beam_*`, `val/rerank_*`, and `val/*_gain` are all available

Relevant implementation:

- [src/models/modules/base_module.py](/home/MMReco2021/liuyu/GRID/src/models/modules/base_module.py:142)
- [src/models/modules/semantic_id/tiger_generation_model.py](/home/MMReco2021/liuyu/GRID/src/models/modules/semantic_id/tiger_generation_model.py:566)

## Data

Current experiments use the Taobao multi-modal recommendation dataset family:

- source: https://taobao-mm.github.io/

Repository-local convention seen in run scripts:

- dataset root: `./data/taobao`
- semantic ID file example: `./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy`

Expected split layout in training config:

- `training`
- `evaluation`
- `testing`

These paths are wired in:

- [configs/experiment/tiger_train_flat.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_flat.yaml:53)

## Main Training Commands

### SID stage

From [run_sid.sh](/home/MMReco2021/liuyu/GRID/run_sid.sh:1):

```bash
python -m src.train "experiment=[tiger_train_flat,tiger_train_sid]" \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096
```

### Joint stage

From [run_click.sh](/home/MMReco2021/liuyu/GRID/run_click.sh:1):

```bash
python -m src.train "experiment=[tiger_train_flat,tiger_train_click]" \
    data_dir=./data/taobao \
    semantic_id_path=./data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy \
    num_hierarchies=4 \
    num_embeddings_per_hierarchy=4096
```

### KV-cache efficiency profiling

From [run_profile_kv_cache.sh](/home/MMReco2021/liuyu/GRID/run_profile_kv_cache.sh:1):

```bash
sbatch run_profile_kv_cache.sh
```

The launcher runs [scripts/profile_rerank_efficiency.py](/home/MMReco2021/liuyu/GRID/scripts/profile_rerank_efficiency.py:1) on fixed test batches and reports model-forward latency only. It is separate from Lightning validation/test loops, so it should be used for paper efficiency claims instead of `perf/val_samples_per_sec`.

Current profiling outputs:

- `logs/profiles/listwise_kv_cache_profile.json`
- `logs/profiles/pointwise_kv_cache_profile.json`
- `logs/profiles/sid_beam_only_kv_cache_profile.json`
- `logs/profiles/separate_rerank_small_trained_profile.json`

For `--profile-mode recochain`, the main cache ablation compares:

- `beam_only`: semantic-ID beam generation only
- `rerank_cache`: beam generation plus RecoChain reranking with generation-stage KV cache reuse
- `rerank_no_cache`: same beam candidates, but reranking recomputes the full history/candidate prefix

For `--profile-mode separate_rerank`, the measured path compares:

- `beam_only`: frozen SID teacher beam generation only
- `separate_rerank`: frozen SID teacher beam generation plus the standalone small reranker scorer

The current separate-reranker profiling uses the trained checkpoint at `logs/train/runs/2026-07-12/23-28-36/checkpoints/checkpoint_000_020000.ckpt`.

## Important Training Config Defaults

Current defaults in [configs/experiment/tiger_train_flat.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_flat.yaml:1):

- backbone hidden size: `1024`
- decoder layers: `2`
- attention heads: `16`
- `top_k_for_generation=20`
- `top_k_for_score=10`
- optimizer: `AdamW`
- scheduler: cosine warmup
- precision: `bf16-mixed`
- gradient accumulation: `4`
- validation interval: every `20000` steps
- checkpoint monitor: `val/recall@5`

## Outputs and Run Artifacts

Each run writes under Hydra output directories and persists:

- checkpoints
- CSV logs
- TensorBoard logs
- metadata directory
- `run_metadata.json`
- `git_commit.txt`
- symlinks to matching Slurm logs when available

Relevant code:

- [src/utils/logging_utils.py](/home/MMReco2021/liuyu/GRID/src/utils/logging_utils.py:102)
- [src/utils/restart_job.py](/home/MMReco2021/liuyu/GRID/src/utils/restart_job.py:149)

## Recommended Files to Read First in a New Session

1. [docs/RECOCHAIN_PROJECT_GUIDE.md](/home/MMReco2021/liuyu/GRID/docs/RECOCHAIN_PROJECT_GUIDE.md:1)
2. [configs/experiment/tiger_train_flat.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_flat.yaml:1)
3. [configs/experiment/tiger_train_sid.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_sid.yaml:1)
4. [configs/experiment/tiger_train_click.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_train_click.yaml:1)
5. [src/models/modules/semantic_id/tiger_generation_model.py](/home/MMReco2021/liuyu/GRID/src/models/modules/semantic_id/tiger_generation_model.py:346)
6. [src/models/modules/base_module.py](/home/MMReco2021/liuyu/GRID/src/models/modules/base_module.py:142)

## Known Caveats

- The root `README.md` is still upstream-oriented and should not be treated as the current project spec.
- The training path is clearly aligned to the decoder-only LLaMA-based RecoChain branch.
- [configs/experiment/tiger_inference_flat.yaml](/home/MMReco2021/liuyu/GRID/configs/experiment/tiger_inference_flat.yaml:1) still contains upstream-style T5 encoder-decoder settings and should be revalidated before using it as the official RecoChain inference entry.
- If a separate rerank-small-model workflow becomes important, it should get its own explicit experiment config and launcher script.
- If the paper discusses joint training, describe it as joint training after discrete beam construction, not as a fully differentiable end-to-end objective through candidate generation.
