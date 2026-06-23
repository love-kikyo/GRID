# RecoChain Project Guide

## Positioning

This repository is no longer a plain upstream GRID checkout. The current working branch is a modified research codebase centered on:

**RecoChain: Unified Generative Retrieval and Ranking**

The original `README.md` still describes baseline GRID. This document is the branch-specific reference for future sessions.

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
3. Use a click head to rerank beam candidates.
4. Optionally train SID generation only, or jointly train SID generation and click reranking.

Additional model components:

- `sid_head`: predicts hierarchical SID tokens.
- `click_head`: predicts candidate click relevance for reranking.
- `item_embedding_table`: dual-hash item embedding used in similar-item context construction.
- `powers` buffer: converts hierarchical SID digits into flattened item keys.

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
  - click head reranks beam candidates
  - validation/test primary metrics come from rerank results
  - gain metrics are logged as `rerank - beam`

### 3. SID model plus a separate small rerank model

This mode is part of the experiment landscape, and is intended as a strong standalone baseline rather than as the main proposed model.

Current research expectation for ranking quality on this branch:

- `joint sid + rerank multitask training` > `small rerank model` > `sid-only`

Interpretation of that expectation:

- the joint multitask setup is the main proposed model and is expected to be strongest overall
- the separate small reranker should still beat SID-only beam retrieval and serve as a meaningful baseline
- the goal of this branch is to keep improving the small reranker, even if it does not surpass the joint model

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
