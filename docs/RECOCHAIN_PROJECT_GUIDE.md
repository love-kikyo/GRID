# RecoChain Project Guide

## Positioning

This repository is no longer a plain upstream GRID checkout. The current working branch is a modified research codebase centered on:

**RecoChain: Unified Generative Retrieval and Ranking**

The original `README.md` still describes baseline GRID. This document is the branch-specific reference for future sessions.

## Baseline and Environment

- Baseline project: `GRID` from Snap Research.
- Current branch: `taobao-dec-only-hash-similar-itemkey`.
- Environment: keep the same Python/CUDA/runtime setup as upstream GRID.
- Dependency source of truth: [requirements.txt](/home/MMReco2021/liuyu/GRID/requirements.txt:1)
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

This mode is part of the experiment landscape, but it is not currently represented by a dedicated first-class config in this branch in the same way as modes 1 and 2.

What is already available in code:

- evaluator/logging structure supports beam and rerank outputs separately
- metrics can expose both base and rerank views

What is not obviously standardized in config:

- a standalone rerank model training/inference pipeline with its own experiment yaml and launcher

If this mode becomes primary, add a dedicated experiment config rather than overloading the joint-training path.

Recommended direction for standardizing this mode:

- treat it as a separate experiment family, e.g. `sid_rerank_small`
- keep existing `training_task=sid` and `training_task=click` semantics unchanged
- do not add a third branch to the current joint `training_step`
- split the workflow into:
  - offline SID candidate generation from a trained SID checkpoint
  - standalone rerank model training/evaluation on cached candidates

#### Proposed workflow for mode 3

Stage A: SID candidate generation

1. Load a trained SID checkpoint.
2. Run beam search on evaluation/training examples without training any rerank head.
3. Save candidate sets to local cache files for later rerank training.

Recommended split-level outputs:

- `training_candidate_cache/predictions_<rank>_<timestamp>.pt`
- `evaluation_candidate_cache/predictions_<rank>_<timestamp>.pt`
- `testing_candidate_cache/predictions_<rank>_<timestamp>.pt`
- `training_candidate_cache/shard_manifest.json`
- `evaluation_candidate_cache/shard_manifest.json`
- `testing_candidate_cache/shard_manifest.json`
- `candidate_manifest.json`

Stage B: standalone small reranker

1. Read cached beam candidates.
2. Train a separate lightweight ranking model on top of those candidates only.
3. Evaluate both beam baseline and reranked results from the standalone reranker.

This keeps the expensive autoregressive retrieval path and the cheap rerank path decoupled.

#### Candidate cache schema

The candidate generator should write one row per user/example. A practical first schema is:

- `user_id`
- `hist_itemkey`
- `target_itemkey`
- `target_sid`
- `beam_candidate_sids`
- `beam_candidate_itemkeys`
- `beam_scores`
- `candidate_labels`
- `label_index`

Notes:

- `candidate_labels` is a binary list aligned with the beam candidates.
- `label_index` is optional if exactly one positive candidate is guaranteed.
- keep `beam_candidate_sids` and `beam_candidate_itemkeys` both, so rerank models can choose SID-level or item-level features.
- first version should cache only beam candidates, not joint-model rerank logits.
- for large-scale training, keep split outputs shard-based instead of merging to a single file.
- current preferred shard format is tensorized `.pt` rather than `pickle(list[dict])`.

#### Required new first-class configs

Recommended new experiment configs:

- `configs/experiment/tiger_infer_sid_candidates.yaml`
- `configs/experiment/tiger_train_rerank_small.yaml`

Recommended new launch scripts:

- `run_sid_candidates.sh`
- `run_rerank_small.sh`

Recommended execution order:

1. Run `run_sid_candidates_all_splits.sh` to export candidate shards for `training`, `evaluation`, and `testing`, and to generate `candidate_manifest.json`.
2. Run `run_rerank_small.sh` to train the standalone rerank model from that manifest.

Role of each launcher:

- `run_sid_candidates_all_splits.sh`
  - primary data-preparation entry for mode 3
  - expected to be used before standalone rerank training
- `run_sid_candidates.sh`
  - single-split candidate export helper
  - use it for debugging, smoke tests, or partial refresh of one split
  - typical cases:
    - verify candidate row schema on `testing`
    - benchmark export throughput on one split
    - regenerate only one split after changing export logic
- `run_rerank_small.sh`
  - standalone rerank training entry
  - assumes `candidate_manifest.json` already exists

Example workflow:

```bash
cd GRID
# regenerate candidate shards after export-format changes
bash run_sid_candidates_all_splits.sh
bash run_rerank_small.sh
```

Example single-split debug workflow:

```bash
cd GRID
bash run_sid_candidates.sh
```

Current branch status:

- candidate export config now exists as `tiger_infer_sid_candidates.yaml`
- first standalone rerank training config now exists as `tiger_train_rerank_small.yaml`
- multi-split candidate export launcher now exists as `run_sid_candidates_all_splits.sh`
- both launchers exist, but the rerank-small path should still be treated as a v1 scaffold until candidate-cache split handling and itemkey indexing strategy are fully validated

The inference config should point to a trained SID checkpoint and a prediction writer output path.

The rerank config should introduce dedicated fields such as:

- `candidate_manifest_path`
- `num_candidates`
- `rerank_model_type`
- `use_beam_score`
- `use_sid_features`
- `pretrained_sid_ckpt_path`
- `freeze_pretrained_embeddings`

Recommended manifest contents:

- `splits.train`
- `splits.val`
- `splits.test`
- `num_hierarchies`
- `num_embeddings_per_hierarchy`
- `num_candidates`

Recommended split manifest contents:

- `manifest_path`
- `num_shards`
- `shards`
- `shard_paths`
- `shard_lengths`
- `total_rows`

#### Required code additions

Recommended new code paths:

- candidate export logic in [src/models/modules/semantic_id/tiger_generation_model.py](/home/MMReco2021/liuyu/GRID/src/models/modules/semantic_id/tiger_generation_model.py:835)
- `src/data/loading/components/dataloading/rerank_candidate_dataset.py`
- `src/data/loading/datamodules/rerank_candidate_datamodule.py`
- `src/models/modules/rerank/sid_rerank_small_model.py`

The current `SemanticIDEncoderDecoder` should only be extended enough to support candidate export via `predict_step`.

The standalone reranker should be a separate module with its own:

- `training_step`
- `eval_step`
- `predict_step`
- evaluator selection logic

#### Rerank model scope for v1

The first implementation should optimize for a stable experiment loop, not architectural completeness.

Recommended v1 model:

- history encoder: mean pooling or attention pooling over history item embeddings
- candidate encoder: candidate item embedding plus optional SID embedding
- extra dense features: beam score and simple similarity-derived features
- scorer: a small MLP producing one click logit per candidate

This is sufficient to establish the mode-3 experiment family. DIN/SIM-style attention can be added later without changing the data contract.

#### Data module shape for rerank-only training

Recommended first training formulation: listwise reranking.

Each batch example should contain:

- history features for one user/example
- `K` candidates
- `K` beam scores
- `K` labels

This matches the actual inference problem and simplifies rerank metric computation.

#### Metric semantics for mode 3

Mode 3 should behave like a rerank experiment from the metric consumer's perspective:

- primary metrics should come from rerank results
- beam metrics should still be logged as baseline
- gain metrics should be computed as `rerank - beam`

Do not bind this to `training_task=click`. If needed, the standalone rerank module should override evaluator-selection methods instead of expanding the existing task enum.

#### Recommended implementation order

1. Add SID candidate export inference.
2. Add candidate-cache dataset and datamodule.
3. Add a minimal standalone rerank model.
4. Add dedicated train/eval configs and shell launchers.
5. Update docs and experiment naming to make the mode first-class.

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
