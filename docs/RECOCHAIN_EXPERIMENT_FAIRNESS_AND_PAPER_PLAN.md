# RecoChain Experiment Fairness and Paper Plan

## Goal

This note summarizes how to present the current RecoChain results fairly, what additional experiments are most important for the paper, and how far the separate small reranker should be upgraded.

The current paper direction should be:

- RecoChain is a cache-aware unified generate-and-rank framework.
- It extends TIGER-style semantic ID generation into an autoregressive reranking stage.
- It should not be framed as a fully differentiable end-to-end joint loss through beam selection.
- The listwise objective is important as a ranking-consistency correction, but it is not the whole paper.

The main comparison should be:

```text
two-stage generate-then-rerank
vs.
unified cache-aware generate-and-rank
```

not:

```text
pointwise loss
vs.
listwise loss
```

The ranking loss is applied after discrete beam construction. It can train the shared decoder states used in candidate scoring and the click/rerank head, but it should not be described as directly differentiating through top-k beam selection.

## Reviewer Concerns to Address

The revision should directly cover these points:

- missing comparison with TIGER
- missing comparison with OneRec, if feasible under available GPU and parameter-budget constraints
- missing comparison with TIGER plus a DIN-like or DIN-sim reranker
- pointwise BCE is not a true ranking objective over beam candidates
- KV-cache reuse is mentioned but not validated with efficiency experiments

Mapping from concern to response:

- TIGER comparison: use `RecoChain w/o rerank` / SID-only generation as the TIGER-style baseline, with naming made explicit in the paper.
- TIGER + DIN-style comparison: make the separate reranker branch a faithful modular reranking baseline, or label it conservatively as DIN-style if it is not exact DIN.
- Pointwise BCE concern: present listwise beam-level reranking as the corrected objective.
- KV-cache concern: add a latency/memory table isolating ranking with and without cache reuse.
- OneRec concern: attempt a fair run if resources permit; otherwise document resource limits and avoid pretending it was evaluated.

## Current Code-Level Reality

Based on the current implementation:

- `joint` adds only a lightweight `click_head` as new scoring parameters.
- `small rerank model` adds a dedicated reranker with substantially more parameters than the joint `click_head`.
- `joint` is slower mainly because the scoring path reuses the autoregressive backbone and runs a longer computation path with cached sequence context.
- `small rerank model` is more parameter-heavy in the reranking component, but computationally lighter at scoring time.

Implication:

- `joint` does **not** win because it has a larger added reranking module.
- `small rerank model` does **not** lose because it is too small in parameter count.
- The main tradeoff is:
  - `small rerank`: more reranking parameters, more modular
  - `joint`: fewer added parameters, deeper shared computation, stronger representation coupling

## Recommended Paper Claim

The safest and strongest claim is:

- RecoChain unifies candidate generation and candidate ranking in one autoregressive computation path.
- RecoChain reuses generation-stage KV cache for ranking, avoiding a full re-encoding of user history for each candidate.
- RecoChain uses a beam-level listwise objective to make the ranking stage mathematically consistent.
- Its gain is parameter-efficient, because the additional reranking parameters in the joint model are very small.

Avoid weaker or potentially misleading claims:

- "`joint` wins because unified models are always better than separate rerankers."
- "RecoChain is fully end-to-end differentiable from ranking loss to beam generation."
- "The main contribution is that listwise always outperforms pointwise."

Instead, keep the conclusion in a controlled form:

- "RecoChain turns a TIGER-style generator into a unified generate-and-rank chain. The ranking stage is trained with a beam-level objective and implemented with KV-cache reuse, giving a principled and efficient alternative to external two-stage rerankers."

## Why `joint-rerank-only` Is Not the Main Fairness Experiment

One tempting experiment is:

- freeze the backbone
- train only the joint `click_head`

This experiment is valid as a small ablation, but it should **not** be the main fairness comparison.

Reason:

- the joint `click_head` is only a linear layer
- the separate reranker is a much richer scorer with more parameters and more engineered interactions

Therefore this comparison mainly answers:

- "Is a linear scorer enough?"

It does **not** cleanly answer:

- "Is unified generation-plus-scoring better than modular reranking?"

Recommended use of this experiment:

- include it only as a minor ablation to show that a linear click head alone is not sufficient
- do not rely on it as the central evidence for the paper

## What Fairness Means in This Project

Fairness should be discussed on at least two axes.

### 1. Parameter fairness

Report:

- total parameters
- trainable parameters
- newly added reranking parameters

Expected interpretation:

- `small rerank model` is likely larger on the reranking side
- `joint` is likely much smaller on added scoring parameters

### 2. Computation fairness

Report:

- training wall-clock time
- average step time
- inference latency
- peak memory
- if possible, approximate FLOPs or tokens processed

Expected interpretation:

- `joint` is likely more expensive in computation
- `small rerank model` is likely cheaper in scoring-time compute

This leads to a fairer overall statement:

- `small rerank` is stronger in parameter budget
- `joint` is heavier in computation budget
- `joint` still gives the best final accuracy

## Most Important Additional Experiments

The paper does not need every imaginable experiment. The highest-value additions are below.

### A. Main comparison table

Include:

- `TIGER` / `RecoChain w/o rerank`
- `TIGER + DIN-style reranker`
- `TIGER + stronger separate reranker`, if available
- `RecoChain-Pointwise`, as the previous formulation
- `RecoChain-Listwise`, as the ranking-consistent formulation
- `OneRec`, if deployment and resource constraints permit a fair run

Report at least:

- Recall@K
- NDCG@K
- trainable params
- added rerank params
- training time
- inference time
- peak memory

This should be the main paper table.

Current reproducible separate-reranker baseline:

- run dir: `logs/train/runs/2026-07-12/23-28-36`
- Slurm job: `GRID-RerankSmall-122455`
- checkpoint: `logs/train/runs/2026-07-12/23-28-36/checkpoints/checkpoint_000_020000.ckpt`
- best validation checkpoint step: `20000`
- validation: R@5 `0.23633`, N@5 `0.23045`, R@10 `0.24078`, N@10 `0.23190`
- test: R@5 `0.23696`, N@5 `0.23187`, R@10 `0.24018`, N@10 `0.23291`
- test gain over beam: R@5 `+0.00349`, N@5 `+0.00217`, R@10 `+0.00217`, N@10 `+0.00174`
- test click AUC: `0.70072`

### B. Efficiency and KV-cache reuse table

This experiment is essential because the method explicitly mentions cache reuse.

Include:

- RecoChain ranking with KV cache reuse
- RecoChain ranking without KV cache reuse, if feasible
- TIGER + external reranker with separate user-history encoding

Report:

- inference latency per batch or per user
- ranking-stage latency
- peak GPU memory
- number of extra tokens or sequence steps processed during ranking

Expected interpretation:

- RecoChain is not merely "one more scorer"; it amortizes history computation through the generation cache.
- The relevant efficiency claim is lower marginal ranking cost than rerunning a full user-candidate encoder for every candidate.

Completed profiling:

- branch: `profile-kv-cache-efficiency`
- Slurm job: `GRID-ProfileKV-122453`, completed on 2026-07-12
- script: `scripts/profile_rerank_efficiency.py`
- launcher: `run_profile_kv_cache.sh`
- measured setting: Taobao test split, batch size 32, `top_k_for_generation=20`, bf16 autocast, 5 warmup batches, 30 measured batches, one GPU on `aias-compute-01`
- output files:
  - `logs/profiles/listwise_kv_cache_profile.json`
  - `logs/profiles/pointwise_kv_cache_profile.json`
  - `logs/profiles/sid_beam_only_kv_cache_profile.json`
  - `logs/profiles/separate_rerank_small_trained_profile.json`

Profiling results:

| model/checkpoint | variant | total ms/batch | gen ms/batch | rerank ms/batch | users/s | active params | added scorer params | peak GPU GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RecoChain-Listwise | beam only | 22.59 | 22.59 | 0.00 | 1416.77 | 144.18M | 1.02K click head | 1.49 |
| RecoChain-Listwise | beam + cache rerank | 56.21 | 23.20 | 33.01 | 569.25 | 144.18M | 1.02K click head | 4.56 |
| RecoChain-Listwise | beam + no-cache rerank | 99.57 | 23.51 | 76.06 | 321.38 | 144.18M | 1.02K click head | 4.66 |
| RecoChain-Pointwise | beam only | 22.40 | 22.40 | 0.00 | 1428.79 | 144.18M | 1.02K click head | 1.49 |
| RecoChain-Pointwise | beam + cache rerank | 55.99 | 23.01 | 32.97 | 571.55 | 144.18M | 1.02K click head | 4.56 |
| RecoChain-Pointwise | beam + no-cache rerank | 99.51 | 23.48 | 76.04 | 321.56 | 144.18M | 1.02K click head | 4.66 |
| SID / TIGER-style | beam only | 22.81 | 22.81 | 0.00 | 1403.01 | 144.18M | none | 1.49 |
| SID + separate small reranker | beam only | 22.91 | 22.91 | 0.00 | 1396.60 | 223.41M | 79.23M reranker | 1.89 |
| SID + separate small reranker | beam + small rerank | 24.94 | 23.13 | 1.81 | 1282.95 | 223.41M | 79.23M reranker | 1.89 |

Immediate interpretation:

- KV-cache reuse reduces the measured RecoChain ranking-stage latency from about `76.0 ms/batch` to about `33.0 ms/batch`, roughly `2.3x` faster for the ranking substep.
- End-to-end generation-plus-ranking is about `1.75x` faster with cache reuse than with full recomputation.
- Beam-only generation is stable across SID-only, pointwise, and listwise checkpoints at about `22-23 ms/batch`, so the speed comparison should not use Lightning `val_samples_per_sec` as a proxy for model-only latency.
- The separate small reranker is much faster at the marginal ranking step, about `1.8 ms/batch`, because it is a shallow standalone scorer rather than the shared autoregressive decoder path.
- The separate small reranker also introduces far more active reranking parameters: about `79.23M` active reranker parameters versus RecoChain's `1.02K` click-head parameters.
- The separate-reranker profiling now uses the reproduced trained checkpoint `logs/train/runs/2026-07-12/23-28-36/checkpoints/checkpoint_000_020000.ckpt`.
- The cache-reuse claim is now supported by a direct ablation inside the same RecoChain scorer. The external small reranker result should be framed as a speed/parameter tradeoff: faster shallow scoring but a separate large reranker module and weaker integration with the generator.

### C. Small reranker scaling study

Build at least three separate reranker sizes:

- small
- medium
- large

Purpose:

- test whether the separate reranker still remains below `joint` when given a stronger parameter budget
- identify whether the current small reranker has obvious unused headroom
- separate architectural advantage from parameter budget

How to interpret outcomes:

- if even the larger reranker remains below `joint`, the unified claim becomes much stronger
- if a slightly larger reranker catches up, then the correct paper claim becomes:
  - RecoChain is better in the parameter-efficient regime

### D. Joint-model ablation table

Recommended ablations:

- beam score only
- click score only
- beam score + click score fusion
- pointwise BCE ranking loss
- listwise beam-level ranking loss
- listwise without no-hit auxiliary loss
- remove similar-item context
- vary `top_k_for_generation`
- vary rerank fusion weight

Purpose:

- show which component actually produces the gain
- show that the final result is not from one accidental heuristic
- show that listwise is a ranking-consistency choice, not just an unexamined replacement for BCE

### E. Stability experiment

Run at least 3 random seeds for the main settings.

Report:

- mean
- standard deviation

Purpose:

- show the ranking order is stable
- reduce the chance that the result is due to variance

### F. Budget-matched comparisons

If resources allow, include one or both:

- equal wall-clock budget
- equal trainable-parameter budget

Purpose:

- address reviewer concerns that one system got an unfair advantage from more time or more capacity

This is useful, but should come after the main comparison, scaling study, and ablations.

## How Far to Upgrade the Small Reranker

The small reranker should be upgraded until it becomes a strong modular baseline, but it should not be upgraded so far that it effectively turns into another version of the joint model.

Recommended upgrade boundary:

- keep the SID generator separate and frozen
- keep reranking as a standalone module
- allow richer feature interactions and slightly stronger ranking architecture
- do not turn it into an end-to-end shared autoregressive scorer

Good upgrade directions:

- stronger feature set based on history-candidate interactions
- better use of beam score and beam rank
- listwise ranking loss over the beam list, rather than pointwise BCE
- a more faithful DIN-style interaction module when used as the `TIGER + DIN` baseline
- lightweight transformer or cross-feature scorer instead of only MLP
- distillation from joint click logits, if desired

Stop upgrading when:

- it is clearly a serious modular baseline
- its parameter and compute profile is documented
- further upgrades would blur the paper's method boundary

## Suggested Paper Narrative

A clean narrative for the experiments section is:

1. TIGER-style semantic ID generation establishes the generative retrieval baseline.
2. External rerankers such as DIN-style modules establish the conventional two-stage generate-then-rerank path.
3. RecoChain uses the same autoregressive decoder to continue from generation into ranking.
4. The listwise beam-level objective makes this ranking stage consistent with the candidate list structure.
5. KV-cache reuse explains why the ranking stage is computationally different from a normal external reranker.

This narrative is more defensible than saying the joint model wins simply because it is larger or because listwise loss alone is stronger.

## Suggested Reviewer-Facing Conclusion

If the future experiments support the current trend, the conclusion can be written as:

- RecoChain unifies candidate generation and ranking score estimation within a cache-aware autoregressive workflow.
- Compared with external two-stage rerankers, RecoChain reuses the generator's hidden context and KV cache for candidate scoring.
- The revised listwise objective directly optimizes a distribution over the beam list, addressing the limitation of pointwise BCE.
- The method is not claimed to be differentiable through beam selection; instead, it provides a principled ranking stage over generated candidates.

## Minimal Experiment Priority Order

If time is limited, prioritize in this order:

1. Main comparison table against TIGER and TIGER + DIN-style reranking
2. KV-cache efficiency table
3. Joint ablation table, especially pointwise vs listwise and score fusion
4. Small reranker scaling study
5. Multi-seed stability
6. OneRec if deployment is feasible
7. Budget-matched fairness experiment

## Practical Bottom Line

For the current paper, the key message should be:

- RecoChain is not primarily an end-to-end loss contribution.
- RecoChain is a unified, cache-aware generate-and-rank architecture.
- Listwise ranking is the correct objective for the beam-level reranking stage.
- The paper should be defended through TIGER / DIN-style / RecoChain comparisons plus KV-cache efficiency evidence.

That is the most coherent and defensible experimental story to build around from the current codebase.
