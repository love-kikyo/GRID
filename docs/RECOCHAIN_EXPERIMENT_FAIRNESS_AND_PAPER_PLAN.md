# RecoChain Experiment Fairness and Paper Plan

## Goal

This note summarizes how to present the current RecoChain results fairly, what additional experiments are most important for the paper, and how far the separate small reranker should be upgraded.

Current observed ranking order:

- `joint sid + rerank multitask training` > `small rerank model` > `sid-only`

This ordering is encouraging, but the paper should avoid over-claiming why `joint` wins unless the comparison dimensions are clearly separated.

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

- RecoChain achieves the best overall accuracy with a unified autoregressive generation-and-scoring workflow.
- Its gain is parameter-efficient, because the additional reranking parameters in the joint model are very small.
- The gain is not free in computation, because the joint scoring path is slower than a standalone small reranker.

Avoid the weaker or potentially misleading claim:

- "`joint` wins because unified models are always better than separate rerankers."

Instead, write the conclusion in a controlled form:

- "`joint` achieves the best final accuracy under the current system design; compared with a stronger modular reranker, it uses fewer additional reranking parameters but more computation."

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

- `sid-only`
- `sid + separate small rerank`
- `joint sid + rerank multitask training`

Report at least:

- Recall@K
- NDCG@K
- trainable params
- added rerank params
- training time
- inference time

This should be the main paper table.

### B. Small reranker scaling study

Build at least three separate reranker sizes:

- small
- medium
- large

Purpose:

- test whether the separate reranker still remains below `joint` when given a stronger parameter budget
- identify whether the current small reranker has obvious unused headroom

How to interpret outcomes:

- if even the larger reranker remains below `joint`, the unified claim becomes much stronger
- if a slightly larger reranker catches up, then the correct paper claim becomes:
  - RecoChain is better in the parameter-efficient regime

### C. Joint-model ablation table

Recommended ablations:

- beam score only
- click score only
- beam score + click score fusion
- remove similar-item context
- vary `top_k_for_generation`

Purpose:

- show which component actually produces the gain
- show that the final result is not from one accidental heuristic

### D. Stability experiment

Run at least 3 random seeds for the main settings.

Report:

- mean
- standard deviation

Purpose:

- show the ranking order is stable
- reduce the chance that the result is due to variance

### E. Budget-matched comparisons

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
- pairwise or listwise ranking loss in addition to pointwise BCE
- lightweight transformer or cross-feature scorer instead of only MLP
- distillation from joint click logits, if desired

Stop upgrading when:

- it is clearly a serious modular baseline
- its parameter and compute profile is documented
- further upgrades would blur the paper's method boundary

## Suggested Paper Narrative

A clean narrative for the experiments section is:

1. `sid-only` establishes the generative retrieval baseline.
2. `small rerank model` shows that a strong modular post-ranker improves over retrieval-only decoding.
3. `joint` delivers the best overall ranking accuracy.
4. The `joint` gain is notable because it uses very few added scoring parameters.
5. The `joint` gain is not free, because it incurs higher computation than the standalone small reranker.

This narrative is much more defensible than saying the joint model wins simply because it is larger.

## Suggested Reviewer-Facing Conclusion

If the future experiments support the current trend, the conclusion can be written as:

- RecoChain unifies candidate generation and ranking score estimation within a single autoregressive workflow.
- Compared with a standalone modular reranker, RecoChain achieves the best ranking accuracy while introducing only a minimal additional scoring head.
- The tradeoff is increased computation, not increased reranking parameter count.

## Minimal Experiment Priority Order

If time is limited, prioritize in this order:

1. Main comparison table with efficiency columns
2. Small reranker scaling study
3. Joint ablation table
4. Multi-seed stability
5. Budget-matched fairness experiment

## Practical Bottom Line

For the current paper, the key message should be:

- `joint` is not a bigger reranker in parameter count
- `joint` is a more unified and more compute-intensive scoring workflow
- it beats a parameter-heavier modular reranker anyway

That is already a meaningful and publishable experimental story if the additional evidence above is filled in.
