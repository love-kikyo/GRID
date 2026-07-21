# RecoChain Pairwise Exploration

本文档只维护当前 `explore-pairwise-ranking` 分支的实验设计、实现路径和结果记录。
全局项目计划、总实验表和论文包装不在本分支维护；listwise 分支的历史记录也不在本分支继续同步。

## 分支目标

本分支专门探索 pairwise ranking objective 是否能比纯 listwise 更适合 RecoChain 的 click head。

核心问题：

> 在 beam 内通常只有一个 ground-truth item、且 no-hit 行较多的情况下，pairwise 是否能提供比 listwise 更稳定、更直接的相对排序信号，同时保留 pointwise BCE 的 residual ranking 优势？

## 背景判断

已有实验显示：

- Pointwise BCE 指标最好，但形式上容易被质疑为“逐候选二分类”，不是 beam-level ranking。
- Listwise 更符合 reviewer 对 ranking objective 的要求，但纯 listwise 仍低于 pointwise。
- 直接用 listwise 优化 final score 后，click AUC 进一步偏低，说明 click logit 更像 weak residual，不再是稳定概率 logit。

因此 pairwise 的定位不是立刻替代 pointwise，而是诊断和寻找折中：

- 如果 pairwise 明显优于 listwise，说明 relative preference learning 是可行方向。
- 如果 pairwise 仍低于 pointwise，后续应转向 `pairwise/listwise + BCE calibration` 的 hybrid loss。

## 首选实验设计

不建议一开始把 `top_k_for_generation` 改成 2。

原因：

- `K=2` 时 hit-in-beam 可能显著下降，主 pairwise loss 有效样本不足。
- 训练只看到 top-2 局部比较，推理仍要 rank top-20，训练/推理分布不一致。
- no-hit 行会更多，auxiliary 人工样本占比过高。

首选方案是：

```text
generate K=20 candidates
score all beam candidates with click head
construct pairwise positives/negatives inside the beam
use pairwise loss to train click head
inference still ranks the full beam list
```

## Pair Construction

### Hit Rows

当 beam 中包含 ground-truth candidate：

```text
positive = candidate whose itemkey == target_itemkey
negative = non-GT beam candidate
```

负样本优先使用 hard negative：

```text
negative = highest beam_log_prob candidate among non-GT candidates
```

后续可扩展为 top-m hard negatives：

```text
negative_set = top m non-GT candidates by beam_log_prob
```

### No-Hit Rows

当 beam 中没有 ground-truth candidate：

复用当前 no-hit auxiliary 思路，teacher-force ground-truth SID 得到 `gt_click_logit`，再与自然 beam 中的 hard negative 构造 pair。

建议 aux 权重从较小值开始：

```text
pairwise_aux_loss_weight = 0.1 或 0.2
```

避免 auxiliary 分布过强地支配训练。

## Candidate Losses

### Pairwise Over Click Logits

首选：

```text
L_pair = softplus(-(click_pos - click_neg))
```

多负样本：

```text
L_pair = mean_j softplus(-(click_pos - click_neg_j))
```

推理仍使用当前融合：

```text
final_score = beam_log_prob + rerank_logit_weight * click_logit
```

这个版本最接近 pointwise 的使用方式：click head 学 residual preference，beam score 仍作为 generator prior。

### Pairwise Over Final Score

备选：

```text
score = beam_log_prob.detach() + rerank_logit_weight * click_logit
L_pair = softplus(-(score_pos - score_neg))
```

风险：

- `rerank_logit_weight=0.1` 会缩小 click head 梯度。
- beam prior 太强时，click head 可能只学到很弱 residual。

因此它不作为第一优先级。

## 实验顺序

建议按以下顺序推进：

1. Pairwise over `click_logits`, `K=20`, hard negative top1。（已实现）
2. Pairwise over `click_logits`, `K=20`, top3 hard negatives + BCE calibration。（已实现）
3. Pairwise over final score, `K=20`, hard negative top1。
4. `K=2` pairwise 仅作为 diagnostic ablation。
5. 若纯 pairwise 不够，测试 hybrid：

```text
L = L_pairwise + lambda * L_BCE
```

其中 `lambda` 先测 `0.1` 和 `0.2`。

## 当前实现

当前分支已实现第一组纯 pairwise 实验：

```text
model.rerank_loss_type=pairwise
model.pairwise_num_negatives=1
model.use_beam_score_in_rerank_loss=false
```

对应配置：

```text
configs/experiment/tiger_train_pairwise.yaml
```

启动脚本：

```bash
sbatch run_click_pairwise.sh
```

训练时仍生成 `top_k_for_generation=20` 个候选。若 beam 内有 GT，则选 GT 作为 positive，并按 `beam_log_prob` 从非 GT 候选里选择 hard negative top1：

```text
L_pair = softplus(-(click_pos - click_neg))
```

no-hit 行继续复用当前 auxiliary 逻辑：teacher-force GT，插入候选列表，再用相同 pairwise loss 训练。

当前分支也已实现 hybrid 实验：

```text
model.rerank_loss_type=pairwise
model.pairwise_num_negatives=3
model.use_beam_score_in_rerank_loss=false
model.no_hit_aux_loss_weight=0.1
model.bce_aux_loss_weight=0.1
```

对应配置：

```text
configs/experiment/tiger_train_pairwise_hybrid.yaml
```

启动脚本：

```bash
sbatch run_click_pairwise_hybrid.sh
```

Hybrid loss：

```text
L = L_pairwise(click_logits) + no_hit_weight * L_no_hit_aux + 0.1 * L_BCE(click_logits)
```

这个版本用于验证当前判断：纯 pairwise/listwise 的 ranking 信号太稀疏，需要 lightweight BCE calibration 保持 click logit 的全局方向。

## 需要记录的指标

除了现有 recall / ndcg / click AUC，建议新增或离线统计：

- `AUC(sigmoid(click_logits))`
- `AUC(final_score)`
- positive rank in beam
- pairwise accuracy over click logits
- pairwise accuracy over final score
- hit-in-beam rate
- no-hit auxiliary ratio

当前实现已在线记录：

- `train/rerank_pairwise_acc`
- `train/rerank_positive_rank`
- `train/rerank_final_pairwise_acc`
- `train/rerank_final_positive_rank`
- `train/rerank_bce_aux_loss`

这些指标可以区分：

- click logit 不校准；
- final score 排序不够；
- beam 本身没有正样本；
- auxiliary 分布影响过强。

## 预期解释路径

如果 pairwise 优于 listwise 但仍低于 pointwise：

> Pairwise ranking provides a more direct relative preference signal than listwise softmax under sparse single-positive beam supervision, but pure ranking losses lack the dense calibration signal of pointwise BCE.

如果 hybrid 最终最好：

> We use pairwise/listwise ranking as the principal beam-level objective and retain a lightweight BCE calibration term to stabilize residual score estimation.

如果 pairwise 仍不如 listwise/pointwise：

> The current click head and candidate-aware prompt are more suitable for calibrated residual scoring than pure pairwise preference learning; pairwise/listwise results are retained as ranking-objective ablations.

## 实验结果记录

更新时间：2026-07-22。

### 参考实验

| Run | 训练目标 | Test Recall@5 | Test NDCG@5 | Recall@5 Gain | NDCG@5 Gain | Click AUC | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `2026-06-26/17-00-10` | Pointwise BCE | 0.242472 | 0.236194 | 0.007272 | 0.005582 | 0.688450 | 当前最强经验参考。 |
| `2026-07-11/15-47-00` | Listwise over click logits | 0.241169 | 0.234219 | 0.003284 | 0.002162 | 0.599137 | 理论形式更像 beam-level ranking，但略低于 pointwise。 |
| `2026-07-20/03-51-34` | Pure pairwise, top1 hard negative | 0.235947 | 0.231266 | 0.001225 | 0.000844 | 0.361195 | 明显失败，click logit 排序/校准均不稳定。 |

注：`2026-07-11/15-47-00` 在部分讨论中被口头作为 pointwise 对照提及，但从配置看它是当前 listwise 路径下的稳定参考；严格的 pointwise BCE 参考仍使用 `2026-06-26/17-00-10`。

### Pairwise Hybrid

Run：`2026-07-21/05-58-48`

配置：

```text
model.rerank_loss_type=pairwise
model.pairwise_num_negatives=3
model.use_beam_score_in_rerank_loss=false
model.no_hit_aux_loss_weight=0.1
model.bce_aux_loss_weight=0.1
model.rerank_loss_weight=0.1
```

截至本次检查，该任务 `123164` 仍处于 `RUNNING` 状态，因此还没有最终 test 行。CSV 中已写出的 validation 结果如下：

| Step | Val Recall@5 | Val NDCG@5 | Recall@5 Gain | NDCG@5 Gain | Val Click AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 29,999 | 0.233562 | 0.230645 | 0.000569 | 0.001170 | 0.679480 |
| 54,999 | 0.236144 | 0.232116 | 0.001981 | 0.001473 | 0.699662 |
| 59,999 | 0.238569 | 0.234721 | 0.001113 | 0.001040 | 0.706263 |
| 69,999 | 0.238694 | 0.233925 | 0.001075 | 0.000935 | 0.702958 |
| 74,999 | 0.234987 | 0.232047 | 0.001400 | 0.000987 | 0.699016 |

观察：

- 加入 `0.1 * BCE(click_logits)` 后，click AUC 从 pure pairwise 的明显异常值恢复到约 `0.70`，说明 BCE calibration 对 click logit 的全局方向是必要的。
- 但 rerank gain 仍偏小，当前可见 validation `Recall@5 Gain` 约 `0.001-0.002`，低于稳定 listwise 和 pointwise BCE。
- Pure pairwise 的最终 test AUC 只有 `0.361195`，说明单纯优化局部 pair preference 会让 click head 失去可解释的全局概率/残差分数语义。
- Pairwise hybrid 虽然修复了 AUC，但没有证明 pairwise 相对比较能更好转化为 full-beam reranking；局部 hard negative 信号和最终 top-k list 排序之间仍存在错位。

### 当前判断

Pairwise 分支目前更像一个诊断性负结果，而不是主线候选：

```text
pure pairwise: click signal collapse
pairwise + BCE calibration: AUC recovered, but top-k gain remains weak
pointwise BCE: empirical strongest
listwise: ranking formulation cleaner, performance close to pointwise
```

因此下一步不建议继续在 pairwise 上扩大搜索空间。更值得回到 listwise 路径，测试：

```text
L = L_listwise(click_logits) + lambda * L_BCE(click_logits)
```

建议保持：

```text
use_beam_score_in_rerank_loss=false
```

也就是 ranking loss 仍只约束 click logits，让 click head 学 residual ranking signal；beam score 继续只在推理阶段作为 generator prior 融合。之前 `joint final score` 和当前 pairwise 失败都提示：把 `beam_log_score + click_logit` 直接放进 ranking loss，容易让 click head 被 generator prior 吸收或压制，导致 AUC 和 gain 同时走低。
