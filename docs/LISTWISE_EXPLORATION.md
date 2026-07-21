# RecoChain Listwise Exploration

本文档只维护当前 `explore-listwise-ranking` 分支的实验路径和结论。
全局项目计划、总实验表和论文包装不在本分支维护，避免不同探索分支反复同步同一份全局 doc。

## 分支目标

本分支专门回答一个问题：

> RecoChain 的 click head 是否能用真正的 beam-level ranking objective 替代 pointwise BCE，并在保留统一生成-重排框架的同时接近或超过 pointwise 结果？

目前结论偏保守：listwise 在形式上更符合 reviewer 对“排序学习”的要求，但纯 listwise 监督仍弱于 pointwise BCE。后续需要进一步验证 pairwise / hybrid loss，而不是继续只调上下文特征。

## 关键实现路径

### Pointwise Reference

历史 pointwise 版本使用 BCE 训练 `click_logits`：

```text
L_pointwise = BCEWithLogits(click_logits, candidate_is_gt)
final_score = beam_log_prob + 0.1 * click_logits
```

这个目标不是真正的 list-level ranking objective，但 dense binary supervision 会把 click head 训练成较稳定的 residual score，因此 empirical accuracy 和 AUC 都较强。

参考 run：

```text
/home/MMReco2021/liuyu/GRID/logs/train/runs/2026-06-26/17-00-10
```

最终 test：

| Method | R@5 | N@5 | R@10 | N@10 | R@5 Gain | Click AUC |
|---|---:|---:|---:|---:|---:|---:|
| Pointwise BCE | 0.24247 | 0.23619 | 0.24337 | 0.23649 | +0.00727 | 0.68845 |

### Listwise Beam Loss

当前 listwise 主线把 beam 视为一个候选 list：

```text
L_listwise = CE(softmax(click_logits over beam), normalized_positive_labels)
final_score = beam_log_prob + 0.1 * click_logits
```

这个实现解决了“逐候选 BCE 不是排序学习”的形式问题，但训练目标和最终融合分数仍有差异。

较稳定的 listwise run：

```text
/home/MMReco2021/liuyu/GRID/logs/train/runs/2026-07-11/15-47-00
```

最终 test：

| Method | R@5 | N@5 | R@10 | N@10 | R@5 Gain | Click AUC |
|---|---:|---:|---:|---:|---:|---:|
| Listwise | 0.24117 | 0.23422 | 0.24496 | 0.23547 | +0.00328 | 0.59914 |

对比 pointwise，listwise 的 gain 明显较小，AUC 也低。这说明问题不仅是 AUC 评价不适配 listwise，最终排序收益也还不够。

### Stabilization Attempts

早期尝试中，等权 loss、较大的 rank loss 权重、以及 joint optimization 容易导致 SID beam 质量下降或训练抖动。
后续稳定性改动包括：

- 降低 `rerank_loss_weight`，让 SID generation 仍是主目标。
- 引入 no-hit auxiliary，缓解 beam 内没有 GT 时 listwise 主 loss 无梯度的问题。
- 隔离 rank loss，不再让 rank objective 过度干扰 generator。

这些改动让 listwise 从明显负作用变成小幅正收益，但仍未超过 pointwise。

### Context Feature Attempts

已探索过的 join context feature 方向：

- 检索上下文不与 candidate 重复，并引入相似度分数。
- 检索上下文不与 candidate 重复，并引入时间距离分数。

其中时间距离的语义是：相似度负责从历史中取出相关 item，额外 scalar feature 恢复该 item 在原始序列中距离当前 generate 位置的时间步距离。

代表 run：

```text
/home/MMReco2021/liuyu/GRID/logs/train/runs/2026-07-17/00-53-13
```

最终 test：

| Method | R@5 | N@5 | R@10 | N@10 | R@5 Gain | Click AUC |
|---|---:|---:|---:|---:|---:|---:|
| Listwise + Context Feature | 0.23862 | 0.23277 | 0.24283 | 0.23414 | +0.00207 | 0.54737 |

未继续保留的方向：

- 检索上下文同时不与 candidate 和自回归 prefix 内容重复，并引入时间距离分数。

当前判断：这个方向不一定合理。因为自回归 prefix 中的历史 item 已经进入 KV cache，但 rerank 阶段显式取回 candidate-aware context 可能仍有 grounding 价值；完全排除 prefix item 可能会去掉最有用的近邻历史。

### Final-Score Listwise Loss

为对齐训练和 inference，测试让 listwise loss 直接优化最终融合分数：

```text
score = beam_log_prob.detach() + 0.1 * click_logits
L = CE(softmax(score over beam), labels)
```

脚本：

```text
run_click_finalscore_loss.sh
```

run：

```text
/home/MMReco2021/liuyu/GRID/logs/train/runs/2026-07-19/02-21-17
```

当前状态：该 job 在 2026-07-20 仍在运行，尚无最终 test 行。

中间 val 观察：

| Step | Val R@5 | Val N@5 | Val R@5 Gain | Val Click AUC |
|---:|---:|---:|---:|---:|
| 84999 | 0.25304 | 0.24681 | +0.00309 | 0.37974 |
| 94999 | 0.24833 | 0.24174 | +0.00294 | 0.38136 |

这个实验没有解决 click AUC 低的问题。更可能的原因是：

- `rerank_logit_weight=0.1` 让 click head 在 final-score loss 中得到的梯度被缩小。
- `beam_log_prob` 是很强的固定 prior，click head 只学到较弱 residual。
- `click_logits` 不再是可校准的 GT 概率 logit，因此 `sigmoid(click_logits)` AUC 会更差。

## 当前问题判断

listwise 不如 pointwise 的核心原因可能不是上下文特征，而是监督密度与 logit 校准：

- Pointwise BCE 对所有候选都有二分类监督，click logit 更容易学成稳定 residual。
- Listwise 只在 beam 内有正样本时有主 loss；no-hit 行依赖 auxiliary，分布更人工。
- Listwise softmax 只关心同一 beam 内相对顺序，不保证跨 user / 跨 beam 的 logit 可校准。
- 当前 inference 仍依赖 `beam_log_prob + 0.1 * click_logits`，纯 listwise 不一定能学出足够强的 residual。

因此论文中如果保留 listwise，不宜表述为纯性能 trick，而应表述为 ranking-objective ablation 或 principled reformulation。

## 下一步计划

### Pairwise Exploration

建议新分支专门测试 pairwise：

```text
generate K=20 candidates
if beam contains GT:
    positive = GT candidate
    negative = hard negative from non-GT beam candidates
    L_pair = softplus(-(click_pos - click_neg))
else:
    use small-weight no-hit auxiliary
```

优先级：

1. Pairwise over `click_logits`, K=20, hard negative top1。
2. Pairwise over `click_logits`, K=20, top3 hard negatives。
3. Pairwise over final score。
4. K=2 pairwise only as diagnostic ablation, not primary setting。

### Hybrid Loss

Pairwise 分支结果显示：pure pairwise 会让 click AUC 明显塌陷，而加入轻量 BCE calibration 后 AUC 可以恢复，但 rerank gain 仍偏弱。
因此当前 listwise 分支优先测试 listwise hybrid：

```text
L = L_listwise(click_logits) + no_hit_weight * L_no_hit_aux + lambda * L_BCE(click_logits)
```

其中 listwise 主损失仍只作用在 `click_logits` 上：

```text
use_beam_score_in_rerank_loss=false
```

也就是训练阶段不直接把 `beam_log_prob + 0.1 * click_logits` 放入 ranking loss，避免 generator prior 过强导致 click head 梯度被压制；推理阶段仍使用：

```text
final_score = beam_log_prob + 0.1 * click_logits
```

当前已实现配置：

```text
configs/experiment/tiger_train_listwise_hybrid.yaml
```

启动脚本：

```bash
sbatch run_click_listwise_hybrid.sh
```

第一组参数：

```text
model.no_hit_aux_loss_weight=0.5
model.bce_aux_loss_weight=0.1
model.rerank_loss_weight=0.1
```

如果 `lambda=0.1` 的 AUC 恢复但 gain 仍不够，下一组优先测试：

```bash
sbatch run_click_listwise_hybrid.sh model.bce_aux_loss_weight=0.2
```

预期判断：

- 若 AUC 接近 pointwise，且 gain 高于纯 listwise，说明 BCE calibration 是 listwise 缺失的关键监督。
- 若 AUC 恢复但 gain 仍低，说明问题不只是校准，而是 listwise softmax 的单正样本监督和 final rerank 融合仍存在错位。
- 若 gain 下降，说明 BCE 项压过 ranking objective，需要降低到 `0.05` 或只对 no-hit auxiliary 之外的自然 beam 使用更小权重。

### Diagnostics

后续实验建议补充：

- `AUC(sigmoid(click_logits))`
- `AUC(final_score)`
- positive rank in beam
- pairwise accuracy over final score

这能区分“click logit 不校准”和“最终排序真的差”。
