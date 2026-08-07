# RecoChain History Occurrence Exploration

本文档只维护 `explore-history-occurrence-analysis` 分支的研究目标、统计口径与后续实现交接。

## 分支来源

- 基线分支：`recochain-final-experiments`
- 基线提交：`d02982c Add RecoChain final experiment analysis`
- 当前分支：`explore-history-occurrence-analysis`

本分支先做诊断性统计，不改变训练目标和主模型结构。目标是回答：RecoChain 的预测有多大比例是在重复用户历史，以及数据集的 ground-truth 本身有多大比例是历史重复 item。

## 核心问题

需要分别回答两个问题：

1. 数据集重复性：目标 item 是否曾在该样本的历史交互中出现。
2. 模型重复偏好：beam generation 与 rerank 后的预测 item 是否曾在该样本的历史交互中出现。

不能只比较两个总比例。还应按 `GT repeated / GT novel` 分组报告推荐指标，以判断模型是在合理利用重复消费规律，还是仅仅偏向复制历史。

## 两种历史口径

当前数据管线中存在两种不同长度的历史，必须分开统计。

### Full Retrieval History

```text
target_itemkey = hist_itemkey[:, -1]
full_history = hist_itemkey[:, :-1]
```

`hist_itemkey` 在 collate 阶段会左侧填充 `masking_token=-1`，但不会像 `sequence_data` 一样裁剪到 `sequence_length`。它是 ranking 阶段 `get_similar_itemkey` 的候选历史池。

### Autoregressive Visible History

默认配置为：

```text
sequence_length=32
num_hierarchies=4
```

因此 `sequence_data` 最多包含 8 个 item block；去掉最后一个 GT block 后，generation 阶段最多直接看到 7 个历史 item。该口径记为 `ar_history`。

实现时优先从 `sequence_data[:, :-num_hierarchies]` 每 4 个 SID 还原 itemkey；也可以用其有效 block 数截取 `full_history` 最右侧相同数量的 item，但必须做一致性断言。

论文或报告中应明确：

- `full_history` 衡量 GT/预测是否出现在 ranking 可检索的长历史中。
- `ar_history` 衡量 GT/预测是否出现在 generation 直接可见的短历史中。

## 精确定义

对样本 `u`，令：

```text
H_full(u) = 有效 full_history itemkey 集合
H_ar(u)   = 有效 ar_history itemkey 集合
y(u)      = ground-truth itemkey
p_beam(u, k)   = generation beam 第 k 个候选
p_rerank(u, k) = rerank 后第 k 个候选
```

有效历史必须排除 `masking_token=-1`，且绝不能把最后一个 GT 位置包含进历史。

### 数据集 GT 重复率

对 `H_full` 和 `H_ar` 分别计算：

```text
GT repeat rate = mean_u 1[y(u) in H(u)]
```

同时统计：

- GT 在历史中的出现次数。
- GT 最近一次出现距目标位置的时间步距离。
- 按有效历史长度分桶后的 GT repeat rate。

### 预测重复率

对 beam 和 rerank、`H_full` 和 `H_ar` 分别计算：

```text
Top1 repeat rate = mean_u 1[p(u, 1) in H(u)]

Candidate repeat rate@K
  = sum_u sum_{k<=K} 1[p(u, k) in H(u)] / (N * K)

Any-repeat@K
  = mean_u 1[exists k<=K, p(u, k) in H(u)]
```

建议至少报告 `K = 1, 5, 10, 20`。`Candidate repeat rate@K` 和 `Any-repeat@K` 含义不同，不能互相替代。

### 条件推荐质量

按 GT 是否重复拆成两个 cohort：

```text
GT repeated: y(u) in H(u)
GT novel:    y(u) not in H(u)
```

分别报告：

- beam Recall/NDCG@5、@10。
- rerank Recall/NDCG@5、@10。
- rerank gain。
- top-1 prediction repeat rate。

这组条件指标比总重复率更关键：

- 若 repeated cohort 明显受益且 novel cohort 不受损，说明历史重复是有效偏好信号。
- 若模型预测重复率远高于 GT 重复率，且 novel cohort 明显变差，说明存在 history-copy bias。
- 若 rerank 相比 beam 主要提高重复预测比例，需要检查 rerank gain 是否只来自 repeated cohort。

## Item Identity 风险

当前代码用以下方式构造 itemkey：

```text
itemkey = sum_h semantic_id[h] * powers[h]
```

因此上述统计首先是“相同 semantic itemkey”的出现率。正式解释成“相同原始 item”之前，必须审计 semantic ID 映射是否一一对应：

```text
num_raw_items
num_unique_itemkeys
collision_count
max_items_per_itemkey
```

若存在碰撞，需要同时提供 raw item ID 口径，或在论文中明确使用 semantic-ID recurrence，而不能直接称为 exact item repetition。

## 推荐实现

新增独立脚本：

```text
scripts/analyze_history_occurrence.py
```

不建议把这批统计直接塞进 `tiger_generation_model.py` 的训练/验证逻辑。独立脚本更适合复用 checkpoint、切换 split，并避免继续增加模型主文件复杂度。

建议提供两个模式：

```text
--mode dataset
```

- 不加载 checkpoint。
- 遍历 train/val/test dataloader。
- 输出 GT recurrence、历史长度、出现次数和最近距离。

```text
--mode prediction
```

- 加载指定 checkpoint。
- 默认只跑 test，必要时补 val。
- 调用现有 `_infer_for_eval` 或等价 inference 路径，取得 `result_dict["beam"]["ids"]` 和 `result_dict["rerank"]["ids"]`。
- 将 SID 转为 itemkey 后计算预测重复率和条件推荐指标。

首选 checkpoint：

```text
logs/train/runs/2026-07-11/15-47-00/checkpoints/checkpoint_000_085000.ckpt
```

这是当前主结果表使用的 RecoChain-Listwise seed42。若要判断 rerank objective 是否改变重复偏好，再补 Pointwise checkpoint 作为对照。

## 输出约定

建议写入：

```text
logs/analysis/history_occurrence/<run-name>/summary.json
logs/analysis/history_occurrence/<run-name>/by_history_length.csv
logs/analysis/history_occurrence/<run-name>/by_gt_recurrence.csv
```

`summary.json` 至少包含：

```text
split
checkpoint
git_commit
history_scope
num_samples
gt_repeat_rate
beam_top1_repeat_rate
beam_candidate_repeat_rate_at_5/10/20
beam_any_repeat_at_5/10/20
rerank_top1_repeat_rate
rerank_candidate_repeat_rate_at_5/10/20
rerank_any_repeat_at_5/10/20
metrics_for_gt_repeated
metrics_for_gt_novel
semantic_itemkey_collision_audit
```

所有分母、有效样本数和 padding 过滤数量都要落盘，避免比例无法复核。

## Sanity Checks

实现后必须通过以下检查：

1. `target_itemkey` 从 `hist_itemkey[:, -1]` 与 target SID 计算得到的 itemkey 完全一致。
2. `ar_history` 与 `full_history` 的右侧对应部分一致。
3. 所有历史集合已排除 `masking_token=-1`。
4. `Candidate repeat rate@K` 位于 `[0, 1]`，且使用的实际 K 不超过候选数。
5. rerank 只改变候选顺序，所以 beam 与 rerank 的 `Any-repeat@K_all` 在 K 等于完整 beam 长度时应一致。
6. 同一个完整候选集合下，beam 与 rerank 的整体 candidate repeat rate 应一致；不同只应出现在 top-k 截断后。
7. `GT repeated` 与 `GT novel` 两组样本数之和等于总样本数。

## 第一阶段产出

第一轮先完成最小但可回答问题的表：

| Split / Model | History scope | GT repeat | Beam top1 repeat | Rerank top1 repeat | Beam candidate repeat@20 | Rerank candidate repeat@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test / Listwise | AR-visible | TBD | TBD | TBD | TBD | TBD |
| test / Listwise | Full retrieval | TBD | TBD | TBD | TBD | TBD |

再补条件推荐质量：

| Cohort | Samples | Beam R@5 | Rerank R@5 | R@5 gain | Beam N@5 | Rerank N@5 | N@5 gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT repeated | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GT novel | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

完成这两张表后，再决定是否扩展到 pointwise、不同 seed、历史长度分桶和时间距离分析。

## 当前交接状态

- 新分支已创建。
- 统计定义与风险边界已确定。
- 尚未实现分析脚本，也尚未运行统计。
- 下一步应先完成 dataset-only 模式，核对两种历史口径和 semantic itemkey 碰撞，再接 checkpoint inference。
