> 历史快照（归档于 2026-09-06）：以下原文仅供追溯，可能含过期状态、旧数据结果或已纠正表述。当前入口见 [文档导航](../../README.md)。原文中的相对路径按归档前目录理解。

# RecoChain 当前实验结果与后续实验规划

更新时间：2026-07-15

## 一句话结论

当前结果支持把 RecoChain 包装为一个 **cache-aware unified generate-and-rank framework**，而不是单纯包装成 listwise loss 的性能提升。Listwise 的绝对指标略低于 pointwise，但它稳定带来正的 beam-level rerank gain，并且回应了 reviewer 对 pointwise BCE “不是排序学习”的质疑。KV cache、candidate list length、candidate-aware historical context 三组实验已经形成闭环；后续重点应转向 **score fusion 消融、OneRec 可行性评估、最终表格/图整理**，不建议继续盲目调 listwise 或单独补 seed44。

## 当前证据强度

可以作为主论文证据：

- TIGER / SID-only baseline 已有完整日志和测试结果。
- RecoChain-Pointwise 已有完整日志和测试结果，可作为旧目标/消融。
- RecoChain-Listwise seed42 已有完整日志和测试结果，适合作为主方法结果。
- RecoChain-Listwise seed43 已完成，可用于说明 rerank gain 稳定，但不建议单独作为主表均值依据。
- Separate small reranker 已重跑，有完整日志、checkpoint 和 profiling，可作为可信 two-stage baseline。
- KV cache profiling 已完成，能够直接回应 reviewer 的效率质疑。
- Candidate list length / beam size 曲线已完成，可用于说明 rerank 收益与候选池大小、推理成本之间的 trade-off。
- 历史相似上下文检索长度的固定槽位 mask 实验已完成，可用于说明 candidate-aware historical context 的必要性和边际收益衰减。

需要谨慎使用：

- `GRID-RerankSmall-121275` 的旧 din-styled 结果较好，但丰富日志已误删，建议只作为历史参考，不作为主表唯一 baseline。
- `2026-07-09/22-20-18` 和 `2026-07-10/10-04-45` 两个 listwise 中间尝试的 train run 目录当前不存在，不能作为可复核主证据。
- OneRec 尚未部署，仍是 reviewer 对比缺口。

## 当前实验状态总览

| 实验模块 | 状态 | 论文用途 | 当前判断 |
| --- | --- | --- | --- |
| 主结果表 | 已基本完成 | 对比 TIGER / separate reranker / RecoChain | RecoChain-Listwise 优于 TIGER 和可复现 separate small；pointwise 作为旧目标/消融 |
| Listwise 稳定性 | 已有 seed42/seed43 | 说明 rerank gain 稳定 | 绝对值受 beam/generator seed 波动影响，gain 稳定 |
| KV cache efficiency | 已完成 | 回应 reviewer 效率质疑 | cache reuse 使 ranking 子步骤约 `2.30x` 加速 |
| Candidate list length | 已完成 | accuracy-latency trade-off 图 | 候选池越大，rerank gain 越明显，latency/memory 也增加 |
| Historical context length | 已完成，采用固定槽位 + mask | 证明 candidate-aware historical context 有效 | `active=0 -> 1` 是主要跃升，`active>=2` 基本饱和 |
| Separate reranker scaling | 已完成 small/medium | two-stage baseline 可信度 | medium 未优于 small，不继续盲目加容量 |
| OneRec | 未完成 | reviewer 指名 baseline | 下一步先做 feasibility check |
| Score fusion 消融 | 未完成 | 解释 beam score 与 click score 数学意义 | 下一步优先做 evaluation-only ablation |

## 主实验结果

以下表格优先使用有完整 `metrics.csv` 的 run。数值均为 test split。

| 方法 | run / 来源 | 目标/说明 | R@5 | N@5 | R@10 | N@10 | R@5 gain | click AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TIGER / SID-only | `2026-06-12/16-29-32` | 仅生成，无有效 rerank | 0.23378 | 0.22981 | 0.23751 | 0.23102 | +0.00031 | 0.35721 |
| TIGER + separate small reranker | `2026-07-12/23-28-36` | 可复现 two-stage baseline | 0.23696 | 0.23187 | 0.24018 | 0.23291 | +0.00349 | 0.70072 |
| TIGER + separate medium reranker | `2026-07-13/09-57-41` | separate scaling | 0.23673 | 0.23177 | 0.24000 | 0.23283 | +0.00327 | 0.70094 |
| RecoChain-Pointwise | `2026-06-26/17-00-10` | 旧 BCE 目标，建议放消融 | 0.24247 | 0.23619 | 0.24337 | 0.23649 | +0.00727 | 0.68845 |
| RecoChain-Listwise seed42 | `2026-07-11/15-47-00` | 当前主方法候选 | 0.24117 | 0.23422 | 0.24496 | 0.23547 | +0.00328 | 0.59914 |
| RecoChain-Listwise seed43 | `2026-07-13/14-28-44` | 稳定性检查 | 0.23507 | 0.23084 | 0.23854 | 0.23198 | +0.00313 | 0.57510 |

补充历史结果：

| 方法 | 来源 | R@5 | N@5 | R@10 | N@10 | R@5 gain | click AUC | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| din-styled / old small reranker | `GRID-RerankSmall-121275.out` | 0.23977 | 0.23371 | 0.24119 | 0.23418 | +0.00630 | 0.69607 | 旧 run，完整 train 日志不可复核 |

## Best validation checkpoint

| 方法 | best step | val R@5 | val N@5 | val R@10 | val N@10 | val R@5 gain | val click AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TIGER / SID-only | 84999 | 0.24863 | 0.24419 | 0.25227 | 0.24537 | -0.00011 | 0.36358 |
| RecoChain-Pointwise | 84999 | 0.25795 | 0.25078 | 0.25943 | 0.25126 | +0.00773 | 0.69636 |
| RecoChain-Listwise seed42 | 84999 | 0.25601 | 0.24857 | 0.26218 | 0.25057 | +0.00402 | 0.61034 |
| RecoChain-Listwise seed43 | 84999 | 0.24839 | 0.24454 | 0.25321 | 0.24613 | +0.00333 | 0.58479 |
| Separate small | 19999 | 0.24733 | 0.24143 | 0.25230 | 0.24306 | +0.00374 | 0.72338 |
| Separate medium | 19999 | 0.24663 | 0.24116 | 0.25222 | 0.24301 | +0.00304 | 0.72316 |

## Listwise seed42/seed43 稳定性

seed43 的绝对 R@5 明显低于 seed42，但 beam 本身也低。因此不能简单解释成 listwise 失效。更准确的判断是：**generator / beam 质量存在 seed 波动，而 listwise rerank gain 稳定**。

| 指标 | seed42 | seed43 | 均值 | sample std |
| --- | ---: | ---: | ---: | ---: |
| test beam R@5 | 0.23788 | 0.23193 | 0.23491 | 0.00421 |
| test rerank R@5 | 0.24117 | 0.23507 | 0.23812 | 0.00432 |
| test R@5 gain | +0.00328 | +0.00313 | +0.00321 | 0.00011 |
| test beam N@5 | 0.23206 | 0.22879 | 0.23042 | 0.00231 |
| test rerank N@5 | 0.23422 | 0.23084 | 0.23253 | 0.00239 |
| test N@5 gain | +0.00216 | +0.00205 | +0.00211 | 0.00008 |
| test click AUC | 0.59914 | 0.57510 | 0.58712 | 0.01699 |

结论：

- 暂时不建议补 seed44。
- 如果论文最终决定报告 `mean ± std`，必须让 TIGER、separate reranker、RecoChain 都补齐多 seed，否则只给 listwise 补 seed44 不公平。
- 当前 seed43 的最大作用是支持一句话：listwise 对 beam list 的相对排序增益稳定，但绝对指标受生成底座质量影响。

## KV cache efficiency profiling

profiling 设置：

- test split
- batch size: 32
- beam / `top_k_for_generation`: 20
- bf16 autocast
- warmup batches: 5
- measured batches: 30
- profiling 脚本：`scripts/profile_rerank_efficiency.py`
- launcher：`run_profile_kv_cache.sh`

| 方法 | variant | total ms/batch | gen ms/batch | rerank ms/batch | users/s | active params | added rerank params | peak GPU GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SID / TIGER-style | beam only | 22.81 | 22.81 | 0.00 | 1403.01 | 144.18M | none | 1.49 |
| RecoChain-Listwise | beam only | 22.59 | 22.59 | 0.00 | 1416.77 | 144.18M | 1.02K click head | 1.49 |
| RecoChain-Listwise | beam + cache rerank | 56.21 | 23.20 | 33.01 | 569.25 | 144.18M | 1.02K click head | 4.56 |
| RecoChain-Listwise | beam + no-cache rerank | 99.57 | 23.51 | 76.06 | 321.38 | 144.18M | 1.02K click head | 4.66 |
| RecoChain-Pointwise | beam + cache rerank | 55.99 | 23.01 | 32.97 | 571.55 | 144.18M | 1.02K click head | 4.56 |
| RecoChain-Pointwise | beam + no-cache rerank | 99.51 | 23.48 | 76.04 | 321.56 | 144.18M | 1.02K click head | 4.66 |
| TIGER + separate small reranker | beam + small rerank | 24.94 | 23.13 | 1.81 | 1282.95 | 223.41M | 79.23M reranker | 1.89 |
| TIGER + separate medium reranker | beam + medium rerank | 24.87 | 23.05 | 1.82 | 1286.51 | 225.41M | 81.23M reranker | 1.90 |

关键解释：

- RecoChain 的 cache reuse 将 ranking-stage latency 从 `76.06 ms/batch` 降到 `33.01 ms/batch`，ranking 子步骤约 `2.30x` 加速。
- end-to-end latency 从 `99.57 ms/batch` 降到 `56.21 ms/batch`，整体约 `1.77x` 加速。
- separate reranker 的 marginal scoring 更快，但需要额外 `79.23M+` reranker 参数；RecoChain 只增加约 `1.02K` click-head 参数。
- 因此论文里不能说 RecoChain 比 separate reranker 更快；更稳妥的说法是：RecoChain 在共享 autoregressive decoder 的统一框架内，通过 KV cache reuse 显著降低生成式 reranking 的额外代价，并以极小新增参数完成排序。

## RecoChain-Listwise 候选列表长度 / 指标 / 时间效率曲线

本组实验使用同一个 RecoChain-Listwise seed42 checkpoint：

- run: `logs/train/runs/2026-07-11/15-47-00`
- checkpoint: `checkpoint_000_085000.ckpt`
- evaluation launcher: `run_eval_retrieval_length.sh`
- profiling launcher: `run_profile_retrieval_length.sh`
- 注意：这里的 length 是 `top_k_for_generation`，即 beam/candidate list length，不是历史相似上下文检索长度。

### 完整 test 指标

| candidate length / beam size | beam R@5 | rerank R@5 | R@5 gain | beam N@5 | rerank N@5 | N@5 gain | beam R@10 | rerank R@10 | R@10 gain | beam N@10 | rerank N@10 | N@10 gain | click AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.23685 | 0.23685 | +0.00000 | 0.23140 | 0.23206 | +0.00066 | - | - | - | - | - | - | 0.56942 |
| 10 | 0.23785 | 0.24003 | +0.00218 | 0.23197 | 0.23364 | +0.00167 | 0.24195 | 0.24195 | +0.00000 | 0.23332 | 0.23427 | +0.00095 | 0.58953 |
| 20 | 0.23788 | 0.24117 | +0.00328 | 0.23206 | 0.23422 | +0.00216 | 0.24293 | 0.24496 | +0.00203 | 0.23372 | 0.23547 | +0.00175 | 0.59914 |

当 candidate length 等于评价 cutoff 时，Recall gain 理论上不会明显增加，因为 rerank 只是重排同一个候选集合。例如：

- length=5 时，@5 recall 的候选集合不变，所以 R@5 gain 为 `0`，但 N@5 gain 为正，说明排序位置变好。
- length=10 时，R@10 gain 为 `0`，但 R@5 gain 为 `+0.00218`，说明 rerank 能把命中的候选提升到更靠前位置。
- length=20 时，R@5/R@10 都有正 gain，说明更长 beam 给 rerank 提供了更多可提升空间。

### 模型 forward 时间效率

| candidate length / beam size | variant | total ms/batch | gen ms/batch | rerank ms/batch | users/s | peak GPU GB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | beam only | 12.35 | 12.35 | 0.00 | 2591.62 | 0.86 |
| 5 | cache rerank | 22.76 | 12.52 | 10.24 | 1406.04 | 1.56 |
| 5 | no-cache rerank | 33.45 | 12.87 | 20.57 | 956.75 | 1.58 |
| 10 | beam only | 15.55 | 15.55 | 0.00 | 2058.20 | 1.07 |
| 10 | cache rerank | 33.51 | 15.78 | 17.73 | 954.94 | 2.56 |
| 10 | no-cache rerank | 54.94 | 16.32 | 38.62 | 582.41 | 2.61 |
| 20 | beam only | 22.99 | 22.99 | 0.00 | 1391.81 | 1.49 |
| 20 | cache rerank | 56.96 | 23.42 | 33.54 | 561.84 | 4.56 |
| 20 | no-cache rerank | 99.82 | 23.83 | 75.99 | 320.57 | 4.66 |

关键结论：

- candidate length / beam size 从 `5 -> 10 -> 20` 增加时，R@5 gain 从 `0.00000 -> 0.00218 -> 0.00328` 增加，说明较长 beam 给 rerank 更多提升空间。
- N@5 gain 在 `5/10/20` 下均为正，分别为 `+0.00066/+0.00167/+0.00216`，说明即使 Recall 集合不变，rerank 仍改善了命中位置。
- cache rerank 的 ranking latency 从 `10.24 -> 17.73 -> 33.54 ms/batch` 增长，基本随 candidate length 增加。
- no-cache rerank 的 ranking latency 从 `20.57 -> 38.62 -> 75.99 ms/batch` 增长更快。
- cache reuse 在不同检索长度上都有效：ranking 子步骤约 `2.0x-2.3x` 加速。

## RecoChain-Listwise 历史相似上下文检索长度 / 指标 / 时间效率曲线

本组实验固定 `top_k_for_generation=20`，改变 `top_k_for_score`。这里的 `top_k_for_score` 才是 ranking 阶段从用户历史交互序列中按相似度检索出的上下文长度。

使用 checkpoint：

- run: `logs/train/runs/2026-07-11/15-47-00`
- checkpoint: `checkpoint_000_085000.ckpt`
- training-time default: `top_k_for_score=10`
- evaluation launcher: `run_eval_context_length.sh`
- profiling launcher: `run_profile_context_length.sh`

注意：该实验只是 inference-time context length sensitivity。由于模型训练时使用的是 `top_k_for_score=10`，`0/5/20` 都存在 train-test context mismatch；同时 decoder 的 click logit 来自最后一个 token 的 hidden state，改变 context length 会改变最后 candidate token 的位置分布。因此这组结果不能说明 “20 个相似历史 item 真的比 10 个差”，只能说明当前固定长度训练的 checkpoint 不能直接泛化到不同检索长度。

### 完整 test 指标

| top_k_for_score | rerank context | R@5 gain | N@5 gain | R@10 gain | N@10 gain | R@20 gain | N@20 gain | click AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | candidate only | +0.00031 | +0.00012 | -0.00014 | -0.00004 | +0.00000 | -0.00001 | 0.35221 |
| 5 | 5 history + candidate | -0.00002 | -0.00004 | +0.00021 | +0.00003 | +0.00000 | -0.00003 | 0.35168 |
| 10 | 10 history + candidate | +0.00328 | +0.00216 | +0.00203 | +0.00175 | +0.00000 | +0.00126 | 0.59914 |
| 20 | 20 history + candidate | +0.00064 | +0.00023 | +0.00012 | +0.00002 | +0.00000 | +0.00001 | 0.39588 |

### 模型 forward 时间效率

| top_k_for_score | variant | total ms/batch | gen ms/batch | rerank ms/batch | users/s | peak GPU GB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | beam only | 22.12 | 22.12 | 0.00 | 1446.44 | 1.49 |
| 0 | cache rerank | 29.99 | 22.12 | 7.87 | 1066.87 | 1.51 |
| 0 | no-cache rerank | 73.75 | 23.51 | 50.24 | 433.90 | 2.27 |
| 5 | beam only | 22.50 | 22.50 | 0.00 | 1422.04 | 1.49 |
| 5 | cache rerank | 50.44 | 22.95 | 27.49 | 634.43 | 4.56 |
| 5 | no-cache rerank | 94.93 | 23.64 | 71.29 | 337.10 | 4.66 |
| 10 | beam only | 22.54 | 22.54 | 0.00 | 1419.90 | 1.49 |
| 10 | cache rerank | 56.53 | 23.28 | 33.26 | 566.05 | 4.56 |
| 10 | no-cache rerank | 99.54 | 23.63 | 75.90 | 321.49 | 4.66 |
| 20 | beam only | 22.78 | 22.78 | 0.00 | 1404.89 | 1.49 |
| 20 | cache rerank | 67.12 | 23.18 | 43.94 | 476.75 | 4.56 |
| 20 | no-cache rerank | 110.00 | 23.76 | 86.24 | 290.92 | 4.66 |

谨慎结论：

- 当前固定 `top_k_for_score=10` 训练的模型，在同样 `scorek=10` 的输入格式下表现最好。
- `scorek=0/5/20` 的退化不能直接解释为这些检索长度本身更差，因为它们改变了 decoder 看到的 prompt 长度、candidate token 位置和 attention 分布。
- `top_k_for_score=20` 虽然包含 top-10 相似历史 item，但 transformer reranker 不是集合单调函数；额外 token 会改变位置编码、attention softmax 和最后 token 表示，因此不具备“超集一定不差”的性质。
- 效率上，cache rerank 的 ranking latency 随上下文长度增加：`7.87 -> 27.49 -> 33.26 -> 43.94 ms/batch`。
- 因此当前实验只支持：在固定 `scorek=10` 训练设置下，默认 `scorek=10` 是安全的评测设置；如果要比较不同历史检索长度，需要重新设计训练或评测协议。

论文里建议这样表述：

- RecoChain 的 reranking stage 不是简单重复生成，而是为每个候选构建 candidate-aware historical context。
- 当前主模型使用固定长度的相似历史上下文；该上下文是 ranking stage 的关键输入。
- 不把 `scorek=0/5/20` 的 inference-only 结果作为主论文结论，最多作为内部诊断或 appendix 中的 sensitivity note。
- 若要正式报告 context length ablation，应训练对应长度的模型，或采用固定最大长度加 mask/dropout 的长度鲁棒训练。

### 正式检索长度实验建议：固定最大长度 + active mask

为了正式讨论“历史相似上下文检索长度影响”，应避免改变 candidate token 位置。当前更合理的 evaluation-only 方案是：

- 固定 `top_k_for_score=10`，与 checkpoint 训练时一致。
- 使用 `active_top_k_for_score` 控制真正可见的相似历史 item 数量。
- 对超过 active length 的历史 item 置为 mask，但保留 10 个历史上下文槽位和最后 candidate token 的位置。

这样测到的是“有效历史检索上下文数量”的影响，而不是 prompt 长度、位置编码和最后 token 位置一起变化的影响。

新增脚本：

- `run_eval_context_mask_length.sh`
- `run_profile_context_mask_length.sh`

建议先测：

| active_top_k_for_score | 含义 |
| ---: | --- |
| 0 | 只保留 candidate item，历史相似上下文全 mask |
| 2 | 很短历史相似上下文 |
| 5 | 中等历史相似上下文 |
| 10 | 训练默认完整上下文 |

### 固定最大槽位 + active mask 实验结果

本组实验固定：

- `top_k_for_generation=20`
- `top_k_for_score=10`
- checkpoint: `logs/train/runs/2026-07-11/15-47-00/checkpoints/checkpoint_000_085000.ckpt`

只改变 `active_top_k_for_score`，即在固定 10 个历史相似上下文槽位中，允许前多少个相似历史 item 对 rerank 可见。超过 active length 的历史 item 被 mask，candidate token 位置保持不变。

#### 完整 test 指标

| active_top_k_for_score | R@5 gain | N@5 gain | R@10 gain | N@10 gain | R@20 gain | N@20 gain | click AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.00037 | +0.00025 | +0.00036 | +0.00024 | +0.00000 | +0.00016 | 0.31761 |
| 1 | +0.00305 | +0.00199 | +0.00176 | +0.00157 | +0.00000 | +0.00115 | 0.57593 |
| 2 | +0.00335 | +0.00218 | +0.00185 | +0.00169 | +0.00000 | +0.00125 | 0.58678 |
| 3 | +0.00335 | +0.00218 | +0.00188 | +0.00169 | +0.00000 | +0.00125 | 0.58857 |
| 4 | +0.00336 | +0.00219 | +0.00196 | +0.00173 | +0.00000 | +0.00126 | 0.58974 |
| 5 | +0.00336 | +0.00218 | +0.00196 | +0.00172 | +0.00000 | +0.00125 | 0.59098 |
| 10 | +0.00328 | +0.00216 | +0.00203 | +0.00175 | +0.00000 | +0.00126 | 0.59914 |

边际变化：

| active step | delta R@5 gain | delta N@5 gain | delta click AUC |
| --- | ---: | ---: | ---: |
| 0 -> 1 | +0.00268 | +0.00174 | +0.25833 |
| 1 -> 2 | +0.00030 | +0.00019 | +0.01084 |
| 2 -> 3 | +0.00000 | -0.00000 | +0.00179 |
| 3 -> 4 | +0.00001 | +0.00001 | +0.00117 |
| 4 -> 5 | +0.00000 | -0.00001 | +0.00124 |
| 5 -> 10 | -0.00008 | -0.00002 | +0.00815 |

#### 模型 forward 时间效率

| active_top_k_for_score | variant | total ms/batch | gen ms/batch | rerank ms/batch | users/s | peak GPU GB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | beam only | 22.35 | 22.35 | 0.00 | 1431.60 | 1.49 |
| 0 | cache rerank | 56.19 | 23.18 | 33.02 | 569.49 | 4.56 |
| 0 | no-cache rerank | 100.91 | 23.50 | 77.41 | 317.12 | 4.66 |
| 1 | beam only | 22.57 | 22.57 | 0.00 | 1417.57 | 1.49 |
| 1 | cache rerank | 56.36 | 23.21 | 33.15 | 567.81 | 4.56 |
| 1 | no-cache rerank | 100.97 | 23.49 | 77.48 | 316.94 | 4.66 |
| 2 | beam only | 22.47 | 22.47 | 0.00 | 1424.15 | 1.49 |
| 2 | cache rerank | 56.03 | 23.05 | 32.98 | 571.10 | 4.56 |
| 2 | no-cache rerank | 101.82 | 24.01 | 77.82 | 314.27 | 4.66 |
| 3 | beam only | 22.41 | 22.41 | 0.00 | 1427.02 | 1.49 |
| 3 | cache rerank | 55.74 | 22.90 | 32.84 | 574.05 | 4.56 |
| 3 | no-cache rerank | 100.60 | 23.42 | 77.18 | 318.09 | 4.66 |
| 4 | beam only | 22.53 | 22.53 | 0.00 | 1420.09 | 1.49 |
| 4 | cache rerank | 56.29 | 23.17 | 33.12 | 568.48 | 4.56 |
| 4 | no-cache rerank | 101.04 | 23.51 | 77.53 | 316.70 | 4.66 |
| 5 | beam only | 22.56 | 22.56 | 0.00 | 1418.49 | 1.49 |
| 5 | cache rerank | 55.85 | 22.99 | 32.86 | 572.97 | 4.56 |
| 5 | no-cache rerank | 102.10 | 24.14 | 77.96 | 313.43 | 4.66 |
| 10 | beam only | 22.79 | 22.79 | 0.00 | 1404.20 | 1.49 |
| 10 | cache rerank | 55.54 | 22.82 | 32.72 | 576.17 | 4.56 |
| 10 | no-cache rerank | 99.73 | 23.18 | 76.55 | 320.88 | 4.66 |

关键结论：

- `active=0 -> 1` 是主要跃升，R@5 gain 增加 `+0.00268`，N@5 gain 增加 `+0.00174`，click AUC 增加 `+0.25833`。
- `active=1 -> 2` 仍有小幅收益；`active>=2` 后 R@5/N@5 基本饱和，呈现清楚的边际收益衰减。
- `active=5` 与 `active=10` 非常接近，甚至 R@5/N@5 略高，说明主要排序收益来自 top-ranked 的少数相似历史 item，而非完整 10 个历史 item。
- `active=0` 明显退化，说明 candidate-aware historical context 是 rerank 有效性的关键来源。
- 由于方案 C 固定了 10 个上下文槽位，只改变 attention mask，因此不同 active length 的 forward latency 和显存基本相同。这组实验适合用于解释“指标随有效检索上下文数量变化”，不适合直接说明动态裁剪后的真实加速。
- 若论文要同时讨论“动态减少检索上下文带来的速度收益”，需要另做位置保持或重新训练方案；当前最严谨的速度结论仍应使用 KV cache/no-cache 和 candidate-list length profiling。

## Separate reranker scaling

| 方法 | run | best val R@5 | test R@5 | test N@5 | test R@10 | test N@10 | reranker params | scorer params | total ms/batch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| separate small | `2026-07-12/23-28-36` | 0.24733 | 0.23696 | 0.23187 | 0.24018 | 0.23291 | 79.23M | 0.30M | 24.94 |
| separate medium | `2026-07-13/09-57-41` | 0.24663 | 0.23673 | 0.23177 | 0.24000 | 0.23283 | 81.23M | 1.23M | 24.87 |

结论：

- medium 没有超过 small，说明瓶颈不只是 MLP 容量。
- 继续盲目加大 separate reranker 不值得。
- separate baseline 的目标是“可信”，不是把它发展成另一篇独立 reranker 工作。
- 主表建议使用可复现的 separate small；medium 放 appendix 或内部报告。

## 当前论文包装建议

主贡献可以这样组织：

1. RecoChain 将 TIGER-style semantic ID generation 扩展为统一的 generate-and-rank chain。
2. Ranking 阶段不是外接独立 reranker，而是在同一个 autoregressive workflow 中继续利用 decoder context。
3. Listwise beam-level objective 直接在 beam candidate list 上归一化，回应 pointwise BCE 缺乏 list-level ranking semantics 的问题。
4. KV cache reuse 让生成式 reranking 避免重复完整编码历史上下文，实验证明 ranking 子步骤约 `2.30x` 加速。
5. 方法新增参数极少，主要是 `1.02K` click head；相比 separate reranker 的 `79.23M+` 参数，更像是参数高效的统一建模。

需要避免的表述：

- 不要说 RecoChain 是 through-beam 的 fully differentiable end-to-end loss。
- 不要说 listwise 一定比 pointwise 指标更高。
- 不要说 RecoChain 比所有外部 reranker 都更快。
- 不要把 old din-styled run 当成唯一 two-stage baseline。

建议主表放：

- TIGER / SID-only
- TIGER + separate small reranker
- RecoChain-Listwise

建议消融表放：

- RecoChain-Pointwise
- RecoChain-Listwise seed42/seed43 stability
- cache vs no-cache
- beam only / click score only / beam+click score fusion

## 后续实验规划

### P0：整理论文主表和核心图

当前最应该做的是把现有实验转成论文可用表格和图，而不是继续大规模训练。

建议主表包含：

- TIGER / SID-only
- TIGER + separate small reranker
- RecoChain-Listwise
- RecoChain-Pointwise 放 ablation，不放主方法对比

建议核心图：

- candidate list length vs. R@5/N@5 gain and latency
- active historical context length vs. R@5/N@5 gain and click AUC
- cache vs. no-cache latency bar chart

### P0：暂不跑 seed44

当前不建议马上跑 seed44。

原因：

- seed43 已经证明 listwise gain 稳定。
- 绝对 R@5 波动主要来自 beam/generator。
- 如果只给 listwise 补 seed44，而不给 TIGER 和 separate reranker 补 seed，不利于公平表述。

只有在最终论文决定所有主方法都报告 `mean ± std` 时，才需要统一补多 seed。

### P1：beam/candidate length 曲线已完成，整理为图

这组实验已经完成，实际含义是 `top_k_for_generation` 曲线，不是历史相似上下文检索长度。

建议设置：

| beam size | 目的 |
| ---: | --- |
| 5 | 低延迟设置 |
| 10 | 中间 trade-off |
| 20 | 当前主设置 |

这个实验把 **candidate list length / beam size** 作为横轴，同时报告效果和效率：

- R@5 / N@5 / R@10 / N@10
- total ms/batch
- rerank ms/batch
- peak GPU memory
- users/s

作用：

- 回应速度问题。
- 解释 attention / candidate list 带来的计算增长。
- 给论文一个 accuracy-latency trade-off 图。
- 说明 RecoChain 的性能收益是否依赖更长的候选列表，以及更长候选列表带来的边际成本。

建议拆成两组运行：

- metric evaluation：用 Lightning test-only 跑完整 test split，得到正式 R@K/N@K。
- model-only profiling：用 `scripts/profile_rerank_efficiency.py` 固定 warmup/measured batches，得到模型 forward latency、ranking latency 和显存。

注意：

- 当 `top_k_for_generation=5` 时，只报告 @5；不能报告 @10。
- 当 `top_k_for_generation=10` 时，报告 @5/@10。
- 当 `top_k_for_generation=20` 时，报告 @5/@10，必要时可额外加 @20。
- trained separate reranker 的 rank embedding 与训练时 beam=20 绑定，暂时不建议拿它做变长检索曲线；先聚焦 TIGER/SID-only 与 RecoChain-Listwise。

### P1：历史相似上下文检索长度实验已完成，采用 mask 方案写入论文

正式论文建议采用固定最大槽位 + active mask 的结果，而不是直接改变 `model.top_k_for_score` 的结果。

含义：

- 生成候选 item 后，ranking 阶段会用 `get_similar_itemkey` 从用户历史交互序列 `hist_itemkey` 中按 candidate item 相似度检索 top-k 个历史 item。
- 这个 top-k 由 `top_k_for_score` 控制。
- 最终送入 rerank decoder 的相似上下文长度约为 `top_k_for_score + 1`，其中 `+1` 是 candidate item 自身。

论文图建议使用：

| active_top_k_for_score | 含义 |
| ---: | --- |
| 0 | 只保留 candidate item，历史相似上下文全 mask |
| 1 | 只保留最相似的 1 个历史 item |
| 2 | 保留最相似的 2 个历史 item |
| 3/4/5 | 观察边际收益衰减 |
| 10 | 训练默认完整上下文 |

需要报告：

- R@5 / N@5 / R@10 / N@10
- click AUC
- total ms/batch
- rerank ms/batch
- peak GPU memory

论文作用：

- 证明 ranking 阶段不是简单“多 prompt 再生成”，而是显式利用 candidate-aware historical context。
- 展示有效历史上下文数量带来的准确率收益和边际收益衰减。
- 说明最相似的少数历史 item 提供主要排序信号。
- 注意：该 mask 方案固定序列长度，不能用来证明动态缩短上下文会加速。

### P1：整理最终主表和 appendix 表

不需要新训练，只需要从现有 CSV/JSON 生成最终表。

主表：

- TIGER / SID-only
- TIGER + separate small
- RecoChain-Listwise

appendix / ablation：

- RecoChain-Pointwise
- RecoChain-Listwise seed43
- separate medium
- old din-styled result，标注日志不完整
- cache vs no-cache profiling
- direct `top_k_for_score` sensitivity，标注为位置/长度分布变化诊断，不作为主结论

### P1：score fusion / ranking score ablation

这个实验与之前的数学担忧直接相关：beam score 是生成概率，click score 是 rerank 预测，二者直接 log 相加的意义需要澄清。

建议优先做 evaluation-only ablation：

- beam score only
- click score only
- beam score + click score
- different fusion weights / temperature，如果当前配置支持

需要回答的问题：

- listwise gain 是否主要来自 click score。
- beam score 是否仍然提供必要先验。
- 等权融合是否合理，是否需要在论文中表述为 calibrated fusion 或 late fusion。

建议候选设置：

| ablation | beam weight | rerank weight | 目的 |
| --- | ---: | ---: | --- |
| beam only | 1.0 | 0.0 | 生成器原始排序 |
| click only | 0.0 | 1.0 | 纯 rerank head 排序能力 |
| current | 1.0 | 0.1 | 当前默认 late fusion |
| stronger click | 1.0 | 0.2/0.3 | 检查 rerank 分数权重敏感性 |

### P2：OneRec feasibility check

先不直接全量部署，先做可行性检查：

- 代码依赖是否能装。
- 3 张 3090 是否能承载。
- 参数量是否能和 TIGER / RecoChain 合理对齐。
- 输入格式改造成本是否可控。

如果可行，再跑小规模 smoke test；如果不可行，需要在论文策略上决定是补替代 generative baseline，还是在 rebuttal 中说明资源限制。

### P3：更强 separate reranker 只作为备选

目前不建议继续 medium/large 容量扩展。

如果 reviewer 压力很大，下一步不应是单纯加大 MLP，而应改：

- listwise separate reranker
- beam rank / beam score normalization feature
- history-candidate interaction
- candidate position feature

但这条线优先级低于 KV cache、beam size 和 OneRec。

### P3：训练型 ablation

这些实验成本较高，放后面：

- listwise without no-hit auxiliary loss
- remove similar-item context
- different rerank loss weight
- joint click-head only / freeze backbone

这些主要服务 appendix，不应阻塞主线。

## 下一步执行建议

推荐顺序：

1. 从现有结果生成论文主表、appendix 表和三张图：cache、candidate length、context length。
2. 做 score fusion evaluation-only ablation，解释 beam score 与 click score 的融合意义。
3. 做 OneRec feasibility check，决定能否跑 reviewer 指名 baseline。
4. 根据 OneRec 可行性决定是否补替代 generative baseline。
5. 只有在最终决定主方法都报告 `mean ± std` 时，再统一补多 seed；不要单独补 seed44。

当前最不建议做的事情：

- 继续盲目调 listwise。
- 只给 listwise 跑 seed44。
- 继续把 separate reranker 做得越来越大。
