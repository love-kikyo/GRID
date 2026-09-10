# 当前状态

核验时间：2026-09-10 14:08，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

Seed 42 四模型全量 test 与配对统计已验收通过，listwise 的融合结果在四项主指标上最高；这不是多 seed 稳定性或实验整体收尾结论。下一优先项是匹配的多 seed 方案与执行。沿用现有 split，三个 reranker 固定 validation 选出的 beta=0.50（网格上界）；不追加去重或按 beta 重训，已知重叠作为协议限制保留。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| 过滤后跨 split 重复审计 | 保留已有结果，不追加内容去重 | 132048 仅同用户范围；其余未知，非通过；[准确范围](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| Teacher 逐样本导出 / user ID | 全量 test 已通过 | 132684；原 preflight 保留，显式兼容记录随导出存档 |
| 新协议 listwise / pointwise / separate | 训练及全量评估验收完成 | 正式 checkpoint 固定，未重选；[主表与台账](RUN_LEDGER_ZH.md) |
| Separate 辅助日志 / 冻结 teacher | 修复与核对完成 | 修复 `6879764`，47 passed；132672 参数逐值核对通过；旧 NaN 日志保留 |
| 固定 checkpoint 全量 validation / beta 校准 | 已完成并验收 | 132673–132678；三份各 347,416 唯一用户，输入 / 目标完全对齐 |
| 四模型全量 test 与配对统计 | 已完成并验收 | 132681–132685；每模型 2,061,101 唯一用户，11 项比较 / 44 个指标差值与区间复核一致 |
| 效率复测 / 多 seed | 待开展 | seed 42 全量基线已具备，本次未提交新任务 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132681–132684：四模型全量 test 导出，均已完成，不是重训。
- 132685：四模型 CPU 配对统计，09-10 01:47:28 正常结束；本次独立复算与产物验收通过。
- 132673–132678 的 validation / 选权已在上一阶段验收，本次只复核冻结文件，没有重跑或根据 test 改权重。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

按 [实验计划](EXPERIMENT_PLAN_ZH.md) 进入匹配的多 seed，不重复已完成的 seed 42 训练、smoke 或全量推理：

1. 固定新增两个训练 seed 及资源安排；每个 seed 训练自己的 SID teacher，再由同一个 teacher 初始化该 seed 的三个对照，数据划分与 SID 映射保持不变。
2. 沿用相同的 validation 选权和 test 配对协议；逐用户区间与跨训练 seed 波动分开报告。效率与必要的解释分析仍待开展。
3. [数据协议](DATA_PROTOCOL_ZH.md) 已补固定资产重建命令与逐步计数；上游下载快照和历史 SID 生成来源的完整绑定仍需补齐。论文正文继续暂停。

补充：不测量 beta 改变早停的影响，也不宣称该影响已证明为零；现有 checkpoint 直接固定使用。多 seed 仍计划在 seed 42 全量结果收尾后开展，不为每个 beta 重训。全量作业状态另记台账。

本次验收记录与解释集中在 [Run 台账](RUN_LEDGER_ZH.md)：包括主表、纯重排与融合的区别、pointwise 纯重排对融合的 Recall@5 区间跨零例外。未重训、重建数据或修改论文正文；旧日志、preflight、checkpoint 和已完成报告保留不变。
