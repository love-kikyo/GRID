# 当前状态

核验时间：2026-09-09 23:25，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

按作者最新决定沿用现有 split，不追加跨 split 内容去重或为 beta 重训。下一步固定三个重排 checkpoint 做全量 validation 导出与推理融合选权，再对四个 checkpoint 做当前完整 test 的配对评估。已知数据重叠保留为协议限制，不作无泄漏声明。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| 过滤后跨 split 重复审计 | 保留已有结果，不追加内容去重 | 132048 仅同用户范围；其余未知，非通过；[准确范围](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| Teacher 逐样本导出 / user ID | Smoke 已通过 | 132022 / 132023；原 preflight 保留，显式兼容记录随导出存档 |
| 新协议 listwise / pointwise / separate | 3 项训练及产物验收完成 | 新 checkpoint 的 val 导出各 64 条通过；不是全量评估；[台账](RUN_LEDGER_ZH.md) |
| Separate 辅助日志 / 冻结 teacher | 修复与核对完成 | 修复 `6879764`，47 passed；132672 参数逐值核对通过；旧 NaN 日志保留 |
| 新 checkpoint 全量导出、配对统计、效率复测 | 待完成 | 训练自带 test 有 batch 上限 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132665 / 132666 / 132667：三个正式 checkpoint 的首次 validation 导出验收，均完成；不是新训练或全量评估。
- 132672：CPU checkpoint 张量检查，已完成；不使用 GPU，不训练模型。
- 132049 / 132050 / 132051：此前的正式训练均已完成。这些验收与训练任务均无待运行项。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

进入 [计划第 3 阶段](EXPERIMENT_PLAN_ZH.md)，不重复已完成的训练与 smoke：

1. 三个重排 checkpoint 各导出完整 validation 347,416 条；按计划中的统一 beta 网格和选优指标，仅做推理融合校准。
2. 固定权重后对四个 checkpoint 的共同 test 全部 2,061,101 条做配对评估。整理当前数据链的复现命令，不新增去重步骤。
3. 配对统计、必要分析及论文改写均未完成。论文正文继续暂停。

补充：不测量 beta 改变早停的影响，也不宣称该影响已证明为零；现有 checkpoint 直接固定使用。多 seed 仍计划在 seed 42 全量结果收尾后开展，不为每个 beta 重训。全量作业状态另记台账。

已修复辅助日志、补回归与精确源码兼容记录，完成小规模导出和 checkpoint 检查；未重训、重建数据或修改论文正文。旧训练日志、preflight 和 checkpoint 保留不变。
