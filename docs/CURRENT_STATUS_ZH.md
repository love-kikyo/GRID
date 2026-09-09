# 当前状态

核验时间：2026-09-09 22:25，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

seed 42 正式训练及 checkpoint 小规模导出验收已收尾：三个新模型均能导出，并与 teacher 的 64 条 validation 样本严格配对。Separate 辅助日志缺陷已修复，47 项回归通过；保存的冻结 teacher 已逐值核对一致。下一步固定全量 test 去重复掩码，再进入全量评估；不重训现有模型。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| 过滤后跨 split 用户 / 样本重复审计 | 全量完成，有重复发现 | 132048；[精确计数与评估要求](DATA_PROTOCOL_ZH.md) |
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

1. 优先生成完整 test 去重复名单 / 掩码及哈希；现有边界报告只有计数和少量示例，不能代替完整名单。具体规则只在数据协议页维护。
2. 按作者最新要求，仅报告四个模型在共同去重复 test 上的全量结果；原始数据保留作追溯。数据处理链稳定后整理可复现命令与计数核验。
3. 配对统计、必要分析及论文改写均未完成。论文正文继续暂停。

补充：全量 validation 不是重测 test 的强制前置条件；仅在明确开展融合权重搜索时，需要三个 reranker 的逐样本 validation 导出。详见实验计划。当前未启动该搜索。

已修复辅助日志、补回归与精确源码兼容记录，完成小规模导出和 checkpoint 检查；未重训、重建数据或修改论文正文。旧训练日志、preflight 和 checkpoint 保留不变。
