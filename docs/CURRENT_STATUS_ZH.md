# 当前状态

核验时间：2026-09-08 10:00，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

seed 42 的 listwise 132049 已完成并通过训练产物验收：跑满 120000 步，best step 100000 已重载完成测试；尚未完成论文全量评估。Pointwise 132050 已接续运行，separate 132051 等待其成功。三者共用 131908 的 best teacher，每次 3 GPU；不重复已通过的 smoke。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| 过滤后跨 split 用户 / 样本重复审计 | 全量完成，有重复发现 | 132048；[精确计数与评估要求](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| Teacher 逐样本导出 / user ID | Smoke 已通过 | 132022 / 132023；原 preflight 保留，显式兼容记录随导出存档 |
| 新协议 listwise / pointwise / separate | 1 项验收通过，1 项运行，1 项等待 | 132049–132051；[验收及指标](RUN_LEDGER_ZH.md#132049listwise-正式训练产物验收通过) |
| 新 checkpoint 全量导出、配对统计、效率复测 | 待完成 | 训练自带 test 有 batch 上限 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132048：CPU 数据边界审计，已完成，不训练模型。
- 132049：RecoChain-listwise（SID + listwise 联合目标）已完成；训练日志融合重排指标优于同模型 beam-only，但不是全量统计结论。
- 132050：正在正式训练 RecoChain-pointwise；核验时 step 2039，loss 有限、无重启，尚未首次验证。
- 132051：正式训练 separate-small，冻结同 teacher，等待 132050 成功。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

当前进入 [计划第 2 阶段](EXPERIMENT_PLAN_ZH.md)，不再回到已经完成的准备步骤：

1. 完成剩余 pointwise / separate 的训练与产物验收；listwise 已验收，不重训。某任务失败则处理原因，不把依赖等待当作正在训练。
2. 全量评估前固定 test 去重复掩码，同时报告原始全量与去重复对照；审计发现不能被忽略，具体规则只在数据协议页维护。
3. 全量导出、融合选权、配对统计、必要分析及论文改写均未完成。论文正文继续暂停。

正在运行或等待的训练任务依赖当前源码，不改动这些任务依赖的代码 / 配置；文档和独立分析可继续维护。没有重建数据或修改论文正文。
