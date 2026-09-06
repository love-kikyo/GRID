# 当前状态

核验时间：2026-09-07 00:36，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

seed 42 正式重训已启动：listwise 132049 运行中，已记录训练 step 214，loss 有限、3 rank、无重启；pointwise 132050 和 separate 132051 按依赖顺序等待。三者共用 131908 的 best teacher，每次 3 GPU。已通过的 smoke 不再重复执行。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| 过滤后跨 split 用户 / 样本重复审计 | 全量完成，有重复发现 | 132048；[精确计数与评估要求](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| Teacher 逐样本导出 / user ID | Smoke 已通过 | 132022 / 132023；原 preflight 保留，显式兼容记录随导出存档 |
| 新协议 listwise / pointwise / separate | 正式重训运行 / 排队中 | 132049–132051；此前三种方法的两步 DDP 均通过 |
| 新 checkpoint 全量导出、配对统计、效率复测 | 待完成 | 训练自带 test 有 batch 上限 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132048：CPU 数据边界审计，已完成，不训练模型。
- 132049：正式训练 RecoChain-listwise（SID + listwise 联合目标）。
- 132050：正式训练 RecoChain-pointwise，等待 132049 成功。
- 132051：正式训练 separate-small，冻结同 teacher，等待 132050 成功。

详细作业记录、teacher 指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

当前进入 [计划第 2 阶段](EXPERIMENT_PLAN_ZH.md)，不再回到已经完成的准备步骤：

1. 检查正式任务的训练 / 验证、early stop 与 best checkpoint；某任务失败则处理原因，不把依赖等待当作正在训练。
2. 全量评估前固定 test 去重复掩码，同时报告原始全量与去重复对照；审计发现不能被忽略，具体规则只在数据协议页维护。
3. 全量导出、融合选权、配对统计、必要分析及论文改写均未完成。论文正文继续暂停。

正在运行或等待的训练任务依赖当前源码，不改动这些任务依赖的代码 / 配置；文档和独立分析可继续维护。没有重建数据或修改论文正文。
