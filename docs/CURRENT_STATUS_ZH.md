# 当前状态

核验时间：2026-09-06 14:24，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

新 teacher 的逐样本导出 smoke、52 项 CPU 回归、三种下游方法的两步 DDP smoke 均已通过，相关 Slurm 任务全部结束。SID teacher 已完成正式训练；三个 reranker 仍只是小规模验收，实验没有整体收尾。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| Teacher 逐样本导出 / user ID | Smoke 已通过 | 132022 / 132023；原 preflight 保留，显式兼容记录随导出存档 |
| 新协议 listwise / pointwise / separate | 两步 DDP smoke 通过；待正式训练 | 132025–132027；加载、梯度、冻结规则与真实验证均通过 |
| 新 checkpoint 全量导出、配对统计、效率复测 | 待完成 | 训练自带 test 有 batch 上限 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 131908：正式训练 SID 生成器，不是完整 RecoChain reranker。
- 132022 / 132023：从新 teacher 的 validation / test 各导出 64 条；不做全量评估，不训练模型。
- 132024：综合回归测试 52 passed；不训练模型。
- 132025–132027：各下游方法的 2 optimizer steps / 3 GPU 验收，不是正式重训。

详细作业记录、teacher 指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步只推进一个小阶段

加载与导出 smoke 已完成。下一步只补“过滤后 train/test 边界审计”，见 [计划第 1 阶段](EXPERIMENT_PLAN_ZH.md)，之后再确定正式重训顺序：

1. 统计 L2 过滤后的 train/test 用户交集，核查相同历史 / 目标样本是否跨 split 重复；现有审计只保证 train/validation 互斥。
2. 全量 validation / test 及跨模型配对尚未完成；64 条导出 smoke 与两步训练均不能用作论文结果。

当前只推进小规模验收；没有启动正式重训，也没有修改论文正文。
