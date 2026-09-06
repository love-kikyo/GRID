# 当前状态

核验时间：2026-09-06 14:20，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

新 teacher 的逐样本导出 smoke、52 项 CPU 回归、listwise 两步 DDP smoke 已通过；正在顺序验收 pointwise / separate。SID teacher 已完成正式训练；三个 reranker 仍只是小规模验收，实验没有整体收尾。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| 新协议 listwise / pointwise / separate | 待正式训练 | 配置解析通过不等于训练通过 |
| 新 checkpoint 全量导出、配对统计、效率复测 | 待完成 | 训练自带 test 有 batch 上限 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 131908：正式训练 SID 生成器，不是完整 RecoChain reranker。
- 132022 / 132023：从新 teacher 的 validation / test 各导出 64 条；不做全量评估，不训练模型。
- 132024：综合回归测试 52 passed；不训练模型。
- 132025–132027：各下游方法的 2 optimizer steps / 3 GPU 验收，不是正式重训。

详细作业记录、teacher 指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步只推进一个小阶段

先完成“新 teacher 的下游初始化与真实导出 smoke 验收”，见 [计划第 1 阶段](EXPERIMENT_PLAN_ZH.md)。目前不能直接把测试通过理解为可以批量提交所有正式任务：

1. 精确哈希兼容记录与 user ID 导出修复已通过真实 teacher smoke；原始 preflight 未改动。
2. 下游各方法的真实初始化、冻结规则、梯度和 DDP smoke 验收进行中。
3. 全量 validation / test 必须核对样本数、唯一用户数和数据版本，不能用训练时的截断指标代替。

当前只推进小规模验收；没有启动正式重训，也没有修改论文正文。
