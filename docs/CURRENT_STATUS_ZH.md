# 当前状态

核验时间：2026-09-06 14:12，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

正在做新 teacher 的逐样本导出 smoke 和 CPU 回归测试（132022–132024），不是正式重训。SID teacher 已完成；joint listwise、joint pointwise、separate reranker 尚未完成新协议正式训练，实验没有整体收尾。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| 新协议 listwise / pointwise / separate | 待正式训练 | 配置解析通过不等于训练通过 |
| 新 checkpoint 全量导出、配对统计、效率复测 | 待完成 | 训练自带 test 有 batch 上限 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 131908：训练生成器，不是完整 RecoChain reranker。其 `training_task=sid`；配置中的默认 `rerank_objective=listwise` 不表示启用了 listwise 训练。
- 131910：综合回归测试，发现空 worker 问题；没有训练新模型。
- 132021：修复后的综合回归测试，38 项通过；附带数据 smoke 审计与两套 Hydra 配置解析，没有训练新模型。

详细作业记录、teacher 指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步只推进一个小阶段

先完成“新 teacher 的下游初始化与真实导出 smoke 验收”，见 [计划第 1 阶段](EXPERIMENT_PLAN_ZH.md)。目前不能直接把测试通过理解为可以批量提交所有正式任务：

1. 已补精确哈希、绑定原 preflight 的显式兼容记录；默认仍严格检查。真实验收进行中。
2. 已修复导出端 user ID 保留，定向测试确认模型输入不变；真实新 checkpoint 导出进行中。
3. 全量 validation / test 必须核对样本数、唯一用户数和数据版本，不能用训练时的截断指标代替。

当前只推进小规模验收；没有启动正式重训，也没有修改论文正文。
