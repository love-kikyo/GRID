# 当前状态

核验时间：2026-09-09 21:07，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

seed 42 的三个正式任务均已 COMPLETED / 0:0，目前这组三个任务没有运行或排队项。Pointwise、separate 均正常 early stopping 并重载 best checkpoint 完成测试。Separate 有两个验证辅助日志字段因累加器未更新而为 NaN，不能宣称全部数值检查通过；主损失和 Recall / NDCG 有限。全量评估仍未完成。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| 过滤后跨 split 用户 / 样本重复审计 | 全量完成，有重复发现 | 132048；[精确计数与评估要求](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| Teacher 逐样本导出 / user ID | Smoke 已通过 | 132022 / 132023；原 preflight 保留，显式兼容记录随导出存档 |
| 新协议 listwise / pointwise / separate | 3 项训练完成 | Listwise 已验收；另两项日志 / 指纹已核对，separate 辅助日志缺陷待处理；[台账](RUN_LEDGER_ZH.md) |
| 新 checkpoint 全量导出、配对统计、效率复测 | 待完成 | 训练自带 test 有 batch 上限 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132048：CPU 数据边界审计，已完成，不训练模型。
- 132049：RecoChain-listwise（SID + listwise 联合目标）已完成；训练日志融合重排指标优于同模型 beam-only，但不是全量统计结论。
- 132050：RecoChain-pointwise 已完成；停止 step 115000，best step 90000。
- 132051：Separate-small 已完成；停止 step 35000，best step 10000；短于 joint 是较早触发 early stopping，不是任务失败。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

当前收尾 [计划第 2 阶段](EXPERIMENT_PLAN_ZH.md)，准备进入全量评估，不重复已完成的训练：

1. 收尾 pointwise / separate checkpoint 验收；单独处理 separate 辅助日志缺陷并回归。本次仅诊断，未修改代码；现有证据不支持仅因这两个空累加器日志就重训。
2. 全量评估前固定 test 去重复掩码，同时报告原始全量与去重复对照；审计发现不能被忽略，具体规则只在数据协议页维护。
3. 全量导出、融合选权、配对统计、必要分析及论文改写均未完成。论文正文继续暂停。

本次只读检查训练产物并更新文档，没有提交新作业、修改代码、重建数据或修改论文正文；旧 preflight 保留不变。
