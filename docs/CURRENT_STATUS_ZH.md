# 当前状态

核验时间：2026-09-10 14:27，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

已进入匹配的多 seed 重训，训练 seed 固定为 42 / 43 / 44。本轮只启动新增 SID teacher：43 正式训练中，44 等待接续；新 teacher 验收后再绑定各自三个下游模型。Seed 42 全量结果已收录，不重复运行；数据划分、编码和训练 / 评估规则保持不变，论文继续暂停。

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
| 多 seed SID teacher 43 / 44 | 已启动 | 132732 运行中，预检和 3-rank 启动通过；132733 依赖等待 |
| 多 seed 下游 / 评估 | 未提交 | 先验收新 teacher，再固定各 seed 的共同初始化 checkpoint |
| 效率复测 / 必要解释分析 | 待开展 | 不与当前 teacher 训练争用 3 张 GPU |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132732：seed 43 SID teacher 从头训练，3 GPU DDP；预检通过，三个 rank 均使用 seed 43，已记录训练步与有限 loss。
- 132733：seed 44 SID teacher，afterok:132732；未开始，不是两个 teacher 并行运行。
- 132681–132685：seed 42 全量 test / 配对统计已验收，不重复执行。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

按 [实验计划](EXPERIMENT_PLAN_ZH.md) 分阶段推进，不重复已完成的 seed 42 训练、smoke 或全量推理：

1. 验收 teacher 43 / 44 的 best checkpoint、配置、preflight 和训练日志，再安排相同 seed 的 listwise / pointwise / separate。当前尚未绑定或提交这六个下游任务。
2. 新 seed 各自使用相同 validation 网格选权与 test 配对协议，不直接搬用 seed 42 的 beta；逐用户区间与跨训练 seed 波动分开报告。
3. [数据协议](DATA_PROTOCOL_ZH.md) 已补固定资产重建命令与逐步计数；上游下载快照和历史 SID 生成来源的完整绑定仍需补齐。论文正文继续暂停。

补充：多 seed 不是为每个 beta 重训。仍不测量 beta 改变早停的影响，也不宣称影响已证明为零；每个 run 完成后固定其原规则选出的 best checkpoint。已知 split 重叠继续按原协议披露。

本轮仅参数化 SID / separate launcher 并补测试（43 passed），未改模型、损失或数据。新作业、seed 42 主表与结果解释集中在 [Run 台账](RUN_LEDGER_ZH.md)；旧日志、preflight、checkpoint 和已完成报告保留不变。
