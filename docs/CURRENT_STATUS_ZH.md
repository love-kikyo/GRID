# 当前状态

核验时间：2026-09-21 23:00，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

三个 seed 的计划训练均已结束；seed 43 的三个重排模型全量 validation / 选权均已通过，各自冻结 beta=0.50。**Seed 44 短任务链 135701 → 135702 → 135703 已提交**：listwise 运行，pointwise / separate 依赖等待；每项包含导出、验收、选权与独立复算，全部通过才启动下一项。同一时刻仅 1 GPU / 总计 4 CPU；没有批量提交长时间 test。Seed 42 全量结果保留，新增 seed 的 test / 配对评估仍未完成，论文继续暂停。剩余数量见 [待办总览](EXPERIMENT_PLAN_ZH.md#当前剩余工作总览2026-09-21)。

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建与全量审计 | 已完成 | 131857；[数据协议](DATA_PROTOCOL_ZH.md) |
| 过滤后跨 split 重复审计 | 保留已有结果，不追加内容去重 | 132048 仅同用户范围；其余未知，非通过；[准确范围](DATA_PROTOCOL_ZH.md) |
| SID teacher，seed 42，3 GPU | 已完成 | 131908；best optimizer step 90000 |
| 空 worker 初始化问题 | 已修复并回归通过 | 修复提交 `1cbf5ba`；132021 为 38 passed |
| Seed 42 teacher 逐样本导出 / user ID | 全量 test 已通过 | 132684；原 preflight 保留，显式兼容记录随导出存档 |
| Seed 42 listwise / pointwise / separate | 训练及全量评估验收完成 | 正式 checkpoint 固定，未重选；[主表与台账](RUN_LEDGER_ZH.md) |
| Separate 辅助日志 / 冻结 teacher | 修复与核对完成 | 修复 `6879764`，47 passed；132672 参数逐值核对通过；旧 NaN 日志保留 |
| Seed 42 全量 validation / beta 校准 | 已完成并验收 | 132673–132678；三份各 347,416 唯一用户，输入 / 目标完全对齐 |
| Seed 42 四模型全量 test 与配对统计 | 已完成并验收 | 132681–132685；每模型 2,061,101 唯一用户，11 项比较 / 44 个指标差值与区间复核一致 |
| 多 seed SID teacher 43 / 44 | 训练完成，产物核验通过 | 132732 / 132733；best step 分别 100000 / 90000；各自全量 test 尚未完成 |
| Seed 43 listwise | 训练及全量 validation / 选权验收通过 | 132786 best 100000；135621 / 135645，beta=0.50；全量 test 尚未开始 |
| Seed 43 pointwise | 训练及全量 validation / 选权验收通过 | 132793 best 100000；135649 / 135666，beta=0.50；全量 test 尚未开始 |
| Seed 43 separate | 训练及全量 validation / 选权验收通过 | 133277 best 10000；135681 冻结 teacher 通过；135686 / 135699，beta=0.50；全量 test 尚未开始 |
| Seed 44 原 listwise / pointwise | 已取消 | 133873 保留 step 20000 完整 checkpoint；133874 从未启动 |
| CPU 预算 / 完整续训短测 | 134290 验收通过 | 88 项测试、3 rank 完整恢复、20000→20100、有限 loss / 验证与后代线程 CPU 绑定均通过 |
| Seed 44 listwise 正式续训 | 134298 完成，产物验收通过 | 105000 正常早停，best 80000；原 133873 step 20000 恢复，不是从 smoke 继续 |
| Seed 44 pointwise | 134299 完成，产物验收通过 | 达到 120000 上限，best 100000；同 seed SID teacher 独立初始化 |
| Seed 44 separate-small | 135459 完成，最终产物验收通过 | 35000 早停、best 10000；两份 teacher 状态各 31 张量与原 teacher 逐值相同 |
| 4 核全量导出入口 | 135620 短测验收通过 | 36 项测试 + 64 条真实 validation；不是全量结果 |
| 新增 seed 全量评估 | Seed 43 三模型 validation / 选权通过；seed 44 三项短任务已串行提交 | 135701 listwise 运行；135702 pointwise / 135703 separate 等待；8 份新增 seed test 尚未开始 |
| 效率复测 / 必要解释分析 | 待开展 | 后续与训练统一安排 GPU 资源 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近阶段在做什么

- 135701：09-21 22:54:43 启动 seed 44 listwise 短阶段；复用已验收 best 80000，前置绑定及四核检查通过，23:00 已导出 124800 / 347416 条；随后自动验收与选权，不是重训。
- 135702 / 135703：分别为 seed 44 pointwise best 100000、separate best 10000；逐级 `afterok` 等待，任何前项失败均不放行。
- 135699：09-21 22:52:12 完成，1m13s；separate 43 全量产物验收、选权与 204 项指标独立复算通过，beta=0.50，未扩网格。
- 135686：09-21 22:41:18 完成，7m46s；separate 43 validation 完整 347416 唯一用户，输入 / 目标与 seed 43 listwise 一致。

按作者 09-21 最新确认，本轮允许多个短任务串行等待；长任务仍最多“1 个运行 + 1 个依赖等待”。当前只核验本账号队列，不代表核验了其他账号的 GPU 占用。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

按 [实验计划](EXPERIMENT_PLAN_ZH.md) 分阶段推进，不重复已完成的 seed 42 训练、smoke 或全量推理：

1. 收尾 seed 44 三项短阶段，核验全部产物与独立复算记录；短链结束不自动提交 test。
2. Validation 选权全部冻结后，分批开展剩余 8 份全量 test，再做两个 seed 的四模型配对统计与三 seed 汇总；逐用户区间与跨训练 seed 波动分开报告。
3. [数据协议](DATA_PROTOCOL_ZH.md) 已补固定资产重建命令与逐步计数；上游下载快照和历史 SID 生成来源的完整绑定仍需补齐。论文正文继续暂停。

补充：多 seed 不是为每个 beta 重训。仍不测量 beta 改变早停的影响，也不宣称影响已证明为零；每个 run 完成后固定其原规则选出的 best checkpoint。已知 split 重叠继续按原协议披露。

本阶段完成 separate 43 validation / 选权，并串行安排 seed 44 三组短任务，不改模型、损失、训练预算、数据或选权规则。09-16 续训的数据游标无法恢复，仍不宣称与不中断训练逐步完全一致。训练自带 test 有 batch 上限，不并入全量主表。较早任务、失败追溯和指标集中在 [Run 台账](RUN_LEDGER_ZH.md)，本页不重复追加历史流水。
