# 当前状态

核验时间：2026-09-21 21:45，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

三个 seed 的计划训练均已结束；**seed 43 listwise 全量 validation 与融合选权验收通过，冻结 beta=0.50**（预设网格内最优）。当前队列为空；下一项为 seed 43 pointwise validation。Seed 42 全量结果保留，新增 seed 的 test / 配对评估仍未完成，论文继续暂停。

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
| Seed 43 pointwise / separate | 训练产物验收通过 | 132793 / 133277；best 分别 100000 / 10000；不等于全量评估完成 |
| Seed 44 原 listwise / pointwise | 已取消 | 133873 保留 step 20000 完整 checkpoint；133874 从未启动 |
| CPU 预算 / 完整续训短测 | 134290 验收通过 | 88 项测试、3 rank 完整恢复、20000→20100、有限 loss / 验证与后代线程 CPU 绑定均通过 |
| Seed 44 listwise 正式续训 | 134298 完成，产物验收通过 | 105000 正常早停，best 80000；原 133873 step 20000 恢复，不是从 smoke 继续 |
| Seed 44 pointwise | 134299 完成，产物验收通过 | 达到 120000 上限，best 100000；同 seed SID teacher 独立初始化 |
| Seed 44 separate-small | 135459 完成，最终产物验收通过 | 35000 早停、best 10000；两份 teacher 状态各 31 张量与原 teacher 逐值相同 |
| 4 核全量导出入口 | 135620 短测验收通过 | 36 项测试 + 64 条真实 validation；不是全量结果 |
| 新增 seed 全量评估 | 首个模型完成 validation / 选权 | Listwise seed 43 共 347416 条；其余模型 validation / 选权和全部新增 seed test 待开展 |
| 效率复测 / 必要解释分析 | 待开展 | 后续与训练统一安排 GPU 资源 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 135645：09-21 21:14:59 完成，42s；listwise 全量 validation 选权与独立复算，51 组 / 204 项指标核对一致；总计 4 CPU，无 GPU。
- 135643：09-21 21:12:15 完成，46s；单核验收导出完整性、跨 seed 输入 / 目标对齐；39 项回归通过。
- 135621：09-21 19:41:37 完成，12m42s；seed 43 listwise 全量 validation 导出，347416 唯一用户，无截断，最终验收通过。
- 135620：09-21 19:25:42 完成，1m30s；4 核 / 单 GPU 导出短测及回归通过，无 checkpoint 缺失 / 多余参数；不构成正式评估结果。
- 135618：09-21 19:23:07 完成，2m39s；单核、无 GPU 验收 separate，冻结 teacher 两份状态逐值一致。
- 135459：09-21 06:10:39 完成，5h34m32s；35000 正常早停、best 10000，已重载测试并通过最终产物验收。
- 135458：09-21 单核 CPU 产物验收完成，2m17s；两份 best checkpoint 的模型 / optimizer 张量有限性、哈希、配置 / 选优和测试记录均通过，无新推理。
- 135457：此前验收入口缺少项目导入路径而失败，已修正为 135458；非训练或 checkpoint 失败，日志保留。
- 134298：09-17 13:50:11 完成，续训耗时 26h55m42s；正常早停，已重载 best 测试并通过产物验收。
- 134299：前序结束后自动启动，09-19 03:53:04 完成，耗时 38h02m53s；达到上限，已重载 best 测试并通过产物验收。
- 134290：09-16 10:46:58 完成，耗时 5m33s；CPU 预算、完整状态恢复、100-step 训练及短验证验收通过。不是正式训练产物或效果验收。
- 133873：09-15 15:27 因 CPU 24 核占用由作者取消；最后 CSV step=20629，保留 best/full-state step 20000。
- 133874：未启动即取消，无训练产物；以后仍从同 seed SID teacher 独立初始化，不用 listwise 输出。
- 132793：seed 43 SID + pointwise 联合训练，09-14 09:41 完成，耗时 32h40m01s；达到 120000 步上限，best 100000。
- 133277：seed 43 separate-small，09-14 20:44 完成，耗时 4h13m51s；正常早停 35000，best 10000；两个验证辅助字段未再出现 NaN。
- 132786：seed 43 SID + listwise 联合训练，09-13 01:01 正常完成，耗时 33h30m40s；此前已验收。
- 132787–132791：09-11 在未启动时取消，后续均用新 ID 分阶段替代；其中 seed 44 separate 原 132791 现由 135459 替代。原记录保留，没有删除文件。
- 132732：seed 43 SID teacher，09-11 02:06 完成；达到 max steps，不是早停。
- 132733：seed 44 SID teacher，09-11 13:10 完成；正常 early stopping，不是故障退出。
- 132681–132685：seed 42 全量 test / 配对统计已验收，不重复执行。

当前账号队列为空；不代表核验了其他账号的 GPU 占用。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

按 [实验计划](EXPERIMENT_PLAN_ZH.md) 分阶段推进，不重复已完成的 seed 42 训练、smoke 或全量推理：

1. 下一项只提交 seed 43 pointwise 全量 validation，仍固定既有 best 100000；验收后再做该模型的 validation-only 选权，不批量排队。
2. 新 seed 各自使用相同 validation 网格选权与 test 配对协议，不直接搬用 seed 42 的 beta；逐用户区间与跨训练 seed 波动分开报告。
3. [数据协议](DATA_PROTOCOL_ZH.md) 已补固定资产重建命令与逐步计数；上游下载快照和历史 SID 生成来源的完整绑定仍需补齐。论文正文继续暂停。

补充：多 seed 不是为每个 beta 重训。仍不测量 beta 改变早停的影响，也不宣称影响已证明为零；每个 run 完成后固定其原规则选出的 best checkpoint。已知 split 重叠继续按原协议披露。

本阶段完成 listwise 全量 validation / 选权及分析资源适配，不改模型、损失、训练预算、数据或选权规则。09-16 续训的数据游标无法恢复，仍不宣称与不中断训练逐步完全一致。训练自带 test 有 batch 上限，不并入全量主表。详细证据集中在 [Run 台账](RUN_LEDGER_ZH.md)，父 run 保留不变。
