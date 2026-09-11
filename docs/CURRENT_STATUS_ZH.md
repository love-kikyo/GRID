# 当前状态

核验时间：2026-09-11 15:22，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

新增 SID teacher 43 / 44 均已正常完成；日志、配置、preflight 与 checkpoint 文件核验通过。下一步是为各自的 listwise / pointwise / separate 绑定同 seed 的 best teacher 并安排训练，目前这六项尚未提交。Seed 42 全量结果保留；本轮只检查及更新记录，不启动新任务，论文继续暂停。

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
| 多 seed SID teacher 43 / 44 | 训练完成，产物核验通过 | 132732 / 132733；best step 分别 100000 / 90000；尚无新 seed 全量评估 |
| 多 seed 下游 / 评估 | 未提交 | 每个 seed 的三个对照须绑定该 seed 的同一 teacher |
| 效率复测 / 必要解释分析 | 待开展 | 后续与训练统一安排 GPU 资源 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132732：seed 43 SID teacher，09-11 02:06 完成；达到 max steps，不是早停。
- 132733：seed 44 SID teacher，09-11 13:10 完成；正常 early stopping，不是故障退出。
- 132681–132685：seed 42 全量 test / 配对统计已验收，不重复执行。

本账号在 aias-compute-01 的队列查询为空；这批任务已无运行 / 等待项，不代表核验了其他账号的 GPU 占用。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

按 [实验计划](EXPERIMENT_PLAN_ZH.md) 分阶段推进，不重复已完成的 seed 42 训练、smoke 或全量推理：

1. 使用已核验的 best teacher（路径 / SHA256 在台账）安排相同 seed 的 listwise / pointwise / separate。当前尚未提交这六个下游任务。
2. 新 seed 各自使用相同 validation 网格选权与 test 配对协议，不直接搬用 seed 42 的 beta；逐用户区间与跨训练 seed 波动分开报告。
3. [数据协议](DATA_PROTOCOL_ZH.md) 已补固定资产重建命令与逐步计数；上游下载快照和历史 SID 生成来源的完整绑定仍需补齐。论文正文继续暂停。

补充：多 seed 不是为每个 beta 重训。仍不测量 beta 改变早停的影响，也不宣称影响已证明为零；每个 run 完成后固定其原规则选出的 best checkpoint。已知 split 重叠继续按原协议披露。

本轮核验未发现 NaN/Inf、异常重启或致命执行错误；没有重复推理、修改模型或数据。训练自带 test 有 batch 上限，不并入全量主表。详细证据集中在 [Run 台账](RUN_LEDGER_ZH.md)，旧日志、preflight、checkpoint 和已完成报告保留不变。
