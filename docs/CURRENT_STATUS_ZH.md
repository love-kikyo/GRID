# 当前状态

核验时间：2026-09-13 13:12，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

Seed 43 listwise 训练产物验收通过；pointwise 正在运行，按作者本次要求仅补交 separate-small 作为唯一依赖等待项。Seed 44 三项下游仍不排队。保持“1 个运行 + 1 个依赖等待”；seed 42 全量结果保留，新 seed 全量评估未开始，论文继续暂停。

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
| Seed 43 listwise | 训练产物验收通过 | 132786；达到 120000 上限，best 100000；不是全量评估验收 |
| 多 seed 其余下游训练 | 1 个运行 + 1 个依赖等待 | 132793 pointwise / 133277 separate-small，均为 seed 43；seed 44 三项暂缓 |
| 新增 seed 全量评估 | 未提交 | 各模型完成验收后，沿用原 validation 选权与 test 协议 |
| 效率复测 / 必要解释分析 | 待开展 | 后续与训练统一安排 GPU 资源 |
| 论文正文 | 暂停修改 | 等实验整体收尾，按作者要求执行 |

## 最近的任务到底在做什么

- 132793：seed 43 SID + pointwise 联合训练，09-13 01:01 自动接续，已运行约 12 小时；13:11 CSV 约 45000 步，数值有限、无异常重启。
- 133277：seed 43 separate-small，13:12 提交，等待 `afterok:132793`；冻结 SID teacher 132732，只训练独立小型 reranker。
- 132786：seed 43 SID + listwise 联合训练，09-13 01:01 正常完成，耗时 33h30m40s；本次核验通过。
- 132787–132791：09-11 在未启动时取消；pointwise / separate 分别由 132793 / 133277 替代，seed 44 三项仍暂缓。原记录保留，没有删除文件。
- 132732：seed 43 SID teacher，09-11 02:06 完成；达到 max steps，不是早停。
- 132733：seed 44 SID teacher，09-11 13:10 完成；正常 early stopping，不是故障退出。
- 132681–132685：seed 42 全量 test / 配对统计已验收，不重复执行。

本次账号队列仅有上述一项 RUNNING 和一项 PENDING（Dependency）；不代表核验了其他账号的 GPU 占用。

详细作业记录、已完成模型的指标与 checkpoint 见 [Run 台账](RUN_LEDGER_ZH.md)。

## 下一步

按 [实验计划](EXPERIMENT_PLAN_ZH.md) 分阶段推进，不重复已完成的 seed 42 训练、smoke 或全量推理：

1. 等待并验收 pointwise；已排队的 separate 按依赖接续。验收后再确认并提交后续项目，始终不超过“1 个运行 + 1 个依赖等待”。相同 seed 的三个对照绑定同一 best teacher。
2. 新 seed 各自使用相同 validation 网格选权与 test 配对协议，不直接搬用 seed 42 的 beta；逐用户区间与跨训练 seed 波动分开报告。
3. [数据协议](DATA_PROTOCOL_ZH.md) 已补固定资产重建命令与逐步计数；上游下载快照和历史 SID 生成来源的完整绑定仍需补齐。论文正文继续暂停。

补充：多 seed 不是为每个 beta 重训。仍不测量 beta 改变早停的影响，也不宣称影响已证明为零；每个 run 完成后固定其原规则选出的 best checkpoint。已知 split 重叠继续按原协议披露。

Listwise 本次通过的是日志 / 配置 / preflight / checkpoint 文件与训练重载测试验收；未另做独立张量扫描或 GPU 推理。训练自带 test 有 batch 上限，不并入全量主表。本轮未修改模型或数据；详细证据集中在 [Run 台账](RUN_LEDGER_ZH.md)，旧产物保留不变。
