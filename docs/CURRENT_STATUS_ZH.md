# 当前状态

核验时间：2026-09-24 10:11，Asia/Shanghai。此页是状态快照，不是实时监控。

## 一句话结论

**Seed 42 / 43 的四模型全量 test 与配对统计均已验收；正在推进 seed 44 最后一组全量 test。** 本轮只提交 136157 listwise → 136158 pointwise，后项依赖前项完整验收；separate / SID 尚未提交。所有训练、全量 validation 与选权已完成，不重复运行。论文正文继续暂停。

## 当前进度

| 工作项 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| A + H1 + L2 数据构建 / 全量审计 | 已完成 | 固定 history-last-user-v1；[数据协议](DATA_PROTOCOL_ZH.md) |
| 三个 seed × 四模型训练 | 已完成 | 既有 best checkpoint 固定，不因选权或 test 重选 |
| 三个 seed 的全量 validation / 选权 | 已完成 | 每个重排 run 独立验收；九组选权均为固定网格边界 beta=0.50；SID 不搜索 beta |
| Seed 42 全量 test / 配对统计 | 已验收 | 四模型各 2061101 条；11 项比较 / 44 个差值区间 |
| Seed 43 全量 test / 配对统计 | 已验收 | 135942 / 135943 补齐 separate / SID；136153 完成统计和独立复核；冻结生成器候选 / 分数逐值一致 |
| Seed 44 listwise / pointwise 全量 test | 136157 已启动 / 136158 依赖等待 | 1 GPU / 整项 4 CPU；原 best 80000 / 100000；完整产物待验收 |
| Seed 44 separate / SID 全量 test | 尚未提交 | 本对长任务验收后再提交，不预排长链 |
| Seed 44 配对统计、三 seed 汇总 | 待完成 | 用户配对区间与训练 seed 间波动分开报告 |
| 效率复测 / 必要解释分析 / 复现来源补齐 | 待开展或收尾 | 范围与验收条件见 [实验计划](EXPERIMENT_PLAN_ZH.md#当前剩余工作总览2026-09-24) |
| 论文正文 | 暂停修改 | 等实验整体收尾后由作者确认 |

## 本轮完成了什么

- 验收 seed 43 最后两份 test：各 2061101 唯一用户，四模型用户 / 输入 hash / GT 对齐；separate 与 SID 的 beam 候选和分数完全相同。
- 配对分析入口适配整项 4 CPU、库线程 1、显式绑定，统计核心不改；136151 的 32 项回归通过。
- 136153 在 3m23s 内完成四模型配对分析及独立复算，11 项比较、44 个差值 / 区间、22 个 Recall 配对检验复核一致。本 seed 的 44 个区间均高于零；不将其解释为训练 seed 稳定性或多重比较同时保证。正式数值只维护在 [Run 台账](RUN_LEDGER_ZH.md)。
- 提交 seed 44 的两份 joint test：只读既有 checkpoint、固定数据与 validation 选权，不重训、不在 test 调参。当前队列限制仍为“1 个运行 + 1 个依赖等待”。

## 下一步

1. 验收 136157 / 136158，再安排 seed 44 separate / SID 两份 test。
2. 完成 seed 44 四模型配对统计和三 seed 主表，再推进效率 / 必要解释分析及复现记录收尾。
3. 实验整体收尾后再修改论文。具体待办见 [实验计划](EXPERIMENT_PLAN_ZH.md)。

口径与限制：重排方法的正式 selected 指标按各自冻结 validation beta=0.50 在 FP32 分数上重排；导出保留训练时 runtime beta=0.1 排序，不能混作主表。SID 只报告 beam。Joint 模型更新生成器，不要求跨方法共享同一候选池。已知 split 重叠继续披露，不追加去重；不为 beta 重训。Seed 44 listwise 续训未恢复数据游标，不宣称与不中断训练逐步一致。训练自带有限 batch test 不并入全量主表。
