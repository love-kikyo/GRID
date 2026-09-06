# 实验收尾计划

执行原则：一次推进一个可验收的小阶段；实验整体收尾后再改论文。本文是待办与验收条件，不是已执行记录；执行结果只登记到 [Run 台账](RUN_LEDGER_ZH.md)。

## 0. 已有基础

- [x] 冻结 A + H1 + L2，完成正式数据构建和全量审计。
- [x] 完成 seed 42 SID teacher 正式训练。
- [x] 修复空 worker 初始化，综合测试 38 passed。
- [x] 实现下游训练入口、逐样本导出、融合选权、配对分析和历史出现分析工具。
- [ ] 在新协议、新 checkpoint 上完成这些工具的真实端到端验收。工具已实现不等于实验已完成。

## 1. 下一个小阶段：teacher 与下游链路验收

先做本阶段，不同时铺开所有重训或消融。

1. 固定 131908 的 best checkpoint，记录 checkpoint 哈希、训练配置、数据 preflight 和执行代码版本；保留历史源码指纹，不修改原始 preflight。
2. 处理空 worker 修复引入的训练 / 导出源码哈希差异：审查兼容性并保留独立记录；当前严格导出器尚无已验收的放行路径。
3. 用新 teacher 分别做 listwise、pointwise、separate 的真实权重加载、少量训练 / 验证和 DDP smoke，检查冻结参数及梯度是否符合设计。现有 Hydra 配置解析未覆盖这些行为。
4. 做新 checkpoint 的少量逐样本导出 smoke。核验 user ID 的保留与配对、样本顺序 / 去重、beam / rerank 输出、配置和哈希绑定。
5. 补核验过滤后 train/test 用户交集，以及重叠用户历史 / 目标的边界，明确是否存在同一样本跨 split 重复；这与既有 train/validation 互斥审计不是一回事。

验收：真实日志、明确的通过 / 失败记录和完整 provenance；发现问题则单独修复并提交，不跳过校验。此阶段完成后再确定正式重训的提交顺序。

## 2. 核心受控重训

| 方法 | 初始化 / 可训练部分 | 在比较中回答什么 |
| --- | --- | --- |
| Generator-only | 已训练的 SID teacher | 不重排的生成基线；称 TIGER-style，不能冒称原版 TIGER |
| RecoChain-listwise | 从同一个 teacher 初始化；联合 SID + listwise rerank | 统一生成与候选竞争式重排 |
| RecoChain-pointwise | 同 teacher；联合 SID + pointwise rerank | 与 listwise 的目标函数对照 |
| Separate-small | 冻结同 teacher；训练独立 candidate-aware reranker | 与分离式 generate-then-rerank 的对照 |

先完成 seed 42 的受控一组并审阅结果；再安排第二 seed 及相匹配的主要对照，避免只给所提方法补 seed。第二 seed 的 teacher 是否共享、各阶段随机种子的含义应先写入配置约定。

共同锁定数据、SID 编码、初始化 checkpoint、beam size、预算 / early-stop 口径、目标标签构造、调参规则。训练时 no-hit 辅助候选可按实现插入 GT；推理候选不得注入 GT。

注意：同一个 checkpoint 的 beam 与 rerank 使用同一批自然候选；不同 joint 模型会更新生成器，不能声称所有方法始终共享完全相同的候选池。

验收：每个模型的 run / best checkpoint / 配置 / preflight / 日志可追溯；失败任务与有效结果分开，不凭目录存在认定成功。

## 3. 全量评估与统计

- [ ] Generator-only 以及各 reranker 独立全量导出，validation 347,416、test 2,061,101 条，核对唯一用户和跨模型配对一致性。
- [ ] 只在 validation 选择融合权重；每种方法使用相同搜索规则，锁定后一次用于 test。旧数据的 beta 不继承。
- [ ] 对逐用户 Recall / NDCG@5、@10 做配对差值与置信区间，明确 beam-only、rerank-only、选权后结果；多 seed 波动与逐用户统计分别呈现。
- [ ] 报告中区分覆盖率、候选内排序能力、整体推荐质量，不把重排提升归因于未经验证的生成能力提升。

当前分析 launcher 采用 5000 次 bootstrap、95% CI、seed 2027。主表还需要 generator-only；现有三模型配对 launcher 不能自动代替完整主表的全部比较。

## 4. 仅补支撑主张的分析

- [ ] 新 checkpoint 上同硬件、batch、beam、上下文长度的 cache / no-cache 延迟与显存复测；独立 reranker 的成本分开报告。
- [ ] 根据核心结果决定是否需要 candidate-aware history 长度或历史重复项分析；不自动扩成全面消融网格。
- [ ] 每项结论有原始输出；旧效率表、旧历史出现率只留作探索参考。

验收：形成可人工审阅的主表、最小必要分析、统计区间与 provenance；没有证据的优势不写成结论。

## 5. 实验冻结后再恢复论文

作者确认可发表结果 → 冻结表格及取数来源 → 更新 short 稿 → 导师审阅 → 匿名 / 页数 / 引用 / ethics 复核。审稿问题见 [评审回应清单](../../WSDM_2027/doc/REVIEW_RESPONSE_MATRIX_ZH.md)。
