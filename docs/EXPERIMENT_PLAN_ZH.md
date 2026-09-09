# 实验收尾计划

执行原则：一次推进一个可验收的小阶段；实验整体收尾后再改论文。本文是待办与验收条件，不是已执行记录；执行结果只登记到 [Run 台账](RUN_LEDGER_ZH.md)。

2026-09-07 作者补充：继续意味着推进进程，不重复已通过的任务；准备就绪后可直接启动重训，无需再次逐项确认。

## 0. 已有基础

- [x] 冻结 A + H1 + L2，完成正式数据构建和全量审计。
- [x] 完成 seed 42 SID teacher 正式训练。
- [x] 修复空 worker 初始化，综合测试 38 passed。
- [x] 实现下游训练入口、逐样本导出、融合选权、配对分析和历史出现分析工具。
- [ ] 在新协议、新 checkpoint 上完成这些工具的真实端到端验收。工具已实现不等于实验已完成。

## 1. 已完成：teacher 与下游链路验收

加载 / 导出 smoke 于 2026-09-06 通过；跨 split 边界审计于 2026-09-07 完成。审计发现 test 重复，处理安排转入第 3 阶段；不更改已冻结训练集。验收证据见 [Run 台账](RUN_LEDGER_ZH.md)。

- [x] 固定 131908 best checkpoint 的哈希、配置与 preflight；原始 preflight 未改动。
- [x] 显式记录空 worker 修复的精确前后源码哈希，绑定原 preflight；兼容检查与真实导出通过，默认仍严格拒绝其他变化。
- [x] Listwise、pointwise、separate 的真实加载、2-step / 3-GPU 训练与验证通过；逐参数检查 teacher，核验预期梯度与冻结行为。
- [x] Teacher 的 validation / test 各 64 条真实导出通过：user ID 唯一，模型输入一致性单测通过，候选与分数结构正确，配置 / checkpoint / 哈希绑定完整。
- [x] 全量核验 train/test、validation/test 用户交集和相同用户的完整 / 生成器视角重复样本；结果与限制见 [数据协议](DATA_PROTOCOL_ZH.md)。

全量跨模型用户配对与实际遍历完整性仍在第 3 阶段验收，不能由上面的 smoke 推断。

验收：真实日志、明确的通过 / 失败记录和完整 provenance；发现问题则单独修复并提交，不跳过校验。此阶段完成后再确定正式重训的提交顺序。

## 2. 已完成：seed 42 核心受控重训及产物验收

已提交 seed 42：listwise 132049 → pointwise 132050 → separate-small 132051；前一任务成功后下一任务运行，各使用 3 GPU，不重新训练 teacher。启动命令、源码版本、数据审计及 checkpoint 绑定见 [启动记录](../logs/training_launch/seed42_2026-09-07.json)。

- [x] Listwise 正式训练产物验收：run、best checkpoint、配置 / preflight、选优及测试日志已核验；不等于第 3 阶段全量评估完成。
- [x] Pointwise / separate 正式训练结束，正常 early stopping 并重载 best checkpoint 测试；日志、配置与 preflight 指纹已核对。
- [x] 两份新 checkpoint 指纹登记与张量有限性核对；separate 保存的冻结 teacher 逐值一致；三个新模型各 64 条 validation 独立导出及跨模型样本对齐通过。
- [x] Separate 两个验证辅助累加器日志缺陷已单独修复，47 项定向回归通过；精确兼容记录绑定旧 preflight。未修改旧日志或重训。

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

## 3. 当前阶段：全量评估与统计

- [ ] 先按 [已固定的数据规则](DATA_PROTOCOL_ZH.md) 生成完整 test 去重复名单 / 掩码及哈希；所有方法共用。按作者更新要求，只报告去重复 test，原始数据保留作追溯；排除名单不依赖模型预测。
- [ ] 去重实现稳定后整理完整数据处理链、运行命令、环境 / 版本与计数核验，使第三方可复现最终评估集。
- [ ] Generator-only 以及各 reranker 在同一去重复 test 上全量评估，预计 2,057,968 条，以最终名单为准；核对唯一用户和跨模型配对一致性。
- [ ] 明确是否另做融合权重搜索：若沿用当前固定 beta=0.1，无须为了重测 test 强制重跑 validation；若做搜索，先导出三个 reranker 的全量 validation（347,416 条），使用相同搜索规则并锁定后用于 test。SID-only 无融合权重，其全量 validation 导出仅为可选一致性检查。现有 best checkpoint 保持不变，不通过 test 重挑权重或 checkpoint。
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
