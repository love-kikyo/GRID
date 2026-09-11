# 实验收尾计划

执行原则：一次推进一个可验收的小阶段；实验整体收尾后再改论文。本文是待办与验收条件，不是已执行记录；执行结果只登记到 [Run 台账](RUN_LEDGER_ZH.md)。

2026-09-07 作者补充：继续意味着推进进程，不重复已通过的任务；准备就绪后可直接启动重训，无需再次逐项确认。

## 0. 已有基础

- [x] 冻结 A + H1 + L2，完成正式数据构建和全量审计。
- [x] 完成 seed 42 SID teacher 正式训练。
- [x] 修复空 worker 初始化，综合测试 38 passed。
- [x] 实现下游训练入口、逐样本导出、融合选权、配对分析和历史出现分析工具。
- [x] 在新协议、新 checkpoint 上完成导出、融合选权及四模型配对工具的真实全量端到端验收。
- [ ] 历史出现分析工具在新 checkpoint 上的应用仍按第 4 阶段决定，不因其他工具通过而视为已完成。

## 1. 已完成：teacher 与下游链路验收

加载 / 导出 smoke 于 2026-09-06 通过；同用户范围的跨 split 边界审计于 2026-09-07 完成。审计发现 test 重复；按作者最新决定保留现有 split，在评估中披露限制，不追加跨用户审计 / 内容去重。验收证据见 [Run 台账](RUN_LEDGER_ZH.md)。

- [x] 固定 131908 best checkpoint 的哈希、配置与 preflight；原始 preflight 未改动。
- [x] 显式记录空 worker 修复的精确前后源码哈希，绑定原 preflight；兼容检查与真实导出通过，默认仍严格拒绝其他变化。
- [x] Listwise、pointwise、separate 的真实加载、2-step / 3-GPU 训练与验证通过；逐参数检查 teacher，核验预期梯度与冻结行为。
- [x] Teacher 的 validation / test 各 64 条真实导出通过：user ID 唯一，模型输入一致性单测通过，候选与分数结构正确，配置 / checkpoint / 哈希绑定完整。
- [x] 全量核验 train/test、validation/test 用户交集和相同用户的完整 / 生成器视角重复样本；结果与限制见 [数据协议](DATA_PROTOCOL_ZH.md)。

全量跨模型用户配对与实际遍历完整性已在第 3 阶段独立验收，不是由上面的 smoke 推断。

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

作者已确认多 seed 有必要，seed 42 全量评估已验收。2026-09-10 固定训练 seed 为 **42、43、44**，新增 43 / 44；每个 seed 训练自己的 SID teacher，再由该 teacher 初始化该 seed 的三种对照，不是只给所提方法补 seed。同一数据划分 / SID 映射固定，训练随机性与数据划分随机性分开；若仅共享 seed 42 teacher 重训下游，只能解释为下游随机性实验。多 seed 不等于为每个推理 beta 重训。

- [x] 固定新增 seed 43 / 44 及分阶段资源安排；SID 与 separate launcher 支持 seed 参数，默认仍为 42，listwise / pointwise 已支持。
- [x] 完成 SID teacher 43 → 44，各使用单节点 3-GPU DDP；以 afterok 串行调度。两者从头初始化，不加载 seed 42 权重。
- [x] 核验两份 teacher 的停止原因、best checkpoint 文件 / SHA256、run / preflight 和训练重载测试证据；范围与未做的独立张量检查见台账，不等于完整 split 推理通过。
- [ ] 为相同 seed 的 listwise、pointwise、separate 绑定同一 best teacher 并提交训练；训练和选优预算与 seed 42 一致。本次状态检查未提交新任务。
- [ ] 对新增 seed 分别沿用原 validation-only 网格选权与完整 test 评估，不能直接搬用 seed 42 选出的 beta，也不根据已见 test 扩网格；汇总 3 个训练 seed 的均值与波动。

启动接口：`run_sid_history_last_user_v1.sh [SEED]`，三个下游均为 `LAUNCHER TEACHER_CKPT [SEED]`。run ID、tags、preflight 名称和实际 Hydra `seed` 同步；本轮只改两个原本写死 42 的 launcher 参数，不改模型、损失或数据规则。43 项无 GPU 的 Bash 参数传递测试通过，覆盖四个入口的 42 / 43 / 44、默认值、非法参数、checkpoint 路径和 3-task 分布式保护；新旧默认入口实际参数逐字一致。

时间仅作资源估计：按 seed 42 的实际耗时，两份 teacher 合计约 23 小时；新增两组完整四模型训练在 3 GPU 串行下约 6–7 天，另加评估时间。实际耗时取决于各 seed 的早停与排队，不能保证相同步数结束。任务状态只在 [Run 台账](RUN_LEDGER_ZH.md) 记录。

共同锁定数据、SID 编码、初始化 checkpoint、beam size、预算 / early-stop 口径、目标标签构造、调参规则。训练时 no-hit 辅助候选可按实现插入 GT；推理候选不得注入 GT。

注意：同一个 checkpoint 的 beam 与 rerank 使用同一批自然候选；不同 joint 模型会更新生成器，不能声称所有方法始终共享完全相同的候选池。

验收：每个模型的 run / best checkpoint / 配置 / preflight / 日志可追溯；失败任务与有效结果分开，不凭目录存在认定成功。

## 3. Seed 42 全量评估与统计：核心验收完成

- [x] 作者确定保留现有 split；不追加跨 split 去重、排除名单或相关重训。已知重叠及未审计范围在 [数据协议](DATA_PROTOCOL_ZH.md) 保留，不把共同数据等同于偏差必然抵消。
- [x] 整理当前固定编码资产到最终样本的数据处理链、重建命令、环境与计数核验；[数据协议](DATA_PROTOCOL_ZH.md) 明确复现范围与来源缺口。
- [ ] 发布前补齐上游快照身份和历史 SID 生成运行的版本 / 参数绑定；不能把有命令文档等同于已完成第三方从零复现。
- [x] 固定三个重排 checkpoint，全量导出 validation 的 347,416 条记录；唯一用户、输入哈希 / 目标对齐及候选 / 分数检查通过。
- [x] 第一层融合校准完成：beam 权重 1，beta 网格 0.00–0.50、步长 0.01；NDCG@10、NDCG@5、Recall@10、Recall@5 依次选优，同分取较小 beta。三者均选 0.50，锁定文件指纹后提交 test；仅为网格内最优，本轮不自动扩搜索。SID-only 不搜索 beta。
- [x] 四个固定 checkpoint 在共同 test 的全部 2,061,101 条记录上评估；唯一用户、输入 / 目标对齐和分数 / 候选完整性通过。当前 checkpoint 是按 beta=0.1 的原验证设置选出的，不另行重选或为 beta 重训，不宣称做过 checkpoint × beta 联合搜索。
- [x] 11 项预声明比较的逐用户 Recall / NDCG@5、@10 差值与置信区间完成，44 个区间复算一致；beam-only、rerank-only、选权后结果分开。此项只完成 seed 42 用户统计，多 seed 波动仍待开展。
- [x] 台账区分自然 beam@20 覆盖率、同一候选池排序和整体质量；保留 listwise 纯重排 NDCG 下降及 pointwise 融合 Recall@5 区间跨零的例外，不作未经验证的归因。

当前四模型分析任务直接调用支持多模型的分析脚本，包含 generator-only 及三个重排模型；5000 次 bootstrap、95% CI、seed 2027。旧三模型 launcher 不代表完整主表。区间为逐项用户配对区间，不等同于多 seed 波动或多重比较同时置信保证。

参数区分：当前三个 reranker 均 `use_beam_score_in_rerank_loss=false`，推理融合 beta=`rerank_logit_weight` 不进入重排优化损失，但用于验证排序，因而可能改变 early stopping / checkpoint 选择。按作者决定不为此追加重训；没有测量其影响，不写成已证明为零。Joint 的 lambda=`rerank_loss_weight` 才是 SID 与重排损失的梯度权重。本轮不增加融合训练目标或 lambda 消融。

## 4. 仅补支撑主张的分析

- [ ] 新 checkpoint 上同硬件、batch、beam、上下文长度的 cache / no-cache 延迟与显存复测；独立 reranker 的成本分开报告。
- [ ] 根据核心结果决定是否需要 candidate-aware history 长度或历史重复项分析；不自动扩成全面消融网格。
- [ ] 作者认可可解释性探索：先按预先定义的历史长度、GT 是否自然进入 beam 等分组分析，再展示代表性案例；案例不替代全样本配对统计，多 seed 波动也不由单 seed 用户 bootstrap 代替。
- [ ] 每项结论有原始输出；旧效率表、旧历史出现率只留作探索参考。

验收：形成可人工审阅的主表、最小必要分析、统计区间与 provenance；没有证据的优势不写成结论。

## 5. 实验冻结后再恢复论文

作者确认可发表结果 → 冻结表格及取数来源 → 更新 short 稿 → 导师审阅 → 匿名 / 页数 / 引用 / ethics 复核。审稿问题见 [评审回应清单](../../WSDM_2027/doc/REVIEW_RESPONSE_MATRIX_ZH.md)。
