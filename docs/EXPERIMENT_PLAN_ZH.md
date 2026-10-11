# 实验收尾计划

执行原则：一次推进一个可验收的小阶段；实验整体收尾后再改论文。本文是待办与验收条件，不是已执行记录；执行结果只登记到 [Run 台账](RUN_LEDGER_ZH.md)。

2026-09-07 作者补充：继续意味着推进进程，不重复已通过的任务；准备就绪后可启动重训。2026-09-21 更新提交节奏：validation / 选权等短任务可一次提交多项，串行依赖且逐项自动验收；训练 / 全量 test 等长任务仍最多“1 个运行 + 1 个依赖等待”，不预排整串长任务。详见 [开发与记录约定](DEVELOPMENT_WORKFLOW_ZH.md)。

## 当前剩余工作总览（2026-10-11）

三组 seed（42 / 43 / 44）的计划训练、全量 validation / 选权、12 份全量 test、逐 seed 配对统计与三 seed 主表均已验收。剩余的是效率与必要解释分析、复现来源补齐，再恢复论文；并非实验整体已经收尾。以下是待办量，不是一次性提交全部实验的清单。

| 类别 | 剩余工作 / 数量 | 完成条件 |
| --- | --- | --- |
| 已完成：validation / 选权 | 剩余 0 组；seed 44 三项均完成并验收，各自冻结 beta=0.50 | 每组 347416 条，51 点固定网格及 204 项指标独立复算通过；135920 复核产物哈希未变；SID-only 不搜索 beta |
| 已完成：全量 test | 剩余 0 份；138897 separate / 138898 SID 完整产物验收通过 | 每份 2061101 条；既有 best / 冻结选权、唯一用户 / 输入和目标对齐通过；同 seed separate / SID 候选与 beam 分数逐值相同 |
| 已完成：配对统计 | 剩余 0 组；138922 补齐 seed 44 并通过独立复核 | 每组 11 项比较 / 44 个差值区间 / 22 个 Recall 检验复核；统计核心不改，未平均或合并不同 seed 的区间 |
| 已完成：多 seed 汇总 | 三 seed 主表已登记于台账 | 报告哈希绑定，均值 / 样本标准差（ddof=1）双实现算术复核；逐用户配对区间与训练 seed 波动分开，不新增跨 seed 显著性结论 |
| 支撑主张：效率与解释 | 正式效率复测 138935 已独立验收；解释分析下一项 | 效率 BF16 主表与 FP32 cache 对照分开，见台账；解释分析先复用 seed44 四份 test 导出按自然 beam 命中分组，不新增 GPU 推理 |
| 复现收尾 | 补上游快照身份与历史 SID 生成版本 / 参数绑定，冻结配置、指标和取数出处 | 固定资产到样本的链路已有文档；来源缺口未补齐前，不宣称已实现第三方从零复现 |
| 论文与转投 | 作者确认结果后再更新 short 稿、逐项回应 review、导师审阅及投稿合规检查 | 实验先收尾，正文仍暂停；提交前重新核验官网 / 投稿系统要求 |

本轮仅串行推进 seed 44 separate / SID，前项完整验收才放行后项，现已全部验收。原节点故障，作者同意两项迁移至 `aias-compute-2` 的 4090，保留 job ID / 依赖及执行参数，迁移记录见台账。Seed 44 冻结 teacher 的逐用户候选 / 分数一致性已通过。不自动补长链，不追加 split 去重、beta 重训、lambda 全网格或新大规模 baseline 训练；跨硬件逐位等价未经验证，效率实验须统一硬件复测。

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
- [x] 固定各 seed 的共同 best teacher 绑定与既有训练预算；先安排 seed 43 listwise 和唯一依赖后继 pointwise，具体 job / 撤销记录见台账。
- [x] Seed 43 listwise 训练产物验收通过：停止 / 选优、配置 / preflight、CSV 数值及 checkpoint 文件 / 重载测试证据已核验；不等于全量评估完成。
- [x] Seed 43 pointwise 训练产物验收通过；达到 120000 步上限，best 100000，已重载测试，配置 / preflight / CSV / 文件指纹通过。
- [x] Listwise 验收后，按 09-13 作者要求仅补交 seed 43 separate-small，作为 pointwise 的唯一依赖等待项；冻结同 seed SID teacher，具体新旧 job 关系见台账。
- [x] Seed 43 separate 训练产物验收通过；正常早停 35000，best 10000，辅助日志有限，有限 batch test 的四项 beam 汇总指标与 teacher 一致；不等于冻结参数逐值核对或全量推理已通过。
- [x] 补齐 seed 43 separate 冻结 teacher 逐张量核验：135681 通过，模型 / optimizer 张量有限，两份 teacher 状态与原 teacher 逐值一致；不修改 09-15 的历史验收记录。
- [x] 09-15 seed 43 验收后，作者确认继续；仅启动 seed 44 listwise，并提交 pointwise 作为唯一依赖后继。共同初始化为 seed 44 SID teacher best 90000，预算 / 数据 / 损失不变。
- [x] 09-16 作者确认整项 CPU 上限 4 核；修复为 3 rank × 1 核并实现完整状态续训，原 133873 / 133874 取消记录保留。
- [x] CPU 预算 / 完整恢复短测 134290 验收通过：计算节点 88 项回归、3 rank 模型 / optimizer / scheduler / 早停与 best 状态、后代线程 affinity、100 个恢复后 optimizer steps 和真实验证均通过；不是仅检查退出码。
- [x] 短测通过后提交 134298，从原 step 20000 checkpoint 新建正式续训，三 rank 启动核验通过；不从 smoke 输出初始化，总上限仍 120000。仅提交 134299 pointwise 作为唯一依赖等待项。
- [x] 完成并验收 seed 44 listwise / pointwise；135458 单核验收通过，包括权重 / optimizer 张量有限性、SHA256、配置 / 数据身份、选优与重载测试记录，不等于全量评估。
- [x] 09-21 按作者确认提交 seed 44 separate-small 135459；实际配置、teacher / 源码、三 rank 与子进程 CPU 绑定及首批有限 loss 核验通过。仅这一项运行，无后继排队，原 132791 取消记录保留。
- [x] 完成并验收 seed 44 separate-small：135618 单核验收通过，best 10000；模型 / optimizer 张量有限，两份冻结 teacher 状态逐值一致，辅助日志及重载测试通过。
- [x] 全量导出入口适配整项 4 CPU 核；36 项回归与 135620 的 64 条真实导出短测通过，不改打分 / 配对 / 选权规则。
- [x] 完成 seed 43 listwise 全量 validation：135621 导出、135643 验收通过，347416 唯一用户与 seed 42 输入 / 目标逐用户一致。
- [x] 融合选权入口适配四核限制，39 项回归通过；135645 完成 listwise validation-only 选权与独立复算，冻结 beta=0.50，未扩网格。
- [x] 完成 seed 43 pointwise 全量 validation 与选权：135649 导出、135665 完整性 / 同 seed 输入对齐验收通过；135666 独立校准并复算 204 项指标，冻结 beta=0.50，未扩网格。
- [x] 完成 seed 43 separate 全量 validation / 同规则选权：135686 导出，135699 完整性 / 对齐验收及 204 项指标独立复算通过，冻结 beta=0.50；三种重排模型的 seed 43 validation 均已完成。
- [x] 完成 seed 44 三个重排模型的全量 validation / 选权：135701 / 135702 / 135703 均完整验收，各自冻结 beta=0.50；135920 再核对现有产物与验收哈希绑定，不重复推理或搜索。
- [x] 完成并验收 seed 43 listwise / pointwise 全量 test：135921 / 135922 各 2061101 唯一用户，输入 / 目标对齐，候选 / 分数及来源检查通过；本项为产物验收，后续统计验收另列如下。
- [x] 完成 seed 43 separate / SID 全量 test：135942 / 135943 产物与冻结生成器候选 / 分数一致性验收通过。
- [x] 配对入口适配四核与完整四模型；136151 的 32 项回归通过，统计核心不改；136153 全量统计及 44 个差值区间 / 22 个 Recall 检验复算通过。
- [x] Seed 44 listwise / pointwise 全量 test 产物验收通过；138899 单核复核完整文件哈希与日志，不重复推理。历史 Slurm 终态因记账库不可用 / 控制器过期未补核，不虚填退出码。
- [x] 完成 seed 44 separate / SID 全量 test：138897 → 138898 经作者确认迁移至 4090 节点；完整产物与冻结 teacher 候选 / 分数一致性验收通过。
- [x] Seed 44 四份 test 齐备后完成同规则配对统计；138922 独立复核通过。
- [x] 全部 test 通过后汇总 3 个训练 seed 的均值与样本标准差。各 run 的独立 validation-only 选权已完成，不直接搬用 seed 42 的 beta，也不根据已见 test 扩网格。

启动接口：`run_sid_history_last_user_v1.sh [SEED]`，三个下游从 teacher 开始时均为 `LAUNCHER TEACHER_CKPT [SEED]`；listwise 现可另传第三个 `RESUME_CKPT` 作完整续训。历史的 09-10 seed 参数改动没有改模型 / 损失 / 数据，43 项启动测试通过；09-16 的 CPU 与恢复改动另有定向测试和真实短测，不将两次修改混作同一基线。资源预算见 [开发约定](DEVELOPMENT_WORKFLOW_ZH.md)。

旧资源配置下的时间估计：按 seed 42 实测，两份 teacher 合计约 23 小时（已完成）；六项下游合计约 5.7 天计算时间，另加评估、排队与验收。CPU 缩减后的吞吐需要重新实测，不沿用旧耗时作承诺。任务状态只在 [Run 台账](RUN_LEDGER_ZH.md) 记录。

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
- [x] 11 项预声明比较的逐用户 Recall / NDCG@5、@10 差值与置信区间完成，44 个区间复算一致；beam-only、rerank-only、选权后结果分开。此项记录 seed 42 用户统计；后续 seed 43 / 44 与三 seed 描述性汇总也已验收，见第 2 阶段。
- [x] 台账区分自然 beam@20 覆盖率、同一候选池排序和整体质量；保留 listwise 纯重排 NDCG 下降及 pointwise 融合 Recall@5 区间跨零的例外，不作未经验证的归因。

四模型分析包含 generator-only 及三个重排模型；5000 次 bootstrap、batch500、95% CI、seed 2027。09-24 launcher 已适配完整 11 项比较与四核限制，复用未改动的统计核心，seed 43 全量验收通过。区间为逐项用户配对区间，不等同于多 seed 波动或多重比较同时置信保证。

参数区分：当前三个 reranker 均 `use_beam_score_in_rerank_loss=false`，推理融合 beta=`rerank_logit_weight` 不进入重排优化损失，但用于验证排序，因而可能改变 early stopping / checkpoint 选择。按作者决定不为此追加重训；没有测量其影响，不写成已证明为零。Joint 的 lambda=`rerank_loss_weight` 才是 SID 与重排损失的梯度权重。本轮不增加融合训练目标或 lambda 消融。

## 4. 仅补支撑主张的分析

- [x] 新 checkpoint 上同硬件、batch、beam、上下文长度的 cache / no-cache 延迟与显存复测；138935 与独立复核通过，BF16 主比较 / FP32 cache 对照分开，独立 reranker 成本单列；没有整体更快 / 更省显存的证据则不宣称。
- [ ] 根据核心结果决定是否需要 candidate-aware history 长度或历史重复项分析；不自动扩成全面消融网格。
- [ ] 作者认可可解释性探索：先按预先定义的历史长度、GT 是否自然进入 beam 等分组分析，再展示代表性案例；案例不替代全样本配对统计，多 seed 波动也不由单 seed 用户 bootstrap 代替。
- [ ] 每项结论有原始输出；旧效率表、旧历史出现率只留作探索参考。

验收：形成可人工审阅的主表、最小必要分析、统计区间与 provenance；没有证据的优势不写成结论。

## 5. 实验冻结后再恢复论文

作者确认可发表结果 → 冻结表格及取数来源 → 更新 short 稿 → 导师审阅 → 匿名 / 页数 / 引用 / ethics 复核。审稿问题见 [评审回应清单](../../WSDM_2027/doc/REVIEW_RESPONSE_MATRIX_ZH.md)。
