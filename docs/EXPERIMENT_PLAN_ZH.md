# 实验收尾计划

执行原则：一次推进一个可验收的小阶段；实验整体收尾后再改论文。本文是待办与验收条件，不是已执行记录；执行结果只登记到 [Run 台账](RUN_LEDGER_ZH.md)。

2026-09-07 作者补充：继续意味着推进进程，不重复已通过的任务；准备就绪后可启动重训。2026-09-21 更新提交节奏：validation / 选权等短任务可一次提交多项，串行依赖且逐项自动验收；训练 / 全量 test 等长任务仍最多“1 个运行 + 1 个依赖等待”，不预排整串长任务。详见 [开发与记录约定](DEVELOPMENT_WORKFLOW_ZH.md)。

## 当前剩余工作总览（2026-10-22）

三组 seed（42 / 43 / 44）的既定四模型实验、统计与统一硬件效率复测均已验收；自然 beam 分组、来源核验及内部取数索引也已完成。作者取消的历史相关探索及其他 seed 效率测试不恢复。**10-12 新授权扩展 SIM / LIGER 对照，论文继续暂停；原冻结候选保留，新增结果另行验收后再合入。** 对外匿名复现包尚未制作。

## 外部基线扩展

范围仅为作者批准的 SID-only + SIM 与 LIGER，不同时扩展 DIN、MUSE、LETTER 或 LLM4Rec。工作分支为 `wsdm27-external-baselines-v1`，基点 `c3ffaba`。一次完成一个验收阶段。**10-19 作者在 SIM seed42 全流程验收后决定停止 SIM seed43 / 44 扩展，将后续算力转向 LIGER。** 此决定替代原“两个基线均补三 seed”的计划；保留 SIM 弱结果及单 seed 局限，不将其删除、称为不可比或宣称原始 SIM 普遍无效。LIGER 先完成适配与规模验证，再放行 seed42，后续 seed 另行验收推进。

| 阶段 | 工作与验收条件 | 状态 |
| --- | --- | --- |
| 1 实现与来源核对 | 查明 Separate 演变；固定 LIGER 官方版本、许可证和执行源码哈希 | 历史已核对；官方版本 `b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524` 已下载至相邻 `liger-upstream`，未安装或升级环境 |
| 2 LIGER 受限短测 | 使用训练 split 的少量真实样本验证输入 / GT、官方模型前后向和生成接口；检查 CPU / GPU，记录全目录开销下界 | 已通过并独立验收，见台账；小目录仅供兼容性检查，不产生效果指标，不等于正式训练可行 |
| 3 SIM 任务适配 | 保留 soft-search GSU、ESU 多头注意力和 GSU 辅助目标；检查 frozen teacher、候选完全一致、GT 不泄漏及空历史 / padding | 已实现；139475 的 25 项回归、三 rank 两步训练 / 验证及独立报告验收通过 |
| 4 正式训练放行 | 固定方法适配说明、训练目标、候选目录 / 负例规则、输入长度、验证选优和预算；短测通过、源码提交后提交 seed42 | SIM seed42 已完成；LIGER 前两轮运行 / 数值验收保留，第三轮 147815 已按作者要求在启动前取消；先审查低指标原因 |
| 5 效果验收与扩 seed | full validation 后锁定参数，全量 test 与配对统计；多 seed 与单 seed 结果分别标识 | SIM 不扩 seed；LIGER 暂停最终选模 / test，不能把当前弱验证结果视为充分训练结论 |

SIM 复用当前 teacher / 数据 / 评估外壳，独立增加 scorer，不改 Separate-small。报告名明确为任务与特征适配后的 `SID + SIM-soft`；使用同 seed 冻结 teacher 的自然 beam20、完整保留历史，检索 K 首轮固定 10。不把 SID、rank embedding 和 beam-score 输入等 Separate 自定义增强混入 SIM 主版本。原始排序和 validation 选定融合分别保留；训练候选仍须生成，旧 test 缓存不能替代训练。

10-13 作者同意缓存候选重排短测，10-14 全量 validation / 选权验收后推进 test，不继续追查跨硬件候选差异。评估接入方案：独立 validation / test 从既有、已验收的同 seed 导出读取自然 `beam_sid` / `beam_score`（validation 使用 Separate 的 frozen-teacher beam 字段，test 使用 SID-only；旧收据已证实其与 Separate 逐值相同），按 user ID、完整输入哈希与 GT 校验逐条对齐；SIM 仅对历史和这些候选打分，不读取旧 rerank logits、不按 test 标签筛候选、不注入 GT。该方案保持与旧 Separate 完全相同的候选池，而不要求新硬件重新生成时逐位复现旧输出；不改 SIM 训练、best checkpoint 或既定选权规则。缓存重排耗时不能当作包含生成阶段的端到端效率。

64 条短测已通过。全量使用 `run_sim_cached_validation_cpu4.sh` 缓存专用入口，延续候选来源与逐样本身份检查，逐值核对全量导出前 64 条与已验收短测，并对所有样本独立复算 runtime 融合分数；本次缓存收据不能用作原 `run_sim_seed42_validation_cpu4.sh full` 重新生成候选路径的放行凭证。

全量 validation 与选权现已验收，seed42 selected beta 固定 **0.09**，不是原三种重排方法的 0.50。Test 使用 `run_sim_cached_test_cpu4.sh`，1 GPU / 整项 4 CPU；运行前验证选权收据、checkpoint 与 test 缓存哈希，覆盖 2061101 条，包括最后 13 条不足 batch。导出仍保存 runtime beta=0.1 的排序与原始 logits；后续正式指标按冻结 beta=0.09 在 FP32 分数上重排。独立完整性验收后才做配对统计，不在 test 搜索或自动提交新 seed。

10-19 test 完整性与配对统计均已验收，原路径类型 / Markdown 顺序问题只修复审核入口，不重复推理或报告生成。配对统计固定 5 项：SID beam → SIM 纯重排、SID beam → SIM selected、SIM 纯重排 → SIM selected、SIM selected → Separate selected、SIM 纯重排 → Separate 纯重排。三者输入与自然候选池逐值对齐；SIM / Separate 各用自己的 validation beta（0.09 / 0.50）。沿用 5000 次 bootstrap、batch500、seed2027、95% CI、Recall / NDCG@5、@10；144442 独立 GT 名次计数复算 20 个差值区间和 10 个 Recall 配对检验通过。单 seed 用户区间不替代训练 seed 波动，两种 reranker 的差距不归因于单个结构增强。SIM 到此按作者决定收尾，不再提交 seed43 / 44。

SIM 首轮规则固定如下，源码入口为 `sim_rerank_model.py` / `sim_soft.py`，配置为 `tiger_train_sim_soft.yaml`：

- 按 [SIM 原文第 3 节](https://arxiv.org/pdf/2006.05639) 保留 GSU 可学习双投影内积、原始内积加权求和辅助分支、TopK 后 ESU 多头注意力及共享 item embedding。GSU 与 ESU 各用 BCE，权重均为 1；不是把现有 Separate 改名或只增加一次 TopK。
- ID-only 适配：沿用 DualHashEmbedding（两张 300000 桶表，输出 128 维），ESU 固定 4 头、每头 128 维；两个预测 MLP 均为 200 → 80 → 1、PReLU。没有类别、时间戳或用户画像，故不加类别检索 / 时间间隔 embedding / 短期 DIEN 分支；采用精确 TopK，不实现线上近似索引。以上不能称原始工业 CTR 系统的完整复现。
- 标签是历史末项 next-item，不是曝光点击。自然 beam 命中行计算 mean BCE；未命中行只在独立候选副本随机替换一个位置为 GT，计算同一双 BCE，辅助列表权重 0.5。GSU loss 与 no-hit loss 是两个不同概念。推理不注入 GT；SIM scorer 不读 beam，因此损失路径无须为插入项另算 teacher 分数。
- Teacher 输入仍为最多 7 个历史 item；SIM 看完整保留历史，最多 999 项，padding 不参与检索或聚合。训练仅更新 SIM scorer；teacher 的权重及 eval 模式均冻结。每候选独立预测，候选顺序不能成为特征。
- 沿用 AdamW（lr=1e-5、weight_decay=1e-4、warmup20000 / cosine120000）、每卡 batch32、累积 4、3-GPU DDP、120000 optimizer-step 上限；每 20000 microbatches 验证、最多 5000 validation batches，按 runtime beta=0.1 的 val/Recall@5 选 checkpoint、patience=5。整项为 3 tasks × 1 CPU。先同配置 2-step DDP 短测，不生成正式 checkpoint；正式训练不附带有限 batch test。
- 后续独立 full validation 使用已有 beta 0.00–0.50 / 步长 0.01 / 既有 tie-break，锁定后 full test；同时报告 beam-only、纯 ESU 重排、选权融合。BCE 与现有 Separate 的 listwise 不同，该外部基线是方法级比较，不是只改结构的受控消融。不为 beta 重训、不根据 test 扩网格。

LIGER 保留“生成损失 + 稠密分类损失、生成候选后稠密排序”的方法结构。官方代码在 `get_target_embed` 使用逐元素广播，且对全目录执行可训练内容投影；直接按 35458467 行资产搬到 4090 不可行。先验证等价矩阵乘法的数值和梯度，后续仍需解决投影 / 全目录损失的分块与计算预算，不能擅自改成 sampled softmax。资产行数不是已核实的训练活跃物品数。

全目录损失适配采用两遍分块：前向遍历全部 active 行累计 log-sum-exp，反向按原 chunk 的 dropout 状态重算并累加 query / 投影参数梯度；被 mask 的行按官方常量 -100 保留分母贡献，不改成负采样。仅支持单设备、一阶梯度、冻结内容特征、dense FP32/FP64；与 SID / 输入侧共享投影参数的梯度须正常相加。无 dropout 时对照整表广播的损失和梯度，有 dropout 时对照相同 chunk 随机实现；不声称分块与整表调用的 RNG 轨迹逐位一致。4090 真实目录两步规模测试已验收，显存可控但反向成本过高，不能直接放行正式训练。

保持数学结构的执行优化已验收：官方 MLP 的全宽单通道残差 Conv1d 写为同一权重的矩阵乘法，不删除参数、dropout 或残差；保留参数对象、张量形状和 checkpoint 字段。FP64 / FP32 / 输入侧 AMP 输出与梯度、两种内核 / 两种分块大小组件计时、矩阵版 chunk65536 的完整目录两步更新均通过；详见台账。其他模型 / 损失 / 数据规则不变；下一步用较大 batch 的真实 train 短测评估吞吐 / 显存，按资源成本规划预算，不根据两步 loss 选模型或报效果。候选协议与正式训练配置仍待确定。

10-19 train-only 目录盘点已通过独立验收，见台账：使用冻结 manifest 中的全部训练分片，统计完整存储序列（含 GT）、模型实际消费的最后 7 项历史 + GT、仅训练 GT 三个行号集合。保持 row0 为真实物品行；消费集合没有 SID 别名。不读取 held-out 样本、不改变 split、不训练、不计算推荐指标；此项不是恢复已取消的 GT 历史出现 / 召回分析。三集合供规模和身份规划使用，**不等于已批准候选全集或 seen / cold-start 定义**。当前规模测试使用完整实际消费集合做两次更新，报告仍固定 `candidate_universe_finalized=false` / `formal_training_ready=false`，通过后再确定正式训练预算与协议。

10-19 作者批准下一项真实 train batch 容量短测：单张 4090 / 整项 4 CPU，batch 固定 128、512、1024，每档 1 次预热更新 + 3 次计时更新，最大消费 6656 条记录。只顺序读取冻结 train 分片，档内 / 档间用户不重复，原始末项 GT 不变；不重复或拼接旧 32 条 fixture。每档重置为同一 seed42 初始化并核对权重哈希，优化器 / 损失 / 目录 / chunk65536 / 矩阵残差不变。总步耗时含 TFRecord 读取、解码、输入打包和更新；初始化、资产哈希、预热不计入吞吐，另记各部分耗时及显存。任一档 OOM 后停止更大档，部分完成不输出该档吞吐；只做容量和成本预估，不凭短测 loss、最大可容纳 batch 或粗略 epoch 外推选正式训练参数。无 checkpoint、held-out 读取或自动训练后继；验收核对样本身份、152 个梯度 / 优化器状态、计时算术、CPU 绑定及源码 / 资产哈希。

该容量短测现已完整验收，三档均通过，结果见台账；不再重复同一短测。当前只验证了单卡短测，未验证更大 batch 的容量上限、收敛或多卡吞吐；不能直接按卡数除以单卡成本，也不机械套用 GRID 的训练步数。

10-19 作者确认预算：单张 4090、batch1024、seed42、最多 3 个完整 epoch，先完成 1 个 epoch 后验收再续训。每轮 6420 个 batch，末批 618 条，不丢弃；总上限 19260 个 batch / 最多同数优化器更新，AMP 跳过更新时另记计数。学习率 3e-4，总调度预算固定 19260，前 963 步 warmup 后 cosine；续训保留原调度，不在验收后重新安排三轮预算。已有短测粗估纯训练约 16.5 小时/轮，验证、保存与数据遍历额外耗时另计。正式训练须先完成候选 / 验证入口与恢复短测；没有获准自动扩 seed。

10-19 作者确认采用完整 cold 补充版，最终比较 Recall / NDCG@5、@10、@20。[LIGER 原文第 3 节](https://arxiv.org/html/2411.18814v2#S3)及[固定版本官方实现](https://github.com/facebookresearch/liger/blob/b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524/src/evaluation.py#L508)将生成候选与 cold 集合合并后进行稠密排序。主外部 baseline 保留这一机制，同一 checkpoint 的 beam20-only 只作补充消融；不据 test 效果选择是否补 cold。Seen 固定为训练实际消费的最后 7 项历史 + GT，cold 为固定资产全集减 seen，不按 held-out 标签挑选；这一定义下的目录外集合不代表按时间划分的新物品。不同方法的自然候选不必逐值相同，但候选范围、输入信息与计算成本必须披露。

[机器可读协议](../configs/baselines/liger_history_last_v1.json)固定自然 beam20、四层 SID、无额外生成约束、有效候选与全部 cold 的并集、FP32 归一化内积 / 温度 0.07 排序、同分时 global row 升序；不融合 beam 分数、不按用户过滤历史物品、无效 SID 不补候选，合并后物品去重。每个 checkpoint 在 eval 模式投影全部 cold 并缓存 FP32 表示，权重变动后缓存失效；分块精确 Top20 再与 seen beam 候选合并，结果等价于完整并集 Top20，不做负采样或近似检索。原四模型的 @20 优先从已验收完整预测补算，尚未完成，不重写既有 @5 / @10 主表。

本轮真实规模放行短测固定为 128 条 validation（4 × 32，首批预热）、单张 4090 / 整项 4 CPU：记录全 cold 缓存建立、数据读入、自然生成与精确排序耗时及显存；GPU 小集合整表对照、GT 改写不影响生成、完整扫描行数、候选身份与独立排名指标复算均须通过。使用随机初始化，指标只作实现诊断；不读取 test、不产生正式 checkpoint、不更新参数或自动提交长训练。

该短测已通过并独立验收，见台账；不重复提交。完整 LIGER validation 按实测外推属于长任务，遵守“1 运行 + 1 依赖等待”，不沿用原四模型短 validation 的批量提交例外。下一放行项是正式训练 / 完整验证与恢复：提前冻结验证频率、选优指标及同分规则；逐项保存并恢复模型、optimizer、固定总预算 scheduler、AMP scaler、RNG 与有界打乱数据游标，核验恢复后的真实样本身份和更新；验证缓存不得跨权重更新复用。通过后才提交已批准的第一轮训练，不能将单独的候选短测视为完整训练放行。

10-20 固定的[训练配置](../configs/baselines/liger_train_seed42_v1.json)：每个完整 epoch 后验证全部 347416 条，主选优为完整 cold 的 Recall@5，同分看 NDCG@5，再同分保留较早 epoch；不以 beam-only、@10 / @20 或 test 选择模型。每 1000 batch 及验证前后保存 create-only checkpoint，第一轮验证与独立审核完成后停止。LR 为下一次成功更新的线性 warmup / cosine，AMP 跳过更新时数据游标照常前进、学习率计数不前进；三轮总数据 / 更新上限保持不变。中断恢复绑定原协议 / 源码 / 资产及完整状态，数据按 epoch / seed 重放前缀后继续；正式与 smoke checkpoint 身份隔离。

首次训练前只做一次同配置恢复验收：真实打乱的 batch1024、完整训练目录损失，连续更新 3 步、第 1 步保存；另一个进程从该点完成第 2 / 3 步。两条路径都在第 2 步后执行 65 条完整 cold validation（32 / 32 / 1），比较完整状态哈希、后续样本 / loss / 梯度范数 / 学习率及全部预测逐值一致；CPU 独立复算验证排名。正式任务从新 seed42 初始化，不能从短测 checkpoint 接着训练；恢复短测未验收时清单拒绝正式提交。

10-21 首轮完整训练与 validation 验收通过后，仅推进第 2 轮。冻结配置文件不改，新增 manifest 的阶段边界为已完成 1 轮 → 完成 2 轮，第三轮仍需验收后另提。使用 `epoch01-accepted.pt` 完整恢复，沿用总调度 19260 / warmup963、epoch 对应的数据打乱和全局累计更新数。续训入口仅允许编排 / 测试文件的显式哈希迁移，训练 / 验证核心、数据、资产、配置不得漂移；启动前对恢复后的模型、优化器、scheduler、scaler、RNG、进度和模式做完整状态哈希一致性检查，旧 checkpoint / manifest 不改写。每轮验证与独立审核后停止，选优在全部已验收 epoch 间比较，不强制用最新一轮。

10-22 第 2 轮验收后曾提交最后一轮，计划沿用 `epoch02-accepted.pt` 及原三轮总调度；作者随后因指标过低要求取消，147815 未启动。该取消决定替代“继续第三轮 → 最终选模 → test”的执行计划，旧 manifest / 配置不改写。

10-22 作者要求改进，先执行受控学习短测，工作分支 `wsdm27-liger-learning-v2`，从 `de1d6dc` 分出。实现两项已查明的适配差异，不改模型容量：内容特征按冻结 train-seen 物品逐维标准化；每条训练样本只在原最后 8 项窗口内展开非空历史前缀，原最终 GT 保留。标准化统计用 FP64、总体方差，只拟合 11542755 个训练消费物品，历史与目录共用同一变换、padding 保零；官方代码拟合全内容目录，本版采用 train-seen 统计，须作为适配差异披露。

一个独立短测作业串行比较“原始 / 标准化特征 × 最终 GT / 多前缀监督”四种设置，见 [配置](../configs/baselines/liger_learning_probe_v2.json)。冻结 train 流前 256 个不同 parent，按 seed20261022 划为 128 个优化、128 个保留样本；保留组未参与优化，但其物品可能参与 train-seen 统计，因此不是独立 validation。若保留组最终任务与优化前缀完全相同，停止诊断。各组从相同 seed42 初始化，batch32、1024 次更新尝试，lr3e-4、warmup51 后 cosine；展开后等 prefix 权重。四组对齐的是尝试次数 / 目标 token 数，不是最终 GT 曝光次数或 parent 权重，须记录实际计数与 AMP 跳步。

短测稠密损失仅用共同的 4096 个训练物品目录（优化样本最后 8 项并集加固定随机 train-seen 补齐，不按保留组 GT 补入）。这只检验小样本可学性，不替换正式全目录目标；另以独立初始化做 2 步标准化 + 多前缀的完整 seen 目录集成检查。四组在第 0 / 256 / 1024 次尝试后，对两组原最终 GT 任务保存分层 teacher-forcing loss / Top1 / Top20 和自然 beam20 的前缀覆盖、Recall / NDCG@5/10/20。小样本自然 beam 不做 cold 排序，不修改已批准的正式完整 cold 协议，也不用于推断正式涨幅或联合损失的因果贡献。

资源为单张 4090 / 整项 4 CPU / 24G，作业内诊断上限 40 分钟、Slurm 时限 45 分钟；无 validation / test 记录读取，无正式 checkpoint 或自动后继。验收重放训练样本 / LR / 更新计数，从冻结 SID 重算 GT 与映射，再复算预测指标及资源约束；通过仅表示诊断产物完整，不放行正式训练。验收后根据四组学习曲线决定下一项，新的长训练预算另定；不恢复旧第三轮、不删除弱结果或恢复历史相关探索。

四组短测 147957 已完整完成，结果见台账。作者随后指出预算可能不足，并批准仅重做 **raw + final** 单条件的 [8192 步预算诊断](../configs/baselines/liger_learning_budget_v3.json)。复用旧收据绑定的完整 parent 快照、拟合 / 保留组顺序及 4096 目录，从相同 seed42 权重初始化；模型、batch32、原始内容、最终 GT 监督、SID+dense 权重 1:1、优化器和 warmup51 都不变。只将总尝试上限 / cosine 终点改为 8192：第 52 步起 LR 轨迹不同，不是接旧权重续训，也不要求第 1024 步复现旧指标。

在第 0 / 256 步及每 512 步保存两组原最终 GT 的 teacher-forcing、自然 beam 指标、分层码本合法率和物品映射率；初始权重、前 1024 步训练身份必须与旧 raw_final 对齐，初始预测及前 51 步 loss 的数值差异单独记录。AMP 跳步不补步，调度只随成功更新；独立进程逐步重算 LR / 身份及全部预测指标。固定跑满 8192 次尝试或遇错误 / 超时停止，不按保留组成绩选权重、追加条件或后继。只复用已验收训练快照，不重扫 TFRecord、重拟合标准化或重复全目录 smoke；资源上限仍为单卡 / 整项 4 CPU / 24G / 45 分钟（程序 40 分钟）。该项先排除优化预算不足；小集有拟合能力仍不等于正式实验差距已解释。

10-22 晚验收：148939 已完成，小集拟合检查通过，保留组未见完整目标命中；曲线与独立复算结果仅在 [台账](RUN_LEDGER_ZH.md#liger-单条件优化预算诊断) 维护。本预算诊断已收尾，不重复提交或继续在同一 128 条上加步。作者随后同意将多前缀监督、标准化和预算核算一起接入新版训练；下述新版方案替代此前继续开展小集受控诊断的待办，不恢复旧第三轮。

### LIGER 新版监督与正式预算

新分支 `wsdm27-liger-prefix-v3` 保留旧实现和结果，使用独立 `liger_prefix_*` 组件。训练样本仍来自原冻结 train：每个父样本只保留最后八项，展开所有非空历史前缀，每个前缀同时计算 SID 与完整 seen 目录的 dense 损失，按前缀等权。原最终 GT 每遍出现一次；不加入没有历史的首项目标。因此它补回了历史位置的监督，但不是与 decoder-only GRID 完全相同的损失或参数更新次数。

复用已验收的 train-seen 标准化统计，历史输入、训练目录及推理 cold 目录共用同一转换；padding 保持零、真实 row0 正常转换。统计仅由训练 seen 物品拟合，不使用 validation / test 标签；这是对固定 SID / SCL 的任务适配，不称从头复现官方 RQ-VAE。原 seen / cold 集合覆盖全部新前缀目标，无须重新构建目录。Validation / test 不展开、不改 split，仍采用自然 beam20 加全部 cold 后排序。

- [x] 实现确定性有界前缀流，保留尾批，断点按已消费 prefix batch 重放；AMP 跳步推进数据游标但不推进成功更新计数。
- [x] 训练 / 标准化 / checkpoint 绑定及 CPU 回归；旧权重不得迁入新版本。
- [x] 从已验收的目录窗口计数派生精确 prefix 数，不再次扫描全部 TFRecord；数字和来源只在台账维护。
- [ ] 一次新版集成 gate：真实 batch1024、完整 seen 损失，连续三步与新进程恢复逐值比较；65 条 validation 含尾批、每条完整 cold 扫描及独立审核。只检查新链路，不做效果判定或小集拟合，不读 test。
- [ ] 根据新吞吐确认正式总预算及首个验收阶段，再完善正式分阶段启动入口并从头训练。冻结总 scheduler，不让每个短阶段各自退火到零；完整 validation 用于看学习曲线及选优，无自动多 seed / 后继。

预算必须分别列出父样本覆盖、前缀 / 原最终 GT 曝光、成功更新与 AMP 跳步、GPU 型号 / 数量及耗时。旧三轮授权不能直接解释成三遍全部展开数据；也不能仅用 epoch 数声称与 RecoChain 预算相同。对照成本包含 SID 初始化阶段，区分最优 checkpoint 经历的训练量和早停前实际花费。当前 gate 配置中的 19260 / warmup963 仅用于检查调度与恢复，最多执行三次尝试，**不是新正式训练总预算**。

已实现独立 train / validation 读取组件：只从冻结 manifest 选择对应分片，训练有界打乱、validation 保序，断点按 seed / epoch / 已消费 batch 重放数据游标，保留尾批；144626 的真实数据及完整 checkpoint 新进程恢复、144629 的首轮全量训练计数和完整 validation 均已验收。全资产 SID 索引已构建并逐行验收，候选查找失败返回 -1，不将无效 SID 映射到真实 row0。

冻结数据划分、用户样本及历史末项 GT 不变，不运行官方 Amazon 的重新划分 / prefix 扩增脚本。复用当前固定 SID / SCL 时须称固定编码适配版，不宣称完整复现原 RQ-VAE 链路。LIGER 不使用 user ID；历史预算为最多 7 个物品并披露其内容输入。候选规则按上述已确认协议执行，不从 test 标签构造候选，不静默删除原方法的 cold 补充机制。

计算仍受整项 4 CPU 限制。首项短测只申请 1 GPU，不改 GRID 环境；正式训练 / 全量 test 最多 1 运行 + 1 依赖等待。新运行不覆盖旧 run，新增方法也不得触发已关闭的历史分析。

## 已完成的原实验范围

| 类别 | 剩余工作 / 数量 | 完成条件 |
| --- | --- | --- |
| 已完成：validation / 选权 | 剩余 0 组；seed 44 三项均完成并验收，各自冻结 beta=0.50 | 每组 347416 条，51 点固定网格及 204 项指标独立复算通过；135920 复核产物哈希未变；SID-only 不搜索 beta |
| 已完成：全量 test | 剩余 0 份；138897 separate / 138898 SID 完整产物验收通过 | 每份 2061101 条；既有 best / 冻结选权、唯一用户 / 输入和目标对齐通过；同 seed separate / SID 候选与 beam 分数逐值相同 |
| 已完成：配对统计 | 剩余 0 组；138922 补齐 seed 44 并通过独立复核 | 每组 11 项比较 / 44 个差值区间 / 22 个 Recall 检验复核；统计核心不改，未平均或合并不同 seed 的区间 |
| 已完成：多 seed 汇总 | 三 seed 主表已登记于台账 | 报告哈希绑定，均值 / 样本标准差（ddof=1）双实现算术复核；逐用户配对区间与训练 seed 波动分开，不新增跨 seed 显著性结论 |
| 已收尾：效率与解释 | 效率及三 seed 自然 beam 分组已验收；后续历史相关探索取消 | 不扩其他 seed 效率，不新增历史侧信息 / 案例；已有分组只作描述性汇总，不新增组内 CI 或因果结论 |
| 已完成：内部复现与冻结交付 | 139022 来源验收及取数索引复核通过；待作者确认范围 | 固定资产到样本的链路已有文档；历史聚类源码 / 随机状态缺口明确披露，不宣称从零逐位复现；公开包另行制作 |
| 论文与转投 | 作者确认结果后再更新 short 稿、逐项回应 review、导师审阅及投稿合规检查 | 实验先收尾，正文仍暂停；提交前重新核验官网 / 投稿系统要求 |

Seed44 separate / SID 已按串行验收顺序完成；原节点故障后的迁移记录见台账。同一物理 4090 的效率比较独立验收，不将混合硬件 test 作业时长作为效率结论。新增 baseline 仅按上面的授权阶段推进；不追加 split 去重、beta 重训或 lambda 全网格。

## 实验冻结清单（待作者确认）

本清单是内部交接入口，不是论文正文或新增实验方案。详细数字只维护在台账与原始 JSON；不在这里再复制一套主表。

| 冻结项目 | 论文取数入口 | 使用边界 |
| --- | --- | --- |
| 数据与编码资产 | [冻结协议与复现命令](DATA_PROTOCOL_ZH.md)、[来源验收](RUN_LEDGER_ZH.md#上游快照与编码资产来源核验) | A + H1 + L2 不变；披露 split 重叠；固定 SID 资产复现，不承诺重新聚类逐位一致 |
| 核心效果主表 | [三 seed 主表](RUN_LEDGER_ZH.md#三-seed-主表) | 四模型 × 三 seed，完整 test；SID 用 beam，重排用各自 validation 选定 beta；均值 ± 训练 seed 样本 SD |
| 配对统计与排序变体 | [全精度汇总](../logs/analysis/paired_evaluation/three_seed_summary_full_v1.json) 的 `paired_comparisons` / `all_variants`，逐 seed 报告由索引定位 | 用户配对 CI 与训练 seed SD 分开；不平均 CI，不把 runtime beta=0.1 混作 selected beta=0.50，不省略已记录的非显著例外 |
| 效率与 cache 对照 | [统一硬件结果](RUN_LEDGER_ZH.md#已验收统一硬件效率复测) | 仅 seed44；BF16 四模型主比较与 FP32 cache 对照分开；不声称联合模型比 Separate 更快 / 更省显存 |
| 自然 beam 分组 | [三 seed 描述性分组](RUN_LEDGER_ZH.md#已验收三-seed-自然-beam-命中分组) | 已完成的候选覆盖分析；不是历史召回分析，不作因果归因，不声称双方命中等于同一候选池 |

[机器可读取数索引](../logs/analysis/paired_evaluation/experiment_freeze_index_2026-10-12.json) 绑定 12 个 checkpoint / 完整 test、9 份 validation 选权、3 份配对报告及效率 / 分组 / 来源验收。97 个小文件的当前 SHA256 已复核；checkpoint 的原始 stat 再核对通过，大 checkpoint / HDF5 哈希引用已验收记录，**没有重复读取大文件或重新计算模型指标**。索引 SHA256 `7ba5a731353b4c51cd237dba8a10d0924f5e22ebb035866cf5233ba3552bf956`；[只读构建脚本](../logs/analysis/paired_evaluation/freeze_results_2026-10-12.py)。索引状态为 `verified_for_author_review`，不是作者已批准。

- [x] 结果、checkpoint / Hydra 配置 / preflight、选权和数据身份相互绑定，内部冻结候选可审阅。
- [x] 排除旧错误协议指标、训练时有限 batch test、已取消的历史分析；不扩其他 seed 效率测试。
- [ ] 作者确认采用此结果范围与结论边界，再开始论文改写；建议主线为固定协议下的效果增益、同模型 cache 收益及质量 / 成本取舍。
- [ ] 论文阶段逐项回应 review，制作匿名复现交付包并检查投稿合规；这是交付工作，不是新的实验计算队列。当前内部路径含账户 / run 身份，不直接用于匿名稿公开材料。

## 0. 已有基础

- [x] 冻结 A + H1 + L2，完成正式数据构建和全量审计。
- [x] 完成 seed 42 SID teacher 正式训练。
- [x] 修复空 worker 初始化，综合测试 38 passed。
- [x] 实现下游训练入口、逐样本导出、融合选权、配对分析和历史出现分析工具。
- [x] 在新协议、新 checkpoint 上完成导出、融合选权及四模型配对工具的真实全量端到端验收。
- 已取消（10-12）：历史出现分析工具在新 checkpoint 上的应用；保留工具与旧记录，不把未做分析记成完成。

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
- [x] 完成本轮可查来源核验：绑定上游缓存 revision / ETag、固定编码资产和已找到的历史 SID 参数日志；不能把有命令文档等同于已完成第三方从零复现。
  - 10-12 的 139022 全量 235 文件通过，22:18 独立报告验收通过。历史 codebook 源码 / 随机状态仍缺，不将该缺口标作已恢复，不因此重跑聚类；按固定编码资产分发的可复现边界交付，发布材料仍需携带资产与清单。
- [x] 固定三个重排 checkpoint，全量导出 validation 的 347,416 条记录；唯一用户、输入哈希 / 目标对齐及候选 / 分数检查通过。
- [x] 第一层融合校准完成：beam 权重 1，beta 网格 0.00–0.50、步长 0.01；NDCG@10、NDCG@5、Recall@10、Recall@5 依次选优，同分取较小 beta。三者均选 0.50，锁定文件指纹后提交 test；仅为网格内最优，本轮不自动扩搜索。SID-only 不搜索 beta。
- [x] 四个固定 checkpoint 在共同 test 的全部 2,061,101 条记录上评估；唯一用户、输入 / 目标对齐和分数 / 候选完整性通过。当前 checkpoint 是按 beta=0.1 的原验证设置选出的，不另行重选或为 beta 重训，不宣称做过 checkpoint × beta 联合搜索。
- [x] 11 项预声明比较的逐用户 Recall / NDCG@5、@10 差值与置信区间完成，44 个区间复算一致；beam-only、rerank-only、选权后结果分开。此项记录 seed 42 用户统计；后续 seed 43 / 44 与三 seed 描述性汇总也已验收，见第 2 阶段。
- [x] 台账区分自然 beam@20 覆盖率、同一候选池排序和整体质量；保留 listwise 纯重排 NDCG 下降及 pointwise 融合 Recall@5 区间跨零的例外，不作未经验证的归因。

四模型分析包含 generator-only 及三个重排模型；5000 次 bootstrap、batch500、95% CI、seed 2027。09-24 launcher 已适配完整 11 项比较与四核限制，复用未改动的统计核心，seed 43 全量验收通过。区间为逐项用户配对区间，不等同于多 seed 波动或多重比较同时置信保证。

参数区分：当前三个 reranker 均 `use_beam_score_in_rerank_loss=false`，推理融合 beta=`rerank_logit_weight` 不进入重排优化损失，但用于验证排序，因而可能改变 early stopping / checkpoint 选择。按作者决定不为此追加重训；没有测量其影响，不写成已证明为零。Joint 的 lambda=`rerank_loss_weight` 才是 SID 与重排损失的梯度权重。本轮不增加融合训练目标或 lambda 消融。

## 4. 仅补支撑主张的分析

- [x] 新 checkpoint 上同硬件、batch、beam、上下文长度的 cache / no-cache 延迟与显存复测；138935 与独立复核通过，BF16 主比较 / FP32 cache 对照分开，独立 reranker 成本单列；没有整体更快 / 更省显存的证据则不宣称。
  - 2026-10-12 作者明确：效率测试到此收尾，不在其他 seed 重测耗时 / 显存。下方跨 seed 的候选命中分析只读取已有预测，属于解释分析，不是效率测试或再次模型推理。
- 已取消（10-12）：candidate-aware history 长度、GT 历史出现与模型历史召回相关分析，以及所需历史侧信息 / 案例。未提交历史数据扫描；不作为实验收尾前置项。
- [x] 自然 beam 命中分组完成；与已取消的“GT 是否在输入历史出现”是不同问题。多 seed 波动不由单 seed 用户 bootstrap 代替。
  - 本次先固定 seed44（沿用上一阶段，不按效果挑 seed），三个对照 SID / pointwise / separate 相对 listwise，各按自然 beam@20 的 GT 覆盖分为双方命中、仅 baseline 命中、仅 listwise 命中、双方未命中；四组穷尽且互斥。
  - 只读原四份完整 HDF5、原 validation beta；保留各组样本数、Recall / NDCG@5、@10 和 0–20 GT 排名联合计数，组内差值与全体加权贡献分开。分组和指标在执行前固定，但整体仍属见过主结果后的描述性探索，不冒称前瞻性预注册。
  - 要求输入字节身份 / 用户目标对齐、独立分数比较重建排序、逐用户与排名直方图双路计算、各组加权和还原原配对报告。没有新的推理 / bootstrap / 显著性检验，不能据模型依赖分组作因果归因；双方命中不等于相同候选池。
  - 当前导出不含历史长度；按 10-12 最新决定不再补这部分侧信息，不重新 GPU 导出、不修改训练数据。
  - [x] Seed44 分组 138973 完成：15 项回归、2061101 用户全量统计、12 组独立算术与全体差值还原通过；结果与限制见台账。
  - [x] 相同规则补 seed42 / 43：139020 / 139021 各自独立验收，三 seed 分组汇总通过；保留 NDCG@10 对 Separate 的跨 seed 正负差异，不根据分组结果扩大训练 / 调参网格。
- [x] 本轮纳入冻结清单的结论均有原始输出与验收绑定；旧效率表、旧历史出现率只留作探索参考，不进入正式取数索引。

验收：形成可人工审阅的主表、最小必要分析、统计区间与 provenance；没有证据的优势不写成结论。

## 5. 实验冻结后再恢复论文

作者确认可发表结果 → 冻结表格及取数来源 → 更新 short 稿 → 导师审阅 → 匿名 / 页数 / 引用 / ethics 复核。审稿问题见 [评审回应清单](../../WSDM_2027/doc/REVIEW_RESPONSE_MATRIX_ZH.md)。
