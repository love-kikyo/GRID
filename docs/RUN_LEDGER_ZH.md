# Run 台账

最后核验：2026-09-08 10:00（Asia/Shanghai）。状态来自 `sacct`，模型用途来自 launcher、run 配置与日志；作业退出码为 0 不自动表示完成了正式训练。

## 新协议作业（最近在前）

| Job | 开始时间 | 在做什么 / 模型 | 结果与解释 |
| --- | --- | --- | --- |
| 132051 | 待开始 | 正式 separate-small，seed 42，3 GPU | PENDING；afterok:132050，冻结 teacher，仅训练独立 reranker |
| 132050 | 09-08 09:26 | 正式 joint pointwise，seed 42，3 GPU | RUNNING；日志 step 2039，loss 有限，3 rank、无重启；尚未首次验证 |
| 132049 | 09-07 00:31 | 正式 joint listwise，seed 42，3 GPU | COMPLETED，32h55m25s；训练产物验收通过，best step 100000；全量评估待做 |
| 132048 | 09-07 00:21 | 首次 CPU 全量跨 split 边界审计，无模型训练 | COMPLETED，7m08s；14 项新工具测试通过；完整计数有重复发现，代码 b838070 |
| 132027 | 09-06 14:20 | Separate 3-GPU staged smoke | COMPLETED，1m00s；冻结 teacher 前后一致，2 optimizer steps，代码 7ae9bb4 |
| 132026 | 09-06 14:19 | Joint pointwise 3-GPU staged smoke | COMPLETED，59s；3 rank 验收通过，2 optimizer steps，代码 7ae9bb4 |
| 132025 | 09-06 14:17 | Joint listwise 3-GPU staged smoke | COMPLETED，1m07s；3 rank 的初始化 / 梯度 / 验证检查通过；不是正式重训 |
| 132024 | 09-06 14:11 | CPU 综合回归，含新导出测试 | COMPLETED，1m20s；52 passed，代码 55dd60f |
| 132023 | 09-06 14:11 | 131908 teacher 的 test 导出 smoke | COMPLETED，43s；64 条，代码 55dd60f |
| 132022 | 09-06 14:11 | 131908 teacher 的 validation 导出 smoke | COMPLETED，43s；64 条，代码 55dd60f |
| 132021 | 09-06 13:31 | CPU 综合测试，无正式训练 | COMPLETED，1m46s；38 passed，smoke 审计及 listwise/pointwise 配置解析通过 |
| 131910 | 09-06 00:01 | CPU 综合测试，无正式训练 | FAILED；28 passed、1 failed，空 worker 提前初始化迭代器 |
| 131909 | 未开始 | 待运行的综合测试 | CANCELLED；被后续测试替代，无模型产物 |
| 131908 | 09-05 12:34 | SID-only teacher，seed 42，3 GPU | COMPLETED，11h27m04s；正常 early stop，有 best checkpoint |
| 131907 | 09-05 12:33 | 数据加载 smoke，无正式训练 | COMPLETED；旧状态文档写“已取消”有误 |
| 131906 | 09-05 12:11 | 同一 SID teacher 的正式训练尝试 | FAILED，21m19s；首次验证遇到空 worker，无可用正式 checkpoint |
| 131905 | 09-05 12:11 | SID 模型 3-rank DDP smoke | COMPLETED；仅 2 optimizer steps，不是完成训练 |
| 131904 | 09-05 12:07 | SID 模型 3-rank DDP smoke | FAILED；2-step smoke 后清理阶段内存不足，后续提高主机内存配置 |
| 131899 | 09-05 11:54 | SID teacher，单 rank 启动 | CANCELLED，12m47s；改用真正的 3-rank 启动，无正式结果 |
| 131898 | 09-05 11:53 | 数据加载 smoke | COMPLETED |
| 131897 | 09-05 11:47 | 早期综合测试，无正式训练 | COMPLETED；当时测试集 10 passed，不代表当前全部测试 |
| 131891 | 09-05 11:44 | 早期综合测试，无正式训练 | FAILED；测试收集阶段循环导入 |
| 131857 | 09-04 22:20 | 新协议全量数据构建与审计 | COMPLETED，3h43m17s |
| 131855 | 09-04 22:18 | 新协议小规模构建与审计 | COMPLETED；不是全量数据 |

本表年份均为 2026。较早的七月训练、124632–124637 等旧数据导出见 [历史归档](archive/README.md)，不属于新协议正式结果。

## 当前正式重训：132049–132051

提交时工作树干净，源码提交 `36e78c57a7f977a2c00d77a1aa5417f34379ed2a`；三个方法依赖的模型 / 配置与已通过的 DDP smoke 指纹一致。完整命令、依赖和 teacher / 审计指纹见 [启动记录](../logs/training_launch/seed42_2026-09-07.json)。

共同设置：teacher 为 131908 的 `checkpoint_000_090000.ckpt`，seed 42，3-rank DDP，max optimizer steps 120000，按 validation early stopping 选优。保留训练脚本的原预算，不使用两步 smoke 的限制。Pointwise 和 separate 依赖前一任务成功，使用 `kill-on-invalid-dep=yes`，前序失败时不会留下永久不可运行的下游依赖。

132049 已完成，验收见下节。132050 于 09-08 09:26:43 自动接续启动；[preflight](../logs/training_preflight/pointwise_seed42_132050.json) 通过且 teacher SHA256 与启动记录一致，实际 `world_size=3`、无重启；核验时 [CSV](../logs/train/runs/2026-09-08/click-pointwise-seed42-132050/csv/metrics.csv) 最新 step 2039、train/loss 13.080742，已记录数值均有限，尚未首次验证。132051 仍等待 132050 成功，不是已经开始训练。

边界审计 [132048 全量报告](../logs/analysis/split_boundary/boundary_132048.json) 的 SHA256 为 `2bb805d4d3305037cae0528c8e5b9c22ca6fbac46eea29915aa2d7b15542c4a6`。精确重复计数、为何可继续训练、以及必须执行的去重复评估规则仅在 [数据协议](DATA_PROTOCOL_ZH.md) 维护。审计是完成了测量，不是证明 test 没有重复。

## 132049：listwise 正式训练产物验收通过

验收范围是正式训练、产物可追溯及训练自带测试；不代表全量评估或论文效果验收通过。本次只读核验日志 / 配置 / CSV / 指纹，没有重复训练、smoke 或启动新评估任务。

| 核验项 | 结果 |
| --- | --- |
| 作业退出 | 09-08 09:26:43 结束；主作业及全部 Slurm steps 为 COMPLETED / 0:0，耗时 32h55m25s |
| 真实训练 | `training_task=click`、`rerank_objective=listwise`，SID + listwise 联合训练；不是 SID-only 或 smoke |
| 初始化 / 分布式 | 三个 rank 均成功加载共同 teacher 权重并从 step 0 开始；world size 3，无重启 |
| 指纹 | preflight 通过；验收时重新计算的 7 个源码 / 配置 / launcher 与 3 个数据清单 / 审计 JSON 哈希全部匹配；未重新扫描全部 TFRecords |
| 停止原因 | 达到 max optimizer steps 120000；不是 early stopping。最优之后仅 4 次验证未改善，尚未达到 patience 5 |
| 选优 | 监控融合重排 `val/recall@5`；24 次验证中 step 100000 最佳，值 0.2850142419 |
| 最终测试 | stderr 明确记录重新加载 `checkpoint_000_100000.ckpt` 后测试；不是使用最终 step 120000 权重 |
| 数值 / 异常 | CSV 共 24025 行，含 24 次 validation 和 1 次 test；所有非空数值均有限；未检出 traceback、OOM、CUDA / NCCL 致命错误 |

Best checkpoint 的训练日志指标如下，均为 0–1 标度：

| Split / 排序方式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| Validation / beam-only | 0.282477 | 0.288758 | 0.260033 | 0.262077 |
| Validation / 融合重排 | 0.285014 | 0.290865 | 0.262006 | 0.263909 |
| Test / beam-only | 0.277564 | 0.283769 | 0.255874 | 0.257891 |
| Test / 融合重排 | 0.280016 | 0.285772 | 0.257739 | 0.259614 |

这里 `rerank_*` 及主指标使用 `1.0 × beam_log_prob + 0.1 × rerank_logits` 排序，不是 rerank-only，也不是完成 validation 搜索后的最优融合权重。同一 checkpoint 的 test 重排相对 beam-only，Recall@5 增加 **0.2452 个百分点**、NDCG@5 增加 **0.1864 个百分点**；是描述性正向结果，尚无全量配对置信区间。`click_auc` 是候选标签上的辅助指标，不是曝光 CTR AUC。

CSV 最优 validation 的 `step=99999` 对应保存时 global step 100000；最终 test 行的 `step=120000` 不代表测试了该步权重。validation / test 仍有每 rank 5000 / 10000 batches 上限，validation 的 2 个 shard 在 3 rank 下还涉及重复分配；以上数字不能直接进入论文全量主表。完整唯一用户导出、固定 test 去重复对照及跨模型统计均待完成。

证据与产物：

- [Best checkpoint](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/checkpoints/checkpoint_000_100000.ckpt)：5,701,613,500 bytes；SHA256 `8c452c412b3a35575516449fd312a06d59b19a3b69d0d40e88320076166e31ad`。验收时 ZIP 目录可读取；实际模型加载证据来自训练结束后的成功测试日志，本次未另做参数逐值 / 全归档 CRC 校验。
- [运行配置](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/.hydra/config.yaml)、[指标 CSV](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/csv/metrics.csv)、[重启记录](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/metadata/restart_metadata.json)、[preflight](../logs/training_preflight/listwise_seed42_132049.json)。
- [标准输出](../logs/slurm/GRID-List-HLastV1-132049.out)、[含停止原因与 best 权重重载记录的 stderr](../logs/slurm/GRID-List-HLastV1-132049.err)。

## 131908：已完成的 SID teacher

模型是修改版 GRID 的 TIGER-style、decoder-only LLaMA SID 生成器：2 层、hidden size 1024、16 attention heads、4 层 SID、每层码本容量 4096、beam size 20。不是原版 TIGER 复现，也不是已经训练好重排阶段的 RecoChain。

“SID-only”指训练任务 / 损失：backbone、embedding 和 SID head 参与训练，click/rerank head 冻结。日志中“sid_head only”的简写不能理解为整个 backbone 冻结。

| 配置 / 进度 | 实际值 |
| --- | --- |
| 数据 / seed | `history-last-user-v1` / 42 |
| 初始化 | 从头训练，preflight 中 initialization checkpoint 为 null |
| 分布式 | world size 3；无重启 |
| Batch / 累积 | 每卡 32，累积 4；名义全局 optimizer batch 为 384 |
| 步数口径 | 每 20000 个训练 minibatch 验证，即每 5000 optimizer steps |
| Early stopping | 监控 beam `val/recall@5`，patience 5 |
| 停止 / 最优 | optimizer step 115000 / 90000；不是达到 max_steps 120000 |
| 最终测试 | 训练结束后重新加载 step 90000 的 best checkpoint |

### 训练日志指标：仅用于检查训练，不是论文全量主表

| Best checkpoint | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| Validation | 0.271914 | 0.276220 | 0.255139 | 0.256532 |
| Test | 0.266965 | 0.271429 | 0.250223 | 0.251670 |

配置限制每 rank validation 最多 5000 batches、test 最多 10000 batches，batch size 32。validation 仅 2 个文件、DDP 有 3 个 rank，当前加载器还涉及跨 rank 重复分配；不能由这些行推导全量唯一用户统计。需要独立全量导出后复核。

Teacher 日志也打印 `rerank_*` / `click_auc`，但重排 head 未训练；这些值既不是正式 reranker 结果，也不是曝光 CTR 指标。最佳 validation 在 CSV 中记为 `step=89999`，对应 checkpoint 保存时的 global step 90000；test 行记为 115000，所测权重仍是 90000，勿混淆。

证据与产物：

- [Best checkpoint](../logs/train/runs/2026-09-05/sid-seed42-131908/checkpoints/checkpoint_000_090000.ckpt)（5,701,604,302 bytes；本地大文件，未加入 Git）。
- [完整运行配置](../logs/train/runs/2026-09-05/sid-seed42-131908/.hydra/config.yaml)、[指标 CSV](../logs/train/runs/2026-09-05/sid-seed42-131908/csv/metrics.csv)、[重启记录](../logs/train/runs/2026-09-05/sid-seed42-131908/metadata/restart_metadata.json)。
- [训练前数据与源码指纹](../logs/training_preflight/sid_seed42_131908.json)、[标准输出](../logs/slurm/GRID-SID-HLastV1-131908.out)、[含 early stop / checkpoint 加载记录的 stderr](../logs/slurm/GRID-SID-HLastV1-131908.err)。

## 132021：修复验收，不是 listwise / pointwise 训练

运行 [综合测试脚本](../run_test_retraining_controls.sh)，代码基于 `1cbf5baeb94bc356ffa0fc5c7379cdf349fa85af`。完整 [输出](../logs/data_build/retraining_controls_132021.out) 显示 38 passed、2 个依赖弃用 warning，小规模数据审计通过，两个 Hydra `--cfg job --resolve` 命令退出成功。

脚本中的六月 teacher 路径只参与配置打印，没有加载权重或执行 fit。测试没有证明新 teacher 的下游实际训练、全量导出或 separate reranker DDP 已成功。

### Teacher 源码兼容记录（已通过真实导出 smoke）

131908 preflight 记录的 `dataloading.py` SHA256 为 `82f949bc8d08684788746b13824d0dcf6de078e47dfd02e73d0a07db0c7e262f`；修复后的当前文件为 `eb26bd1f4bae1581480d48196940b7a7a7d53fed5344443bead0e49277a1b1e6`。修复将空 worker 返回提前到 iterator 初始化前；其余 131908 preflight 源文件本次核验未变。

默认导出仍严格比较源码哈希。提交 55dd60f 增加显式、精确哈希的兼容记录：[131908 记录](../configs/evaluation/teacher_131908_source_compatibility.json)，绑定原 preflight SHA256 `869939827019f2950aee726e6ba9b89548d2903c0ece2dcdd28d5a2327007003`。仅允许上面这次已审查的空 worker 修复；不接受其他源码 / manifest 变化，不覆盖旧 preflight，例外记录与哈希随导出 metadata 保存。

导出修复 0c0c445 将 user ID 与原 sequence collator 分开处理；单元测试确认输入张量 / mask 不变，支持 datamodule 参数传递。相关定向回归 37 passed。真实 smoke 任务为 132022 / 132023，输出标签 `sid_seed42_131908_{val,test}_smoke_v1`；不代表全量评估。

两个真实导出均加载 `missing=0, unexpected=0`；每份 64 个唯一 int64 user ID，beam shape `[64,20,4]`，分数有限，beam/rerank 是同一候选集的排列。共同 checkpoint SHA256：`0cdb246af381e15988acd4116509c2d0c8e162b5db704438d450012ed4c69664`。证据：[val metadata](../logs/analysis/paired_evaluation/sid_seed42_131908_val_smoke_v1.json)、[test metadata](../logs/analysis/paired_evaluation/sid_seed42_131908_test_smoke_v1.json)、[52 项回归输出](../logs/data_build/retraining_controls_132024.out)。

## 132025–132027：真实下游初始化与 DDP 验收

使用 [两步 launcher](../run_smoke_staged_history_last_user_v1.sh) 和正常 `src.train` 入口，不保存 checkpoint。每个 rank 都比较了 teacher 的 144,183,296 个参数值；三种方法绑定同一个上述 checkpoint SHA256。每种方法每 rank 完成 2 optimizer steps、2 个验证 batches，验证指标有限。

| 方法 | 训练行为检查 | 逐 rank 报告 | Preflight |
| --- | --- | --- | --- |
| Listwise | backbone、SID head、click head 均存在有限非零梯度 | [报告目录](../logs/stage_smoke/listwise_132025) | [指纹](../logs/training_preflight/stage_smoke_listwise_132025.json) |
| Pointwise | 同上，重排目标为 pointwise | [报告目录](../logs/stage_smoke/pointwise_132026) | [指纹](../logs/training_preflight/stage_smoke_pointwise_132026.json) |
| Separate | 仅 reranker 可训练且有有限非零梯度；teacher 无梯度、训练后参数逐值不变 | [报告目录](../logs/stage_smoke/separate_132027) | [指纹](../logs/training_preflight/stage_smoke_separate_132027.json) |

这证明了加载和少量优化 / 验证链路可运行，不证明模型收敛、长时间运行稳定性、全量配对或论文效果。报告中的验证数字只供 smoke 检查，不进入主表。后续正式重训与全量边界审计见本页前文；全量模型评估仍待做。

收尾定向回归：user identity、paired evaluation、staged smoke、empty worker 四组共 41 passed（登录节点 `USE_TF=0`）；此前计算节点综合回归 132024 为 52 passed，尚不包含随后新增的 staged smoke 四项单测。两组有重叠，不相加计数。
