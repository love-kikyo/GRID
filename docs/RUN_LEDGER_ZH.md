# Run 台账

最后核验：2026-09-10 14:08（Asia/Shanghai）。状态来自 `sacct`，模型用途来自 launcher、run 配置与日志；作业退出码为 0 不自动表示完成了正式训练。

## 新协议作业（最近在前）

| Job | 开始时间 | 在做什么 / 模型 | 结果与解释 |
| --- | --- | --- | --- |
| 132685 | 09-10 01:46 | 四模型全量 test 的配对统计，CPU | COMPLETED / 0:0，1m23s；01:47:28 结束，11 项比较 |
| 132684 | 09-10 00:34 | SID teacher best 的全量 test 导出，单 GPU | COMPLETED / 0:0，1h11m51s；01:46:05 结束，2,061,101 条 |
| 132683 | 09-09 23:45 | Separate best 的全量 test 导出，单 GPU | COMPLETED / 0:0，48m51s；09-10 00:34:14 结束，2,061,101 条 |
| 132682 | 09-09 23:45 | Pointwise best 的全量 test 导出，单 GPU | COMPLETED / 0:0，1h13m08s；09-10 00:58:31 结束，2,061,101 条 |
| 132681 | 09-09 23:45 | Listwise best 的全量 test 导出，单 GPU | COMPLETED / 0:0，1h13m36s；09-10 00:58:59 结束，2,061,101 条 |
| 132678 | 09-09（选权） | Separate validation-only beta 选权，CPU | COMPLETED，50s；51 点网格中选 beta=0.50 |
| 132677 | 09-09 23:28 | Separate best 的完整 validation 导出，单 GPU | COMPLETED，8m55s；347,416 唯一用户，验收通过 |
| 132676 | 09-09（选权） | Pointwise validation-only beta 选权，CPU | COMPLETED，47s；beta=0.50 |
| 132675 | 09-09 23:28 | Pointwise best 的完整 validation 导出，单 GPU | COMPLETED，12m58s；347,416 唯一用户，验收通过 |
| 132674 | 09-09（选权） | Listwise validation-only beta 选权，CPU | COMPLETED，46s；beta=0.50 |
| 132673 | 09-09 23:28 | Listwise best 的完整 validation 导出，单 GPU | COMPLETED，13m05s；347,416 唯一用户，验收通过 |
| 132672 | 09-09（验收） | CPU 检查 pointwise / separate checkpoint | COMPLETED，36s；模型张量有限，separate 冻结 teacher 两份别名状态均逐值一致 |
| 132667 | 09-09（验收） | Separate best checkpoint 的 val 导出，单 GPU | COMPLETED，1m22s；64 条，显式兼容记录绑定日志修复 |
| 132666 | 09-09（验收） | Pointwise best checkpoint 的 val 导出，单 GPU | COMPLETED，1m18s；64 条；不是重训 |
| 132665 | 09-09（验收） | Listwise best checkpoint 的 val 导出，单 GPU | COMPLETED，1m19s；64 条；不是重训 |
| 132051 | 09-09 16:27 | 正式 separate-small，seed 42，3 GPU | COMPLETED，4h25m05s；early stop 35000，best 10000；两个验证辅助字段 NaN，见下文 |
| 132050 | 09-08 09:26 | 正式 joint pointwise，seed 42，3 GPU | COMPLETED，31h01m13s；early stop 115000，best 90000；已重载测试 |
| 132049 | 09-07 00:31 | 正式 joint listwise，seed 42，3 GPU | COMPLETED，32h55m25s；训练产物验收通过，best step 100000；全量评估见 132681 / 132685 |
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

## 已验收：seed 42 四模型全量 test 与配对统计

Validation 验收通过后，提交前工作树干净，源码提交 `3db28b13914846e18cf0c0aa506aec3d0ee302ed`。四模型仍使用现有 best checkpoint，每份导出覆盖当前 test 的 2,061,101 条，不去重、不重训。首批三个 reranker 各用单 GPU，SID 依赖 separate 成功后接续；配对统计依赖四份导出全部成功。完整命令、依赖、冻结选权文件 SHA256 见 [test 启动记录](../logs/training_launch/full_test_seed42_2026-09-09.json)。

Test 标签为各方法的 `*_test_full_v1`；`MAX_BATCHES` 为空、batch size 32、单 worker、BF16。Teacher 和 separate 分别沿用已验收的精确源码兼容文件，joint 默认严格匹配。导出保留原 run 的候选、raw logits 和 runtime 排序；配对分析从原始分数按已冻结的 validation beta=0.50 重新排序，而不是把 run 原 beta=0.1 的 `rerank_sid` 直接当作 selected 结果。SID 只使用 beam 基线，不使用其未训练的 rerank head 作方法结果。

132685 直接使用支持任意模型数的 `scripts/analyze_paired_evaluation.py`，不是仅支持三模型参数的旧 launcher。预先固定 11 项比较：各 selected 对 SID beam（3）、listwise selected 对 pointwise / separate selected（2）、各方法 selected 对自身 beam / click-only（6）。Recall / NDCG@5、@10；5000 次 paired bootstrap、95% CI、seed 2027，输出为 `paired_report_seed42_four_models_full_v1`。五项作业及 Slurm steps 均 COMPLETED / 0:0，从首批导出启动到统计完成共 2h02m05s。

### 验收证据与主表

四份导出各 2,061,101 个唯一 int64 user ID；逐条 user ID、输入 example hash、target SID 完全一致。所有分数有限、形状 / SID 范围正确；每行 20 个候选无重复，同一模型 beam 与 runtime rerank 候选多重集一致，分数按降序保存。Separate 与 SID 的全量 beam SID 和分数逐值完全一致。

重新读取四个 checkpoint、三份 validation 输入 / sidecar 和冻结选权，SHA256 与既有记录一致；本次额外为四份 test HDF5 / sidecar 登记 SHA256。源码 / 运行配置 / 数据清单指纹与报告绑定也通过，未重扫原 TFRecords。SID / separate 精确源码兼容文件及原 preflight 绑定再次核对通过。Separate 的 missing=0 / unexpected=62 仍仅为此前已验收的延迟 teacher 别名；不是新权重丢失。stderr 未检出致命错误；只读 NumPy 转 tensor 警告仍在，当前路径先转 long 并做非原地加法，没有向只读映射写入。

从完整 HDF5 用独立的单 GT 命中名次公式重算指标及逐样本差值；11 项比较中 44 组指标的均值 / 差值与原报告在绝对容差 1e-12 内一致。独立累计差值分布后复用既有 bootstrap 函数与同一随机序列，44 个区间复算一致；22 组 Recall 胜负计数及精确二项检验也一致。没有再次进行 GPU 推理或权重搜索。

证据：[机器可读验收](../logs/analysis/paired_evaluation/test_acceptance_seed42_full_v1.json)、[只读验收脚本](../logs/analysis/paired_evaluation/accept_test_seed42_full_v1.py)、[完整配对报告](../logs/analysis/paired_evaluation/paired_report_seed42_four_models_full_v1.md)、[原始统计 JSON](../logs/analysis/paired_evaluation/paired_report_seed42_four_models_full_v1.json)。验收脚本从任意目录用 GRID 环境 Python 执行，仅向 stdout 输出结果，不覆盖原产物。

下表为当前完整 test、seed 42、0–1 标度；SID 使用 beam，其余使用 validation 冻结的 beta=0.50、FP32 融合。这一主表与下面的有限 batch 训练日志分开保留。

| 方法 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| SID / beam-only | 0.267156 | 0.271581 | 0.250433 | 0.251868 |
| RecoChain-listwise / 融合 | 0.284292 | 0.287060 | 0.261166 | 0.262070 |
| RecoChain-pointwise / 融合 | 0.280813 | 0.284096 | 0.259177 | 0.260254 |
| Separate-small / 融合 | 0.272527 | 0.274050 | 0.254558 | 0.255053 |

Listwise 相对 SID：Recall@5 相对增加 6.414%，NDCG@10 增加 4.051%。相对 pointwise：Recall@5 差值 +0.003479（+0.3479 个百分点，相对 +1.239%），95% 配对 CI 为 [+0.003363, +0.003599]；NDCG@10 差值 +0.001816，CI 为 [+0.001743, +0.001889]。Listwise 对三个主表基线的四项指标差值区间均高于零。这是本 seed / 本 test 的逐用户证据，不替代多 seed。

### 不能混淆的结果解释

自然 beam@20 的 GT 覆盖率：listwise 0.289569（596,831 条命中）、pointwise 0.286810（591,144）、SID / separate 0.275886（568,629）。Joint 模型更新了生成器，因此跨模型的整体差异同时包含候选覆盖和排序差异；不能把融合总收益全部归因于纯重排头。对同一模型，重排不改变候选集及该覆盖率。

对每个模型自身的同一候选池，三种打分方式的 NDCG@10 如下；报告中的 `click` 是 raw rerank logit 排序，不是曝光 CTR：

| 模型 | Beam-only | Rerank-only | 固定融合 |
| --- | ---: | ---: | ---: |
| Listwise | 0.258045 | 0.254695 | 0.262070 |
| Pointwise | 0.256797 | 0.259444 | 0.260254 |
| Separate | 0.251868 | 0.253829 | 0.255053 |

- Listwise 的纯重排 NDCG@10 低于自身 beam-only，融合后才达到主表最好结果；不能声称纯重排头在所有指标上更优。上表中未预声明的额外对比只作描述，不补做事后显著性检验。
- 44 个预声明差值区间中，43 个高于零；例外是 pointwise 融合对纯重排的 Recall@5：差值 +0.000046，CI [-0.000036, +0.000131]，p=0.277，不能宣称该项显著改善。
- 区间为逐项用户重采样区间，不是训练 seed 波动或多重比较同时覆盖保证。极小的 Recall 精确检验 p 值在原报告中有数值下溢为 `0` 的情况，不写成数学意义上的 p=0；NDCG 只报告 bootstrap CI。
- 现有 split 重叠限制继续按 [数据协议](DATA_PROTOCOL_ZH.md) 披露；beta=0.50 是已冻结 validation 网格上界，不是全局最优。验收通过意味着本轮评估完整且可追溯，不意味着实验整体或论文已收尾。

## 已完成：全量 validation 与固定 checkpoint 融合校准

按作者最新决定保持现有 split，不追加去重、checkpoint 重选或为 beta 重训。提交前工作树干净，源码提交 `458b68f6ad782acb7b2fc62ea2275337c063b44e`；实际导出及模型源码与已通过的新 checkpoint smoke 一致。三个 GPU 导出分别接续一个 CPU 选权任务，依赖为 `afterok` 且失败依赖自动取消。完整启动命令、输出标签与预声明选权规则见 [启动记录](../logs/training_launch/full_validation_seed42_2026-09-09.json)。

导出：`MAX_BATCHES` 显式清空、`OUTPUT_TAG=_full_v1`，batch size 32、单 GPU / 单 worker、BF16；每模型应覆盖 347,416 个唯一 validation 用户。Listwise / pointwise 严格匹配训练源码；separate 沿用已验收的精确日志修复兼容记录。输出名称分别为 `listwise_seed42_132049_val_full_v1`、`pointwise_seed42_132050_val_full_v1`、`separate_seed42_132051_val_full_v1`，后缀 `.h5` / `.json`。

选权：beam 权重固定 1，beta=0.00–0.50、步长 0.01；按 NDCG@10、NDCG@5、Recall@10、Recall@5 依次比较，同分取较小 beta，完整 validation、FP32 离线融合。脚本拒绝 smoke 或非 validation 输入；本轮不根据 test 扩展搜索。六项任务全部 COMPLETED / 0:0。

全量验收通过：每模型 347,416 个唯一 int64 user ID；三个模型逐条 user ID、输入 example hash、target SID 一致；SID 候选形状正确、分数有限，beam/rerank 候选多重集相同；运行配置与导出源码指纹一致。选权读取完整 validation、51 个候选权重，按声明规则选出的最大项与输出一致；输入 HDF5 / sidecar / checkpoint SHA256 绑定正确。[机器可读验收记录](../logs/analysis/paired_evaluation/validation_acceptance_seed42_full_v1.json)。

| 方法 | 固定 beta | Validation Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| Listwise | 0.50 | 0.288150 | 0.290885 | 0.264714 | 0.265607 |
| Pointwise | 0.50 | 0.284440 | 0.287756 | 0.262570 | 0.263657 |
| Separate | 0.50 | 0.276096 | 0.277561 | 0.257934 | 0.258411 |

三者均选到网格上界，因此只能称为**预设网格内最优**，未证明更大 beta 不会更好；本次不自动扩大搜索。上表是用于选权的 validation 结果，不是 test 或无偏泛化结论，也不能与此前有限 batch / BF16 runtime 融合日志直接混为同一口径。完整网格及文件指纹：[listwise](../logs/analysis/paired_evaluation/listwise_seed42_132049_fusion_val.json)、[pointwise](../logs/analysis/paired_evaluation/pointwise_seed42_132050_fusion_val.json)、[separate](../logs/analysis/paired_evaluation/separate_seed42_132051_fusion_val.json)。

## 已完成的正式重训：132049–132051

提交时工作树干净，源码提交 `36e78c57a7f977a2c00d77a1aa5417f34379ed2a`；三个方法依赖的模型 / 配置与已通过的 DDP smoke 指纹一致。完整命令、依赖和 teacher / 审计指纹见 [启动记录](../logs/training_launch/seed42_2026-09-07.json)。

共同设置：teacher 为 131908 的 `checkpoint_000_090000.ckpt`，seed 42，3-rank DDP，max optimizer steps 120000，按 validation early stopping 选优。保留训练脚本的原预算，不使用两步 smoke 的限制。Pointwise 和 separate 依赖前一任务成功，使用 `kill-on-invalid-dep=yes`，前序失败时不会留下永久不可运行的下游依赖。

三项均已完成。132050 于 09-09 16:27:56 结束并自动接续 132051，后者于 20:53:01 结束；主作业及各 Slurm steps 全部 COMPLETED / 0:0。本次核验时三项均已不在队列中。

边界审计 [132048 全量报告](../logs/analysis/split_boundary/boundary_132048.json) 的 SHA256 为 `2bb805d4d3305037cae0528c8e5b9c22ca6fbac46eea29915aa2d7b15542c4a6`。精确重复计数及当前保留 split 的评估取舍仅在 [数据协议](DATA_PROTOCOL_ZH.md) 维护。审计是完成了同用户范围的测量，不是证明 test 没有重复。

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

CSV 最优 validation 的 `step=99999` 对应保存时 global step 100000；最终 test 行的 `step=120000` 不代表测试了该步权重。训练内 validation / test 有每 rank 5000 / 10000 batches 上限，validation 的 2 个 shard 在 3 rank 下还涉及重复分配；以上数字不能直接进入论文全量主表。后续完整导出及跨模型统计见本页前文。

证据与产物：

- [Best checkpoint](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/checkpoints/checkpoint_000_100000.ckpt)：5,701,613,500 bytes；SHA256 `8c452c412b3a35575516449fd312a06d59b19a3b69d0d40e88320076166e31ad`。验收时 ZIP 目录可读取；实际模型加载证据来自训练结束后的成功测试日志，本次未另做参数逐值 / 全归档 CRC 校验。
- [运行配置](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/.hydra/config.yaml)、[指标 CSV](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/csv/metrics.csv)、[重启记录](../logs/train/runs/2026-09-07/click-listwise-seed42-132049/metadata/restart_metadata.json)、[preflight](../logs/training_preflight/listwise_seed42_132049.json)。
- [标准输出](../logs/slurm/GRID-List-HLastV1-132049.out)、[含停止原因与 best 权重重载记录的 stderr](../logs/slurm/GRID-List-HLastV1-132049.err)。

## 132050 / 132051：训练完成，日志与指纹核验

2026-09-09 检查：两项实际 world size 均为 3、无重启；preflight 通过，初始化 teacher SHA256 均为 `0cdb246af381e15988acd4116509c2d0c8e162b5db704438d450012ed4c69664`。重新计算各自 preflight 记录的源码与数据清单哈希，无不一致；未重扫全部 TFRecords。日志未检出 traceback、OOM 或 CUDA / NCCL 致命错误。

| 核验项 | Pointwise 132050 | Separate-small 132051 |
| --- | --- | --- |
| 停止 / best optimizer step | 115000 / 90000 | 35000 / 10000 |
| 停止原因 | 连续 5 次验证 Recall@5 未改善 | 同左；不是超时或崩溃 |
| Best validation 融合 Recall@5 | 0.2816953361 | 0.2739449441 |
| CSV validation / test 行数 | 23 / 1 | 7 / 1 |
| 数值核验 | 所有非空数值均有限 | 14 个 NaN，仅分布于两个验证辅助字段；其余数值有限 |
| 最终测试权重 | 日志确认重载 best 90000 | 日志确认重载 best 10000 |
| Checkpoint 大小 | 5,701,613,500 bytes | 6,128,885,734 bytes |

Best checkpoint 的训练自带 test 指标（0–1 标度；有 batch 上限，不是论文全量结果）：

| 方法 / 排序 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| Pointwise / beam-only | 0.274972 | 0.280851 | 0.254766 | 0.256672 |
| Pointwise / 融合重排 | 0.276553 | 0.282448 | 0.256078 | 0.257991 |
| Separate / beam-only | 0.266965 | 0.271429 | 0.250223 | 0.251670 |
| Separate / 融合重排 | 0.269048 | 0.273179 | 0.251678 | 0.253025 |

两者融合仍为 `1.0 × beam_log_prob + 0.1 × rerank_logits`，未做 validation 权重搜索。Separate 的 beam 指标与冻结 teacher 的日志结果一致，但这不替代正式 checkpoint 参数逐值检查。CSV 的 best validation step 分别为 89999 / 9999；test 行的 115000 / 35000 不是实际测试权重步数。

Separate 的历史日志缺陷：继承的 `on_validation_epoch_end` 会输出 `val/beam_log_prob` 与 `val/click_log_prob`，但训练当时的 separate `eval_step` 只更新 loss、beam/rerank evaluator 与 AUC，没有更新这两个 MeanMetric 累加器；7 次验证各产生 2 个 NaN。stderr 同时有 `compute before update` 警告。它们不参与优化损失或选优监控，因此证据指向辅助日志缺失，不是训练数值发散；不能宣称旧 CSV 的“所有数值均有限”。21:07 检查仅诊断；随后已修复并回归，见下节。原始日志与 preflight 未改写。

证据：[pointwise 配置](../logs/train/runs/2026-09-08/click-pointwise-seed42-132050/.hydra/config.yaml)、[CSV](../logs/train/runs/2026-09-08/click-pointwise-seed42-132050/csv/metrics.csv)、[preflight](../logs/training_preflight/pointwise_seed42_132050.json)、[stderr](../logs/slurm/GRID-Point-HLastV1-132050.err)、[best checkpoint](../logs/train/runs/2026-09-08/click-pointwise-seed42-132050/checkpoints/checkpoint_000_090000.ckpt)；[separate 配置](../logs/train/runs/2026-09-09/rerank-small-seed42-132051/.hydra/config.yaml)、[CSV](../logs/train/runs/2026-09-09/rerank-small-seed42-132051/csv/metrics.csv)、[preflight](../logs/training_preflight/separate_small_seed42_132051.json)、[stderr](../logs/slurm/GRID-Sep-HLastV1-132051.err)、[best checkpoint](../logs/train/runs/2026-09-09/rerank-small-seed42-132051/checkpoints/checkpoint_000_010000.ckpt)。

### 收尾：日志修复、checkpoint 指纹与独立导出通过

修复 `6879764` 仅在 [separate eval_step](../src/models/modules/semantic_id/separate_rerank_model.py) 增加 7 行：用自然推理分数更新两项辅助 MeanMetric，口径与 joint 一致；不改优化、候选、排序或选优。新增 3 项测试在修复前因 NaN 失败，修复后通过；含目标函数、配对导出、用户身份、DDP smoke 辅助及空 worker 的定向回归共 **47 passed**、2 个依赖弃用警告。这里只新增单测，没有重复正式训练或 staged training smoke。

Separate 原 preflight SHA256 `07268ef4eec80fc999b232d20993b5caba38ff97e640d1feac399757edad1481` 保留；[精确源码兼容记录](../configs/evaluation/separate_132051_source_compatibility.json) 仅允许该日志修复，记录 SHA256 `95d821f304ec68255a4c6b5a191d2b1d5a711ea1ef82041a336764721c595831`。实测默认严格检查拒绝源码变化，提供该记录后通过；兼容信息随 separate 导出 metadata 保存。Joint 源文件未改，不需要此例外。

新增 best checkpoint SHA256（与独立导出 metadata 再次计算结果一致）：

- Pointwise：`edf4eec9bb908a3f0dec797628c66f45ff7668e0e9e978e50ea397bde9df3e39`。
- Separate：`4fa94bd5d1efd3ccd1d3c5feae598c936c70cd5baf96b64df406957ac41f0caf`。

[CPU 检查 132672](../logs/slurm/GRID-CkptAudit-132672.out) 读取 global step 分别为 90000 / 10000，31 / 105 个 state tensors 均有限。Separate 的 `teacher_model.*` 和 `teacher.sid_model.*` 各 31 个张量、144,183,296 个值，与原 teacher 逐值完全一致。本地反序列化曾以 Illegal instruction 退出，故改在计算节点完成；不是新训练失败，也未改动 checkpoint。

三个真实导出使用干净源码提交 `5c92de9df514db143b23e204068c5a1fe60fc25f`，现有 `run_export_paired_evaluation.sh`，`MAX_BATCHES=2`、`OUTPUT_TAG=_smoke_v1`、split=val、每 GPU batch 32；仅 separate 指定上述兼容文件。均正常退出，输出各 64 条，验收 metadata：[listwise 132665](../logs/analysis/paired_evaluation/listwise_seed42_132049_val_smoke_v1.json)、[pointwise 132666](../logs/analysis/paired_evaluation/pointwise_seed42_132050_val_smoke_v1.json)、[separate 132667](../logs/analysis/paired_evaluation/separate_seed42_132051_val_smoke_v1.json)；同名 `.h5` 为逐样本产物。

检查通过：每份 64 个唯一 int64 user ID；SID 候选形状 `[64,20,4]`，分数有限；各模型 beam/rerank 为相同候选多重集。三个新模型与原 teacher val smoke 的用户 ID、输入样本哈希、target 完全一致；separate 自然 beam 的 SID 与分数也与 teacher 一致。Joint 加载 missing=0 / unexpected=0；separate 为 missing=0 / unexpected=62，仅来自延迟 setup 的两个 teacher 别名前缀，已由 132672 逐值核对，setup 从指纹绑定的原 teacher 重建，不是 reranker 权重丢失。

融合数值核对需遵循实际 AMP 运算：raw logits 在 BF16 下乘 0.1 后与 beam 相加，导出后全部转存 FP32。直接在 NumPy 用 FP32 重算出现最大约 0.003125 的舍入差异；按实际 BF16 乘法重算后三个新模型的融合分数通过 `rtol=atol=1e-6` 检查。后续全量离线选权沿用统一 FP32 分析口径，不把它与训练日志结果混为同一数值口径。

上述 smoke 阶段只验收小规模产物，当时不代表全量 validation / test、融合选权或论文统计完成；随后全量验收见本页前文。真实导出走 INFER；日志修复由上述定向单测验证，当时未重跑 DDP validation。

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
