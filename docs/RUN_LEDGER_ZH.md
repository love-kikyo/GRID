# Run 台账

最后核验：2026-09-21 21:45（Asia/Shanghai）。状态来自 `squeue` / `sacct`，模型用途来自 launcher、run 配置与日志；作业退出码为 0 不自动表示完成了正式训练。

## 新协议作业（最近在前）

| Job | 开始时间 | 在做什么 / 模型 | 结果与解释 |
| --- | --- | --- | --- |
| 135645 | 09-21 21:14 | seed 43 listwise validation-only 选权及独立复算，4 CPU / 0 GPU | COMPLETED / 0:0，42s；51 组 beta / 204 项指标复核通过，冻结 beta=0.50，未扩网格 |
| 135643 | 09-21 21:11 | 135621 全量导出验收及融合资源回归，1 CPU / 0 GPU | COMPLETED / 0:0，46s；39 passed，347416 唯一用户与 seed 42 输入 / 目标逐用户一致 |
| 135621 | 09-21 19:28 | seed 43 listwise 全量 validation 导出，1 GPU / 4 CPU | COMPLETED / 0:0，12m42s；347416 条无截断，完整性 / 来源 / CPU 采样验收通过 |
| 135620 | 09-21 19:24 | seed 43 listwise validation 导出短测，1 GPU / 4 CPU | COMPLETED / 0:0，1m30s；36 项回归、64 条真实样本与 CPU 绑定验收通过，非全量结果 |
| 135618 | 09-21 19:20 | seed 44 separate 最终产物 / 冻结 teacher 核验，1 CPU / 0 GPU | COMPLETED / 0:0，2m39s；两份 teacher 状态逐值相同，产物验收通过 |
| 135459 | 09-21 00:36 | 正式 separate-small，seed 44，3 GPU / 3 CPU | COMPLETED / 0:0，5h34m32s；06:10:39 结束，35000 早停、best 10000；最终产物验收通过 |
| 135458 | 09-21 00:29 | seed 44 两项训练的 checkpoint / 记录验收，1 CPU / 0 GPU | COMPLETED / 0:0，2m17s；模型和 optimizer 张量扫描、指纹及选优记录通过，无新推理 |
| 135457 | 09-21 00:27 | 同上，首次独立验收入口 | FAILED / 1:0，1m24s；缺少项目导入路径，修正后以 135458 重跑；非训练失败 |
| 134299 | 09-17 13:50 | 正式 joint pointwise，seed 44，3 GPU / 3 CPU | COMPLETED / 0:0，38h02m53s；09-19 03:53:04 结束，达到 120000 上限，best 100000；产物验收通过 |
| 134298 | 09-16 10:54 | 正式 joint listwise，seed 44，3 GPU / 3 CPU | COMPLETED / 0:0，26h55m42s；09-17 13:50:11 结束，原 step 20000 续训至 105000 早停，best 80000；产物验收通过 |
| 134290 | 09-16 10:41 | seed 44 listwise CPU 预算 / 完整续训短测，3 GPU | COMPLETED / 0:0，5m33s；10:46:58 结束；3 CPU、88 项测试、完整恢复及 100-step 短测验收通过 |
| 133874 | 未开始 | 正式 joint pointwise，seed 44，3 GPU | CANCELLED，09-15 15:27:36 结束，耗时 0；无 run / preflight |
| 133873 | 09-15 10:06 | 正式 joint listwise，seed 44，3 GPU | CANCELLED by 1018，09-15 15:27:29；5h20m49s，作者因 CPU 24 核占用取消；保留 step 20000 完整 checkpoint |
| 133277 | 09-14 16:30 | 正式 separate-small，seed 43，3 GPU | COMPLETED / 0:0，4h13m51s；20:44:13 结束，正常早停 35000、best 10000；训练产物验收通过 |
| 132793 | 09-13 01:01 | 正式 joint pointwise，seed 43，3 GPU | COMPLETED / 0:0，32h40m01s；09-14 09:41:22 结束，达到 120000 上限、best 100000；训练产物验收通过 |
| 132791 | 未开始 | 原计划 separate-small，seed 44 | CANCELLED，15:36:34 撤销，耗时 0；09-21 由 135459 替代 |
| 132790 | 未开始 | 原计划 joint pointwise，seed 44 | CANCELLED，09-11 15:36:34 撤销，耗时 0；09-15 由 133874 替代 |
| 132789 | 未开始 | 原计划 joint listwise，seed 44 | CANCELLED，09-11 15:36:34 撤销，耗时 0；09-15 由 133873 替代 |
| 132788 | 未开始 | 原计划 separate-small，seed 43 | CANCELLED，09-11 15:36:34 撤销，耗时 0；09-13 由 133277 替代 |
| 132787 | 未开始 | 原计划 joint pointwise，seed 43 | CANCELLED，15:36:34 撤销，耗时 0；后以 132793 重新提交 |
| 132786 | 09-11 15:30 | 正式 joint listwise，seed 43，3 GPU | COMPLETED / 0:0，33h30m40s；09-13 01:01:21 结束；best 100000，训练产物验收通过 |
| 132733 | 09-11 02:06 | 新增 seed 44 SID teacher 正式训练，3 GPU | COMPLETED / 0:0，11h03m30s；13:10:05 结束，早停 115000、best 90000 |
| 132732 | 09-10 14:25 | 新增 seed 43 SID teacher 正式训练，3 GPU | COMPLETED / 0:0，11h41m23s；09-11 02:06:35 结束，达到 120000 上限、best 100000 |
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

## 已验收：seed 43 listwise 全量 validation 与融合选权

135620 短测验收通过后，仅提交 **135621**，09-21 19:28:55 开始；无后继等待项。提交时工作树干净，源码 `fbeb7c2e6e182e407eb3aa234df784f495cb10b4`（相对短测只改文档）。固定既有 132786 best 100000，不重选 checkpoint、不重训，单 GPU / 总计 4 CPU、20G、batch32、worker1、BF16；`MAX_BATCHES` 显式清空，无 sample 截断，目标完整 347416 个 validation 用户。

135621 于 19:41:37 正常完成，耗时 12m42s；135643 单核验收通过。最终 HDF5 / sidecar 各为完整 **347416 条**，用户和输入哈希唯一；按 user ID 排序后，全部用户、输入 hash、目标 SID 与已验收 seed 42 listwise validation 完全一致。这不是比较两模型的候选池，也不是追加跨 split 去重。每行 20 个自然候选无重复、重排前后为同一集合，SID 范围、分数有限性 / 排序通过；自然 beam@20 命中 102761 条（覆盖率 0.295787）。

Checkpoint SHA256 重新读取核对通过，配置 / preflight、清单及 8 个导出源码指纹通过；正确加载（missing=0 / unexpected=0），无需兼容豁免。导出开始 / 结束采样中主进程和 worker 的全部线程均在 `[0,1,16,17]` 四核内，库线程上限 1。HDF5 SHA256 `05655c1f323b0a1491c4374859f42ad795df9fd592cb2c7691881e27df9372e8`；sidecar SHA256 `117f00c58c900323236341bcbae740184bb84e8137a9a926f1aa5549b82a7455`。

导出证据：[启动命令与冻结规则](../logs/training_launch/full_validation_listwise_seed43_2026-09-21.json)、[全量验收](../logs/training_launch/listwise_seed43_validation_acceptance_2026-09-21.json)、[只读验收脚本](../logs/training_launch/accept_listwise_seed43_validation_2026-09-21.py)、[stdout](../logs/slurm/GRID-ValList43-CPU4-135621.out)、[stderr](../logs/slurm/GRID-ValList43-CPU4-135621.err)。

融合 launcher 原已申请 4 CPU，提交 `2cae4f6` 仅补单节点 core binding、库线程池 1、资源检查 / 进程树采样和源码指纹，未改打分或选优算法。135643 的 39 项回归全部通过，包括限制启用前后选权一致、拒绝 test / 截断输入和拒绝覆盖输出。随后工作树干净地提交 135645，21:14:17–21:14:59 完成；无 GPU、总计 4 CPU，独立验收 step 仅用 1 CPU。选权进程开始 / 结束均为单线程，绑定 `[0,1,2,3]`，累计 CPU / 耗时约 0.93 核。

沿用预声明的 beta=0.00–0.50、步长0.01、beam 权重1，以 NDCG@10 → NDCG@5 → Recall@10 → Recall@5 依次选优，同分取较小 beta；**本 seed 独立选出 0.50**，未搬用 seed 42 结果。固定权重文件 SHA256 `4c2076e0d151309f355fc012a90662f7454531251d4cba63260981ebd515cfda`；独立以目标分数的超越计数与稳定同分规则复算 51 × 4 项指标，最大绝对差 `1.11e-16`。结果处于预设网格上界，只称网格内最优，不扩网格、不重选 checkpoint、不重训。

以下仅为该 checkpoint 的**全量 validation**结果，不能作为独立 test 提升：

| 推理方式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Beam-only，beta=0 | 0.284852 | 0.290677 | 0.260934 | 0.262821 |
| 原 runtime beta=0.1 | 0.286979 | 0.292479 | 0.262818 | 0.264607 |
| Validation 选出 beta=0.5 | 0.290516 | 0.293372 | 0.265931 | 0.266860 |

选权证据：[启动 / 来源绑定](../logs/training_launch/fusion_listwise_seed43_2026-09-21.json)、[完整 51 组指标](../logs/analysis/paired_evaluation/listwise_seed43_132786_fusion_val.md)、[机器可读冻结结果](../logs/analysis/paired_evaluation/listwise_seed43_132786_fusion_val.json)、[独立复算验收](../logs/training_launch/listwise_seed43_fusion_acceptance_2026-09-21.json)、[复算脚本](../logs/training_launch/accept_fusion43_2026-09-21.py)。本阶段未提交其他模型或 test。

## 已验收：seed 44 separate 最终产物与 4 核导出短测

135459 主作业 / 训练 step `135459.0` 正常完成，best 10000、连续 5 次验证未改善后于 35000 早停，已重载 best 完成训练自带 test。此前辅助探针 `.1` 失败、修正后的 `.2` 通过按下节记录保留，不把辅助步骤失败误判为训练失败。总分配 3 CPU，累计 CPU / 耗时约 2.99 核。

135618 在单核、无 GPU 条件下完成独立核验：checkpoint 的 **105 个模型张量、36 个 optimizer 张量全部有限**；`teacher_model.` 和 `teacher.sid_model.` 各 31 个张量、144,183,296 个数值，均与 seed 44 SID teacher best 90000 逐值一致。配置 / preflight / 源码 / 数据清单、SID / item2row / teacher 哈希核对通过；模型与 scheduler step=10000、best 记录与日志一致。CSV 7008 行、7 次 validation、1 次 test，非空数值均有限，两个辅助验证字段各 7 次正常，无加载超时或作业重启。

[Separate best checkpoint](../logs/train/runs/2026-09-21/rerank-small-seed44-135459/checkpoints/checkpoint_000_010000.ckpt)：6,128,885,734 bytes，SHA256 `00fec1959cd97c57b9f45d0f4cd77f0026e2ace1fefb97f81874b28cdd0181a6`。Best validation Recall@5=0.272732；有限 batch、runtime beta=0.1 的 test 为 Recall@5/10=0.267852/0.271783、NDCG@5/10=0.250885/0.252167；这些数值不并入全量主表。证据：[完整验收](../logs/training_launch/separate_seed44_acceptance_2026-09-21.json)、[只读核验脚本](../logs/training_launch/accept_separate_seed44_2026-09-21.py)、[验收 launcher](../logs/training_launch/run_separate_seed44_audit_2026-09-21.sh)。

导出入口原申请 8 CPU，本轮提交 `3dc6d9d` 改为单 task / 4 CPU / 单 GPU、显式 core binding、库线程 1 / 数据 worker 1；复用已有 TensorFlow 单线程数据选项，未修改模型 / 候选 / 打分 / batch32 / BF16。新增首次 / 结束进程树采样和实际 CPU / iterator 源码指纹；原训练 preflight 未覆盖的依赖范围不伪装成已经覆盖，也未放宽其原有必需源码校验。

135620 提交时工作树干净，源码 `3dc6d9d084abeab0d8dff786204c3899ebdbc8b2`；计算节点 **36 passed**。Seed 43 listwise best 100000 的 64 条 validation 导出验收通过：64 唯一用户和输入 hash、候选一一对应 / 无重复、SID 范围及分数有限 / 排序正确；自然 beam 命中 21 条，未将短测当全量。Checkpoint 加载 missing=0 / unexpected=0，指纹与既有验收一致。主进程和 worker 所有线程限制于 `[0,1,16,17]` 四核，两个采样点通过。证据：[启动](../logs/training_launch/cpu4_export_smoke_2026-09-21.json)、[短测验收](../logs/training_launch/cpu4_export_smoke_acceptance_2026-09-21.json)、[核验脚本](../logs/training_launch/accept_cpu4_export_smoke_2026-09-21.py)。

## 历史：seed 44 separate-small 启动（现已验收完成）

09-21 两项 joint 产物验收通过后，按作者确认仅提交 **135459**，00:36:07 开始；当时只有这一项训练，没有后继等待任务。提交时源码 `dacfb5bee570f523ebece8c95a6aef43a4bf5bd5`、工作树干净；该次没有修改训练代码。

固定 seed 44 SID teacher 132733 best 90000（SHA256 `3f8b60dad3172ffc666b71d5ad0708e348363c44589d31f9d6eb3639c8d85da4`），冻结 teacher、只训练独立 small reranker，不使用 listwise / pointwise 的输出。3 GPU / 3 rank × 1 CPU、60G、BF16、全局 batch 384；max steps / scheduler 120000、runtime beta=0.1、原 Recall@5 / patience=5 规则不变，训练 / validation / test worker 为 1 / 2 / 3。

启动核验通过：13 个源码 / launcher 指纹及数据清单匹配启动记录，fresh preflight 的 teacher 身份一致；与 seed 43 separate 配置仅 seed / id / tags / teacher / train worker 五项差异。三 rank seed=44、world_size=3、current_run=0、restarts=[]。模型摘要显示 79.2M 可训练、288M 不可训练参数；最终冻结参数逐值核验仍须等 checkpoint 产出。00:42:55 CSV 共 141 行、最近 step=704、train/loss=4.456452，所有非空数值有限。

00:42:00 在原分配内做一次只读进程检查：三个 rank 和各自数据 worker 共 6 个进程、74 个线程，分别仅绑定 CPU `[4]` / `[20]` / `[21]`；库线程上限为 1。检查不额外申请 CPU / GPU，这是一次实时采样，不等于全程资源审计。辅助检查 step `135459.1` 曾因把 `srun` 也识别为训练进程而退出 1，修正枚举后的 `135459.2` 通过；主训练 step `135459.0` 始终运行，未停止或重启。后续不能把该已说明的辅助 step 当作正式训练失败，也不能漏掉真正的训练退出异常。

证据：[启动记录](../logs/training_launch/seed44_separate_2026-09-21.json)、[启动验收](../logs/training_launch/seed44_separate_startup_acceptance_2026-09-21.json)、[CPU 实时采样](../logs/training_launch/seed44_separate_cpu_snapshot_2026-09-21.json)、[只读核验脚本](../logs/training_launch/accept_seed44_separate_startup_2026-09-21.py)、[run](../logs/train/runs/2026-09-21/rerank-small-seed44-135459)、[preflight](../logs/training_preflight/separate_small_seed44_135459.json)。此处仅是启动验收，不是训练产物 / 全量评估完成。

## 已验收：seed 44 listwise / pointwise 训练产物

两项主作业及全部 Slurm steps 均 COMPLETED / 0:0；pointwise 在 listwise 成功结束后自动启动，没有同时占用两份训练资源。每项 AllocCPUS=3，累计 CPU 时间 / 作业耗时约为 2.99 核。Listwise 三 rank 各 189 次训练 / 验证阶段采样，rank 与所有后代线程均在所属单核内；pointwise 的启动 core binding 及 Slurm 分配记录同样为总计 3 核，不将其冒称为也有 189 次运行中审计。

| 模型 / job | 停止 step / 原因 | Best step | Best validation Recall@5 | CSV / validation / test 条数 |
| --- | --- | ---: | ---: | --- |
| Listwise / 134298 | 105000 / 连续 5 次未改善 | 80000 | 0.288896 | 17018 / 17 / 1（仅续训部分） |
| Pointwise / 134299 | 120000 / 达到上限 | 100000 | 0.285033 | 24025 / 24 / 1 |

135458 用单核、无 GPU 独立验收：每份 best checkpoint 的 **31 个模型张量、93 个 optimizer 张量全部有限**，global step、scheduler 与 best / early-stopping 状态和日志一致；两份模型状态结构一致。均明确重载各自 best 后完成训练自带 test，未重新选择 checkpoint 或做新推理。

源码指纹分别 15 / 13 项，与提交时 Git `53c6faf` 和当前文件一致；数据清单、teacher、SID 和 item2row 均重读 SHA256 核验，未重扫 TFRecords。Listwise 与原取消 run 的配置仅 5 项恢复 / CPU 相关差异；pointwise 与 seed 43 仅 seed / id / tags / teacher / train worker 五项差异。三 rank seed=44、world_size=3、无作业级重启，所有 CSV 非空数值有限。Listwise 有 1 次、pointwise 有 2 次已恢复的 DataLoader timeout retry，原日志保留，不写成完全没有警告。

两份固定 best 各 5,701,613,500 bytes：

- [Listwise best 80000](../logs/train/runs/2026-09-16/click-listwise-seed44-134298/checkpoints/checkpoint_000_080000.ckpt)，SHA256 `494287038e220b5032b331dd887c28a7519a31f980e3b349846cea60eab93c9e`。
- [Pointwise best 100000](../logs/train/runs/2026-09-17/click-pointwise-seed44-134299/checkpoints/checkpoint_000_100000.ckpt)，SHA256 `122ad80cc9506b3f677b77406ed7ac7ff12be104c0cbaf8ed255450a61b46d20`。

原 133873 step 20000 及正式续训的初始副本哈希均未改变，完整恢复与 CPU 审计通过；数据游标未保存，仍不宣称与未中断训练逐步等价。以下仅为 runtime beta=0.1、test 上限每 rank 10000 batches 的日志结果，不能并入全量主表或直接比较 seed 42 的全量校准结果：

| 模型 / 融合 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Listwise | 0.283794 | 0.288222 | 0.259267 | 0.260710 |
| Pointwise | 0.279875 | 0.285096 | 0.257382 | 0.259082 |

证据：[机器可读验收](../logs/training_launch/seed44_artifact_acceptance_2026-09-21.json)、[只读验收脚本](../logs/training_launch/accept_seed44_artifacts_2026-09-21.py)、[输入与源码绑定](../logs/training_launch/seed44_artifact_audit_input_2026-09-21.json)、[验收启动 / 失败追溯](../logs/training_launch/seed44_artifact_audit_launch_2026-09-21.json)。首次 135457 因独立脚本的 Python 导入路径缺失而失败；补齐后 135458 完成，未更改训练代码或产物，未执行完整 ZIP CRC 检查。

## 历史：seed 44 正式续训启动（两项现已完成）

短测 134290 验收通过后，09-16 10:54:28 提交 **134298 listwise**，10:54:29 开始；启动核验通过后于 10:57:37 补交 **134299 pointwise**，`afterok:134298`、`kill-on-invalid-dep=yes`。当时 10:58 队列为一项 RUNNING、一项 PENDING / Dependency；这是历史启动快照，不是当前状态。

两项提交源码为 `53c6faf409819e24aa3b80ab17ff5fb3f9605ddb`，各自提交时工作树干净；相对已通过短测的 `6343824` 仅更新文档。各申请 3 tasks × 1 CPU、3 GPU、60G；线程池 1，训练 / validation / test worker 每 rank 为 1 / 2 / 3。数据、模型、损失、全局 batch 384、runtime beta=0.1、120000 总步数 / scheduler 预算及原选优规则不变。

134298 由原 133873 的 `checkpoint_000_020000.ckpt` 完整恢复，不是 weights-only 重训，也不使用 smoke best 20100；新副本仅迁移 ModelCheckpoint 路径。134299 从同 seed SID teacher 132733 best 90000 独立加载初始权重并重建 optimizer，不使用 listwise 的任何输出。

正式启动验收通过：15 个源码 / launcher 指纹与短测一致；数据清单、SID / item2row 及 teacher 身份相同。实际配置对父 run 仅 5 项差异：新增审计 callback、续训路径、run ID、关闭 weights-only、train worker 8→1；验证 interval=20000 batches、limit=5000、test=true 均恢复正式值。三个 rank 的模型、optimizer、scheduler、early-stopping / best 状态逐值一致，恢复 step=20000，world_size=3；rank CPU 分别 `[0]` / `[16]` / `[17]`，所有后代线程绑定通过。首批 CSV 11 行、最近 step=20054、train/loss=13.427476，所有非空数值有限，无 traceback。此项仅为启动验收，不是新训练产物或效果验收。

追溯：[两项启动记录](../logs/training_launch/seed44_cpu_resume_2026-09-16.json)、[启动验收](../logs/training_launch/seed44_cpu_resume_startup_acceptance_2026-09-16.json)、[只读核验脚本](../logs/training_launch/accept_seed44_cpu_resume_startup_2026-09-16.py)、[正式 run](../logs/train/runs/2026-09-16/click-listwise-seed44-134298)、[preflight](../logs/training_preflight/listwise_seed44_134298.json)。计算节点 run metadata 的 Git 字段为空，源码身份由提交时本地 Git 记录和实际 preflight 文件哈希绑定，不以空字段推断源码干净。数据游标未恢复，不能宣称与未中断训练逐步等价。

## 已验收：CPU 预算修复与 seed 44 完整续训短测

作者确认整项作业最多 4 核。133873 原申请 3 tasks × 8 CPU，实际 AllocCPUS=24；训练 DataLoader 每 rank 8 workers。现改为 3 tasks × 1 CPU、显式 core binding、库线程池 1、训练 worker 每 rank 1；validation / test worker 2 / 3 不变。模型、损失、全局 batch 384、seed、120000 步总训练 / 调度预算和验证选优规则不变。修复提交 `d289330`、`6343824`，提交短测时工作树干净。

133873 最后训练 CSV step=20629；保留 `checkpoint_000_020000.ckpt`，元数据确认 optimizer、scheduler、early stopping 和 best 记录存在。完整恢复不能使用原 weights-only/reset 模式；新副本只迁移 ModelCheckpoint 的输出路径，不修改父文件。数据游标未保存，降低 worker 会改变训练序列，非逐步等价恢复。

134290 仅做短测：同一正式 launcher 从 20000 到 20100，scheduler 总长度仍 120000，短验证 2 batches/rank、test=false。先执行扩展回归，再逐 rank 核对模型 / optimizer / scheduler / 选优状态及所有后代线程的 CPU affinity。短测模型与指标不进入正式主表，也不作为正式续训起点；后续正式任务另见上节。

轻量测试 65 passed；含 TensorFlow 的扩展测试在登录环境触发 Illegal instruction，移到计算节点后 **88 passed**（53.45s）。短测及所有 Slurm steps 均 COMPLETED / 0:0。总分配 3 CPU；rank 0 / 1 / 2 分别绑定 `[0]` / `[16]` / `[17]`，11 个采样点覆盖恢复、训练、验证与结束，所有后代进程的每个线程均在所属 rank 的单核内。各 rank 验证了 400 个有限 loss（梯度累积 4，共 100 optimizer steps）、2 个真实验证 batches，最终 step=20100；CPU 累计时间 / 采样间隔得到每 rank 平均约 0.75 / 0.84 / 0.84 核，不把线程数当占用核数。

模型、optimizer、scheduler 和 early-stopping / best 状态恢复逐值核对通过。父 checkpoint SHA256 为 `30e273d9b5c3323c068f4d3bdee5cd97d800c404409476029d458cc82eb3284b`，短测后重读确认未改；15 个源码指纹、父配置和数据清单核对通过。短测新 run 的初始副本已被自身 top-1 规则正常替换，不影响父 checkpoint；不能拿 smoke best 20100 续训。

证据：[启动记录](../logs/training_launch/cpu_resume_smoke_2026-09-16.json)、[机器可读验收](../logs/training_launch/cpu_resume_smoke_acceptance_2026-09-16.json)、[只读验收脚本](../logs/training_launch/accept_cpu_resume_smoke_2026-09-16.py)、[逐 rank 审计](../logs/train/runs/2026-09-16/cpu-resume-smoke-listwise-seed44-134290/cpu_resume_audit)、[stdout](../logs/slurm/GRID-CPUResume44-Smoke-134290.out)、[stderr](../logs/slurm/GRID-CPUResume44-Smoke-134290.err)。

## 历史：seed 44 首次启动（已取消）

09-15 09:59 完成 seed 43 pointwise / separate 验收，当时未提交新任务；随后作者确认继续。10:06:40 提交并启动 **133873 listwise**，10:07:02 仅追加 **133874 pointwise**，`afterok:133873`、`kill-on-invalid-dep=yes`。Seed 44 separate 尚未重新提交，新 seed 全量 validation 选权 / test 配对也未启动；不预排后续任务。

两项提交时源码 `81cf4c23123e45565b029e4e1ecda805047c9263`，工作树干净；正式 SID + rerank 联合训练，不是 SID-only 或 smoke。各用 3-GPU / 3-task、60G、BF16、全局 batch 384、max optimizer steps / scheduler 120000，原 Recall@5 早停与 runtime beta=0.1 不变。独立加载同一个 seed 44 SID teacher 132733 best 90000，仅加载权重、重新初始化 optimizer；pointwise 不从前一个 listwise 的输出初始化。

提交前重新核验 9 个不同源码 / 配置 / launcher、3 份数据清单和 teacher 文件的 SHA256，13 项均匹配已验收记录；SID / item2row 当前 size / mtime 一致，不重扫 TFRecords。未改数据、损失、模型或预算；共享文件系统当时可用约 586G，不是本任务独占配额。完整命令、来源哈希和新旧 job 替代关系见 [seed 44 启动记录](../logs/training_launch/seed44_joint_2026-09-15.json)。

133873 启动核验通过：fresh preflight 的 7 个源码文件、数据 / SID / 映射身份及初始化路径与预期一致，teacher SHA256 为 `3f8b60dad3172ffc666b71d5ad0708e348363c44589d31f9d6eb3639c8d85da4`。实际配置相对 seed 43 listwise 仅 `seed`、`id`、`tags`、`model.manual_ckpt_path` 四项变化；三个 rank seed 均为 44，成功加载 teacher 并重置到 step 0，world_size=3、current_run=0、restarts=[]。10:10 CSV 首批 3 行、最近 step=14，train/loss=14.133067，全部非空数值有限，无致命日志错误。这只是启动核验，不是训练产物或效果验收。

133873：[run](../logs/train/runs/2026-09-15/click-listwise-seed44-133873)、[配置](../logs/train/runs/2026-09-15/click-listwise-seed44-133873/.hydra/config.yaml)、[preflight](../logs/training_preflight/listwise_seed44_133873.json)（SHA256 `ae5633e1beb93b68a4bf9c0882155be1a7e1da62ee492898d67f42be7e54e30e`）、[CSV](../logs/train/runs/2026-09-15/click-listwise-seed44-133873/csv/metrics.csv)、[stderr](../logs/slurm/GRID-List44-HLastV1-133873.err)。

133874 在开始前取消，没有 run / preflight。Seed 43 四模型训练已完成，原验收证据见下文；上述 09-15 启动核验不是当前队列状态。

调度沿革：09-11 以源码 `19a4fcbda4ce36f0983230bed170ce7b00dcd9b4`、干净工作树提交下游；缩小队列后，pointwise 132793 替代旧 132787。09-13 listwise 验收后，仅补交 separate 133277 替代旧 132788，`afterok:132793`、`kill-on-invalid-dep=yes`；该次提交源码 `1c47d01c598c77a4f8072a02b6e50eb621adc1e8`、工作树干净。原 [09-11 启动快照](../logs/training_launch/multiseed_downstream_2026-09-11.json) 与 [separate 启动记录](../logs/training_launch/seed43_separate_2026-09-13.json) 保留，不当作实时队列。Separate 实际于 09-14 16:30 启动，训练耗时不含依赖满足后的调度等待；此处不推断等待原因。

## 已验收：seed 43 pointwise / separate 训练产物

两个主作业及全部 Slurm steps 均 COMPLETED / 0:0；stderr 均明确重载对应 best checkpoint 后测试。配置、原选优预算与训练 beta=0.1 不变，没有 checkpoint 重选、重训或新 GPU 推理。

| 模型 / job | 停止 step / 原因 | Best step | Best validation Recall@5 | CSV 行数 / validation 次数 / test 次数 |
| --- | --- | ---: | ---: | --- |
| Pointwise / 132793 | 120000 / 达到上限，best 后仅 4 次未改善 | 100000 | 0.286196 | 24025 / 24 / 1 |
| Separate / 133277 | 35000 / 连续 5 次未改善，正常早停 | 10000 | 0.276433 | 7008 / 7 / 1 |

- 两者与各自 seed 42 实际配置逐字段对比，仅 seed / id / tags / 同 seed teacher 路径四项不同；三 rank seed 均为 43，world_size=3、current_run=0、restarts=[]。Pointwise 是 SID + pointwise 联合训练、仅加载初始权重并使用新 optimizer；separate 使用冻结 teacher 的独立小型 reranker 实现。
- 两者初始化均绑定 SID teacher 132732 best 100000，不使用 listwise / pointwise 输出或 seed 42 teacher。全部预检源码哈希与各自启动记录一致；本次重新计算各 8 个源码 / 配置 / launcher、3 份数据清单及共同 teacher 哈希，均匹配。SID / item2row 身份与 seed 42 preflight 一致，当前 size / mtime 相符；未重扫 TFRecords 或重读 SID / 映射全量哈希。
- 所有非空 CSV 数值有限，无 traceback / OOM / CUDA / NCCL 致命错误或异常重启。Separate 的 `val/beam_log_prob` / `val/click_log_prob` 在全部 7 次验证中都有有限值，未复现旧版辅助日志 NaN。
- 两份 checkpoint 的 ZIP 目录可读、全文件 SHA256 已登记。未另做逐张量有限性、separate 冻结参数逐值核对或完整 ZIP CRC 检查；运行实现和训练重载证据通过，不将它们冒称为独立参数验收。

训练自带 test 的结果如下，0–1 标度；validation / test 仍分别有每 rank 5000 / 10000 batches 上限。融合为 runtime beta=0.1，不是全量 validation 选权后的主表，不能直接与 seed 42 全量 beta=0.50 结果比较。

| 模型 / 排序方式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Pointwise / beam | 0.279478 | 0.285149 | 0.256374 | 0.258215 |
| Pointwise / 融合 | 0.281080 | 0.286608 | 0.258004 | 0.259801 |
| Separate / beam | 0.269751 | 0.273784 | 0.251623 | 0.252930 |
| Separate / 融合 | 0.271549 | 0.275466 | 0.252979 | 0.254254 |

Separate 的四项 test beam 汇总数值与 teacher 132732 的训练自带 test 完全一致；这不等于逐样本候选 / 分数或冻结参数逐值核对。两份 test 的 CSV step=120000 / 35000 是停止时记账，实际测试的是 best 100000 / 10000。

- Pointwise：[best checkpoint](../logs/train/runs/2026-09-13/click-pointwise-seed43-132793/checkpoints/checkpoint_000_100000.ckpt)，5,701,613,500 bytes、137 个 ZIP 成员，SHA256 `5b7d4e9721c273339da97de6158ca31347698377d8fe1f71156317f8447b0f02`；[配置](../logs/train/runs/2026-09-13/click-pointwise-seed43-132793/.hydra/config.yaml)、[CSV](../logs/train/runs/2026-09-13/click-pointwise-seed43-132793/csv/metrics.csv)、[preflight](../logs/training_preflight/pointwise_seed43_132793.json)、[stderr](../logs/slurm/GRID-Point43-HLastV1-132793.err)。
- Separate：[best checkpoint](../logs/train/runs/2026-09-14/rerank-small-seed43-133277/checkpoints/checkpoint_000_010000.ckpt)，6,128,885,734 bytes、122 个 ZIP 成员，SHA256 `4e5751819d402f366346499ae1e2479611bfbea4110adae0e2cd80000c1b270b`；[配置](../logs/train/runs/2026-09-14/rerank-small-seed43-133277/.hydra/config.yaml)、[CSV](../logs/train/runs/2026-09-14/rerank-small-seed43-133277/csv/metrics.csv)、[preflight](../logs/training_preflight/separate_small_seed43_133277.json)、[stderr](../logs/slurm/GRID-Sep43-HLastV1-133277.err)。

完整精度指标、Slurm 状态及来源哈希见 [机器可读验收](../logs/training_launch/pointwise_separate_seed43_acceptance_2026-09-15.json)；[只读核验脚本](../logs/training_launch/accept_pointwise_separate_seed43_2026-09-15.py) 仅输出 JSON，不加载张量、覆盖旧产物或提交任务。

## 已验收：seed 43 listwise 132786 训练产物

主作业及全部 Slurm steps 均 COMPLETED / 0:0；09-13 01:01:21 结束，耗时 33h30m40s。达到 max optimizer steps 120000，不是早停；best 100000 后仅 4 次未改善，尚未达到 patience=5。stderr 明确重载 best checkpoint 后执行训练自带 test，测试 CSV step=120000 是停止时记账，不是用了最终权重。

- 正式 SID + listwise 联合训练，seed 43，3 rank；从 teacher 132732 best 100000 仅加载权重，fresh optimizer，非断点续训。实际配置相对 seed 42 listwise 仅 seed / id / tags / 初始化路径四项变化；三个 rank 的 seed、加载路径与无重启状态一致。
- CSV 共 24025 行，24 次 validation、1 次 test；所有非空数值有限，无 traceback / OOM / CUDA / NCCL 致命错误。数据遍历结束提示和已有只读 NumPy 警告不误报为崩溃。
- Preflight 数据 / SID / 映射身份与 seed 42 一致；本次重新计算 7 个源码文件、3 个数据清单及初始化 teacher 的 SHA256，均匹配。SID / item2row 另核对当前 size / mtime，未重读其全量哈希或重扫 TFRecords。
- Best 文件 5,701,613,500 bytes，ZIP 目录 137 成员可读；SHA256 `dea447e38c839d457ec80ef4bc9134226954f238968fe364dc781371380f2467`。本次未另做逐张量有限性 / ZIP 完整 CRC 检查或独立 GPU 导出，训练重载测试证据已通过。

下表为训练日志、0–1 标度；validation 每 rank 5000 batches、test 每 rank 10000 batches 上限，融合仍为训练 runtime beta=0.1。不是全量选权或全量 test 主表，不直接与 seed 42 全量 beta=0.50 结果比较。

| Split / 排序方式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Best validation / beam | 0.285958 | 0.291758 | 0.261787 | 0.263665 |
| Best validation / 融合 | 0.288052 | 0.293471 | 0.263657 | 0.265417 |
| Test / beam | 0.281046 | 0.286633 | 0.257415 | 0.259227 |
| Test / 融合 | 0.283060 | 0.288332 | 0.259214 | 0.260927 |

证据：[best checkpoint](../logs/train/runs/2026-09-11/click-listwise-seed43-132786/checkpoints/checkpoint_000_100000.ckpt)、[配置](../logs/train/runs/2026-09-11/click-listwise-seed43-132786/.hydra/config.yaml)、[CSV](../logs/train/runs/2026-09-11/click-listwise-seed43-132786/csv/metrics.csv)、[preflight](../logs/training_preflight/listwise_seed43_132786.json)、[stderr](../logs/slurm/GRID-List43-HLastV1-132786.err)。[机器可读验收](../logs/training_launch/listwise_seed43_acceptance_2026-09-13.json) 同时保存 pointwise 阶段快照、Slurm 状态与哈希；[只读核验脚本](../logs/training_launch/accept_listwise_seed43_2026-09-13.py) 仅输出 JSON，不加载张量或覆盖 run。

## 已完成：匹配多 seed 的 SID teacher 43 / 44

固定训练 seed 为 42 / 43 / 44；seed 42 全量结果保留，09-10 提交新增 teacher 43 → 44。提交前工作树干净，源码提交 `91fa232aea1397967195139a94a2c7f7600505d9`。两个任务各用 aias-compute-01 上 3 GPU、3 tasks × 8 CPU、60G 内存，max optimizer steps / scheduler 120000、BF16、全局 batch 384、原 validation Recall@5 早停规则不变。完整命令、依赖与后续匹配原则见 [启动记录](../logs/training_launch/multiseed_teachers_2026-09-10.json)。

启动前仅为 SID 与 separate launcher 添加 seed 参数；listwise / pointwise 原已支持。默认仍为 42，run ID、tags、preflight 文件名和 Hydra seed 一致。新增 43 项无 GPU 的 Bash 调用测试：修改前 16 failed / 27 passed，修改后 43 passed；四份 launcher 语法检查通过，新旧默认 seed 42 的实际 preflight / training 参数逐字一致。模型及核心配置未改；新任务使用当前已修复的空 worker 保护，不改写旧 teacher 的 preflight。

09-11 检查：两项主作业及所有 Slurm steps 均 COMPLETED / 0:0，串行总耗时 22h44m53s。两份实际运行配置与 seed 42 逐字段比较，仅 `seed`、`id`、`tags` 不同；每个任务的三个 rank 均使用对应 seed，world_size=3、current_run=0、restarts=[]，从头初始化而非加载旧 teacher。全部非空 CSV 数值有限；未检出 traceback、OOM、CUDA / NCCL 致命错误或空累加器警告。`OUT_OF_RANGE: End of sequence` 出现在正常数据遍历结束处，作业随后继续执行并正常结束，不是训练崩溃。

| Seed | 停止 step / 原因 | Best optimizer step | Best validation Recall@5 | CSV 行数 / validation 次数 / test 次数 |
| --- | --- | ---: | ---: | --- |
| 43 | 120000 / 达到 max steps，best 之后仅 4 次未改善 | 100000 | 0.274226 | 24025 / 24 / 1 |
| 44 | 115000 / 连续 5 次未改善，正常早停 | 90000 | 0.270626 | 23024 / 23 / 1 |

两个 best checkpoint 均存在，各 5,701,604,302 bytes，ZIP 目录可读（134 个成员）；本次登记 SHA256。训练 stderr 明确记录重新加载下列 best 后测试，并正常退出。本轮未另做逐张量有限性检查、完整 ZIP CRC 扫描或独立 GPU 推理；训练文件核验通过不等于新 seed 全量评估完成。

- Seed 43：[best checkpoint](../logs/train/runs/2026-09-10/sid-seed43-132732/checkpoints/checkpoint_000_100000.ckpt)，SHA256 `6514f4a6af6811239f7d0ec66032eaaa8283c79f41f91c223b7657175df9e446`；[配置](../logs/train/runs/2026-09-10/sid-seed43-132732/.hydra/config.yaml)、[CSV](../logs/train/runs/2026-09-10/sid-seed43-132732/csv/metrics.csv)、[stderr](../logs/slurm/GRID-SID43-HLastV1-132732.err)、[preflight](../logs/training_preflight/sid_seed43_132732.json)。
- Seed 44：[best checkpoint](../logs/train/runs/2026-09-11/sid-seed44-132733/checkpoints/checkpoint_000_090000.ckpt)，SHA256 `3f8b60dad3172ffc666b71d5ad0708e348363c44589d31f9d6eb3639c8d85da4`；[配置](../logs/train/runs/2026-09-11/sid-seed44-132733/.hydra/config.yaml)、[CSV](../logs/train/runs/2026-09-11/sid-seed44-132733/csv/metrics.csv)、[stderr](../logs/slurm/GRID-SID44-HLastV1-132733.err)、[preflight](../logs/training_preflight/sid_seed44_132733.json)。

两份 preflight 均通过，数据计数、SID / item2row 身份及三个清单指纹与 seed 42 完全一致，initialization checkpoint 为 null；当前重新计算的全部预检源码与清单 SHA256 匹配，未重扫全部 TFRecords。完整指纹及本次核验范围见 [机器可读状态记录](../logs/training_launch/multiseed_teachers_status_2026-09-11.json)。

训练自带 test 的 beam-only 指标如下（0–1 标度，有每 rank 10000 batches 上限；不是全量主表）。SID 的 rerank head 未训练，不使用其辅助 rerank 指标作为方法结果。

| Seed | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 43 | 0.269751 | 0.273784 | 0.251623 | 0.252930 |
| 44 | 0.265751 | 0.270093 | 0.249298 | 0.250707 |

Validation 的 CSV step=99999 / 89999 对应保存时 optimizer step=100000 / 90000；test 行的 step=120000 / 115000 是停止时记账，不代表测试了最后权重。上述 validation 仍有每 rank 5000 batches 上限，不与全量选权结果混用。

15:22 的 teacher 核验阶段只检查与记录；随后下游提交及队列调整见前节，新增 seed 的评估仍未提交。该 seed 的 listwise / pointwise 初始化、separate 冻结绑定同一个已核验 best teacher，不使用 seed 42 权重替代；各自遵循同一 validation 网格选权，不直接复制 seed 42 的 beta=0.50，也不根据 test 扩网格。

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
