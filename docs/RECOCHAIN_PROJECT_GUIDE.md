# 代码与模型说明

本文只解释稳定的实现结构。任务进度见 [当前状态](CURRENT_STATUS_ZH.md)，数字见 [Run 台账](RUN_LEDGER_ZH.md)，原始 GRID 使用说明见 [仓库 README](../README.md)。

## 模型名称不能混用

| 名称 | 实现 / 训练含义 |
| --- | --- |
| SID-only / generator-only / teacher | 修改版 TIGER-style decoder-only LLaMA；`training_task=sid`，训练生成任务，冻结 click head |
| Joint listwise / RecoChain-listwise | `training_task=click`、`rerank_objective=listwise`；SID 损失与候选集合上的重排损失联合训练 |
| Joint pointwise | 同一联合框架，将重排目标替换为 pointwise；用于目标函数对照 |
| Separate-small | 冻结 SID teacher，另训 candidate-aware 小型重排器；不是 joint 模型的别名 |
| Smoke / controls tests | 验证代码或少量步骤；不构成训练完成的模型 |

`training_task=click` 是历史命名，当前标签来自 next-item GT 与候选匹配，不是上游曝光 click label。配置里出现 `rerank_objective=listwise` 也不保证启用了该损失，必须同时看 `training_task`。

## 数据与计算路径

1. TFRecord 的 `sequence_data` 经 SID 映射 / collate 拆出目标和输入历史。
2. Decoder 根据历史自回归生成 SID beam 候选，并保留可复用的 KV cache。
3. Joint 模型结合候选相关历史给候选打分；同 checkpoint 的 beam-only / rerank 比较共享自然候选。
4. Listwise 训练考虑候选竞争；训练的 no-hit 辅助列表可插入 GT，正式推理不允许注入 GT。
5. Beam 分数与重排分数可融合，权重只由 validation 选取。

离散 top-k / beam selection 不能被描述成完全可微；KV-cache 的作用是减少重复编码，不能由此直接推断比所有独立 reranker 更快。实际损失与冻结行为以实现和真实训练检查为准。

## 代码地图

| 负责什么 | 入口 |
| --- | --- |
| 数据构建 / 审计 | [build](../scripts/build_taobao_next_item_dataset.py)、[audit](../scripts/audit_taobao_next_item_dataset.py)、[训练 preflight](../scripts/verify_taobao_training_data.py) |
| 跨 split 用户 / 样本边界 | [boundary audit](../scripts/audit_taobao_split_boundary.py)；只读冻结数据，结果与处理规则见 [数据协议](DATA_PROTOCOL_ZH.md) |
| 文件 / rank / worker 分配 | [dataloading.py](../src/data/loading/components/dataloading.py) |
| 输入、目标与用户 ID 组织 | [collate_functions.py](../src/data/loading/components/collate_functions.py)、[pre_processing.py](../src/data/loading/components/pre_processing.py) |
| SID 与 joint 模型 | [tiger_generation_model.py](../src/models/modules/semantic_id/tiger_generation_model.py) |
| 独立重排模型 | [separate_rerank_model.py](../src/models/modules/semantic_id/separate_rerank_model.py) |
| 逐样本导出 / 配对统计 | [export](../scripts/export_paired_evaluation.py)、[analysis](../scripts/analyze_paired_evaluation.py) |
| 效率 / 历史出现分析 | [profile](../scripts/profile_rerank_efficiency.py)、[history occurrence](../scripts/analyze_history_occurrence.py) |

## 当前协议的操作入口

先看脚本 Usage 和 [验收计划](EXPERIMENT_PLAN_ZH.md)，不要直接复制归档里的旧 checkpoint 命令。新正式任务必须显式绑定 teacher、数据与代码版本。

旧协议的 12 个根目录入口已移入 [历史脚本目录](../scripts/legacy/2026-09-15/README.md)，不属于下表的现用入口；产物去向见 [清理记录](archive/2026-09-15/REPOSITORY_CLEANUP_ZH.md)。

| 用途 | 脚本 |
| --- | --- |
| 综合测试 | [run_test_retraining_controls.sh](../run_test_retraining_controls.sh) |
| 下游 teacher 初始化 / 梯度 / DDP 两步验收 | [staged smoke](../run_smoke_staged_history_last_user_v1.sh)；逐 rank 检查，不保存正式 checkpoint |
| SID 3-GPU smoke / 正式训练 | [smoke](../run_smoke_sid_ddp_history_last_user_v1.sh)、[train](../run_sid_history_last_user_v1.sh) |
| Joint listwise / pointwise | [listwise](../run_click_listwise_history_last_user_v1.sh)、[pointwise](../run_click_pointwise_history_last_user_v1.sh) |
| 3 核完整续训短测 | [CPU resume smoke](../run_smoke_cpu_resume.sh)；100 optimizer steps + 短验证，不构成正式结果 |
| Separate-small | [separate](../run_rerank_small_history_last_user_v1.sh) |
| 逐样本导出 / 配对分析 | [export](../run_export_paired_evaluation.sh)、[paired analysis](../run_paired_analysis.sh) |
| 历史出现分析 | [history occurrence](../run_history_occurrence.sh) |

环境沿用 `/home/MMReco2021/.conda/envs/GRID/bin/python`。训练与较重测试在 Slurm 计算节点执行，不在登录节点直接跑训练。正式 launcher 使用 3 GPU DDP；需检查实际 world size，不能仅以申请了 3 GPU 判断用满三卡。

四个正式训练入口及 CPU resume smoke 使用整项作业 3 核（上限 4 核）。逐样本导出入口现使用 1 task / 4 CPU / 1 GPU，显式 core binding、库线程池 1、数据 worker 1，并把首次 / 结束时的进程树 CPU 绑定写入 sidecar；不改 batch=32、BF16 或打分公式。导出 sidecar 另绑定实际 iterator 和 CPU 控制源码，旧训练 preflight 未覆盖这些文件的范围仍按原记录保留，不假称旧 preflight 覆盖了全部依赖。其他旧 smoke / 分析入口运行前仍须调整并核验 CPU 预算。登录环境不支持当前 TensorFlow 二进制所需的指令，依赖 TensorFlow 的测试放在计算节点执行。

Listwise 可传第三个参数 `RESUME_CKPT`：先在新 run 内准备完整续训副本，仅迁移 best-checkpoint 路径；保留 optimizer / scheduler / early stopping，关闭 weights-only reset。父 run 和原 checkpoint 不覆盖、不删除。恢复时逐 rank 对比真实状态，训练结束后测试新选出的 best，而非最初的 resume 输入。数据读取游标未保存，worker 改变也影响训练顺序，不宣称与未中断训练逐步完全一致。

导出器默认要求源码指纹与训练 preflight 一致。只有经过审查的精确哈希差异才可显式传入 `SOURCE_COMPATIBILITY=/abs/record.json`；兼容记录必须绑定原 preflight，不能作为通用跳过开关。131908 的特定记录与真实验收见 [Run 台账](RUN_LEDGER_ZH.md)。
