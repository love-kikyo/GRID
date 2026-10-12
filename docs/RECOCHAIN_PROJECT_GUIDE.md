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

### Separate 的历史与 SIM 的区别

10-12 核对所有现存分支中相关文件的历史：`d713c35` 的离线候选版和 `bf38cf0` 的在线 teacher 版均采用历史均值池化；`56c6761` 将后者增强为候选相关历史注意力，并增加候选 SID 表示、交互特征和外部 beam 分数融合。当前版本还包含后续候选 / 损失与日志修复。归档称早期结果为 `DIN-style`，但完整训练日志已缺失；尚未定位到可证明“标准 SIM 重排后下降”的代码与完整 run，不能据此否定作者对未留存实验的记忆，也不能把旧协议指标作为新协议证据。

| 结构 | 当前 Separate-small | SIM-soft 的核心结构 |
| --- | --- | --- |
| 历史筛选 | 对全部有效 `hist_itemkey` 做单头点积注意力，无 Top-K 搜索阶段 | GSU 通过学习到的相关性选 Top-K 历史 |
| 历史精排 | 上述一次注意力汇聚后送入 MLP | ESU 对搜索子序列做多头目标注意力 |
| 训练 | 自定义 next-item listwise 目标及 no-hit 辅助列表 | 原文包含 GSU 与 ESU 两个预测损失；迁移任务时须说明适配 |
| 增强特征 | SID 向量及其交互、候选原始名次 embedding、beam 分数 | 不能将这些自定义增强默认为原版 SIM 的组成部分 |

原文依据：[SIM 第 3 节](https://arxiv.org/abs/2006.05639)。因此补 SIM 不等于简单关闭几个 Separate 特征。新基线计划见 [实验计划](EXPERIMENT_PLAN_ZH.md#外部基线扩展)。

输入细节必须按代码解释：当前 `sequence_length=32` 是包含 GT 的 **SID token 数**，四级编码下生成器最多看到 7 个历史物品；`hist_itemkey_length=100` 是 padding 下限，不是截断上限，独立排序器可看到保留的全部历史。Joint 的现用检索由可训练 `DualHashEmbedding` 进行点积 Top-K，不是直接检索冻结 SCL 向量；SCL 仍用于固定 SID 编码资产。论文旧描述与口头调研不得覆盖这一实现事实。

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
| 逐样本导出 / 配对统计 | [export](../scripts/export_paired_evaluation.py)、[统计核心](../scripts/analyze_paired_evaluation.py)、[四核入口](../scripts/analyze_paired_cpu4.py) |
| 效率 / 历史出现分析 | [四核同 GPU profile](../scripts/profile_efficiency_cpu4.py)、[原前向组件](../scripts/profile_rerank_efficiency.py)、[history occurrence](../scripts/analyze_history_occurrence.py) |

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
| 统一硬件效率复测 | [profile CPU4](../run_profile_efficiency_cpu4.sh)；先 smoke，再用其验收记录放行 formal |
| 历史出现分析 | [history occurrence](../run_history_occurrence.sh) |
| 已验收 test 的自然 beam 分组 | [4 CPU / 0 GPU 入口](../run_beam_cohorts_cpu4.sh)、[流式分析](../scripts/analyze_beam_cohorts_cpu4.py)；不调用模型 |

自然 beam 分组入口接受 `MANIFEST OUTPUT_JSON`，提交时绑定 `SOURCE_GIT_COMMIT` / `COHORT_MANIFEST_SHA256`。支持既定 seed42 / 43 / 44，逐模型核对冻结 Hydra 配置哈希及顶层 seed；兼容 seed42 原验收的 `verified_metrics` 字段，不重写旧记录。输入来自已验收配对报告及其验收文件；按两个模型是否各自覆盖 GT 分成四组，保存联合 GT 排名计数、组内指标与对全体差值的贡献。流式对齐 / 独立排名计数 / 原指标还原全部通过才输出 `passed`；分组是描述性的，不是共享同一候选池的反事实比较。历史长度不在当前 HDF5 中，不能由候选命中组推断历史长度。

环境沿用 `/home/MMReco2021/.conda/envs/GRID/bin/python`。训练与较重测试在 Slurm 计算节点执行，不在登录节点直接跑训练。正式 launcher 使用 3 GPU DDP；需检查实际 world size，不能仅以申请了 3 GPU 判断用满三卡。

四个正式训练入口及 CPU resume smoke 使用整项作业 3 核（上限 4 核）。逐样本导出入口现使用 1 task / 4 CPU / 1 GPU，显式 core binding、库线程池 1、数据 worker 1，并把首次 / 结束时的进程树 CPU 绑定写入 sidecar；不改 batch=32、BF16 或打分公式。导出 sidecar 另绑定实际 iterator 和 CPU 控制源码，旧训练 preflight 未覆盖这些文件的范围仍按原记录保留，不假称旧 preflight 覆盖了全部依赖。

融合选权入口 `run_select_score_fusion.sh` 使用 1 task / 4 CPU / 无 GPU，库线程池 1、显式 core binding；复用单 task 四核检查，把开始 / 结束的进程树采样与分析源码指纹写入结果，不改 validation 网格或打分 / 选优规则。直接调用选权脚本不自动启用资源检查；正式任务必须用受限 launcher。

配对分析入口 `run_paired_analysis.sh` 同样使用 1 task / 4 CPU / 无 GPU、库线程池 1、显式绑定。参数为 `OUTPUT_TAG LISTWISE_TEST LISTWISE_SELECTION POINTWISE_TEST POINTWISE_SELECTION SEPARATE_TEST SEPARATE_SELECTION SID_TEST`（数据与选权路径均为绝对路径）。读取四份已验收 full-test 和三份冻结的 validation 选权，执行与 seed 42 相同顺序的 11 项比较、5000 次 bootstrap、batch500、随机 seed2027；SID 只使用 beam。新四核包装器记录首尾进程树、源码指纹和提交端传入的 `SOURCE_GIT_COMMIT`，统计核心不改。拒绝覆盖报告或使用截断样本，正式报告仍需独立复算验收；资源 / 数值一致性回归见 `tests/test_paired_cpu_budget.py`。

效率入口参数为 `MANIFEST OUTPUT_DIR smoke|formal [SMOKE_ACCEPTANCE]`，提交环境绑定 `SOURCE_GIT_COMMIT` / `PROFILE_MANIFEST_SHA256`。固定 seed 44 的四个已验收 best，1 GPU / 整项 4 CPU、库线程 1、worker1；同一 GPU 分配内串行运行，逐模型 / 重复使用新进程，UUID 必须相同。模型与数据源码不修改；首批输入与对应正式 test HDF5 的用户 / 输入哈希核对。计时为 batch32 / beam20 / BF16，原 sequence_length=32、hist_itemkey_length=100、top_k_for_score=10；重排使用冻结 validation beta=0.50，无新选权。短测 2 warmup + 3 measured batches；正式 10 + 100 batches × 3 次重复，模型 / 变体顺序轮换。

分别记录生成、重排 CUDA 时间及同步墙钟前向时间，数据加载、H2D、checkpoint 加载与验证不计时。一次只保留一个 GPU 输入 batch；报告 allocated / reserved 峰值及相对基线增量。Separate 仅实际 teacher + reranker 放入 GPU，未使用的继承 wrapper 留在 CPU，不能与旧脚本的原始整模块显存直接比较。四模型主比较统一 BF16；严格 cache / no-cache 对照单独统一 FP32（关闭 TF32），同一自然候选 logits 须满足 atol/rtol=2e-4。首轮短测显示 BF16 no-cache 超出原容差，因此该路径不进入计时对照；原 BF16 阈值、超限个数 / 最大差和排序差异仅保留为诊断，不放宽其原阈值后混入结果。加速比只在同精度内计算，不宣称逐位等价；严格短测不通过不得进入正式阶段。3 次计时重复不等于 3 个训练 seed，也不是在线单请求延迟。实际验收情况只看台账。

其他未适配的旧 smoke 入口运行前仍须核验 CPU 预算。登录环境不支持当前 TensorFlow 二进制所需的指令，依赖 TensorFlow 的测试放在计算节点执行。

Listwise 可传第三个参数 `RESUME_CKPT`：先在新 run 内准备完整续训副本，仅迁移 best-checkpoint 路径；保留 optimizer / scheduler / early stopping，关闭 weights-only reset。父 run 和原 checkpoint 不覆盖、不删除。恢复时逐 rank 对比真实状态，训练结束后测试新选出的 best，而非最初的 resume 输入。数据读取游标未保存，worker 改变也影响训练顺序，不宣称与未中断训练逐步完全一致。

导出器默认要求源码指纹与训练 preflight 一致。只有经过审查的精确哈希差异才可显式传入 `SOURCE_COMPATIBILITY=/abs/record.json`；兼容记录必须绑定原 preflight，不能作为通用跳过开关。131908 的特定记录与真实验收见 [Run 台账](RUN_LEDGER_ZH.md)。
