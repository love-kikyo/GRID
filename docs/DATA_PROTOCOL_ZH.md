# 冻结数据协议：history-last-user-v1

作者选择：A + H1 + L2。本页只记录已经采用的规则；改变规则需另起数据版本，不能覆盖当前数据。

## 样本如何构造

| 规则 | 精确定义 |
| --- | --- |
| A：样本单位 | 每个官方 split 内每用户一条。按确定的分片 / 行顺序保留首次出现的用户记录，跳过其余曝光记录 |
| H1：不可映射 item | 先移除没有 SID item-to-row 映射的历史 item；保持剩余历史的原有顺序，最多保留尾部 1000 项 |
| L2：最短序列 | 过滤后至少 2 项（含目标），否则丢弃该用户样本 |
| Ground truth | 保留序列的最后一个 item；模型输入只取此前的历史，再按模型配置处理上下文 |
| 曝光 target / click | 不读取、不参与样本选择、不参与监督；无论是否点击都使用同一规则 |
| Validation | 从官方 train 按 MD5(user_id) 稳定划分约 95% / 5% |
| Test | 保留官方 test 的来源边界，独立应用相同规则；不把它改成用户冷启动测试 |

转换器只读取上游 `129_1`（user ID）、`150_2_180`（历史序列），不读取曝光 item `205` 或 click `label_0`。不额外按时间戳重排历史，也不从输入历史中删除与目标同 ID 的较早交互。

TFRecord 仅含 `user_id`、`sequence_data`。后者仍包含最后的目标项，目标拆分由模型数据处理链完成。构建上限 1000 与模型实际使用的历史长度是两层不同设置。

## 正式版本与审计范围

数据根目录：[data/taobao_history_last_user_v1](../data/taobao_history_last_user_v1)。

| Split | 实际子目录 | 样本 / split 内唯一用户 | 文件数 |
| --- | --- | --- | --- |
| Train | `training` | 6,573,674 | 33 |
| Validation | `evaluation` | 347,416 | 2 |
| Test | `testing` | 2,061,101 | 11 |

全量构建 / 审计任务为 131857，审计 `status=passed`、`partial_run=false`。检查包括字段一致、长度至少 2、各目的 split 内用户唯一、train/validation 用户互斥、manifest 一致；记录的 item row 范围为 0–35,458,466。

重要纠正：train/validation 互斥不等于 train/test 互斥。A 保留官方 train/test 的用户重叠；不要声称“三个 split 用户完全互斥”或“冷启动泛化”。

## 2026-09-07 跨 split 边界核验

任务 132048 对全部冻结记录完成精确数组比较，重新核对 split 内用户唯一及 train/validation 互斥。不是抽样结论，也不是只比较哈希。原始 [边界报告](../logs/analysis/split_boundary/boundary_132048.json) 绑定 teacher 配置、SID 映射和原 preflight；未修改数据或原审计报告。

| Test 与哪个 split 比较（同用户） | 用户交集 | 完整历史 + 目标相同 | 生成器输入 + 目标 SID 相同 |
| --- | --- | --- | --- |
| Train | 184,608 | 2,748 | 2,978 |
| Validation | 9,677 | 141 | 155 |
| 合计（两来源用户互斥） | 194,285 | 2,889 | 3,133 |

“生成器视角”由 131908 配置确定：32 SID tokens / 4 层，即尾部 8 个 item，含 7 个历史 item 和 1 个目标；比较时计入 SID 映射碰撞。完整记录重复占 test 的 0.140168%，生成器视角重复占 0.152006%。上游原始交集 194,382 是过滤前历史数字，不能替代这里的 194,285。

训练决策：train/validation 互斥，三种方法共用相同冻结数据和已验收 teacher，现阶段无需重建训练集；正式 seed 42 重训已获授权。以上 test 重复必须在评估中处理，不能声称当前 test 无重复或无泄漏。

评估要求（2026-09-09 作者更新）：最终只报告所有方法共用的去重复 test，不再要求同时报告原始全量 test 的指标。原始数据 / manifest / 历史日志只保留作追溯，不覆盖删除。排除规则仍为：test 用户在 train 或 validation 中具有相同的尾部 8-item SID 输入与目标；仅由数据决定，不看预测或模型优劣。应构建并固定完整的 3,133 用户排除名单 / 哈希，再用于各方法；预计保留 2,057,968 条，以最终名单核验为准。现有计数报告只含少量例子，完整名单尚未生成。不能凭少量例子充当完整掩码，也不能把该规则称为已消除所有类型的泄漏。

复现交付：去重实现及处理规则稳定后，在本页统一整理“上游来源与版本 → 字段提取 / SID 映射依赖 → A + H1 + L2 → train/validation 划分 → test 排除名单 → 最终评估输入”的完整处理链，提供可执行命令、环境依赖、源码版本、文件哈希、各步保留 / 排除计数及核验方法。去重后的评估版本单独命名并绑定原训练数据，不改写训练时的历史记录。

审计边界：未检查跨不同用户的相同序列；数据不含事件时间戳，历史中出现同一 item 不等于已证明未来信息泄漏。生成器视角相同也不等于 candidate-aware reranker 的完整输入相同。

## 可追溯源文件

- [上游数据 README](../../Taobao/taobao/README.md)：原始数据说明。
- [构建器](../scripts/build_taobao_next_item_dataset.py)、[审计器](../scripts/audit_taobao_next_item_dataset.py)、[训练前核验器](../scripts/verify_taobao_training_data.py)。
- [Train manifest](../data/taobao_history_last_user_v1/protocol_manifest_train.json)、[Test manifest](../data/taobao_history_last_user_v1/protocol_manifest_test.json)、[全量审计报告](../data/taobao_history_last_user_v1/protocol_audit.json)。
- [131908 preflight](../logs/training_preflight/sid_seed42_131908.json)：绑定 manifest、item-to-row 映射、SID 数组及源码哈希。

旧 `data/taobao` 的 TFRecord 在已点击样本上曾提前删除历史末项，导致目标错位；旧 `Taobao/grid_tfrecord` 即使序列行为正确，也仍是曝光行粒度。两者都不能替代本协议。SID 映射数组仍位于旧目录的 `semantic_id/` 下，这只是静态编码资产路径，不表示使用了旧 TFRecord。

本任务是从历史构造的 next-item recommendation，不是原始曝光点击率预测。旧数据上的指标与融合权重只能用于历史排错，不能作为本版本论文结果。
