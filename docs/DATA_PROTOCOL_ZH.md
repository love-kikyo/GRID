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

重要纠正：train/validation 互斥不等于 train/test 互斥。A 保留官方 train/test 的用户重叠；历史上游审计得到原始 train/test 重叠用户 194,382，不能直接把这个数字当成 L2 过滤后新版本的重叠数。当前全量审计没有检查 train/test 互斥，过滤后的精确交集仍需单独统计。不要声称“三个 split 用户完全互斥”或“冷启动泛化”。历史计数来源见 [旧审计记录](../../WSDM_2027/doc/archive/2026-09-06/TASK_STATUS_2026-09-04_ZH.md)。

## 可追溯源文件

- [上游数据 README](../../Taobao/taobao/README.md)：原始数据说明。
- [构建器](../scripts/build_taobao_next_item_dataset.py)、[审计器](../scripts/audit_taobao_next_item_dataset.py)、[训练前核验器](../scripts/verify_taobao_training_data.py)。
- [Train manifest](../data/taobao_history_last_user_v1/protocol_manifest_train.json)、[Test manifest](../data/taobao_history_last_user_v1/protocol_manifest_test.json)、[全量审计报告](../data/taobao_history_last_user_v1/protocol_audit.json)。
- [131908 preflight](../logs/training_preflight/sid_seed42_131908.json)：绑定 manifest、item-to-row 映射、SID 数组及源码哈希。

旧 `data/taobao` 的 TFRecord 在已点击样本上曾提前删除历史末项，导致目标错位；旧 `Taobao/grid_tfrecord` 即使序列行为正确，也仍是曝光行粒度。两者都不能替代本协议。SID 映射数组仍位于旧目录的 `semantic_id/` 下，这只是静态编码资产路径，不表示使用了旧 TFRecord。

本任务是从历史构造的 next-item recommendation，不是原始曝光点击率预测。旧数据上的指标与融合权重只能用于历史排错，不能作为本版本论文结果。
