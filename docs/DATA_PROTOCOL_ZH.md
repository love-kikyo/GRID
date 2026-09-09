# 冻结数据协议：history-last-user-v1

作者选择：A + H1 + L2。2026-09-09 最新决定：沿用当前冻结的 train / validation / test，不追加跨 split 内容去重或为此重训；已知重叠及审计范围如实保留。本次评估规模仍为 validation 347,416、test 2,061,101 条。

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

实现细节：按 `train-shard-*.parquet` / `test-shard-*.parquet` 文件名字典序、文件内原行序处理。用户首次出现时即记为 seen；即使该首条历史过滤后过短，也不会从后面的曝光记录补选。Validation 的精确条件是 `int(md5(str(user_id).encode()).hexdigest()[:8], 16) % 10000 < 500`，不是运行时随机抽样。

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

当时的训练决策依据是 train/validation **用户 ID** 互斥、共同冻结数据及已验收 teacher；正式 seed 42 重训因此启动并已完成。这不证明 train/validation 的模型输入内容互斥。作者现决定保留这一协议做受控比较，不为跨 split 重复追加重训；不能声称当前 test 无重复或无泄漏。

评估要求（2026-09-09 最新决定，替代此前去重方案）：四个模型在现有共同 test 的全部 2,061,101 条记录上评估，不生成排除名单、不裁剪 test、不另报去重复对照。所有方法使用同一冻结数据与预声明选权规则，做该协议下的相对比较。共同数据不保证重叠偏差在不同模型之间完全抵消，因此不据此宣称无泄漏的泛化能力或没有重叠影响。

重要更正：构建器仅跳过各官方 split 内同一用户的额外曝光记录，**没有做跨 split 的 hist + GT 内容去重**；132048 审计也只比较相同 user ID。当前模型不把 user ID 作为输入，因此“用户不同”不能排除样本重复。3,133 是旧审计范围内的重复计数，不是内容去重后的最终排除数；此前 2,057,968 的剩余数只是减去这部分后的算术值，撤回其作为最终评估规模的预期。

此前提出的全局内容审计及去重掩码不再列为本轮评估前置任务。跨不同用户的内容重复数量仍未知；这是保留的审计限制，不记作已检查通过。完整历史相同与仅末尾生成器窗口相同也不能混为一谈。

复现交付：在本页统一整理“上游来源与版本 → 字段提取 / SID 映射依赖 → A + H1 + L2 → train/validation 划分与官方 test 来源保留 → 最终评估输入”的完整处理链，提供可执行命令、环境依赖、源码版本、文件哈希、各步保留 / 排除计数及核验方法。明确本链路只有 split 内按用户去额外曝光记录，没有跨 split 内容去重；不虚构去重步骤，不改写历史记录。

审计边界：未检查跨不同用户的相同序列；数据不含事件时间戳，历史中出现同一 item 不等于已证明未来信息泄漏。生成器视角相同也不等于 candidate-aware reranker 的完整输入相同。

## 数据处理复现入口

处理链为：官方预拼接 Parquet → 仅提取 user / history → 原 item ID 映射为 SID 行号 → 上述用户保留与 H1/L2 → MD5 划分 train/validation、独立保留 test → GZIP TFRecord → SID 映射与模型输入拆分。没有额外的跨 split 去重步骤。

### 上游与固定编码资产

上游本地 [README](../../Taobao/taobao/README.md) 指向 `TaoBao-MM/Taobao-MM`。直接使用其 `train/`、`test/` 中已经拼接的 shards；README 说明它们由 raw samples、user features、item features 拼接而来，不需要再次 join。现有构建 manifest 记录实际读取 train 161 个 / test 49 个文件及其路径、大小、mtime；上游 test `metadata.json` 写 48，复现时以实际 manifest 文件列表为准，不能只信该汇总字段。

本轮固定两个相互对应的编码资产，不能仅凭同名文件替换：

| 资产 | 用途 | SHA256 |
| --- | --- | --- |
| [item2row](../../Taobao/cache/sid_item2row_scl_emb_int8_p90.pkl) | 原始历史 item ID → SID 数组行号 | `ad40ae0e1d2ebf7f149ef12b4c9bd600291b34e2acae23843b2376256bade1d9` |
| [SID 数组](../data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy) | 行号 → 4 层 SID，形状 `[35458467,4]`、int32 | `02397277e46f1ac8f0c67e0ffe293af3ef35dc433be9e7fdd9a912b18bf2cf54` |

上游提供 SCL embedding 的 keys / values；本地 [SID 生成入口](../../Taobao/src/generate_sid_taobao.py) 调用 [res_kmeans.py](../../Taobao/src/res_kmeans.py)，其 reindexing 逻辑给三层聚类编码追加桶内 item 排序编号。历史 launcher 配置为 `512 512 512`；它不是本模型四层均容量 4096 的配置，两者不能混淆。尚未把历史 SID 生成运行的全部版本 / 随机种子与当前固定数组逐项绑定，因此本轮精确复现以**分发并校验上述编码资产**为前提，不承诺重新聚类能按位生成同一 SID 数组。

同样，现有上游文件记录没有完整的下载 revision 与逐 shard SHA256；发布复现材料时应补齐所使用快照的身份，不把本地路径 / mtime 当作完整的下载版本证明。以下命令复现的是“同一上游 shards + 同一编码资产 → 处理后的样本”，不是已经完成第三方从零复现实验的声明。

### 环境与构建命令

当前环境实查：Python 3.10.19、NumPy 2.0.2、PyArrow 20.0.0、tensorflow-cpu 2.18.0；模型 / 导出使用 torch 2.6.0+cu124、lightning 2.5.0、hydra-core 1.3.2、omegaconf 2.3.0、h5py 3.14.0。[requirements.txt](../requirements.txt) 为仓库依赖清单，不保证它与后续安装过的当前环境完全相同。TensorFlow 读取和全量构建应在兼容的计算节点运行，不在登录节点重跑全量数据。

历史构建命令在 [launcher](../run_build_taobao_history_last_user_v1.sh)。下面只改为**新的输出路径**，防止覆盖现有训练输入；示例未在本次重新执行。先将路径改为实际位置，在 GRID 仓库根目录运行：

```bash
set -euo pipefail
DATA_PYTHON=/home/MMReco2021/.conda/envs/GRID/bin/python
TAOBAO_SOURCE=/mnt/beegfs/home/MMReco2021/liuyu/Taobao/taobao
ITEM_MAPPING=/mnt/beegfs/home/MMReco2021/liuyu/Taobao/cache/sid_item2row_scl_emb_int8_p90.pkl
SID_ARRAY=/mnt/beegfs/home/MMReco2021/liuyu/GRID/data/taobao/semantic_id/sid_reindexing_scl_emb_int8_p90.npy
REPRO_ROOT=/path/to/new/history-last-user-v1-reproduction
test ! -e "$REPRO_ROOT"
sha256sum "$ITEM_MAPPING" "$SID_ARRAY"

"$DATA_PYTHON" scripts/build_taobao_next_item_dataset.py \
  --split train --data-dir "$TAOBAO_SOURCE" --item2row-path "$ITEM_MAPPING" \
  --output-root "$REPRO_ROOT" --min-sequence-length 2 --max-sequence-length 1000 \
  --eval-ratio 0.05 --samples-per-shard 200000 --batch-size 8192
"$DATA_PYTHON" scripts/build_taobao_next_item_dataset.py \
  --split test --data-dir "$TAOBAO_SOURCE" --item2row-path "$ITEM_MAPPING" \
  --output-root "$REPRO_ROOT" --min-sequence-length 2 --max-sequence-length 1000 \
  --samples-per-shard 200000 --batch-size 8192
"$DATA_PYTHON" scripts/audit_taobao_next_item_dataset.py \
  --data-root "$REPRO_ROOT" --expected-protocol history-last-user-v1
"$DATA_PYTHON" scripts/verify_taobao_training_data.py \
  --data-root "$REPRO_ROOT" --semantic-id-path "$SID_ARRAY" \
  --num-hierarchies 4 --num-embeddings-per-hierarchy 4096 \
  --report-path "$REPRO_ROOT/reproduction_preflight.json"
```

构建器 / 审计器在当前源码与原始 manifest 中的哈希一致，分别为 `c55f0471869688a5af285f590db3556fe3b6c28a7fa96487d413eef58f42775c` / `55b359eb903a3368961c3937363897b502d5a7db826755b87a85e042b6d29857`。不要使用旧 `Taobao/src/build_grid_dataset.py` 代替该构建入口。

### 逐步计数与下游输入

| 原始 split | 曝光行数 | 首次出现用户数 | 跳过的额外曝光行 | 映射后不足 2 项而丢弃 | 输出样本 |
| --- | --- | --- | --- | --- | --- |
| Train | 76,015,123 | 6,929,606 | 69,085,517 | 8,516 | 6,921,090，随后分为本页 train / validation |
| Test | 22,979,465 | 2,063,682 | 20,915,783 | 2,581 | 2,061,101 |

两边空原始历史丢弃数均为 0；移除的不可映射历史 item 次数分别为 748,072,812 / 220,880,000，计数只针对每用户首次保留行。应核对样本内容、用户集合和以上计数；新目录的路径 / 时间戳会使 manifest 哈希不同，不要求新 manifest 与历史 JSON 按位相同。

TFRecord 保存的是含 GT 的行号序列；[预处理](../src/data/loading/components/pre_processing.py) 映射 SID，[collator](../src/data/loading/components/collate_functions.py) 将生成器序列左填充 / 截断到 32 个 SID tokens，模型再拆出最后 4 tokens 作为 GT，因此生成器最多看到 7 个历史 item。`hist_itemkey` 只做最低长度填充，不按 32 tokens 截断，重排器可使用更长历史。user ID 仅用于数据划分与评估配对，不是模型特征。

## 可追溯源文件

- [上游数据 README](../../Taobao/taobao/README.md)：原始数据说明。
- [构建器](../scripts/build_taobao_next_item_dataset.py)、[审计器](../scripts/audit_taobao_next_item_dataset.py)、[训练前核验器](../scripts/verify_taobao_training_data.py)。
- [Train manifest](../data/taobao_history_last_user_v1/protocol_manifest_train.json)、[Test manifest](../data/taobao_history_last_user_v1/protocol_manifest_test.json)、[全量审计报告](../data/taobao_history_last_user_v1/protocol_audit.json)。
- [131908 preflight](../logs/training_preflight/sid_seed42_131908.json)：绑定 manifest、item-to-row 映射、SID 数组及源码哈希。

旧 `data/taobao` 的 TFRecord 在已点击样本上曾提前删除历史末项，导致目标错位；旧 `Taobao/grid_tfrecord` 即使序列行为正确，也仍是曝光行粒度。两者都不能替代本协议。SID 映射数组仍位于旧目录的 `semantic_id/` 下，这只是静态编码资产路径，不表示使用了旧 TFRecord。

本任务是从历史构造的 next-item recommendation，不是原始曝光点击率预测。旧数据上的指标与融合权重只能用于历史排错，不能作为本版本论文结果。
