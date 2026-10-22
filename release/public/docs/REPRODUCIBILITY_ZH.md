# 四模型复现协议

范围：固定资产上的 `history-last-user-v1` 四模型受控比较。本文记录已采用的协议和代码入口，不表示该源码草稿已经在新机器上完整复现。原始数据和训练资产不随包分发。

## 1. 数据来源与样本构造

上游数据集标识为 `TaoBao-MM/Taobao-MM`。使用其已拼接的 Parquet shards，不重新 join 原始特征表。本地下载记录绑定 revision `f185841b6e9612edd06f5112e06416f1bdf6c6d0`；历史构建实际读取 161 个 train、49 个 test shards，不能用上游汇总 metadata 中的 48 替代后者。这是已审计的本地快照身份，不是本次远端可访问性承诺。

处理顺序不可互换：

1. 按 split 内分片名字典序及原行序读取，仅取 `129_1`（user ID）、`150_2_180`（历史）。不读取曝光 item `205` 或 click `label_0`。
2. **A**：每个官方 split 内只取首次出现的用户记录；立刻标记 seen，即使首条历史太短也不从后续曝光补选。
3. **H1**：移除 item-to-row 映射中不存在的历史项，保持原顺序，取最后至多 1000 项。
4. **L2**：剩余序列至少 2 项。最后一项是 GT，此前项为输入历史；不按时间戳重排，不删除历史中较早的同 ID 项。
5. 官方 train 内以 `int(md5(str(uid).encode()).hexdigest()[:8], 16) % 10000 < 500` 划出 validation。官方 test 独立处理，不改为用户冷启动评测。
6. 保存 GZIP TFRecord：`user_id`、`sequence_data`。序列为包含 GT 的 item 行号，模型预处理阶段才拆分目标。

| Split | 目录 | 样本数 | shards |
| --- | --- | ---: | ---: |
| Train | `training` | 6,573,674 | 33 |
| Validation | `evaluation` | 347,416 | 2 |
| Test | `testing` | 2,061,101 | 11 |

构建计数：官方 train 76,015,123 曝光行 → 6,929,606 首次用户 → 丢弃 8,516 条短序列 → 6,921,090 条再划分；官方 test 22,979,465 行 → 2,063,682 首次用户 → 丢弃 2,581 条 → 2,061,101 条。过滤不可映射 item 的次数分别为 748,072,812 / 220,880,000，仅计首次保留用户行。

**没有跨 split 的内容去重。** Train/validation 用户 ID 互斥，但不保证模型输入内容互斥。对同 user ID 的旧全量审计发现 test 中 2,889 条完整历史+GT 与 train/validation 相同，3,133 条在生成器窗口+GT SID 上相同；跨不同 user ID 的内容重复尚未全量核验。保留全量 test 做共同协议下比较，不声称“无泄漏”或重叠对所有模型影响相同。

## 2. 固定编码资产与构建入口

资产必须成对核验，不能只比文件名：

| 资产 | SHA256 |
| --- | --- |
| 原 item ID → SID 行号映射（item2row） | `ad40ae0e1d2ebf7f149ef12b4c9bd600291b34e2acae23843b2376256bade1d9` |
| SID 数组，int32 `[35458467, 4]` | `02397277e46f1ac8f0c67e0ffe293af3ef35dc433be9e7fdd9a912b18bf2cf54` |

SID 来自 SCL embedding 的历史聚类资产，三层聚类宽度为 `[512,512,512]`，第四层是桶内重编号；模型四层容量均为 4096 是另一层设置。原 codebook 训练的完整源码与随机状态没有完整绑定，**不承诺从 embedding 重新训练可逐位得到相同 SID**。固定资产下载/授权渠道未交付前，复现仍有阻塞。

下列命令是已有转换入口的参数化说明，需在兼容计算节点、受限 CPU 分配中运行。先核验上述资产哈希及上游文件清单；输出根目录必须是新目录。只加载可信来源的 pickle 映射/模型 checkpoint。

```bash
# 替换这些占位路径；不要覆盖既有实验输入。
TAOBAO_SOURCE=/path/to/taobao
ITEM_MAPPING=/path/to/item2row.pkl
SID_ARRAY=/path/to/fixed_sid.npy
REPRO_ROOT=/path/to/new/history-last-user-v1
test ! -e "$REPRO_ROOT" || exit 1
python scripts/build_taobao_next_item_dataset.py \
  --split train --data-dir "$TAOBAO_SOURCE" --item2row-path "$ITEM_MAPPING" \
  --output-root "$REPRO_ROOT" --min-sequence-length 2 --max-sequence-length 1000 \
  --eval-ratio 0.05 --samples-per-shard 200000 --batch-size 8192
python scripts/build_taobao_next_item_dataset.py \
  --split test --data-dir "$TAOBAO_SOURCE" --item2row-path "$ITEM_MAPPING" \
  --output-root "$REPRO_ROOT" --min-sequence-length 2 --max-sequence-length 1000 \
  --samples-per-shard 200000 --batch-size 8192
python scripts/audit_taobao_next_item_dataset.py \
  --data-root "$REPRO_ROOT" --expected-protocol history-last-user-v1
python scripts/verify_taobao_training_data.py \
  --data-root "$REPRO_ROOT" --semantic-id-path "$SID_ARRAY" \
  --num-hierarchies 4 --num-embeddings-per-hierarchy 4096 \
  --report-path "$REPRO_ROOT/reproduction_preflight.json"
```

此处 preflight 只说明数据检查，不替代训练的源码/初始化绑定清单。训练前应增加显式 `--source-file`、初始化 checkpoint 指纹并核验记录完整。新路径/时间戳会改变 JSON，不要求新 manifest 与旧 JSON 逐字节相同；样本、计数与固定资产身份必须一致。

## 3. 四模型与训练边界

所有配置先叠加 `tiger_train_flat`：

| 模型 | 后续配置 | 初始化与作用 |
| --- | --- | --- |
| SID | `tiger_train_sid` | 生成器从头训练，不接收 teacher 参数 |
| Listwise | `tiger_train_click` | 载入 SID 权重，联合生成与 listwise 重排训练 |
| Pointwise | `tiger_train_click,tiger_train_click_pointwise` | 载入 SID 权重，联合生成与 pointwise 重排训练 |
| Separate | `tiger_train_rerank_small` | 固定 SID teacher，训练独立重排器；不是原始 SIM |

SID 基线包含本项目的 DualHash 历史嵌入，不能标为原样 TIGER。生成器 `sequence_length=32` 个 SID tokens 对应含 GT 的 8 项，实际最多 7 项历史。重排器的 `hist_itemkey_length=100` 是最低 padding 长度，不是硬截断上限，可读更长历史。User ID 用于数据划分与配对，不作为模型特征。

当前正式实验覆盖 seeds 42/43/44，seed S 的三个下游模型使用同 seed S 已验收的 SID teacher。预览入口目前不验证 teacher 的 seed/身份，须由后续资产记录核验；换 seed 或换 teacher 是新实验，不是已有结果复现。历史训练为三卡 DDP、每卡 batch 32、梯度累积 4，有效 batch 384，scheduler horizon / max_steps 为 120,000（不代表所有 run 都训练到该上限，仍按各 run 选优与早停）。Teacher 权重初始化与优化器状态 resume 不是同一操作，不互相替代。

`scripts/reproduce_training.py` 只生成四模型参数预览。它显式设 `test=false`，将训练内有限 test 与后续完整评估分开。当前资源适配仍有 Slurm 约束：整项作业 CPU ≤4，历史三卡启动为三任务各 1 CPU、显式 affinity，所有库线程为 1。单卡参数预览的有效 batch 是 128，不能冒充等价复现，也不能关闭 CPU guard 或伪造 Slurm 环境来强行运行。

## 4. 评估顺序与可报告指标

1. 固定每个 run 的最佳 checkpoint：训练时按 `val/recall@5`、`mode=max` 选择，SID 使用 beam，三个重排模型使用训练配置中的 runtime beta=0.1 重排指标。使用 `scripts/export_paired_evaluation.py` 全量导出 validation（347,416 条），保存候选、生成分数、原始重排分数及配对身份。必须绑定 run 配置、训练 preflight、数据/SID 指纹；源文件不匹配时应失败，不能绕过校验。
2. 用 `scripts/select_score_fusion.py` 在 validation 上为各重排 run 独立选择 `生成 log-prob + beta × 原始重排 logit`（不加 sigmoid）。生成权重 1，beta 网格 0.00–0.50，步长 0.01；按 NDCG@10、NDCG@5、Recall@10、Recall@5 字典序选优，同分优先更小 beta。这一步只为已固定 checkpoint 选 beta，不按选权后的 NDCG 重新挑 checkpoint。固定后不再看 test 调参，不为 beta 重训。
3. 同一最佳 checkpoint 全量导出 test（2,061,101 条），应用已冻结选权。原 runtime beta=0.1 指标与 validation-selected 指标分开；当前九个重排 run 的已选 beta 均为 0.5，不意味着可以事后省略选权流程。
4. `scripts/analyze_paired_evaluation.py` 对齐用户/输入哈希与目标，用同一条样本的指标差计算 paired bootstrap。正式口径为 @5/@10、5,000 次 bootstrap、batch 500、seed 2027、95% CI；调用须显式传 `--bootstrap-batch-size 500`，脚本默认是 50。跨 seed 的结果另作汇总；不是只挑几个成功案例。@20 需独立验收后再报告，草稿不补造该指标。

导出参数至少包含 `--run-dir`、`--checkpoint`、`--training-preflight`、`--expected-data-root`、`--expected-protocol history-last-user-v1`、`--split val|test`、`--output`；正式导出 batch 32、worker 1，资源分配与线程限制须另经验证。不要使用上游通用 `src.inference` 配置替代此评估链。

本草稿带出评估实现与单元测试，不带包含真实用户 ID 的 HDF5。正式发布仍需提供脱敏的结果汇总及其证据映射、可执行的评估资源入口；当前不能宣称下载代码即可重现主表。统一硬件效率脚本仍在内部，待参数化与脱敏后另行验收，不在此草稿中承诺复现。
