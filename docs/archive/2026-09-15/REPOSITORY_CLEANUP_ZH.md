# 仓库产物清理记录（2026-09-15）

范围仅为 GRID 仓库内的旧协议产物和入口；不改论文、冻结的数据规则或当前训练实现。清理前 Git 基线为 `8671f50`。作者明确选择“永久删除旧派生产物，保留追溯记录”后执行；**清理与复核已完成，旧数据和旧权重无法直接恢复。**

## 已完成

| 对象 | 判定依据 | 处理状态 |
| --- | --- | --- |
| 12 个根目录旧训练 / 评估 / 效率脚本 | 固定使用旧 TFRecord 或六月、七月 checkpoint | 移入 [历史脚本目录](../../../scripts/legacy/2026-09-15/README.md)，保留参数及 Git 历史 |
| 现用 controls 测试入口的旧 teacher 路径 | 仍引用六月 SID checkpoint | 改为正式 seed 42 SID 的 best90000；两次 `src.train` 调用仍仅打印配置，不启动训练 |
| 116 份旧 Slurm 日志 | 旧任务明细清单，均早于正式数据 smoke 131855 | 压缩、逐文件比对 SHA256 后移除原件，可从归档恢复 |
| 32 个旧逐样本评估 / 报告文件 | sidecar 指向六月、七月旧协议 run；报告输入也是这些旧导出 | 原样移入旧评估目录；正式 seed 42 全量产物留在原位 |
| 旧 `data/taobao/{training,evaluation,testing}`，共 496 个文件 | 历史尾项 / 目标错配，见 [数据协议](../../DATA_PROTOCOL_ZH.md) | **已永久删除**；删除前占用约 32.65 GiB |
| 34 个旧 checkpoint | 逐个检查对应 run 的 `.hydra/config.yaml`，`data_dir=./data/taobao` | **已永久删除**；删除前占用约 112.63 GiB；不是仅按日期筛选 |
| 34 个旧 run 的 402 个配置 / 指标 / 训练记录文件 | 与已删除权重逐一对应，删除前登记文件身份及配置哈希 | 保留并移入 `logs/archive/2026-09-15/legacy_runs/`；原位不再与正式 run 混放 |

当前正式入口仍见 [代码与模型说明](../../RECOCHAIN_PROJECT_GUIDE.md)。旧数据问题不是指跨 split 重复；本次不执行跨 split 去重，也不把已接受的正式数据判为错误数据。

## 明确保留

- `data/taobao_history_last_user_v1/` 全目录及构建 / 审计证据。
- **整个 `data/taobao/semantic_id/`**。它仍在旧父目录内，不能整体删除 `data/taobao/`。
- `data/taobao_history_last_user_v1_smoke_131855/`：现用测试入口依赖，不能仅因名称含 smoke 删除。
- 九月已完成的 9 个正式最佳 checkpoint，以及当前任务新生成的所有 checkpoint；其中 seed 44 SID teacher 是两个当前任务的共同依赖。
- 正式 run 配置、指标、日志、preflight、提交 / 验收记录、全量评估与配对统计。
- GRID 之外的 Taobao 上游数据和 item→row 映射，不在本次清理范围。
- 34 个旧 checkpoint 对应的配置、指标和训练记录；已按原 `logs/` 层次归档，内容不改写。

## 归档与恢复

完整清单：`logs/cleanup/2026-09-15/manifest.json`；删除前路径 / 大小 / inode / 修改时间及配置哈希见同目录 `deletion_validation.json`，执行后验收见 `deletion_result.json`。这些是本机运行记录，不进入 Git。

- 旧日志归档：`logs/archive/2026-09-15/legacy_slurm.tar.gz`，SHA256 为 `896384a983433be3fe3eebb8b78e519818efe664f138683bf864a20f18261521`；116 个成员的路径、大小和内容均与删除前原件一致。
- 日志原文按需用 `tar -xOf logs/archive/2026-09-15/legacy_slurm.tar.gz logs/slurm/具体文件名` 查阅。恢复时解压到新建的空目录，不向现有 run 批量覆盖。旧 run 内若有指向原日志的符号链接，需按归档成员查阅；本次不重写历史记录。
- 旧评估目录：`logs/archive/2026-09-15/legacy_paired_evaluation/`；报告中的历史输入路径未改写，重分析时显式指定新路径。
- 旧 run 记录：`logs/archive/2026-09-15/legacy_runs/`；源 → 目标映射见 `logs/cleanup/2026-09-15/legacy_run_moves.json`。配置文件和指标可追溯，不代表被删除的权重可以从归档或 Git 恢复。
- 旧脚本原文和原路径仍可从 Git `8671f50` 追溯；不建议恢复为当前默认入口。

错误数据和旧权重删除前合计占用 **145.28 GiB**；日志压缩另减少约 **67.95 MiB** 已分配空间。合计约 **145.35 GiB**，旧评估 / run 记录移动不释放空间。空间统计按 `st_blocks × 512`，不把文件表观大小当作实际磁盘占用，也不以共享文件系统的瞬时空闲变化归因。清理后 `data/` 约 24 GiB、`logs/` 约 41 GiB。

## 清理验收

- 现用 / 历史 launcher 相关测试：46 项通过；未占用 GPU，也未提交新测试任务。
- 清理后复核：496 个旧数据文件与 34 个旧权重已删除；9 个正式 checkpoint 均在原位，`logs/` 剩余权重均来自九月正式 run。
- 正式 / smoke 数据、整个 SID 目录和正式 checkpoint 共 69 个文件的大小、修改时间及 inode 与清理前一致；402 个旧 run 记录归档后也与删除前身份一致，已记录的小文件 SHA256 和全部 34 份 run 配置哈希复核通过。
- 两个已提交任务依赖的 12 份源码 / 配置 / 数据 manifest 哈希与提交记录一致；teacher、SID 和 item→row 映射的大小 / 修改时间未变。此次未重新扫描正式 TFRecord 内容。
- 10:37 清理验收时训练 CSV 已推进至 step 1894，loss 有限；本次队列核验仍为 133873 listwise seed 44 运行、133874 pointwise seed 44 依赖等待，未提交 / 取消任务。此为清理时快照，实时状态见 [当前状态](../../CURRENT_STATUS_ZH.md) 及 Slurm。

本次仅对作者批准的确切清单执行永久删除；没有按日期、smoke 名称或宽泛通配符扩展范围。后续新增产物需重新确认用途和引用，不复用本次授权自动删除。
