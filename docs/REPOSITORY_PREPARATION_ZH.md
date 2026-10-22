# 2026-10-22 仓库整理与复现准备

作者已确认：先整理四模型主线，SIM / LIGER 暂留内部。本次不改论文，不重训，不 push，不更换许可。

## 工作区边界

- `GRID`：原实验工作区，分支 `wsdm27-liger-learning-v2`。活动 LIGER 任务继续使用它；本次不修改其 tracked 源码/配置，不移动输入或运行目录。
- `GRID-repro-prep`：从 `8301284f520f9f31d96b0f7449eab9255d47d008` 创建的独立 worktree，分支 `wsdm27-repro-prep`。整理工具、文档与预览入口在这里提交。
- 对外候选：由 `release/manifest.json` 白名单从干净 HEAD 导出为不含 Git 历史的新目录。内部工作树无需删除科研证据来获得干净发布包。

副本中的 CURRENT_STATUS / RUN_LEDGER 是分叉时快照，不是活动任务的最新台账；原工作区继续维护作业事实。本页不复制模型结果或诊断结论。

## 已采取的组织方式

| 内容 | 去向与处理 |
| --- | --- |
| 对外介绍 / 协议 / 发布门槛 | `release/public/`，导出后为根 README + 两份 docs |
| 内部状态、作业台账、历史问题 | 保留原 docs，不进入公开候选 |
| 旧根 README | 保存至 `docs/archive/UPSTREAM_GRID_README.md`，根入口改为本分支导航 |
| 集群运行脚本 | 内部原位保留；不复制到候选，不破坏历史 preflight 路径绑定 |
| 四模型训练参数 | 新 `scripts/reproduce_training.py`，仅预览，源码与配置绑定显式列出 |
| `.project-root` | 将原先未跟踪的空目录需求补成 tracked 标记文件，修复 clone 缺失问题 |
| 依赖 / 许可 | 历史 requirements 原样保留；不删 LICENSE / notices，不改现有环境 |

## 存储盘点与删除边界

以下为整理时的实际占用近似值，不是本次删除量：`logs/train/runs` 56.22 GiB、`logs/analysis/paired_evaluation` 10.86 GiB、`logs/archive/2026-09-15` 6.58 GiB、`logs/slurm` 1.10 GiB、`logs/training_launch` 831 MiB、`logs/exp_result` 190 MiB。后者 apparent size 为 2.08 GiB，不能按表观大小声称释放了相同磁盘空间。

保留正式 TFRecord、整个 semantic_id、四模型最佳 checkpoint、validation/test/选权/配对产物以及可验证恢复过程的 checkpoint 链。活动 LIGER manifest 引用的 probe、catalog、索引和所有输入不动。旧符号链接和已删除 checkpoint 的空占位目录不自动“修复”，它们可能用于对应归档记录。

大体积候选不等于无用：

| 候选 | 大小 | 本次处理 |
| --- | ---: | --- |
| LIGER 前两轮每千步中间权重 | 约 1.35 GiB | 保留；须确认保留最佳/末尾及恢复链后另批删除 |
| 旧协议 legacy paired HDF5 | 约 6.34 GiB | 保留；上次明确留作追溯，旧删除授权不延伸到这批 |
| CPU resume smoke / 中断重训链 | 多 GiB | 保留；不是无用 smoke，绑定恢复验收与正式 best 来源 |
| 小型 bytecode 缓存 / 明确空目录 | 几十 KiB | 已删除 11 个 pyc、其 5 个空缓存目录及 3 个原空目录；见下方清单 |

清理记录：[CACHE_CLEANUP.json](archive/2026-10-22/CACHE_CLEANUP.json)，包含每个删除文件的路径、SHA256、mtime、字节与实际分配空间。删除前文件总大小 69,125 B，实际分配 26,112 B，不构成明显的 GB 级空间回收。源码全部保留，bytecode 可重新生成，空目录可重建；本次未备份这些缓存字节，也未删除任何数据、checkpoint 或指标证据。

## 提交与后续

继续按“一个能独立解释/验证/回退的逻辑改动一次提交”：训练参数预览及测试一组，安全导出器及测试一组，文档/白名单机械整理一组，验收及清理记录一组。只暂存明确文件，不使用 `git add .`，不混入活动 LIGER 修改。

下一步不是发布：先审阅候选说明，再完成环境依赖拆分、资源入口、固定资产获取/授权以及小数据端到端验收。详见 [维护入口](../release/README.md)。无关旧内容先通过发布白名单隔离，未证明可删的科研产物不永久删除。
