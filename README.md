# RecoChain 实验仓库：四模型复现整理

本分支用于整理基于 GRID 的 Taobao next-item 实验，当前仍是**内部准备分支**，不是已发布或已完成第三方复现的仓库。

## 从哪里开始

- **审阅对外草稿**：[四模型 README](release/public/README.md) → [数据、训练与评估协议](release/public/docs/REPRODUCIBILITY_ZH.md)。
- **维护发布候选**：[导出方式与发布门槛](release/README.md)。仅白名单源码进入候选包，不复制本仓库的 Git 历史。
- **查看内部实验**：[文档导航](docs/README.md)。当前状态、job 台账、原始证据仍留在内部；SIM / LIGER 不进入本次候选包。
- **查看本次整理边界**：[2026-10-22 整理记录](docs/REPOSITORY_PREPARATION_ZH.md)。
- **查看原始框架说明**：[上游 GRID README](docs/archive/UPSTREAM_GRID_README.md)，不是本分支四模型的复现入口。

## 本轮范围

| 对外模型名 | 配置叠加（均先加载 `tiger_train_flat`） |
| --- | --- |
| SID | `tiger_train_sid` |
| Listwise | `tiger_train_click` |
| Pointwise | `tiger_train_click` + `tiger_train_click_pointwise` |
| Separate | `tiger_train_rerank_small` |

冻结数据协议为 `history-last-user-v1`（A + H1 + L2）。不更改样本、模型或已验收指标；不混入旧错误协议结果。

## 许可与发布状态

原 GRID 的 [LICENSE](LICENSE) 限定非商业研究用途，并要求保留声明；[notices.txt](notices.txt) 原样保留。不能自行改为 MIT / Apache，也不将当前许可描述为不受限制的开源许可。基于 Snap Research 的 GRID，原作者引用信息见上游 README。

发布前仍需核验资产再分发权限、依赖与第三方声明，并完成干净环境的小型端到端验收。当前参数预览不是可直接执行的正式复现 launcher。未经确认不 push、不发布数据或 checkpoint。
