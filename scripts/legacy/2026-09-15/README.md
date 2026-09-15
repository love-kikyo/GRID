# 旧数据协议启动脚本（仅供追溯）

这些脚本固定使用旧 `data/taobao` TFRecord 或六月 / 七月 checkpoint，不适用于当前 A + H1 + L2 实验。2026-09-15 从仓库根目录归档；保留历史参数，仅增加提示注释并规范化文件末尾空行。不要直接提交或将其中参数当作当前设置。

- 训练旧入口：`run_sid.sh`、`run_click.sh`、`run_click_seed43.sh`、`run_rerank_small.sh`、`run_rerank_medium.sh`。
- 旧评估入口：`run_eval_context_length.sh`、`run_eval_context_mask_length.sh`、`run_eval_retrieval_length.sh`。
- 旧效率入口：`run_profile_context_length.sh`、`run_profile_context_mask_length.sh`、`run_profile_kv_cache.sh`、`run_profile_retrieval_length.sh`。

当前正式入口见 [代码与模型说明](../../../docs/RECOCHAIN_PROJECT_GUIDE.md)。效率分析的核心实现 `scripts/profile_rerank_efficiency.py` 仍保留；后续应显式绑定正式 run / checkpoint，不恢复这些旧默认值。

原始内容和原路径可从 Git 提交 `8671f50` 追溯。历史文档中的旧路径保留原貌，不作为当前可执行导航。
