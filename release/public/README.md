# RecoChain: four-model reproducibility draft

本目录是基于 Snap Research GRID 整理的**源码复现草稿**，仅覆盖 SID、Listwise、Pointwise、Separate。它不是已完成第三方复现的正式发行版。SIM、LIGER、内部调度脚本、运行日志和逐用户预测不在分发范围。

## 阅读顺序

1. [复现协议与当前限制](docs/REPRODUCIBILITY_ZH.md)：数据 → 固定 SID → 四模型 → validation 选权 → 全量 test。
2. [环境与发布门槛](docs/RELEASE_GATES_ZH.md)：哪些已具备，哪些仍不能承诺。
3. `RELEASE_MANIFEST.json`：导出时自动生成的来源 commit 与逐文件 SHA256。

## 可以先检查什么

参数预览仅使用 Python 标准库，不加载数据或模型、不创建输出目录、不提交任务：

```bash
python scripts/reproduce_training.py --model sid --seed 42 --devices 3 \
  --data-root /path/to/history-last-user-v1 \
  --sid-path /path/to/fixed_sid.npy \
  --output-root /path/to/new/run --dry-run
python -m unittest discover -s tests -p 'test_reproduce_training.py'
```

其他三个模型必须显式提供 `--teacher-checkpoint /path/to/sid_teacher.ckpt`。该入口只生成待核验命令，**没有执行模式**。三卡历史配置的有效 batch 为 384；单卡预览不是正式结果的等价复现。

训练、数据构建及推理须在 TensorFlow/CUDA 兼容的计算环境中完成，整个作业最多 4 CPU、库线程为 1；实际 launcher 和新环境端到端测试尚待补齐。不要把参数预览输出直接当作已验收的启动脚本。

## 来源与许可

基于 GRID，原始文献：Ju et al., *Generative Recommendation with Semantic IDs: A Practitioner's Handbook*, CIKM 2025。本草稿增加 Taobao 历史末项协议、联合/独立重排及配对评估；不能将 SID 基线表述为未经改动的原始 TIGER。

[LICENSE](LICENSE) 为上游非商业研究许可；[notices.txt](notices.txt) 原样保留。此草稿没有更改许可，不提供数据、固定 SID、映射或模型权重的再分发授权。发布这些资产前需要分别核验权限。
