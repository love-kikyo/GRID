# 环境与发布门槛

状态：**draft / 未发布 / 未完成干净环境端到端验收**。命令预览、单元测试通过与正式复现完成是不同状态。

## 已具备

- 四模型配置、共享模型/数据实现、数据构建/审计/预检、validation 选权与配对评估实现。
- 受版本管理的 `.project-root`，避免 clone 缺少 rootutils 根标记。
- 无执行能力的统一参数预览，显式输入/输出路径与 teacher 初始化。
- 仅允许清单中已提交文件的导出流程；候选包不含 Git 历史、私有数据、checkpoint、日志和内部作业表。
- 固定数据规则、资产哈希及已知审计限制；保留原许可与第三方声明。

## 还未完成，不能省略

| 门槛 | 验收条件 |
| --- | --- |
| 许可与资产分发 | 作者确认代码发布渠道；分别核验上游数据、item2row、SID、checkpoint 的再分发条件 |
| 固定资产交付 | 有合法可获取渠道、逐文件清单/哈希及来源说明；不能用重新聚类替代冻结 SID |
| 可维护依赖 | 补齐直接依赖源声明，形成验证过的 core/eval/dev 分组与环境锁；不改现有运行环境 |
| 资源入口 | 参数化实际训练/评估 launcher，保留整项 CPU4、线程1和实际 affinity 验收；保留三卡 batch 语义 |
| 新环境复现 | 新目录安装 → 合成小数据 → 四模型短训/导出 → validation 选权 → paired test；成功后再验证正式资产 |
| 结果交付 | 脱敏的正式指标与 checkpoint/配置/源码身份映射；效率测量入口独立验收 |
| 安全与声明 | 扫描最终候选内容、检查外链与个人路径；补核实际分发依赖的声明，不擅自删除旧 notices |

## 依赖现状

`requirements.txt` 是保留的历史完整锁，266 个 pinned distributions，引用的上游 `requirements.in` 未随原仓库提交。它不是此次已验收的最小安装方案。当前实验环境 `pip check` 通过，但不代表该文本在全新机器可直接安装成功。

实查的核心版本：Python 3.10.19、NumPy 2.0.2、PyArrow 20.0.0、tensorflow-cpu 2.18.0、torch 2.6.0+cu124、lightning 2.5.0、transformers 4.47.0、hydra-core 1.3.2、omegaconf 2.3.0、h5py 3.14.0。该列表用于核验，不是完整依赖锁，也不承诺任意硬件/驱动兼容。

不要仅按“源码未直接 import”删除包，传递依赖仍需核验。`src/utils/inference_utils.py` 顶层依赖 `google.cloud.storage`，目前不能直接去掉全部云相关依赖。原 `notices.txt` 部分版本与现依赖不同，应保留原声明并补核实际分发清单。

无需 ML 环境即可运行 README 的参数预览与相应 unittest。其余测试按依赖分层执行；含 TensorFlow 的构建/数据读取测试须在兼容计算节点执行，不能在不兼容登录节点运行全套 pytest。
