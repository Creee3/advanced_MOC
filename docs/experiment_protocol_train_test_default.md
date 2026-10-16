# 后续实验划分协议：默认 train–test，AQuA 保留开发验证

更新时间：2026-10-16

这份文档规定后续实验的默认数据划分。它不改写已经完成的实验，也不修改任何历史 `config.json`、锁文件、checkpoint 或结果。

## 默认规则

除 AQuA 外，后续数据集默认只做：

`固定训练集 → 训练并冻结 policy/checkpoint/config → 作者 test`

训练完成后，测试集只用于一次最终评估。测试答案保持本地，不进入 prompt、训练、policy 选择、阈值调整或重训流程；如果测试过程中出现无效条件、超时或缺失，保留终态并按原协议处理，不为了提高正确率换题或重跑。

因此，后续新数据集不再额外建立 validation split，也不把 test 当作开发集使用。训练样本、训练标签、方法版本、超参数和解析器必须在 test 前固定。没有 validation 并不意味着可以根据 test 结果迭代；test 是冻结后的最终测量。

这条规则是“未来默认”。历史上已经跑过的 GSM8K val、MMLU-Pro val，以及已经完成的独立 Pro val 补测，都必须按真实历史记录保留，不能在报告中改写成从未使用过 validation。

## 唯一例外：AQuA

AQuA 是后续唯一保留开发验证阶段的数据集。沿用已经完成并审计过的 v2/v2.1 配置：

`原始 AQuA train47 → fit35 + calibration12 → 冻结选择器 → AQuA dev-val60 → 冻结最终方法 → AQuA official test254`

AQuA 的 val 是开发验证，不是独立的首次未见确认。一个方法版本只运行这一轮固定的 dev-val；val 结果用于确认链路和发现明显退化，不能在看到 val 后继续改 policy、阈值、题目、checkpoint 或提示词，再把同一轮 val 当作独立验证。完成该轮后，方法和配置冻结，再进入 test。

AQuA 历史配置的关键数字如下：

| 项目 | 冻结值 |
|---|---|
| 原始训练问题 | 47 题；35 题 fit，12 题 calibration |
| 开发验证 | 60 题 × 2 密度 = 120 条方法条件；原 v2.1 还保留作者角色和固定 DAG |
| 官方测试 | 254 题；两密度、adapted 与原 MOC 两组，共 1016 条条件 |
| 主模型 | Gemma2 27B Q4_0；digest `53261bc9c192c1cb5fcc898dd3aa15da093f5ab6f08e17e48cf838bb1c58abfe` |
| 主模型解码 | temperature 0，seed 42，`num_predict=2000`，`ctx=8192`，不使用 CoT |
| 通信预算 | 每条消息预算 1024 tokens |
| 子程序限制 | 代码执行 5 秒；保持原 Landlock/seccomp 和隔离执行器语义 |
| 选择器 | `one_shot_expected_group_utility_v2_1`，compact22，ridge regularization 10，alpha 0.1，bootstrap 32，最大组大小 2，cost penalty 0.02 |
| 候选结构 | 空集、单条或连接二元组；最大祖先 hop 4；空集是合法动作 |
| 原 MOC 对照 | Gemma2 9B Q4_0 merger，temperature 0.1，seed 42；`r=1.0`、`epsilon=0.1`、`kppa=45` |
| 训练后状态 | checkpoint、policy、source、data split、parser 和 decode 全部冻结；不根据 val/test 重训或调参 |

对应的历史配置文件是：

- [AQuA 训练配置](../experiments/moc_v2_evidence_bundles_train_20261014/config.json)
- [AQuA 训练清单](../experiments/moc_v2_evidence_bundles_train_20261014/training_manifest.json)
- [AQuA v2.1 开发验证配置](../experiments/moc_v21_expected_set_val_20261014/config.json)
- [AQuA 官方测试配置](../experiments/moc_v21_official_test_20261014/config.json)

## 报告口径

以后写实验表格时，统一显式写出 `train / val / test` 三列：

- 普通新数据集：`N / 0 / M`，并标注“train–test only”。
- AQuA：写出 `47 train（35 fit + 12 calibration）/ 60 dev-val / 254 test`，并说明 val 是开发验证。
- 历史 GSM8K、MMLU-Pro 和 Pro val 补测：按实际数字报告，不能为了统一口径删除历史 val。

这使得实验叙事保持一致：AQuA 用一次受控开发验证确认方法是否值得进入最终测试；其余数据集直接检验冻结方法在作者 test 上的表现。
