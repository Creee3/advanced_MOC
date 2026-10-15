# 数据核对记录（2026-10-15）

## 作者 test 文件

| 数据集 | 路径（相对仓库） | 题数 | SHA256 |
|---|---|---:|---|
| GSM8K | coding_base/MOC-official/datasets/test/gsm8k_test_n300.csv | 300 | b0b19fcfcb0a1a72f79a868b6070c5abf040aab5d53effc5379692bb1f8712da |
| MMLU-Pro | coding_base/MOC-official/datasets/test/mmlu_pro_test_n20.csv | 280 | ac3fe3cef1c4c95329d35a1cced1a0839e95e070f6709576114d6a0717db54de |

`n20` 是每学科20题，14学科合计280，不是整个数据集20题。

## 官方来源与核验

- GSM8K 官方卡：https://huggingface.co/datasets/openai/gsm8k/raw/main/README.md
- 官方 train：https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
  - 7473题，下载字节 SHA256：17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465。
- 官方 test：https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
  - 1319题，下载字节 SHA256：3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14。
- GSM8K 作者300题在官方 train 的规范化题文匹配数0，在官方 test 匹配数300。
- MMLU-Pro 官方卡：https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/raw/main/README.md
- 官方 validation API：https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FMMLU-Pro&config=default&split=validation&offset=0&length=100
  - 共70题；math/health/physics/business/biology/chemistry/computer science/economics/engineering/philosophy/other/history/psychology/law，各5题。
  - 与作者280题的 question_id 交集为空；规范化题文交集为空。
- 题文规范化只使用空白合并与 casefold；此检查不等于排除所有语义近重复。冻结前需补近重复清单审查，不按答案或模型表现筛题。
- 本次网络读取用于规划和核验，没有生成模型调用。执行前需锁定官方仓库 revision、数据字节与最终题号清单；不能依靠浮动 main/master 直接执行。

## 工程差异

- GSM8K 的原作者路径为 MathSolver / FinalRefer，现有适配器支持。
- MMLU-Pro 的原作者路径为 AnalyzeAgent / FinalRefer，现有适配器明确不支持 AnalyzeAgent，需补薄适配与原生提示词/分支预检，不能改用数学角色代替。
- v2.1 的角色/hop/组大小支持检查保持原规则；缺少训练支持的弃权要显式统计，不私自关闭。
- 额外跨任务的表现应区分：冻结迁移（目标任务零训练）与少样本适配（使用目标任务训练标签）。验证集更大本身不能替代冻结和数据独立性。
