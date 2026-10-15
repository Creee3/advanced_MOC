# 事实与来源

- MMLU作者test285题，57学科各5题；官方dev285题可用作训练池。固定seed42先取40个学科，每学科1题。40题不覆盖57科，须报告覆盖限制。
- SVAMP作者test300题包含在官方全量1000题中；从排除作者test及近重复后的剩余池固定抽40题，是研究用训练划分，不称官方train。全量官方文件与本地SHA一致5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e。
- 旧runtime只支持gsm8k/mmlu_pro；新目录需明确MMLUDataset/SVAMPDataset映射，目标键分别correct_answer/Answer，保持作者原prompt/parser。
- 原始GPU现场：8张4090均不满足主卡util<=15%、free>=21000MiB；两张低利用率卡只有约3.2GiB空闲，不得视为空闲可用。
- 官方来源：https://huggingface.co/datasets/cais/mmlu 与 https://github.com/arkilpatel/SVAMP 。
