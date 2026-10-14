# advanced_MOC

MOC 接收方条件的多阶消息选择研究。当前版本为 **2026-10-14 完成验证的 v1**。

目标：在作者多智能体图中，按接收方任务、角色、已选消息和 token 预算选择最多四跳的祖先材料，保持或提升质量并降低全流程模型 token。

## 当前方法

- 作者 MOC 的 DAG、角色、提示词、执行及 FinalRefer 汇总由固定版本的 Git submodule 提供。
- MiniLM 生成消息 sketch；LADIES 的层条件支持质量提供拓扑特征。
- 用固定上游材料的配对回放训练 ridge 模型，预测读取一条消息的**局部答案增益**。
- 选择器逐条执行读取、展开或停止；正文上限 1024 token，最大四跳，成本惩罚固定。
- 当前实现不导出真实 KV cache；局部增益不等于最终图效用。

主代码：[stateful.py](coding_base/moc_adaptive_sampling/stateful.py)；迁移边界：[SOURCES.md](coding_base/moc_adaptive_sampling/SOURCES.md)；作者对照：[PAPER_BASELINES.md](coding_base/moc_adaptive_sampling/PAPER_BASELINES.md)。

## 已完成实验与限制

训练原选 48 题，47 题有效，1 题因输出截断排除，376 个配对标签；val 同 60 题在两种图密度下运行，总计 540 条件。

| 密度 | 原 MOC 正确数 | v1 正确数 | v1 推理 token 变化 |
|---|---:|---:|---:|
| 0.3 | 24/60 | 19/60 | -10.88% |
| 0.7 | 28/60 | 28/60 | -67.02% |

稀疏图质量下降。稠密图全部 360 个普通非根接收节点选择空集，FinalRefer 仍汇总所有节点输出；因此省 token 不能证明有效多阶选择。计入训练、排除和废弃调用的 657,528 token 后，本轮两种密度合计比原 MOC 多 7.82%。观察正确数持平不构成非劣结论。

一条稀疏 MOC 调用达到 2000 输出上限，完整性门槛未全部通过；原始结果保留，报告含统一剔除同题的敏感性分析。当前 v1 尚未达到可推广的质量与成本目标。

完整结果：[完成结果与首次分析](experiments/moc_stateful_sampling_val_20261014/完成结果与首次分析_20261014.md)。冻结协议、config、checkpoint、policy lock、分组结果和审计随版本保存。训练与 val 已结束，自有云端服务已清理，监控已暂停；没有 test 运行或新任务启动。

## 目录与复用

```text
coding_base/MOC-official/              作者源码 submodule，固定 commit
coding_base/moc_adaptive_sampling/     当前扩展、测试及迁移说明
experiments/moc_stateful_sampling_val_20261014/
  remote_snapshot/runs/               540 个现有条件结果及 metadata
  remote_snapshot/utility_model.json  已冻结轻量 checkpoint
  remote_snapshot/policy_lock.json    冻结配置和执行源码 SHA256
research_ideation_20261013/moc_adaptive_neighborhood_sampling_mainline/
                                      方法设计及来源记录
SNAPSHOT_MANIFEST.json                本次逐文件哈希与本地 raw 归档指针
docs/next_direction.md                v2 候选思路，尚未实现或验证
```

已完成且条件匹配的基准直接复用，不反复生成。只有模型、题目、图、角色、提示词、解码及执行协议匹配时才可配对比较。当前 val 已用于诊断，不能再次包装为 v2 的独立验证集。

## 使用此版本

```bash
git clone --recurse-submodules https://github.com/Creee3/advanced_MOC.git
cd advanced_MOC
python -m pip install -r coding_base/moc_adaptive_sampling/requirements.txt
```

作者运行依赖还需按 submodule 的 requirements 安装。主实验使用已有 Gemma2-27B、Gemma2-9B 和 MiniLM，模型权重不随 Git 保存。实验脚本和配置保留原云端绝对路径及恢复约束，**是冻结实验快照，不是任意机器直接 launch 的安装器**。重部署前需设置环境和新运行目录，并保护已有结果。

可在装有 numpy、torch、shortuuid 的环境中运行不涉及模型调用的 v1 单元检查：

```bash
PYTHONPATH=coding_base python -m unittest discover -s coding_base/moc_adaptive_sampling/tests -p test_stateful.py -v
```

完整历史离线检查另需本地 Gemma tokenizer 和原取源目录中的参考采样器；其历史验证报告已保留。大型 raw trace、模型、完整数据集及 LOCAL_ONLY 金标仍在原工作区，未上传。全量 raw 归档 SHA256 为 `869146b31a36282a3241fb5bb9deb1ace578e3da012e547219bd4f9d2c587f37`；160 文件的归档清单见实验 remote_snapshot。

## 来源和许可

作者 MOC 来源：https://github.com/yao-guan/MOC ，固定 commit `9c67c92507570704a7df73e452552a3f49e83897`。以 submodule 保留原仓库归属，不替作者补写许可证。GNN 来源及独立数学迁移见 SOURCES.md；未将无许可证的第三方训练代码复制进本仓库。此快照未新增整体开源许可证。
