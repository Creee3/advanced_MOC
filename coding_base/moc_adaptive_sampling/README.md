# MOC 接收方条件的多阶邻域采样：代码原型

2026-10-14 v2.1离线修订：`expected_set.py`在最多四跳的单条/连接双条候选中一次选择总期望净收益最大的证据集，空集为0；`expected_utility.py`按题/episode/动作等权、正确加权中心化拟合帮助/损害两头。bootstrap仅作稳定性诊断，不用于置信停止。`initial_features.py`为旧训练singleton继承pair.path的错误生成可审计派生修复。原train35问题隔离五折验收仍失败，新cal/val/GPU未启动；所有原基准复用。代码修复不等于方法有效，结果在 `experiments/moc_v21_expected_set_offline_20261014/`。

v2完整train/cal/val已经完成，其宽单次结果预测区间导致近全读/零组读取；冻结原件及费用保留在 `experiments/moc_v2_evidence_bundles_train_20261014/`。本文后续的数据流与公式描述历史v0，不代表当前v2.1策略。

2026-10-14 v1：新增 `stateful.py`。原 `LayerDependentSampler` 保留为v0；v1使用固定材料配对的本地接收答案增量训练轻量ridge模型，随已选sketch重算语义残差/互补特征，执行read/expand/stop。最多四跳可达；不设固定正文fanout名额。实测与协议在 `experiments/moc_stateful_sampling_val_20261014/`，训练48题、val60、原文baseline及w/o Phi归因消融。模型预测是局部效用代理，不是最终图准确率保证；下面公式与数据流描述v0。

这是本轮讨论主线的实际代码接入，允许最大3–4跳，逐接收节点选择有限的原始祖先材料。主内核迁移自 LADIES 官方实现；AS-GCN proposal 另有显式权重的数学移植，但本轮运行路径不使用未经训练的权重。完整映射在 [SOURCES.md](SOURCES.md)，原文基准线在 [PAPER_BASELINES.md](PAPER_BASELINES.md)。作者 checkout 不修改。

## 已实现的数据流

```text
作者原 Graph.arun 的 DAG 执行
    ↓
发送方完成生成和工具后缀 → 原文保存 + 一次 MiniLM sketch
    ↓
接收方 task/role sketch + 已到达一跳 sketch 形成状态
    ↓
作者 shortest-hop 祖先池，最大 Kmax=4
    ↓
LADIES 当前前沿的列平方范数 × 接收方语义重加权
    ↓
不放回采样路由前沿 → 互补性/实际 token 代理量决定读取正文
    ↓
预算内原文，按作者顺序子序列构造 → 作者 MathSolver 一次生成
    ↓
作者 FinalRefer 汇总全部普通节点（不改）
```

路由前沿与正文分开：一跳长正文不够预算时仍可使用其小 sketch 找到二跳材料，不必先读它。反过来，没有被采到的路径之后的祖先可能被漏掉，这是该逐层采样的限制。

## 当前候选公式

\[
q_j=\operatorname{Norm}\{E(Q,role_j)+\rho\,\operatorname{Mean}(z_{\text{available 1-hop}})\}
\]

\[
p_{j,k}(i)\propto\Big(\sum_{v\in F_{k-1}}U_{vi}^{2}\Big)
\exp\{\cos(q_j,z_i)/\tau\}.
\]

`F` 是上一层已抽到的路由前沿；`U` 是入边行归一化支持。被抽到的原文再按

\[
g(i|A)=\max\{0,(1+\cos(q_j,z_i))/2-\beta\max_{a\in A}\max(0,\cos(z_i,z_a))\}
\]

的增量 token 比值依次准入。正文总量含来源标签、分隔符和 MathSolver 消息包裹。没有未来候选能在剩余预算中产生正代理量时停止扩展。

这些都是可运行的代理规则；`g` 不是已训练或校准的期望准确率提升。`configs/prototype.json` 中的2048token、(4,3,2,1)等是实现默认值，尚未固定为实验最优参数。可以分别配置 fanout 和各 hop token cap，不强制单调递减；真正的学习式 hop 停止和可靠性建模仍缺训练目标/校准数据。

## 直接接入作者代码

在已有 MOC 环境中，令 `coding_base` 加入 PYTHONPATH。使用本地 tokenizer，不下载模型：

```python
from moc_adaptive_sampling import SamplingConfig, LayerDependentSampler
from moc_adaptive_sampling.features import LocalTokenizerCounter
from moc_adaptive_sampling.paper_baselines import (
    PaperRecipe, load_author_api, make_comparison_graphs, run_single_llm,
)

api = load_author_api("/home/liuxin/ICML2027/MOC-official")
from src.llm.profile_embedding import get_sentence_embedding
counter = LocalTokenizerCounter("/path/to/existing/gemma/tokenizer.json")
sampler = LayerDependentSampler(SamplingConfig(max_hops=4), counter)
graphs, adapter, roles = make_comparison_graphs(
    api, PaperRecipe(domain="aqua"), sampler, get_sentence_embedding,
    counter_name=counter.name,
)
# 这里只构造。经正式实验协议允许后，才调用 await graphs[arm].arun(...)
# Single LLM 使用 await run_single_llm(api, recipe, question)，一次原 direct 调用。
```

独立图实例使用独立方法绑定，不对 Graph 类全局 monkeypatch。候选原摘要为空时明确清空旧原文输入；选择异常即失败，不让作者的异常吞掉逻辑悄悄退回完整通信。多轮/CodeWriting/Wiki路径会提前报不支持，避免改变工具旁路。

## 成本与表示接口

`adapter.ledger` 分开保存普通/最终调用正文 token、编码 token 与时间、路由时间、sketch 逻辑字节量和每跳选择。HF tokenizer 保证的是本地文本计数上限，不等同于 Ollama 的 chat-template 或实际 prefill token；后台 actual usage 必须另计。最终聚合器正文、路由成本都不能从论文总成本中扣掉。

当前 MiniLM 在 sender 完成后编码完整最终输出，所以 Programming Expert 的执行后缀也被包含。`features.py` 提供外部 hidden/KV 的 mask pooling、显式位置一致性检查；未改 Ollama，未抽到任何真实 KV，也未假定早层更好。使用内部表征时 sender/receiver 必须共享模型、层与位置基准，并记录额外 prefill/抽取成本。

## 本地验证

```powershell
$env:PYTHONPATH='D:\liuxin_workspace\ICML2027\coding_base;D:\liuxin_workspace\ICML2027\maintenance\task_runtime\moc_sampling_20261014'
python -X utf8 -W ignore::SyntaxWarning -m unittest discover -s coding_base/moc_adaptive_sampling/tests -p 'test_*.py' -v
```

16项检查已通过。直接执行固定作者源码 AST 中的 `Graph.arun`、`Node.async_execute`、MathSolver、FinalRefer、AQuA提示词及 direct 函数，替换服务/图特征依赖；使用已存在 Gemma2 tokenizer 检查正文预算，直接执行 LADIES 原采样函数核对 proposal。`python -X utf8 -W ignore::SyntaxWarning coding_base/moc_adaptive_sampling/verify_local.py` 可重跑并保存 `verification.json`、`local_checks.log`、`offline_trace.json` 和 `asset_audit.json`。

这是结构接入验证；MiniLM/GPU/backend/tool execution 使用离线替身，不能据此声称已跑主模型、已复现精度或新方法有效。本原型在同一运行时存储所有原文，路由阶段也会读取其文本做精确 token 计数；语义评分不把全部正文交给接收 LLM。当前的节省定义是接收方 prompt 正文，未实现跨机器“只取 sketch 再请求正文”的传输协议，不能把逻辑字节量当实际网络带宽。下一个有意义的阶段是固定训练/校准目标与模型评估协议，而不是无终点地重复一跳/两跳诊断。本轮没有恢复旧云端任务。
