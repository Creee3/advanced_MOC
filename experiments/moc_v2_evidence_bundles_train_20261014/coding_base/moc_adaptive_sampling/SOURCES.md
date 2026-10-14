# 官方来源与迁移映射（2026-10-14）

资产根目录：`research_ideation_20261013/moc_adaptive_neighborhood_sampling_mainline/code_migration_20261014/`。`source_manifest.json` 保存每份论文/源码的 URL、SHA256、下载时间和字节数；下表的 stars 为取源时快照。

| 官方实现 | 固定 commit | stars | 本轮用途 |
|---|---|---:|---|
| [LADIES, UCLA](https://github.com/acbull/LADIES/tree/020c8e91f4124612a24f142ec33aed8162905f51) | `020c8e91f4124612a24f142ec33aed8162905f51` | 75 | 主内核：按上一层路由前沿构造支持矩阵、计算列平方范数、按概率不放回采样 |
| [AS-GCN, Tencent AI Lab / ANU](https://github.com/huangwb/AS-GCN/tree/4aa6e7d51c9b4082032f10e1af116d77c885dd95) | `4aa6e7d51c9b4082032f10e1af116d77c885dd95` | 89 | 迁移 proposal 数学内核，显式需要权重；没有接入未经训练的权重来冒充有效策略 |
| [GraphSAGE, Stanford](https://github.com/williamleif/GraphSAGE/tree/a0fdef95dca7b456dab01cb35034717c8b6dd017) | `a0fdef95dca7b456dab01cb35034717c8b6dd017` | 3729 | 分层 fanout 参考；不是本轮主采样器，也不是 MOC 论文 baseline |
| [FastGCN](https://github.com/matenure/FastGCN/tree/b8e6e6412d8cb696e0adb7910832ed1ae7243ac9) | `b8e6e6412d8cb696e0adb7910832ed1ae7243ac9` | 529 | 重要性采样近邻参考；未装入本轮文本选择流程 |

这些旧官方仓库 stars 不都高；选主内核依据是 NeurIPS 正式论文、可核实的作者机构，以及与当前层条件采样的匹配，不能把相关仓库的总 stars 归给单篇实现。

## 主内核逐项映射

1. LADIES [`ladies_sampler`, 行127–160](https://github.com/acbull/LADIES/blob/020c8e91f4124612a24f142ec33aed8162905f51/pytorch_ladies.py#L127)：`U = lap_matrix[previous_nodes,:]`、`pi = sum(U*U,axis=0)`、`p=pi/sum(pi)`、不放回选择、更新 `previous_nodes`。
   - 对应 `kernels.py::ladies_probabilities/weighted_without_replacement` 和 `sampler.py::LayerDependentSampler.select`。
   - 改动：无向 GCN Laplacian → 作者已执行的有向祖先图的入边行归一化支持；列限定于下一最短 hop；全局 RNG → 每请求局部 RNG；路由前沿与正文读取分开；正文受真实 token 上限约束。
   - 自行补充：任务/角色/局部 sketch 的接收方状态，语义重加权，互补性代理量，成本与停止处理，作者顺序子序列、成本账本及空消息回退防护。
2. AS-GCN [`SamplerAdapt.one_layer_sampling`, 行64起](https://github.com/huangwb/AS-GCN/blob/4aa6e7d51c9b4082032f10e1af116d77c885dd95/sampler.py#L64)：列 L2 范数 × 注意代理 × 特征代理，再归一化。
   - 对应 `kernels.py::asgcn_probabilities`；迁移的是 NumPy 数学操作，不是整个 TensorFlow 1 训练器。
   - 保留 `sampler.py` 中父节点用权重第2列、候选用第1列的约定。`models.py::_attention` 的列约定不同，不能悄悄视为相同实现。
   - 原论文还有分类交叉熵与方差正则；没有把该损失当成离散文本接收方的训练目标，没有可复用的 MiniLM 路由权重。

## 许可与归属

GraphSAGE `LICENSE.txt` 为 MIT 文本，包含 William L. Hamilton、Rex Ying 和 Thomas Kipf 的声明。其完整官方文件保留在上面的快照目录。

本次 GitHub repo/tree 核查未在 AS-GCN、LADIES、FastGCN 中发现许可证（GitHub metadata 为 null）。快照用于本地可追溯核查，不为这些仓库补写 MIT；本扩展按公开数学操作独立重写，不复制旧 trainer，不声明拥有原论文的理论。

## 理论边界

GCN 连续线性聚合的无偏/方差结论不适用于有序文本与 LLM 生成。没有给文字乘 `1/p`、复制文字以恢复“无偏输出”，也没有迁移 LADIES 的 self-loop 和逆概率邻接重归一化到接收方文本。预算拒绝与语义重加权又改变了原分布；这里明确只迁移层条件采样结构。

## 引用核验

四篇论文已通过 arXiv 原始记录核验，报告见资产目录 `citation_check.json`。

`PASS: 4/4 verified, 0 errors, 0 warnings (0 checks skipped)`

范围说明：首次默认的同名规范实例查询被 Semantic Scholar HTTP 429 中止。第二次使用 `--no-canonical-instance`，直接核验 arXiv 元数据成功；该 PASS 不包含被选项关闭的同名规范实例检查。离线报告的 `PARSED_ONLY` 不作为在线核验证据。
