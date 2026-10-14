# Findings

## 2026-10-14 首轮真实开发集结果

dev30、两种原文密度、三原文baseline与候选共210条件已完成；真实GPU生成和完整usage审计通过，结束自有进程/子进程全部退出。原MOC与四跳候选：0.3密度18/30对16/30、候选成本+15.8%；0.7密度16/30对15/30、成本-52.6%。符号检验和bootstrap不足以建立非劣。四跳候选分别仅省1.4%/3.3%候选消息条数，fanout从未限制任何层；1024token预算多数未满，零代理阈值导致大部分全读。0.7节省几乎全部来自没有9B候选生成，非语义选择有效的证据。主线继续保留，但下一版须真正实现有取舍的增量准入/停止。详见 `experiments/moc_adaptive_sampling_20261014/首次结果分析_20261014.md`。当前没有自动启动后续GPU或val/test。

## 2026-10-14 官方代码迁移核查与实现结果

- MOC主表基准线是Single LLM、Vanilla MAS、MOC；无通信多代理不是Single LLM。Figure3有K=1/2/3和w/o Φ消融。保留作者原DAG、角色分配、prompt和FinalRefer，只给候选装独立实例适配器。
- 主内核选择LADIES `ladies_sampler`，已从固定官方函数中迁移当前前沿条件的列平方质量与不放回采样，并直接执行官方函数进行数值对照。AS-GCN原采样权重为[input_dim,2]，需要训练，不能把节点分类权重当当前消息路由器。
- 接收方状态使用任务/角色embedding和一跳可用sketch；发送方只在完整回复（包括工具后缀）完成后编码一次。未把全部候选正文送入接收LLM后再选。
- 每跳路由前沿与读正文分开；保留对被抽路径后远端祖先的访问，未抽路径可能漏掉祖先。短路/停止依据是未校准语义互补代理量与真实文本token预算，不是已学习期望准确率。
- 作者序列化顺序是最短hop远到近+同hop BFS，不严格等于全局拓扑序。候选保留这个子序列；来源标签/分隔/MathSolver envelope计入token上限。
- 发现作者Node吞摘要异常且空摘要会落回原前驱正文。候选已同时拦截这两条路径；baseline实例不装该拦截。官方AQuA `postprocess_answer`静态方法签名有self，direct路径需扩展层绑定修复，详细披露见PAPER_BASELINES.md。
- 16项离线检查全部通过；27份取源资产哈希一致，45份作者src文件和Git checkout保持干净。既非GPU验证，也非MiniLM质量验证，不给采样优势作证。
- 表征pooling接口支持caller-supplied hidden/KV和显式位置对齐检查；当前Ollama仍未导出真实hidden/KV。当前运行时可读取保存原文做token计数，尚不是跨主机按需传输实现。
- 引用报告：`PASS: 4/4 verified, 0 errors, 0 warnings (0 checks skipped)`。此轮使用`--no-canonical-instance`，同名规范实例检查因首次429未完成；仅将arXiv身份/标题/作者核验称已完成。

## 已确认事实

1. MOC 作者代码的 `get_neighbors_by_hops(node_id, max_hops)` 已支持任意最大 hop；当前论文实验使用 1/2 跳不等于算法只能使用两跳。
2. `async_execute` 在接收方调用 `get_neighbor_summary_with_ism` 后才把摘要放入最终 prompt；选择器应位于正文被接收方消费之前。
3. MOC 当前句子编码器是 `all-MiniLM-L6-v2`。它服务于消息相似度排序、配对和候选合并结果评分，不是 tokenizer，也不负责答案正确性验证。
4. 当前 Ollama wrapper 只读取 `response['message']['content']`。直接得到主模型 hidden/KV 需要换用或改造可导出中间状态的后端。
5. Programming Expert 的执行结果会在生成后追加到文本；该工具后缀不能被默认视为原始生成 KV 的一部分。

## 理论迁移

- GraphSAGE：分层邻居预算；
- FastGCN/LADIES：按目标节点和层条件化的采样；
- GAT：接收方条件的邻居权重，但注意权重本身不等于通信节省；
- Jumping Knowledge：不同节点可以使用不同的有效感受野；
- Personalized PageRank：作为多跳影响的先验，可用于候选排序，但不能替代任务效用。

## 待证假设

- 3/4 跳中存在少量相对于一跳状态具有增量价值的祖先；
- 增量价值可以在不消费完整正文的情况下由廉价状态和 sketch 预测；
- 递减 hop 预算比固定 K 或等预算随机选择更有成本—质量优势；
- hidden/KV sketch 能比独立 MiniLM 更好地表达同一开源模型接收方的需求。

以上均为假设，当前没有新增实验支持。

## 已停止的方向不作为主线

摘要丢失反例、目录加一次展开、验证迁移和成本感知合并都保留为历史分析或工程基线，不继续追加小规模诊断来替代方法设计。
