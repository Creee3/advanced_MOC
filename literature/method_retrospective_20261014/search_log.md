# 检索与范围日志

2026-10-14；定向叙述复盘；一个审阅者；非PRISMA系统综述。初始问题：GNN迁移是否仍在执行？何种监督能区别相关性与接收模型效用？哪些MAS近邻约束创新？

搜索引擎发现，结论仅用原始来源。实际查询：

1. LADIES layer dependent importance sampling graph convolutional networks NeurIPS 2019
2. LLM multi agent communication pruning AgentPrune 2024 2025
3. Learning to Retrieve In-Context Examples for Large Language Models LM supervised retriever LLM-R
4. site.arxiv.org MOC multi order communication multi agent yao guan
5. site.aclanthology.org REPLUG retrieval augmented black box language models
6. site.proceedings.mlr.press Ross Gordon Bagnell reduction imitation learning structured prediction 2011
7. site.aclanthology.org "Learning to Retrieve In-Context Examples for Large Language Models"
8. site.github.com "SafeSieve"
9. site.github.com "AgentPrune" "Cut the Crap"

读取来源：MOC/arXiv HTML；LADIES/NeurIPS原始记录及此前锁定代码；LLM-R/arXiv方法4.1–4.4、ACL正式记录、官方实现；REPLUG/ACL原始论文；DAgger/PMLR原文和记录；AGP/arXiv方法正文；AgentPrune/arXiv记录、官方README和训练源码；SafeSieve/AAAI正文PDF、官方代码。工具致谢arXiv2609.00065当前记录另核对。

纳入8篇研究来源，工具致谢单列，不当第9篇方法研究。不报告未完整记录的搜索命中/排除总数。第三方摘要、Reddit、聚合站不作为机制证据；GraphSAGE/FastGCN/AS-GCN沿用原迁移文档，仅背景，不新增并列主线。LLM-R、REPLUG、DAgger都是迁移线索，不能转移其原任务效果或理论前提。

访问限制：GitHub网页工具读取LMOps/REPLUG返回错误，随后官方API和raw访问确认LMOps；REPLUG API明确404。AgentPrune OpenReview PDF遇浏览器验证、ICLR PDF本轮读取失败，用官方仓库README会议引用及代码、arXiv记录交叉核对，不能声称已精读其正式会议PDF。SafeSieve正式AAAI标题与早期arXiv标题次序有差异，本稿用正式记录。

GitHub代码获取只读指定文件，未下载数据集、未执行仓库命令。仓库metadata及code_manifest记录commit和文件SHA。Star数量不是论文质量或正确性的证明。

学术索引机器检查使用用户先前明确提供的联系邮箱，仅用于本地进程请求头；不写入本报告或仓库配置。首次默认canonical查询Semantic Scholar持续429退出，受限重试关闭canonical查询完成，随后默认检查再试一次。详细结果见citation_status.md。
