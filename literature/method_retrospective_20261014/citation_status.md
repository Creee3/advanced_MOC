# 引用核验状态

VERDICT-LINE: PASS: 9/9 verified, 0 errors, 1 warnings (0 checks skipped) — 1 MISSING_DOI

这是使用 --no-canonical-instance 的受限机器检查原文，详见citation_check.json。9条包含8篇研究论文和1篇工具致谢。它不代表默认完整检查通过：默认canonical-instance查询两次均因Semantic Scholar HTTP429在MOC条目中途退出，exit1，未产生完整JSON。关闭选项的检查不进入脚本checks_skipped计数，故本项目另外明确记录该范围缺失；没有把默认检查失败清洗成全通过。

唯一机器警告：AgentPrune会议条目MISSING_DOI，检索返回10.48550/arXiv.2410.02506。官方README提供ICLR2025/OpenReview引用，本稿保留作者会议版本，不把预印本DOI填成会议DOI。该警告尚未作为投稿引用问题关闭。其他正式记录优先使用ACL/AAAI/PMLR/NeurIPS来源，人工核对不替代未完成的自动同名版本检索。

本轮未发现身份错误或撤稿标记，不等于全面撤稿保证。此稿是内部研究对齐，正式投稿前应补做默认全量检查。用户提供的联系邮箱仅用于请求头，不保存在报告与Git。

默认失败证据：HTTP429 api.semanticscholar.org/graph/v1/paper/search，query为MOC标题；指数退避2/4/8秒后失败。第二次相同，停止进一步重复请求。
