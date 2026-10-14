# Progress

2026-10-14用户明确授权开展v2.1实验，并要求先启动再简要解释版本演进。创建独立val-only目录，核对现有云端资源/残留与冻结checkpoint，改造新方法运行入口；尚未启动GPU。原v1/v2/v2.1离线原件保持完整。

只读云端两次10秒检查：旧PID不在、11448/49/50空，author commit/status匹配；仅GPU0符合util<=15/free>=21000，所以冻前设1main+原2workers/shards串行共享，保持原门槛。prepare_val复制60原问题无gold，联合5224历史请求、4899可复用、325矛盾键全排除。run_val支持终态条件不重复和原费用断点保留；监督器有界2h/1200调用/150万新token，warmup也预留计费，不重置预算/越过STOP。checkpoint SHA e2ce9cd7f69d3b4e6c2e6be7af1df0a852d44b972396989e5b02428a655fd210；35文件部署包SHA0843b5357a9db380f3c843a27af41282c5d315f84d0e028f563eace77d937c2b。模型无调用预检与GPU实际门槛尚待。

协议源码config/model于commit c42af01冻结并推送后部署；远端同SHA包通过，preflight验证0.3/0.7两原DAG（7/15edges）、模型/60输入/cache，无main调用/GPU分配。4项管理mock通过（warmup实际费用、预算不足不发请求、活动末行不误报、坏内行不吞）。21:27:11北京时间真实监督器1798930启动，模型服务1799002、worker1799575/1799576共享私有11448；运行时GPU1已释放且连续满足原门槛，由冻结调度器选中，并非更改GPU限制。CUDAcuMemAlloc0、真实OK.短生成、size_vram=size=16757418884通过。21:31:16核对9/120提交、全有效、无错误，raw新鲜且PIDcmdline匹配，细节runtime_check.json；完整结果/统计尚未完成。本批仅新方法，不新训练/cal/baseline/test，不创建或恢复旧监控。

2026-10-14 22:04:52北京时间监督器正常complete，120/120全部有效；22:17现场核对所有自有PID消失、11448/49/50关闭。生成终态只读raw归档65文件，SHA bfe2e50d8dbe5bb2eff672328e73cee64dc5f0f49a333b0144de1efee681a509，已下载并逐项核对。新建val-only本地analyze/audit，不重跑推理。

分析：稀疏29/60、301925token；稠密35/60、298387token。对旧MOC两密度分别−10.91%/−62.41%token且多5/7题，准确率bootstrap95%仍跨0。原稀疏MOC旧截断题共同剔除敏感性保持正向。普通节点空集104/130、单条252/174、双条4/56；不是v1全停或v2全读，尚无选取机制因果归因。

审计：33冻结hash、缓存逐调用/计费、原最终答案解析、决策argmax/预算、无截断/超时、0预留、资源清理全部通过。检查发现本地/远端作者AQuA完整文件SHA不同，定位为CRLF/LF；归一化全文相同，远端作者commit与clean status已独立验证，证据author_parser_remote_check.json。原文字符串的SyntaxWarning来自既有非法转义，未修改作者源码。完整报告v2.1完成结果与首次分析_20261014.md。
