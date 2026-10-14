# Progress

2026-10-14用户明确授权开展v2.1实验，并要求先启动再简要解释版本演进。创建独立val-only目录，核对现有云端资源/残留与冻结checkpoint，改造新方法运行入口；尚未启动GPU。原v1/v2/v2.1离线原件保持完整。

只读云端两次10秒检查：旧PID不在、11448/49/50空，author commit/status匹配；仅GPU0符合util<=15/free>=21000，所以冻前设1main+原2workers/shards串行共享，保持原门槛。prepare_val复制60原问题无gold，联合5224历史请求、4899可复用、325矛盾键全排除。run_val支持终态条件不重复和原费用断点保留；监督器有界2h/1200调用/150万新token，warmup也预留计费，不重置预算/越过STOP。checkpoint SHA e2ce9cd7f69d3b4e6c2e6be7af1df0a852d44b972396989e5b02428a655fd210；35文件部署包SHA0843b5357a9db380f3c843a27af41282c5d315f84d0e028f563eace77d937c2b。模型无调用预检与GPU实际门槛尚待。

协议源码config/model于commit c42af01冻结并推送后部署；远端同SHA包通过，preflight验证0.3/0.7两原DAG（7/15edges）、模型/60输入/cache，无main调用/GPU分配。4项管理mock通过（warmup实际费用、预算不足不发请求、活动末行不误报、坏内行不吞）。21:27:11北京时间真实监督器1798930启动，模型服务1799002、worker1799575/1799576共享私有11448；运行时GPU1已释放且连续满足原门槛，由冻结调度器选中，并非更改GPU限制。CUDAcuMemAlloc0、真实OK.短生成、size_vram=size=16757418884通过。21:31:16核对9/120提交、全有效、无错误，raw新鲜且PIDcmdline匹配，细节runtime_check.json；完整结果/统计尚未完成。本批仅新方法，不新训练/cal/baseline/test，不创建或恢复旧监控。
