# Progress

2026-10-14用户明确授权开展v2.1实验，并要求先启动再简要解释版本演进。创建独立val-only目录，核对现有云端资源/残留与冻结checkpoint，改造新方法运行入口；尚未启动GPU。原v1/v2/v2.1离线原件保持完整。

只读云端两次10秒检查：旧PID不在、11448/49/50空，author commit/status匹配；仅GPU0符合util<=15/free>=21000，所以冻前设1main+原2workers/shards串行共享，保持原门槛。prepare_val复制60原问题无gold，联合5224历史请求、4899可复用、325矛盾键全排除。run_val支持终态条件不重复和原费用断点保留；监督器有界2h/1200调用/150万新token，warmup也预留计费，不重置预算/越过STOP。checkpoint SHA e2ce9cd7f69d3b4e6c2e6be7af1df0a852d44b972396989e5b02428a655fd210；35文件部署包SHA0843b5357a9db380f3c843a27af41282c5d315f84d0e028f563eace77d937c2b。模型无调用预检与GPU实际门槛尚待。
