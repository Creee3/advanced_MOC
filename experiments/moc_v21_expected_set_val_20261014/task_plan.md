# v2.1 云端 val

用户2026-10-14明确要求展开v2.1实验。该授权取代前轮“OOF门槛失败则不部署”的本轮工作选择，原失败结果保留；新实验是诊断性全图验证，不宣称训练内门槛通过。

1. 核对云端残留、原val60、基准匹配与冻结模型 — complete；原服务与端口清理，当前1GPU满足原门槛。
2. 实施val-only脚本、冻结协议/config/checkpoint/源码并提交 — complete；compact22固定，保留2shards共享1serial GPU。
3. 模型无调用预检、部署SHA核验、CUDA/短生成/全显存驻留门槛 — complete；2原DAG、35文件hash、CUDA实际分配/OK生成/size_vram=size通过。
4. 启动有界120条件，核对真实生成、进度与错误 — complete；2026-10-14 21:31北京时间核对9/120全部有效、raw持续更新，无中断。全批仍在运行。
5. 完成后下载raw/hash、分析质量成本与行为、核验资源清理 — pending。

不重采训练、不重跑既有基准、不跑test、不用val调参。新唯一运行目录，禁止重置状态/预算或恢复旧任务。

## Errors
- 初次读取误用了不存在的evaluate_val.py文件名；已核对真实入口为run_val.py，无运行发生。
- 冻结前审查修正warmup未提前预留费用及活动results末行部分写入风险；无运行发生。
- 管理检查首次远端正式目录尚未部署，因此测试读取失败；后用独立maintenance staging完成4项model-free模拟，不触碰运行目录/模型/GPU。
