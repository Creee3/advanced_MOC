# v2.1 云端只读预检查

检查时间：2026-10-14 21:16 北京时间。SSH 目标 DevContainer，hostname liuxin-1，uid 1033（liuxin）。本次没有修改远端文件、杀进程、启动 Ollama 或分配 CUDA。

## 旧任务与私有端口

- v1 状态 complete，540 条件已提交；保存的 supervisor、worker、服务 PID 均不存在。
- v2 状态 complete，120 条件已提交；保存的 supervisor、worker、服务 PID 均不存在。
- 调用了两批现有 inspect_run.py，核对实际进程和提交记录，未仅凭旧 status 推断。
- 当前 uid 的 /proc 中未发现这两批目录相关的进程或 Ollama；11448、11449、11450 均无监听。
- nvidia-smi 显示一些 GPU 进程为 [Not Found]，LXC 中不可见。不能把这些显存当成本任务残留，也未尝试清理。

## 资源

两次快照相隔 10 秒。按原规则 util ≤15%、free ≥21000 MiB 连续两次满足，仅 GPU 0 合格（free 21227 MiB）。其余低利用率卡多为 20743 MiB，低于现有阈值；GPU 1、6 有较高利用率。

因此当前不满足原两个 main worker 同时启动条件。没有降低显存阈值，没有 CPU 回退。单 worker 是否采用，应由新协议明确；实际 launch 前仍需重新核验资源、CUDA 分配、短生成和全显存驻留。

## 现有依赖与复用入口

- gemma2:27b manifest SHA256 = 53261bc9c192c1cb5fcc898dd3aa15da093f5ab6f08e17e48cf838bb1c58abfe，与冻结模型 digest 相同。
- tokenizer SHA256 = 3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922，与冻结记录相同。
- Python、Ollama、作者源目录、模型目录、MiniLM 缓存、现有 cuda_v13 库均存在。
- 可复用运行管理源在 /home/liuxin/ICML2027/experiments/moc_v2_evidence_bundles_train_20261014/：inspect_run.py、gpu_runtime.py、launch.py、supervise.py、runtime_controls.py、common.py、trace_hooks.py。run_val.py 可作为新 val-only runner 的参考。
- 完整快照、已存 PID 验证、提交记录计数和两次 GPU 快照见 cloud_preflight.json。

本报告仅说明当前资源与旧任务状态，不是新策略性能或 CUDA 推理验证。
