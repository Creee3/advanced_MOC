# v2：成组证据与下游效用

## 目标
按用户注释启动 v2 改进。用户随后要求工作重心回到 train→val→test，取消独立 motivation 门槛；保留已冻结 v1、复用兼容基准和训练材料，完成成组/下游效用监督的正式训练与冻结后 val。

## 阶段
1. 核验原始训练材料、缓存可复用范围、图与角色覆盖 — complete。
2. 实现证据组构造、下游干预回放及可审计缓存 — complete。
3. 实现帮助/损害预测与不确定性停止的可替换接口 — complete。
4. 冻结 train/calibration/val 协议、离线验证并提交 Git — complete。
5. 在资源允许时有界采集、拟合/冻结及120个新方法val条件，下载审计并收口 — complete。

## Final outcome
47/47 train questions,378 labels,35fit/12calibration,120/120 val complete. No baseline reruns or test. Both per-density operational targets failed; poor calibration leads to almost-full reading and zero group reads/confidence stops. Raw79 files/hash verified, cost/label audit passed, owned cloud resources cleaned. Do not equate task execution complete with research hypothesis supported.

## 约束
- 不重复整批 Single/Vanilla/MOC/w/o Phi 基准；精确输入及解码匹配的调用直接复用。
- v2用原有效train47（35fit/12calibration），val60仅做开发验证，不用其结果改动本批模型/超参；本批不跑test，test等最终方法确定。
- 仅在分配 LXC 内，沿用 Gemma2/MiniLM/cuda_v13；禁止 CPU 回退、安装驱动/新模型及操作他人任务。
- 最多2张main GPU、固定2000输出上限，新批至多3h/2400新调用/250万新token；STOP或预算到期清理，不恢复旧监控。v1已终止，无重置其运行限额。
- 每个新结果保留原始请求/响应、模型实际 token 与缓存复用证据。未完成/截断单元不进训练。

## 反思时机
训练/校准完成后冻结并报告学习信号；val完成后报告收益、失败和未决限制。没有以3个案例筛选再额外加motivation批次；不无限扩样本。

## Errors
- Strict cache admission found conflicting historical outputs for the same exact request; exclude every ambiguous historical key and record provenance, never pick by correctness.
- Model-free preflight output capture conflicted with author's stdout.reconfigure; remove capture during import. No GPU allocation or model call had occurred.
- Fixed-material Graph hydration lacked arun-created spatial_logits. Assert original binary/nonoptimized masks and initialize unused zero logits before original edge construction; val still runs full arun. No model call/GPU yet.
