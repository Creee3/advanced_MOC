# Findings

不重划已接触test；用户明确沿用官方test254并保留限制。当前GPU只有1张达到27B门槛，可另用闲置但显存较少的GPU放9B。两组共用27B服务，不能用墙钟宣称独立双GPU延迟比较。

旧六组全图test仅two_hop_raw/independent50全254和no_communication33；它们不是本次同配方MOC的完整结果。可复用精确调用但不能把不同算法结果冒充MOC，避免重做兼容调用同时保留费用。

## 2026-10-15 用户批准恢复

用户明确回复“继续跑呗”，授权按剩余22125秒窗口续跑，累计此前6674.112595秒，保留原30000调用/12000000 token上限。已再次检查旧5个PID均不存在、私有服务未运行。运行修复增加费用未知项的独立保守上界预算检查，并排除超时工具结果缓存，防止跨条件缓存把超时当成功；算法、checkpoint、输入、角色、解码与5秒工具上限不变。五项本地回归通过，待云端原Graph四条件fake预检通过后启动。恢复清单见runtime_recovery_manifest_20261015.json；原324终态全部保留，已证明的代码超时MOC条件将原样补记invalid，不重生成。

## 2026-10-15 12:08发现新中断并恢复

11:58:35北京时间监督器在737/1016终态时失败并完成全部自有进程/端口清理。此次是新的已核实根因：shard0原MOC密度0.7、题67969c...ccb6da83的9B合并候选两次触发Ollama token repeat limit（HTTP500），没有完整usage；worker自己的费用未知保护正确停止。其他worker一个merger请求被监督器终止，合计新未结算49152（3x16384）token上界。

先唯一归档80文件，归档fcf6e51d3a88b765aeb2f14434d16ba5e586031d46c9772024eeefcc05e2cd3a，下载并逐文件SHA全部核对。原已知实际4148592 token/6109已知调用与账本6113调用吻合（含旧1+新3未知调用）。将三笔预留保守分账，累计unknown_cost_upper_bound_tokens=59344、unknown_call_count=4；actual_tokens仍4148592，调用不重置，冻结reserve已对上界计入原12M cap。原始请求/错误/费用和trace均保留，给三reservation追加reconciled_unknown审计事件。不能把缺usage的调用当零或猜精确usage。

同一失败MOC条件从原8条main回复与所有已知merger记录补记invalid，不采信作者fallback最终答案，也不重新生成；原737终态不动。另worker只是被停的未提交工作，从兼容精确缓存续跑。一次性状态恢复脚本recover_repeat_error_20261015.py已审查/语法检查，核验所有源码/模型/数据锁无变化和STOP/清理/费用/原deadline后启动；未修改任何冻结可执行文件/算法/模型/输出上限。

12:10:36 launch --resume，监督器2239983。现738终态（补记1 invalid），待准入/新CUDA验证后继续；绝对截止15:47:54不变，未补回停机时间。新重复token根因恢复尝试1，后续同根因按协议至多2次、不能无限重启。恢复明细repeat_error_recovery_20261015.json，原状态见repeat_error_snapshot_20261015/manifest.json。
