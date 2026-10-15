# 进度

- 2026-10-15：用户已批准 GSM8K 24/120/300 与 MMLU-Pro 14/56/280，并授权最多两张 GPU 并行适配阶段、每 10 分钟监控。
- 数据清单已生成并冻结候选内容：GSM8K 官方 train7473，MMLU-Pro 官方 validation70；作者 test 文件 SHA 与 split_manifest 已记录。三 split 规范化题文交集为 0。
- 已完成原仓库扩展：MMLU-Pro AnalyzeAgent/FinalRefer 原生路径适配；MathSolver 与 AnalyzeAgent 的执行观察器、受限代码执行接口共用。
- 正在完成运行时预算异常传播、断点审计、CPU 预检和 per-dataset bundle。尚未启动云端 GPU。
- 后续顺序：回归与 CPU 预检 → 本地提交 → 上传快照 → 两批适配阶段并行 → 串行原 MOC baseline → 下载原始记录、评分和审计。

## 2026-10-15 19:14 北京时间：现场监控

- 协调器 PID 2439532 的真实 cmdline 为当前 coordinator.py，仍处于 parallel_adaptive。
- GSM8K：24/24 训练题有效；frozen-transfer val 144/240 条条件（两密度各72）；adapted val/test 尚未开始。
- MMLU-Pro：14/14 训练题有效；frozen-transfer val 112/112、adapted val 112/112 均完成；已进入 adapted author-test，46/560 条条件（密度0.3为24，0.7为22）。
- 两批 supervisor、各两个 worker、主服务均以 /proc/cmdline 核实存活；活动 raw 最后写入分别为0.3/3.1秒和16.5/6.0秒前；无STOP或supervisor错误。
- 私有11448/11449的实时 /api/ps 均为正确27B digest，size_vram=size=16757418884；保持两张主GPU，未启动第三张或重复launch。
- GSM实际1145401 token、pending20384、unknown0；Pro实际1397274、pending20384、unknown0；各自16000000上限内，绝对窗口剩余23684/23703秒。
- 完整现场记录：healthcheck_20261015_191433.json。未读取测试答案、未评分、未调参；继续既定调度。

## 2026-10-15 19:30 北京时间：新授权的独立原MOC val

- 用户针对val对比请求补原MOC，随后明确允许临时新增两张空闲GPU（总上限四张）。新增moc_mmlu_pro_val_baseline_20261015，原两批源码/配置/结果不改；同val56、两个密度共112条件，无训练/test。
- GPU0/4连续两次利用率/空闲显存满足门槛；43冻结文件上传SHA核对、父val字节同一、CPU6图合成条件与隔离预检通过。
- 19:30:18新supervisor2497046启动，19:30:59已进入val/moc；GPU0的27B/11451与GPU4的9B/11452均实际CUDA/短生成与全显存驻留通过，两个worker真实存活。原GSM val与Pro adapted test继续。
- 新批7200秒/6000调用/4000000token独立限制；本聊天监控moc工具更新成功，包含本补测与新增GPU授权、费用和清理规则。
- 准备与部署期间首次读取OpenAI Docs技能路径误用skills根目录，已更正至.system目录；未因此中断实验。详情新批progress与launch_inspection.json。

19:33 heartbeat：三批均健康，无恢复/重启/调参。GSM frozen-transfer val207/240；Pro adapted test96/560；新增原MOC val4/112（两密度各2）。全部已提交条件ok、无重复，所有监督器/worker/服务真实cmdline活着；活动raw年龄分别6.4/2.0、12.6/4.0、8.0/2.2秒。四服务实时digest与size_vram=size正常，无STOP/error。actual token分别1383830、1657860、40275；各pending20384、unknown0，窗口剩余22563/22583/7024秒。完整证据healthcheck_20261015_193314.json。阶段未变，按监控约定不发额外状态通知。

19:43 heartbeat：三批继续健康。GSM frozen-transfer val238/240；Pro adapted test108/560；新增原MOC val37/112（0.3:19，0.7:18）。全部终态ok，无重复/错误/STOP；协调器及全部登记进程cmdline存活。活动raw年龄分别5.6/2.1、6.7/4.0、2.8/0.9秒；三批全部source/data锁逐文件SHA核对通过，四服务实时全显存驻留正常。actual tokens1502295/1762157/276524，各pending20384、unknown0；原窗口剩余21970/21989/6431秒。证据healthcheck_20261015_194307.json。未重启、未扩样本、未读取test gold或评分。阶段未变，保持安静。

19:53 heartbeat阶段变化：GSM frozen-transfer val240/240完成，已转adapted val58/240（两密度各29）。Pro adapted test130/560；原MOC补测val74/112（各37）。三批source/data哈希全匹配、实际进程cmdline存活，四模型全显存驻留；活动raw均9秒内，无重复/invalid/STOP/error。actual token1644108/1896150/525836，各pending20384、unknown0；原截止剩余21354/21373/5814秒。证据healthcheck_20261015_195324.json。无重启或策略改动。新增check_cloud.py为本地只读监控入口，--capture读取云端三批inspect与锁、费用及实时模型信息并保存完整快照，不读取任何gold或计算准确率。

20:03 heartbeat：GSM adapted val113/240，Pro adapted test约164/560仍健康；新增原MOC val从111完成至112，全部有效，两个新增服务/子进程/端口独立核实清理。完整raw归档及77文件SHA核对，费用、缓存、原解析器、112覆盖和冻结锁审计passed，实际802056token/1400call，unknown/pending/截断均0。原MOC val29/56(.3)、28/56(.7)；adapted30/56、27/56，推理logicaltokens−10.26%/−52.94%，训练408512实际token另列。三组报告在Pro父实验val_review_20261015/三组val对照结果.md。新增补测已完成且不得重启；总监控保持ACTIVE继续原两批。原现场healthcheck_20261015_200254.json的inspect与覆盖读取发生在worker提交间，Pro163/164差异为只读快照时序，不是丢失或重复。
