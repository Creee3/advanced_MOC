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

20:12 heartbeat：GSM adapted val174/240（此前frozen240已完成），Pro adapted test198/560，仍为各两密度；原MOC val补测保持complete112/112与清理完成，已关闭端口拒绝连接是正常终态，不恢复。两批活进程及原始日志更新正常（raw2.4/5.5秒、15.1/2.3秒），无invalid/重复/STOP/错误；三批source/data锁仍匹配。运行两批actual1914066/2206888tokens、各pending20384/unknown0，剩余原窗口20186/20205秒。完整证据healthcheck_20261015_201251.json。无阶段变化/新行动，保持安静。

20:22 heartbeat：GSM adapted val235/240，Pro adapted test215/560；两批实际进程与raw持续活跃（5.2/1.6秒、4.4/2.2秒），source/data锁全匹配、模型digest/全显存驻留正常，无invalid/重复/STOP/error。两批actual2057467/2339096tokens、各pending20384/unknown0；原窗口剩余19586/19606秒。补测仍complete且已清理，不恢复。证据healthcheck_20261015_202251.json。阶段未变，未评分test或调参，未重复launch。

20:32 heartbeat阶段变化：GSM两轮val均240/240完成（合计480/480有效），已进入adapted作者test40/600（各密度20）；Pro adapted test235/560。两批source/data锁及真实进程/模型全显存驻留正常，活动raw分别5.1/2.2和7.4/6.0秒，无invalid/重复/STOP/error。actual2177392/2467871tokens、各pending20384/unknown0，原截止剩余18986/19005秒。补测保持已审计complete/已清理，不恢复。证据healthcheck_20261015_203252.json。没有读取test答案或调参，未重复launch。

20:42 heartbeat：GSM adapted test80/600，Pro adapted test268/560；两批持续健康、实际cmdline存活，raw1.9/4.1及1.5/0.3秒前更新。源码/数据锁和GPU模型digest/全显存驻留正常，无invalid/重复/STOP/error。actual2298553/2627932tokens，各pending20384/unknown0，原窗口剩余18387/18406秒。补测保持complete/清理完成，不恢复。证据healthcheck_20261015_204251.json。无阶段变化或可行动故障，不通知、不重启、不评分test。

20:52 heartbeat：GSM adapted test121/600，Pro adapted test308/560。两批登记进程真实存活、raw5.8/0.7及0.5/3.1秒前更新；源码/数据锁、模型digest及全显存驻留均正常，全部终态ok无重复，无STOP/error。actual2417354/2824684tokens、各pending20384/unknown0；原窗口剩余17785/17804秒。补测保持已完成和清理，不恢复。证据healthcheck_20261015_205253.json。无阶段变化/可行动异常，保持安静、不重启、不查看test gold或调参。

21:02 heartbeat：GSM adapted test164/600；Pro inspect335/560、覆盖读取时336/560（并发提交时序），所有终态ok且无重复。两批真实进程存活、raw1.5/5.0及0.2/1.3秒前写入；source/data锁与GPU模型digest/全显存驻留正常，无STOP/error。actual2541530/3007499tokens、pending20384/10192、unknown均0；原窗口剩余17183/17202秒。补测仍已完成且清理，不恢复。完整证据healthcheck_20261015_210255.json。阶段未变，无需通知或重启；未读取test gold或调参。

21:12 heartbeat：GSM adapted test207/600、Pro365/560；实际进程、raw更新（4.1/2.3秒和6.3/2.7秒）、source/data锁与两模型全显存驻留均正常。全部终态ok且无重复，无STOP/error；actual2667875/3169150tokens、各pending20384/unknown0；原窗口剩余16583/16602秒。补测保持complete及清理完成。证据healthcheck_20261015_211255.json。无阶段变化、无恢复、无新实验或test评分，保持安静。

21:22 heartbeat：GSM adapted test255/600、Pro392/560；所有终态ok无重复。登记进程真实存活、活动raw0.0/2.8秒及5.8/3.7秒前更新，source/data锁、模型digest与全显存驻留正常，无STOP/error。actual2796223/3307222tokens、pending10192/20384、unknown均0；原窗口剩余15983/16002秒。补测保持已完成且清理，不恢复；其剩余窗口与完成任务无关。证据healthcheck_20261015_212255.json。阶段未变、不重启、不读取test gold、不调参，保持安静。

21:32 heartbeat：GSM adapted test299/600、Pro430/560。两批实际进程与raw（6.1/0.5秒、3.4/2.3秒）持续活跃，冻结source/data哈希及GPU模型digest/全显存驻留正常，终态均ok无重复、无STOP/error。actual2921298/3473924tokens、各pending20384/unknown0；原窗口剩余15385/15404秒。补测仍complete且清理，其原窗口现已过期亦不影响已审计完成，不恢复。证据healthcheck_20261015_213253.json。无阶段变化或可行动异常，保持安静、未重启或评分test。

21:42 heartbeat：GSM adapted test340/600、Pro460/560；所有终态ok无重复，两批登记cmdline存活，raw4.6/0.1与2.4/0.7秒前更新。源码/数据锁、模型digest和全显存驻留正常，无STOP/error。actual3038414/3620375tokens、各pending20384/unknown0；原窗口剩余14786/14805秒。补测保持已审计complete与资源清理，绝不恢复。证据healthcheck_20261015_214252.json。阶段未变，无新行动，保持安静；未评分test或改动冻结策略。

21:52 heartbeat：GSM inspect381/600、覆盖读取时382/600（并发提交时序），Pro490/560；终态均ok且无重复。登记cmdline、raw3.4/6.4秒及3.0/1.0秒前更新、源码/数据锁和模型digest/全显存驻留正常，无STOP/error。actual3160977/3766882tokens、各pending20384/unknown0；原窗口剩余14185/14204秒。补测保持complete且清理，不恢复。证据healthcheck_20261015_215253.json。阶段未变、无可行动异常，不重复launch、不查看test gold或调参，保持安静。

22:02 heartbeat：GSM adapted test425/600；Pro inspect507/560、覆盖读取508/560（并发提交），均ok且无重复。真实进程、活动raw3.5/5.7与0.2/1.5秒前更新、源码/数据锁及GPU模型digest/全显存驻留正常，无STOP/error。actual3284220/3887684tokens、pending20384/10192、unknown均0；原窗口剩余13586/13604秒。补测已完成、清理与费用不变，不恢复。证据healthcheck_20261015_220253.json。无阶段变化或可行动问题，保持安静、不重启、不评分test或调参。

22:12 heartbeat：GSM adapted test470/600；Pro inspect538/560、覆盖读取539/560（并发提交时序），均ok无重复。两批cmdline真实存活，活动raw7.2/4.1与0.6/1.7秒前更新；源码/数据锁、模型digest及全显存驻留正常，无STOP/error。actual3401769/4043314tokens、各pending20384/unknown0；原窗口剩余12985/13003秒。补测保持complete且清理，不恢复。证据healthcheck_20261015_221254.json。阶段未变，无额外通知/重启/新实验或test评分。

22:22 heartbeat阶段变化：MMLU-Pro已adaptive_complete，作者test560/560（各密度280）全有效，val224/224、train14/14全有效；所有自有worker/服务/子进程及11449端口已清理，actual4127233tokens/5134newcalls，pending0/unknown0，原窗口剩余12401秒。此为适配阶段完成，尚有原MOC作者test对照，不能将正常退出当作失败恢复或提前评分部分test。GSM adapted test514/600仍健康，真实cmdline和raw2.5/5.2秒前更新，actual3527546tokens、pending20384/unknown0，原窗口剩余12383秒。三批锁哈希正常，已完成val补测保持收口。协调器真实存活，按原计划等待GSM也清理后串行原MOC作者test，不另行启动GPU或重复launch。证据healthcheck_20261015_222256.json。

22:32 heartbeat：GSM adapted test inspect560/600、覆盖读取561/600（并发提交），均ok无重复；cmdline真实存活，raw2.9/0.1秒前写入，actual3654365tokens、pending10192/unknown0，原窗口剩余11783秒。Pro仍adaptive_complete560/560且已清理，pending/unknown0，不恢复；等待协调器在GSM完成后按序跑原MOC。三批锁哈希匹配，唯一活动主服务digest及全显存驻留正常，无STOP/error。补测仍complete/已审计，不恢复。证据healthcheck_20261015_223256.json。阶段未变，无需通知、无重复launch、未评分test或调参。

22:42 heartbeat发现运行管理故障：GSM已adaptive_complete600/600全有效，cleanup完成、actual3760366tokens/5211newcalls、pending/unknown0；Pro已adaptive_complete560/560且cleanup完成。协调器22:41:34在开始GSM原MOC前退出，报existing live supervisor; do not duplicate。其检查仅用/proc/PID/cmdline.exists，与最后cleanup状态发布/监督器退出存在时序竞争。独立探针确认原两批全部登记PID已不存在，11448/49/50全部关闭，无STOP、无未知/未结算费用；不是模型/算法/题目失败。原baseline未启动、未提交，不重跑适配结果。

22:46已完成同根因第1次有界恢复：归档两批全部原始请求响应/费用/结果/管理文件及原协调器源码/错误，241文件在本地逐个SHA核对，归档SHA4dfc8e9f12c338ab2a3d7212ddef7707a0712822c3eb59979879da96091c1949，唯一云端attempt=transition_recovery_20261015T144545_646549Z。本地归档与manifest在recovery_transition_20261015/。只改coordinator.py：读取真实cmdline/进程状态，忽略空cmdline/zombie；cleanup完成后仍活的旧监督器最多等30秒，不信号旧PID；pending/清理/STOP/原截止均阻止启动。7项CPU隔离回归通过，涵盖禁止重复launch、截止、STOP、pending、cleanup与只续baseline。首个大上下文apply_patch未匹配，改小上下文成功；没有修改数据集任何冻结源码/配置/策略锁。

22:46:51协调器2612991恢复，保留两批原budget/run_start字节及全部终态，按原序GSM原MOC→Pro原MOC串行，各自仍最多27B+9B两张，绝不延长原窗口。恢复评审/launch记录及patchSHA39edfa90df4e1b6b8fa50d14cad153b911994d4e95d11198067d376057366a19已保存。等待真实GPU/worker核验后再报告启动。

22:47:34恢复后现场确认：协调器处于serial_moc_baseline/GSM，GSM监督器和两个test/moc worker的真实cmdline活着，11448的27B、11450的9B均正确digest、CUDA短生成与全显存驻留通过；原始raw已持续写入（5.2/4.6秒），首批MOC条件处理中，全部600个适配终态保留。实际费用3760398tokens（新增热身32已计入），pending20384/unknown0，原绝对窗口仍剩余10905秒。Pro仍adaptive_complete560/560、pending0，等待串行下一阶段，不重跑。数据集source/data锁均通过；旧cleanup.json是前阶段已归档记录，当前活动status.cleanup_complete=false，不能误读为新阶段清理完成。完整证据healthcheck_20261015_224734.json。

22:52 heartbeat：GSM原MOC test16/600（.3:9，.7:7）有效，适配600终态保留，总test616；真实协调器/监督器/两个worker和两模型正常，raw5.3/0.7秒前更新，冻结锁匹配，无invalid/重复/STOP/error。累计actual3888581tokens、pending32768、unknown0，原窗口剩余10584秒；Pro仍adaptive_complete且清理、actual4127233、pending/unknown0、剩10602秒，依原序等待MOC。独立val补测仍已完成，不恢复。证据healthcheck_20261015_225255.json。阶段无变化，无重复launch或test评分，保持安静。

23:02 heartbeat：GSM原MOC46/600（.3:24，.7:22）有效，总test646；真实协调器/监督器/worker正常，raw5.2/12.5秒前更新，source/data锁和两模型digest/全显存驻留匹配，无invalid/重复/STOP/error。累计actual4111981tokens、pending32768/unknown0，原窗口剩9983秒。Pro适配阶段仍560/560已完成且清理，pending/unknown0、原窗口剩10001秒，按原序等待MOC；已完成val补测不恢复。证据healthcheck_20261015_230256.json。无阶段变化/可行动异常，不重复launch、不评分部分test、不延长原截止。

23:10用户要求下午至今全部实验汇总：核对AQuA/HumanEval完成报告、analysis与audit的receipt哈希；现场GSM原MOC70/600仍健康（适配600保留），Pro适配560且清理、原MOC尚待串行。补做已完成GSM val的本地CPU评分，下载12文件SHA/冻结与gold/原解析器核对，适配116/120与115/120、冻结迁移113/120与112/120，无新生成或test评分/调参。含5批/4数据集的汇总、所有口径限制和整批累计费用在../daily_summary_20261015/实验汇总_20261015下午至晚间.md。现场证据healthcheck_20261015_231036.json；账单是各批自启动累计，不伪称严格12:00后的消费。
