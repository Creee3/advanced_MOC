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

23:28 heartbeat：GSM原MOC122/600（.3:62，.7:60）有效，总test722；真实协调器/监督器/worker和两模型正常、raw3.9/5.5秒前更新，冻结锁匹配，无invalid/重复/STOP/error。累计actual4700369tokens、pending20384/unknown0，原窗口剩8421秒；Pro仍adaptive_complete560且清理，待原序MOC，pending/unknown0，窗口剩8439秒。val补测已完成，绝不恢复。证据healthcheck_20261015_232858.json。阶段未变，无重复launch/测试评分/截止延长，保持安静。

23:39 heartbeat：GSM原MOC153/600（.3:77，.7:76），总test753均有效无重复；协调器/监督器/worker真实存活，raw3.8/6.3秒前更新，source/data锁及两模型digest/全显存驻留正常，无STOP/error。累计actual4939772tokens、pending20384/unknown0，原窗口剩7817秒；Pro仍适配完成560、已清理、等待原序MOC，pending/unknown0，窗口剩7835秒。已审计val补测不恢复。证据healthcheck_20261015_233902.json。无阶段变化、无可行动故障，不重复launch、不评分test、不延长原截止，保持安静。


23:50 heartbeat：GSM原MOC185/600（.3:93，.7:92），含适配共785个test终态，全部有效无重复；协调器、监督器、两个test/moc worker和两服务真实cmdline存活，raw2.7/0.6秒前更新，source/data锁与模型digest/全显存驻留正常，无STOP/error。累计actual5184792tokens/7448新调用、pending20384/unknown0，原窗口剩7158秒。当前status.cleanup_complete=false，旧cleanup.json为上阶段记录，未误判。Pro仍适配完成560/560且已清理，actual4127233、pending/unknown0，窗口剩7176秒，按原序等待原MOC；已审计val补测保持complete且不恢复。证据healthcheck_20261015_235001.json。主协调器目录没有单独protocol.md，现有task_plan/progress及三批各自protocol/config/lock已读取核查。阶段未变、不重复launch、不评分test、不延长原截止，按约定不额外通知。

2026-10-16 00:20 heartbeat：GSM原MOC275/600（.3:138/.7:137），适配600终态保留，总test875全有效无重复；真实协调器/监督器/worker正常，raw2.0/3.1秒前更新，两服务digest/全显存驻留与source/data锁正常，无STOP/error。累计actual5880441tokens/8539新调用，pending20384/unknown0，原截止剩5361秒；当前status cleanup未完成，旧cleanup记录不作为活动阶段清理结论。Pro仍适配560/560已清理、待原序MOC，actual4127233/pending0/unknown0，原窗口剩5379秒。val补测已完成不恢复。证据healthcheck_20261016_001958.json。无阶段变化、不重启、不读test gold或评分、不延长截止。

2026-10-16 00:30 heartbeat：GSM原MOC304/600（.3:153/.7:151），适配600保留，总test904均有效无重复；真实协调器/监督器/worker存活，raw8.5/12.7秒前更新，冻结source/data锁与两模型digest/全显存驻留匹配，无STOP/error。actual6096653tokens/8892新调用、pending32768/unknown0，原窗口剩4760秒。Pro适配560已完成清理，pending/unknown0，原窗口剩4778秒，按原序待MOC；已审计val补测不恢复。证据healthcheck_20261016_002959.json。无阶段变化/可行动故障，不重复launch，不读test gold/评分，不延长原截止。

2026-10-16 00:40 heartbeat：GSM原MOC332/600（.3:167/.7:165），总test932全有效无重复；真实协调器/监督器/worker正常，raw9.9/12.3秒前更新，source/data锁及27B/9B digest/全显存驻留匹配，无STOP/error。actual6328588tokens/9253新调用、pending26576/unknown0，原窗口剩4161秒。Pro适配560已完成且清理，pending/unknown0，窗口剩4179秒，依原序等待原MOC。已完成val补测不恢复。证据healthcheck_20261016_003959.json。旧两批无阶段变化或故障，不重复launch、不评分test、不延长截止。

2026-10-16 00:50 heartbeat：GSM原MOC356/600（.3:179/.7:177），适配600保留，总test956均有效无重复；真实协调器/监督器/worker正常，raw4.9/7.4秒前更新，source/data锁及两服务digest/全显存驻留匹配，无STOP/error。actual6520337tokens/9543新调用、pending32768/unknown0，原截止剩3558秒。Pro适配560已清理、待原序MOC，actual4127233/pending0/unknown0，原截止剩3576秒；已审计val补测不恢复。证据healthcheck_20261016_005002.json。无阶段变化/故障，不重复launch，不读test gold/评分，不延长原截止。

2026-10-16 01:00 heartbeat：GSM原MOC382/600（两密度各191），适配600保留，总test982均有效无重复；真实协调器/监督器/worker健康，raw2.9/0.5秒前更新，source/data锁及两服务digest/全显存驻留正常，无STOP/error。actual6728634tokens/9871新调用、pending20384/unknown0，原截止剩2960秒。Pro适配560已清理、待原序MOC，pending/unknown0，原截止剩2978秒；已审计val补测不恢复。证据healthcheck_20261016_005959.json。阶段未变、不重启、不读test gold/评分、不延长原截止。

2026-10-16 01:10 heartbeat：GSM原MOC409/600（.3:205/.7:204），含适配总test1009均有效无重复；真实协调器/监督器/worker健康，raw8.4/0.3秒前更新，source/data锁及两模型digest/全显存驻留正常，无STOP/error。actual6947185tokens/10219新调用、pending20384/unknown0，原窗口剩2360秒。Pro适配560已清理、原MOC仍待原序，actual4127233/pending0/unknown0，原窗口剩2378秒；已审计val补测不恢复。证据healthcheck_20261016_011000.json。无阶段变化或可行动故障，不重复launch、不读取test gold/评分、不延长原截止。

2026-10-16 01:20 heartbeat：GSM原MOC433/600（.3:217/.7:216），总test1033全部有效无重复；真实协调器/监督器/worker健康，raw2.3/7.2秒前更新，source/data锁及两模型digest/全显存驻留匹配，无STOP/error。actual7150147tokens/10501新调用、pending20384/unknown0，原截止剩1760秒。Pro适配560已清理、待原序MOC，pending/unknown0，原截止剩1778秒；已审计val补测不恢复。证据healthcheck_20261016_011959.json。无阶段变化或可行动故障，不重复launch、不读test gold/评分、不延长原截止。

2026-10-16 01:30 heartbeat：GSM原MOC458/600（.3:230/.7:228），总test1058全有效无重复；实际协调器/监督器/worker、raw5.9/11.9秒前更新、source/data锁及两服务全显存驻留正常，无STOP/error。actual7349963tokens/10819新调用，pending32768/unknown0，原窗口剩1158秒。Pro适配560已清理、待原序MOC，pending/unknown0，原窗口剩1176秒；已完成val补测不恢复。证据healthcheck_20261016_013001.json。旧两批阶段未变，不重复launch、不评分test、不延长原截止。

2026-10-16 01:40 heartbeat：GSM原MOC481/600（.3:241/.7:240），总test1081全有效无重复；真实协调器/监督器/worker健康，raw14.0/0.8秒前更新，source/data锁及两模型digest/全显存驻留正常，无STOP/error。actual7542179tokens/11089新调用、pending26576/unknown0，原窗口剩557秒。Pro适配560已清理、待原序MOC，actual4127233/pending0/unknown0，原窗口剩575秒。已完成val补测不恢复。证据healthcheck_20261016_014002.json。阶段未变，维持原deadline；到期必须停并清理，不能为补齐对照延长或重置。不重复launch、不读test gold/评分。

2026-10-16 01:50到期收口：GSM原MOC504/600（253/251），适配600保留；Pro适配560保留、原MOC尚未启动。原run_start截止已过，worker根因BudgetExceeded: STOP or GPU wall deadline，协调器保护退出；真实PID/自有子进程/端口11448/49/50全部清理。GSM实际7736331tokens/11386calls，Pro4127233/5134；非缓存raw+热身与账本逐项一致，trace reservation全部settled，pending/unknown0。唯一archive261成员本地SHA全核对，SHA655af57509630c51a4e6ee2b09c6f3160919f52260f601d707c922137291fc08，文件在expiry_closure_20261016/；未更改预算/deadline/终态，未读取test gold或评分，不续跑、不延长。窗口到期收口不等于完整对照完成；之后只检查保留状态，不反复通知同一到期等待条件。新两批继续，总监控ACTIVE。

2026-10-16 02:00 heartbeat：到期旧两批与已完成val补测状态保持，GSM原MOC504/600、适配600；Pro适配560、原MOC未启动。真实进程均已退出、自有端口拒绝连接正常，清理完成，pending/unknown0，累计费用与已核验归档一致；source/data锁匹配，无新调用或终态改变。证据healthcheck_20261016_020002.json。不将既有deadline错误当成新故障、不恢复或重复通知、不重跑归档和评分、不延长窗口。

2026-10-16 02:10 heartbeat：原到期两批与已审计val补测保持已清理、不再调用状态；GSM原MOC504/600、适配600，Pro适配560且原MOC未启动，累计actual/调用与01:50归档一致、pending/unknown0，source/data锁匹配，旧端口关闭正常。证据healthcheck_20261016_021001.json。不视同一到期状态为新故障、不恢复/重复通知、不重跑归档或部分对照评分、不延长窗口。

2026-10-16 02:20 heartbeat：旧到期两批与已审计val补测保持清理和费用/终态不变，GSM原MOC504/600、适配600，Pro适配560且原MOC未启动；source/data锁匹配、真实进程均退出、旧端口关闭正常，pending/unknown0。证据healthcheck_20261016_022001.json。不恢复或重复通知同一到期状态、不重跑归档、不评分部分对照、不延长窗口。

2026-10-16 02:30 heartbeat：到期旧两批与已完成val补测保持已清理及原费用/终态不变，pending/unknown0，source/data锁匹配，真实进程退出/旧端口关闭正常。证据healthcheck_20261016_023003.json。没有新授权窗口，不恢复、不重归档或重复通知，不评分部分对照。

2026-10-16 02:40 heartbeat：旧到期两批与已审计val补测仍保持清理、原终态/费用不变，pending/unknown0，source/data锁匹配，真实进程退出与旧端口关闭正常。证据healthcheck_20261016_024000.json。不恢复、不重复到期通知或归档、不评分部分对照、不延长窗口。

2026-10-16 02:50 heartbeat：原到期两批与已审计val补测保持已清理、费用/终态不变、pending/unknown0，source/data锁匹配，真实进程退出与旧端口关闭正常。证据healthcheck_20261016_025001.json。无新窗口授权，不恢复、不重复通知或归档，不评分部分对照、不延长截止。

2026-10-16 03:00 heartbeat：原到期两批与已审计val补测保持已清理，原终态/费用不变、pending/unknown0，source/data锁匹配，进程退出/旧端口关闭正常。证据healthcheck_20261016_030009.json。没有新窗口授权，不恢复、不重复通知或归档，不评分部分对照、不延长截止。

2026-10-16 03:10 heartbeat：原到期两批与已审计val补测保持清理，费用/终态不变，pending/unknown0，source/data锁匹配，进程退出与旧端口关闭正常。证据healthcheck_20261016_031010.json。无新窗口授权，不恢复、不重复通知或归档、不评分部分对照、不延长截止。

2026-10-16 03:20 heartbeat：原到期两批与已审计val补测保持清理，终态与费用不变，pending/unknown0，source/data锁匹配，真实进程退出与原端口关闭正常。证据healthcheck_20261016_032003.json。无新窗口授权，不恢复、不重复归档或通知、不评分部分对照、不延长截止。

2026-10-16 03:30 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用和清理状态不变，pending/unknown0，source/data锁匹配，原自有进程退出及端口关闭正常。证据healthcheck_20261016_033000.json。无新窗口授权，不恢复、不重复通知或归档、不评分部分对照、不延长截止。

2026-10-16 03:40 heartbeat：原到期GSM/Pro与已审计Pro val补测终态/费用不变，pending/unknown0，source/data锁匹配，自有进程/端口保持已清理。证据healthcheck_20261016_034000.json。无新窗口授权，不恢复、不延长截止、不评分部分对照、不重复通知或归档。

2026-10-16 03:50 heartbeat：原到期GSM/Pro与已审计Pro val补测状态/终态/费用均不变，pending/unknown0，source/data锁匹配，自有进程与端口保持已清理。证据healthcheck_20261016_035000.json。无新窗口授权，不恢复或延长、不评分部分对照、不重复通知或归档。

2026-10-16 04:00 heartbeat：原到期GSM/Pro与已审计Pro val补测状态、终态和费用不变，pending/unknown0，source/data锁匹配，自有进程/端口保持清理。证据healthcheck_20261016_040000.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 04:10 heartbeat：原到期GSM/Pro与已审计Pro val补测终态/费用/清理状态不变，pending/unknown0，source/data锁匹配，真实进程退出及原端口关闭正常。证据healthcheck_20261016_041001.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 04:20 heartbeat：原到期GSM/Pro与已审计Pro val补测终态/费用/清理状态不变，pending/unknown0、source/data锁匹配、实际进程与端口关闭正常。证据healthcheck_20261016_042000.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 04:30 heartbeat：原到期GSM/Pro与已审计Pro val补测状态/终态/费用不变，pending/unknown0，source/data锁匹配，自有进程和端口保持清理。证据healthcheck_20261016_043002.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 04:40 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用与清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与旧端口关闭正常。证据healthcheck_20261016_044001.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 04:50 heartbeat：原到期GSM/Pro与已审计Pro val补测终态/费用/清理状态保持不变，pending/unknown0，source/data锁匹配，原进程退出与端口关闭正常。证据healthcheck_20261016_045001.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 05:00 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用与清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_050001.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 05:10 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用和清理状态保持，pending/unknown0，source/data锁匹配，真实进程退出与原端口关闭正常。证据healthcheck_20261016_051002.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 05:20 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用和清理状态不变，pending/unknown0，source/data锁匹配，真实进程退出及原端口关闭正常。证据healthcheck_20261016_052002.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 05:30 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用和清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出及原端口关闭正常。证据healthcheck_20261016_053002.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 05:40 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用和清理状态不变，pending/unknown0，source/data锁匹配，真实进程退出与原端口关闭正常。证据healthcheck_20261016_054002.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 05:50 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_055003.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 06:00 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_060002.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 06:10 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_061004.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 06:20 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_062004.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 06:30 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_063021.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 06:40 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_064002.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 06:50 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_065004.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 07:00 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用与清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_070005.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 07:10 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_071005.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 07:20 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_072005.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 07:30 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_073006.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 07:40 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_074004.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 07:50 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_075004.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 08:00 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_080005.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 08:10 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_081007.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 08:20 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_082007.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 08:30 heartbeat：原到期GSM/Pro与已审计Pro val补测终态、费用及清理状态不变，pending/unknown0，source/data锁匹配，实际进程退出与原端口关闭正常。证据healthcheck_20261016_083008.json。无新窗口授权，不恢复、不延长、不评分部分对照、不重复通知或归档。

2026-10-16 08:45最新状态（覆盖早先运行进度，不改变授权）：MMLU新批1140/1140全有效并完成raw/成本/缓存/本地原解析器评分及资源清理审计；115文件归档SHA98b5cb78eb26766a812963aa5411253f1870e35671eeaf65dfe68b789969f4b4，completion_audit passed、实际7870810token/12846calls、pending/unknown0，永不自动重跑。SVAMP于原deadline08:40:16（1792111216.6382632）停止并08:40:24完成全部自有进程/子进程及11454/11455清理；适配600/600保持，原MOC仅428/600（0.3为215、0.7为213），总1028有效、无重复/已提交invalid，缺172，不评分部分test、不声称完整比较。实际3908073token、7444新调用，其中7443次有返回用量（含3热身）；一个截止时被中断请求无返回，原budget的10192 pending保持不改，独立closure_report以reservation_id核对started7441-settled7440后记录reconciled_unknown，审计pending0、独立unknown保守上界10192，cap合计3918265，不能再与原pending重复相加或冒充actual。126文件完整唯一归档已下载逐文件SHA核验，SHA9395abc1dfbaf27e533def227e6fa0ce621b39640c287027445634c7c3b93e46；本地SVAMP expiry_closure_20261016/closure_report.json和窗口到期收口记录.md，远端新计划expired_archives/moc_v21_svamp_train40_test_20261015/。原GSM/Pro到期未完成与已完成Pro val不变。现在没有活动生成；moc仍ACTIVE只读核查保留证据，无变化保持安静，不重复到期提醒。SVAMP和原GSM/Pro均禁止自动dispatch/恢复/延长；只有新的可信人类明确窗口授权才重新评审续未提交。原budget中该10192预留是已独立核销为unknown的同一项，后续不得当作新故障反复通知或重复收费。未达到所有五批完整实验完成审计且用户未取消，不PAUSED。

2026-10-16 08:50 heartbeat: all five runs unchanged; locked source/data match, no STOP, all recorded owned process cmdlines absent and cleanup complete. MMLU1140 complete; SVAMP adapted600/MOC428 unchanged, actual3908073/7444calls. Original reserved10192 remains the same independently reconciled unknown upper bound (audit pending0, cap3918265), never double counted or treated as a new failure. Original GSM/Pro expired coverage/cost and completed Pro val unchanged. Evidence healthcheck_20261016_085008.json / healthcheck_20261016_085006.json. Read-only only; no dispatch/restart/extension/scoring/automation change. ACTIVE; no repeated notification.

2026-10-16 09:00 heartbeat: all five states, coverage, budgets and owned cleanup unchanged from08:50. Source/data locks match; no STOP, no recorded live owned processes, no owned children/ports. SVAMP original reserved10192 is the same independently reconciled unknown upper bound (audit pending0, actual3908073, cap3918265), not an additional charge or new failure. MMLU and Pro val remain audited complete; expired GSM/Pro/SVAMP remain preserved incomplete. Evidence healthcheck_20261016_090012.json / healthcheck_20261016_090010.json. No dispatch/recovery/extension/scoring/automation mutation; ACTIVE, no repeated notification.

2026-10-16 09:10 heartbeat: five runs unchanged from09:00, including all terminal coverage, actual costs/calls, source/data locks and cleanup. Recorded owned cmdlines absent; no owned children/ports or STOP. MMLU/Pro val remain audited complete; original GSM/Pro and SVAMP remain expired incomplete. SVAMP raw reserved10192 remains the same independently reconciled unknown bound, effective pending0/cap3918265, not a new fault or duplicate charge. Evidence healthcheck_20261016_091008.json / healthcheck_20261016_091006.json. No dispatch/restart/extension/scoring/automation change; ACTIVE, quiet.

2026-10-16 09:20 heartbeat: all five runs unchanged from09:10; terminal coverage, budgets, source/data locks and owned cleanup verified. All recorded owned cmdlines absent, no owned children/ports or STOP. MMLU/Pro val remain audited complete; GSM/Pro/SVAMP remain expired incomplete. SVAMP original reserved10192 is the same independently reconciled unknown upper bound (effective pending0, cap3918265), not an additional charge or new failure. Evidence healthcheck_20261016_092013.json / healthcheck_20261016_092011.json. No dispatch/recovery/extension/scoring/automation mutation; ACTIVE, quiet.

# 无时间窗口的并行续跑

2026-10-16用户明确要求立即开始、不设窗口、尽可能使用GPU跑完。管理入口保留原累计预算、STOP、冻结算法/数据/policy与已提交终态，只移除common及merger各一处实验截止条件；原文件和run_start没有改写。原请求超时、trace静默与隔离代码5秒仍保留。原MMLU1140及Pro val112完整批次不重跑。

启动前363文件全量原始归档逐文件SHA下载核验通过，SHA576c6a6d64c3e27e04bff2b9e6c01ba40afb1ba33c0652a00550dac34c9f6a5b。8项CPU检查（六模块AST只改时间条件、STOP、预算函数不变、终态续跑限制、GPU准入/不重叠/不足卡）本地及远端通过；初始运行管理本地提交c50928f。未推送。

09:37:11同时启动三批。Pro监督器3007459，GPU1主27B/4合并9B，端口11449/11456；GSM监督器3007460，GPU2/5，11448/11450。SVAMP首次监督器3007461在第二次GPU利用率准入检查失败，在启动任何模型服务前退出；自有资源清理。完整119文件归档下载SHA核验通过，SHA00d4dd4b47f272e80779be368fd66d7f2b841cb86f4c49ed5c82663c23334f00。针对同一根因仅一次换卡恢复，09:39:39启动监督器3011499，GPU6/0，11454/11455，旧记录保留，未重复核销费用。

launch_verification.json已核对六张卡不重叠、所有27B/9B的CUDA分配及短生成、digest、size_vram>=size、各批两worker实际cmdline、raw更新时间。三批phase=test/arm=moc，experiment_wall_deadline=null，当前无error，旧run_start及已有终态字节前缀和policy/source/data锁均保持。原MOC新增终态：GSM513/600，Pro16/560，SVAMP432/600，全有效无重复；适配600/560/600保留。

该快照累计预算：GSM actual7816362token/11513calls、pending20384/unknown0；Pro actual4236036/5315、pending20384/unknown0；SVAMP actual3932541/7496、pending20384/unknown10192。SVAMP旧10192已经移入unknown，当前pending是新进行中请求，须同时计cap，不能把旧独立closure初值当作当前费用。预算未重置；所有新增启动热身计入费用。

监控ACTIVE、十分钟及原线程不变，健康状态安静，重要阶段/恢复/完成/失败才通知。只使用resume_no_deadline_20261016管理入口，旧协调器与dispatch保留作历史，不重启。GPU3/7上的资源未触碰；并行三批已使用可准入的三个主卡和三个独立合并卡。
