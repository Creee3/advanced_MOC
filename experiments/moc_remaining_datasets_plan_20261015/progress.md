# 2026-10-15 用户批准40 train + author test

已核查现有两批：GSM原MOC205/600，适配600终态保留；Pro适配560已清理等待原序MOC。旧两批健康、锁匹配，无恢复或改截止。

新两批尚未启动模型调用；当前无符合主GPU门槛卡。读取旧协调器config.json、runtime/train_policy.py及advanced_MOC/tests均不存在，已确认配置在各实验目录、训练逻辑在runner.py，不依赖这些误路径。

按最新指令替代先前114/48训练和val草案；不新增验证或迁移比较组。

CPU preflight: MMLU 6 Graph conditions plus4 math namespace/4 isolation/timeout passed. SVAMP failed before model calls: author registry has no svamp prompt key; author run_experiment explicitly uses prompt_domain=gsm8k. Original no-call bundle archived preflight_attempt1_LOCAL_ONLY; repair only native prompt-domain mapping, no algorithm/author source changes. Test gold stays local.

Both native preflights passed (6 Graph conditions each;4 namespace/4 denial/timeout). Original regression first attempt had13 missing fixture-path errors;8 admission tests passed. Corrected only copied test AUTHOR/ASSETS paths and supplied unchanged ladies fixture;8 admission and53 original CPU regressions now passed. One local ladies fixture lookup mistakenly included advanced_MOC prefix, corrected to workspace root. All production source/data locks verified locally and44 uploaded files per batch verified on cloud. No test gold uploaded.

2026-10-16 00:05 dispatch首个准入tick，两次10秒检查仍无符合main卡门槛资源，waiting_for_gpu；两批run_start不存在、模型调用与费用均0。00:06只读核查两批冻结源码/数据锁全匹配，test0；未启动不应将空进程列表all([])当成真实存活，已修正新只读helper摘要。moc十分钟监控工具已成功更新ACTIVE，保留旧三批并新增两批等卡/首次启动/分阶段监控。dispatch是有界tick，不是常驻进程，退出或actual_cmdline=null正常。一次本地plan读取默认GBK报错，改显式UTF-8后继续，未影响云端锁。

2026-10-16 用户指出定时指令乱码；确认上次Python stdout=gbk，以ensure_ascii=False输出再解码损坏旧prompt与name，原prompt有2006个U+FFFD。已保留原TOML字节备份（LOCAL_ONLY），从可信计划重建monitor_prompt_utf8.txt并用JSON ASCII转义读取后经automation_update修复。UTF-8回读除工具裁剪的一个末尾换行外逐字符一致、replacement0，name继续、原10分钟/ACTIVE/原线程均通过；五批约束/旧截止保留，未改实验进程或锁。初次严格校验因工具自动去掉末尾换行而失败，已明确该唯一差异。验证在automation_encoding_repair_20261016/verification.json。

2026-10-16 00:20 heartbeat：新两批inspect均未首次启动，source/data锁全匹配、无STOP，训练/test提交0、调用/actual/pending/unknown均0。准入tick连续两次10秒检查仍无main free>=21000MiB且util<=15的卡，dispatch=waiting_for_gpu；空闲条件未变，保持安静、不启动窗口、不重复prepare或重启。证据healthcheck_20261016_001955.json与dispatch_20261016_002016.json。定时中文本轮正常。

2026-10-16 00:30首次启动：准入两次10秒检查GPU4 UUID89523765...空闲24081MiB/util0，通过后dispatch于00:30:14启动MMLU adaptive监督器2677784；run_start00:30:15，绝对截止08:30:15（deadline_unix1792110615.056002），不重置延长。00:30:54核验两个worker、监督器与11453的27B真实cmdline存活，CUDA分配/短生成/正确digest/size_vram=size通过，training raw2.0/3.3秒前更新，actual1857tokens/8新调用、pending20384/unknown0。40题开始标签生成，尚0题提交，不伪称训练完成。source/data锁匹配，无STOP/error。SVAMP仍未启动且费用0，等下一次空闲卡准入；当前总共原两卡+新一卡=3张。完整证据healthcheck_20261016_003054.json，run_start/lease/launcher本地已取回。无恢复或旧批变更。

2026-10-16 00:40首次启动：GPU7 UUIDe230f5a4...连续两次10秒检查util0/free24076MiB，通过后SVAMP adaptive监督器2685031于00:40:16启动；run_start00:40:16.638274，固定截止08:40:16.638263（deadline1792111216.6382632），不延长。00:41:02核验监督器/worker2685697、2685698和服务2685169的真实cmdline活着，11454的27B CUDA分配/短生成/正确digest/size_vram=size通过，raw1.1/0.2秒前更新，actual3267tokens/11新调用、pending20384/unknown0，无STOP/error。尚0/40题提交。MMLU训练4/40有效，actual164212/229新调用、pending20384/unknown0，raw6.6/3.3秒；两新批锁均匹配。原cohort两卡+新两卡=总四卡，严守上限。run_start/lease/launcher已下载；现场healthcheck_20261016_004102.json与dispatch_20261016_004016.json。未修改旧批锁、算法/参数/数据或原截止。

2026-10-16 00:50 heartbeat：MMLU训练10/40、SVAMP训练8/40，均有效；两个监督器/各worker/11453和11454的27B真实cmdline活着，raw1.8/0.5秒与1.2/0.1秒前更新，source/data锁及digest/全显存驻留正常，无STOP/error。actual308940/150723tokens、新调用407/242、pending20384/10192、unknown均0；原窗口剩27618/28219秒。两批仍training，无test结果或val，保持新两卡/总四卡上限。证据healthcheck_20261016_004957.json。两批均活跃，无准入或阶段切换需要，不执行dispatch、不重复launch、不评分或调参。按约定无阶段变化保持安静。

2026-10-16 01:00 heartbeat：MMLU训练14/40、SVAMP16/40，已提交均有效；监督器/worker/两主服务真实cmdline存活，raw2.6/0.7秒与5.2/1.7秒前更新，source/data锁及模型digest/全显存驻留正常，无STOP/error。actual465548/296437tokens、新调用595/459、各pending20384、unknown均0，原窗口剩27020/27621秒。两批仍training，未进入fit/test，不启动新服务或dispatch，维持新两卡和总四卡上限。证据healthcheck_20261016_005955.json。无阶段变化或可行动问题，按约定不额外通知。

2026-10-16 01:10 heartbeat：MMLU训练19/40、SVAMP24/40，提交均有效；两监督器/worker/主服务真实cmdline正常，raw0.3/1.7秒及0.2/2.4秒前更新，source/data锁与digest/全显存驻留匹配，无STOP/error。actual621692/458166tokens，新调用784/700，各pending20384、unknown均0；原窗口剩26420/27021秒。仍training，未fit/test，不执行dispatch或启动新服务，维持新两卡和总四卡上限。证据healthcheck_20261016_010955.json。阶段未变、无额外通知、未重启或据结果调参。

2026-10-16 01:20 heartbeat：MMLU训练24/40、SVAMP34/40，提交均有效；两监督器/worker/主服务真实cmdline正常，raw1.3/6.8秒及1.2/3.2秒前更新，source/data锁和模型digest/全显存驻留匹配，无STOP/error。actual778035/622599tokens，新调用1002/962，各pending20384、unknown均0，原窗口剩25820/26421秒。两批仍training，未fit/test，不执行dispatch或另启服务，保持新两卡/总四卡。证据healthcheck_20261016_011955.json。阶段未变，按约定安静、不重启、不调参或评分test。

2026-10-16 01:30阶段变化：SVAMP40/40训练有效，无excluded；80个receiver episode、200条label，已fit并冻结policy SHA728a99c19645a2ff230132ddd220ec243a7ea644bc11776e261ae193e4d4633e。远端模型及训练提交SHA核对、no_val_or_test_fitting/frozen_before_test通过，policy/report/model已下载。已进入adapted作者test36/600（两密度各18），全有效无重复，raw4.6/2.0秒前更新；actual765575tokens/1195新调用、pending20384/unknown0，原窗口剩25820秒。MMLU仍training32/40有效，actual950530/1274新调用、pending20384/unknown0，raw1.9/3.0秒，窗口剩25218秒。两批实际进程/主服务、source/data锁及全显存驻留正常，无STOP/error；不执行dispatch，维持新两卡/总四卡，不评分部分test或调参。证据healthcheck_20261016_012957.json。

2026-10-16 01:40 heartbeat：MMLU训练38/40有效；SVAMP适配test148/600（两密度各74）全有效无重复，训练40/40及冻结模型保留。两批监督器/worker/主服务真实cmdline健康，活动raw4.3/1.8秒及1.0/3.6秒前更新，source/data锁及digest/全显存驻留匹配，无STOP/error。actual1126558/894719tokens，新调用1521/1478，各pending20384、unknown均0；原窗口剩24617/25218秒。无阶段变化、无dispatch/新服务/恢复，维持新两卡和总四卡上限。证据healthcheck_20261016_013958.json。保持安静、不评分部分test、不调参。

2026-10-16 01:50阶段变化：MMLU40/40训练有效，无excluded；80个receiver episode、214条label，已fit并冻结policy SHA71dcd90b4f216475cebd5312e844ff1d258ddb2befac48787ab0b793bb92ef7f。远端模型/训练提交SHA及no_val_or_test_fitting核对通过，policy/report/model已取回本地。适配作者test24/570（两个密度各12）全有效无重复，raw1.2/4.8秒前更新；actual1262838tokens/1723新调用、pending20384/unknown0，原窗口剩24008秒。SVAMP适配test248/600（各124）全有效，actual1016123tokens/1731新调用、pending20384/unknown0，raw1.8/0.4秒，窗口剩24609秒。两新批真实进程/服务、source/data锁与全显存驻留正常，无STOP/error，继续原两张GPU，无dispatch、不评分部分test或调参。证据healthcheck_20261016_015008.json。旧GSM/Pro到期停止不影响两新批窗口。

2026-10-16 02:00 heartbeat：MMLU适配test63/570（32/31）、SVAMP360/600（各180），全部有效无重复，训练各40和冻结policy保持。真实监督器/worker/主服务正常，raw2.8/4.8秒与1.4/3.8秒前更新，source/data锁及digest/全显存驻留匹配，无STOP/error。actual1403599/1146293tokens，新调用1969/2004，各pending20384、unknown均0；原窗口剩23417/24018秒。两批仍适配test，不执行dispatch，不启额外服务、不评分部分test、不调参。原旧两卡清理后当前本聊天GPU仅新两张。证据healthcheck_20261016_015959.json。无阶段变化，按约定保持安静。

2026-10-16 02:10 heartbeat：MMLU适配test87/570（44/43）、SVAMP464/600（各232）全有效无重复。两监督器/worker/主服务真实cmdline健康，raw5.6/2.3秒与1.7/3.5秒前更新，source/data锁和digest/全显存驻留匹配，无STOP/error。actual1523100/1272799tokens、新调用2128/2268、各pending20384、unknown均0，原窗口剩22818/23419秒。训练40/40与冻结policy保持，仍适配test；无dispatch/新增服务/评分/调参，GPU维持新两张。证据healthcheck_20261016_020957.json。无阶段变化，按约定不额外通知。

2026-10-16 02:20 heartbeat：MMLU适配test118/570（各59）、SVAMP580/600（各290）均有效无重复；两批仍test/adapted，未到阶段切换。真实监督器/worker/主服务健康，raw2.8/3.3秒和1.4/4.3秒前更新，source/data锁和digest/全显存驻留匹配，无STOP/error。actual1655553/1407770tokens、新调用2329/2554，各pending20384、unknown均0；原窗口剩22218/22818秒。保持两张新GPU，不dispatch/重复launch，不评分部分test或调参。证据healthcheck_20261016_021959.json。无阶段变化按约定保持安静。

2026-10-16 02:30阶段变化：SVAMP适配test600/600（各密度300）全有效，精确question_id/arm/density覆盖核对300题、重复/invalid均0；40训练和policy SHA保持。已adaptive_complete，自有监督器/worker/子进程与11454/11455均清理，actual1431706tokens/2602新调用，pending/unknown0，非缓存raw+热身与账本逐项一致。独立核验回执adaptive_stage_receipt.json，无test gold读取/评分。原MOC test尚未开始，按原计划等MMLU适配完成清理后串行两批原MOC，不能把本阶段完成称为完整对照完成。MMLU适配test160/570（81/79）全有效，真实进程及11453模型健康，raw3.5/1.9秒、锁/全显存驻留正常，actual1810153/2622新调用、pending20384/unknown0，窗口剩21615秒；SVAMP窗口剩22216秒，不延长。现场healthcheck_20261016_023001.json。未执行dispatch，当前仅MMLU一张GPU活动；不因新空闲卡改变授权调度或增加服务。

2026-10-16 02:40 heartbeat：MMLU适配test196/570（各98）全有效无重复，真实监督器/worker/11453主服务健康，raw5.5/4.4秒前更新，source/data锁及digest/全显存驻留匹配，无STOP/error；actual1952435tokens/2850新调用、pending20384/unknown0，原窗口剩21018秒。SVAMP仍adaptive_complete600/600且清理，actual1431706/2602新调用、pending/unknown0，窗口剩21619秒，按既定顺序等MMLU适配完成后串行原MOC。未满足双批适配完成条件，不执行dispatch或加服务，当前仅一张主GPU活动。证据healthcheck_20261016_023958.json。无阶段变化或可行动问题，保持安静、不评分test或调参。

2026-10-16 02:50 heartbeat：MMLU适配test236/570（各118）全有效无重复，监督器/worker/11453主服务真实cmdline健康，raw2.0/3.1秒前更新，source/data锁与digest/全显存驻留匹配，无STOP/error；actual2109792tokens/3113新调用、pending20384/unknown0，窗口剩20418秒。SVAMP仍adaptive_complete600/600且清理，费用1431706/2602calls不变，pending/unknown0，窗口剩21019秒；等两批适配均清理后按原计划串行原MOC。当前一张主GPU活动，未执行dispatch或新增服务，不评分部分test、不调参。证据healthcheck_20261016_024958.json。无阶段变化，按约定保持安静。

2026-10-16 03:00 heartbeat：MMLU适配test273/570全有效无重复，真实监督器/worker/11453主服务健康，raw2.9/2.0秒前更新，source/data锁与digest/全显存驻留匹配，无STOP/error。actual2259186tokens/3362新调用、pending20384/unknown0，原窗口剩19810秒。SVAMP仍adaptive_complete600/600且清理，费用1431706/2602calls不变、pending/unknown0，窗口剩20411秒，等待两批适配都完成清理后原顺序原MOC。未执行dispatch或额外GPU服务，当前一张主卡；不评分部分test、不调参。证据healthcheck_20261016_030006.json。阶段未变，按约定保持安静。

2026-10-16 03:10 heartbeat：MMLU适配test inspect304/570、覆盖读取305/570（153/152，为并发提交时序），终态全有效无重复；真实监督器/worker/11453主服务健康，raw0.1/1.0秒前更新，source/data锁及digest/全显存驻留匹配，无STOP/error。actual2407759tokens/3574新调用、pending10192/unknown0，原窗口剩19208秒。SVAMP仍adaptive_complete600/600、清理与费用1431706/2602calls不变、pending/unknown0，窗口剩19809秒。未满足双批适配清理条件，不执行dispatch或加服务；一张GPU活动，未评分test/调参。证据healthcheck_20261016_031008.json。阶段未变，保持安静。

2026-10-16 03:20 heartbeat：MMLU适配test347/570（174/173）全有效无重复，真实监督器/worker/11453主服务健康，raw2.5/0.8秒前更新，source/data锁及digest/全显存驻留匹配，无STOP/error。actual2579211tokens/3860新调用、pending20384/unknown0，原窗口剩18615秒。SVAMP仍adaptive_complete600/600且清理，费用1431706tokens/2602新调用不变、pending/unknown0，窗口剩19216秒。未满足两批适配完成清理条件，不执行dispatch或加服务，当前仅MMLU一张主GPU；不读取test gold、不评分部分test、不调参、不延长截止。证据healthcheck_20261016_032001.json。阶段未变，按约定保持安静。

2026-10-16 03:30 heartbeat：MMLU适配test384/570（各192）全有效无重复，40训练及冻结锁保持，真实监督器/worker/11453服务健康，raw1.8/4.2秒前更新，source/data锁与模型digest/全显存驻留匹配，无STOP/error。actual2726276tokens/4107新调用、pending20384/unknown0，原窗口剩18019秒。SVAMP仍adaptive_complete600/600且清理，actual1431706tokens/2602新调用、pending/unknown0不变，窗口剩18619秒。尚不满足双批适配完成清理条件，不执行dispatch或新增服务；当前仅MMLU一张主GPU活动。证据healthcheck_20261016_032957.json。阶段无变化，保持安静，不评分部分test、不调参、不延长截止。

2026-10-16 03:40 heartbeat：MMLU适配test434/570（218/216）全有效无重复，40训练/冻结锁保持，监督器/worker/11453服务真实cmdline健康，raw0.9/2.5秒前更新，source/data锁与digest/全显存驻留匹配，无STOP/error。actual2894108tokens/4443新调用、pending20384/unknown0，原窗口剩17418秒。SVAMP仍adaptive_complete600/600且清理，费用1431706tokens/2602新调用、pending/unknown0不变，原窗口剩18019秒；等待两批适配均完成清理后串行原MOC。证据healthcheck_20261016_033958.json。当前仅一张主GPU，不执行dispatch或新增服务，不评分部分test、不调参、不延长截止；阶段未变，保持安静。

2026-10-16 03:50 heartbeat：MMLU适配test474/570（各237）全有效无重复，40训练与冻结锁保持，监督器/worker/11453主服务真实进程健康，raw2.0/1.3秒前更新，source/data锁与digest/全显存驻留匹配，无STOP/error。actual3045509tokens/4708新调用、pending20384/unknown0，原窗口剩16818秒。SVAMP仍adaptive_complete600/600且清理，actual1431706tokens/2602新调用、pending/unknown0不变，原窗口剩17419秒。未满足双批适配完成清理条件，未执行dispatch或加服务，当前仅一张主GPU；不评分部分test、不调参、不延长截止。证据healthcheck_20261016_034958.json。阶段未变，保持安静。

2026-10-16 04:00 heartbeat：MMLU适配test509/570（255/254）全有效无重复，训练40/40及冻结锁保持，监督器/worker/11453主服务真实cmdline健康，raw3.7/4.4秒前更新，source/data锁与digest/全显存驻留匹配，无STOP/error。actual3200965tokens/4927新调用、pending20384/unknown0，原窗口剩16218秒。SVAMP仍adaptive_complete600/600且清理，actual1431706tokens/2602新调用、pending/unknown0不变，原窗口剩16819秒。未满足双批适配完成清理条件，不执行dispatch或加服务，当前仅一张主GPU。证据healthcheck_20261016_035958.json。阶段未变，保持安静，不评分部分test、不调参、不延长截止。

2026-10-16 04:10 heartbeat：MMLU适配test550/570（各275）全有效无重复，训练40/40及冻结锁保持，真实监督器/worker/11453主服务健康，raw2.2/0.4秒前更新，source/data锁与digest/全显存驻留匹配，无STOP/error。actual3359269tokens/5206新调用、pending20384/unknown0，原窗口剩15618秒。SVAMP仍adaptive_complete600/600且清理，actual1431706tokens/2602新调用、pending/unknown0不变，原窗口剩16218秒。尚不满足双批适配完成清理条件，不执行dispatch或加服务，当前仅一张主GPU活动。证据healthcheck_20261016_040959.json。阶段未变，保持安静，不评分部分test、不调参、不延长截止。

2026-10-16 04:20阶段变化：MMLU适配test570/570（每密度285）全有效，精确285题×两密度覆盖、重复/invalid0；训练40与policy SHA71dcd90b4f216475cebd5312e844ff1d258ddb2befac48787ab0b793bb92ef7f保持。适配清理已核实，actual3426041tokens/5339新调用、pending/unknown0，非缓存raw加首个热身逐项对齐，adaptive_stage_receipt.json已落本地，无gold读取/评分。两批适配均完成清理后，dispatch单次有界准入于04:20:30启动MMLU baseline监督器2818908；随后真实两worker/两服务cmdline通过，GPU0/1 UUID分别0a075077与d7d04192、端口11453/11455，27B/9B均真实CUDA分配+短生成+正确digest+全显存驻留，raw持续更新（04:23:26为6.0/2.5秒），phase=test/arm=moc，status.cleanup_complete=false（旧cleanup.json属于适配阶段）。actual3433289tokens/5354新调用、pending20384/unknown0，source/data锁匹配，无STOP/error。原MMLU截止08:30:15不变、剩14811秒。SVAMP适配600/600保持清理、actual1431706/2602新调用、pending/unknown0，等MMLU原MOC清理后串行准入；原截止08:40:16不变。当前最多两张GPU，未新增方法/题目/val或改锁。证据healthcheck_20261016_041958.json、healthcheck_20261016_042301.json、healthcheck_20261016_042326.json与baseline_transition_20261016.json。
