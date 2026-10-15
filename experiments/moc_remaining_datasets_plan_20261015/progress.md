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
