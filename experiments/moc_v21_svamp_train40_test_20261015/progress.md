Prepared; no model calls or GPU launch. Await CPU preflight and idle resources.

2026-10-16 00:07: uploaded44 files verified, native CPU preflight passed; admission8 and method regressions53 passed. Core algorithm byte-identical to existing frozen cohort. Ready, waiting idle GPU; no supervisor/run_start/model calls yet. Ten-minute monitor admits first launch only after stable resource checks; never blindly retries failed jobs.

2026-10-16 00:40首次启动：GPU7 UUIDe230f5a4...连续两次10秒检查util0/free24076MiB，通过后SVAMP adaptive监督器2685031于00:40:16启动；run_start00:40:16.638274，固定截止08:40:16.638263（deadline1792111216.6382632），不延长。00:41:02核验监督器/worker2685697、2685698和服务2685169的真实cmdline活着，11454的27B CUDA分配/短生成/正确digest/size_vram=size通过，raw1.1/0.2秒前更新，actual3267tokens/11新调用、pending20384/unknown0，无STOP/error。尚0/40题提交。MMLU训练4/40有效，actual164212/229新调用、pending20384/unknown0，raw6.6/3.3秒；两新批锁均匹配。原cohort两卡+新两卡=总四卡，严守上限。run_start/lease/launcher已下载；现场healthcheck_20261016_004102.json与dispatch_20261016_004016.json。未修改旧批锁、算法/参数/数据或原截止。

2026-10-16 01:30阶段变化：SVAMP40/40训练有效，无excluded；80个receiver episode、200条label，已fit并冻结policy SHA728a99c19645a2ff230132ddd220ec243a7ea644bc11776e261ae193e4d4633e。远端模型及训练提交SHA核对、no_val_or_test_fitting/frozen_before_test通过，policy/report/model已下载。已进入adapted作者test36/600（两密度各18），全有效无重复，raw4.6/2.0秒前更新；actual765575tokens/1195新调用、pending20384/unknown0，原窗口剩25820秒。MMLU仍training32/40有效，actual950530/1274新调用、pending20384/unknown0，raw1.9/3.0秒，窗口剩25218秒。两批实际进程/主服务、source/data锁及全显存驻留正常，无STOP/error；不执行dispatch，维持新两卡/总四卡，不评分部分test或调参。证据healthcheck_20261016_012957.json。

2026-10-16 02:30阶段变化：SVAMP适配test600/600（各密度300）全有效，精确question_id/arm/density覆盖核对300题、重复/invalid均0；40训练和policy SHA保持。已adaptive_complete，自有监督器/worker/子进程与11454/11455均清理，actual1431706tokens/2602新调用，pending/unknown0，非缓存raw+热身与账本逐项一致。独立核验回执adaptive_stage_receipt.json，无test gold读取/评分。原MOC test尚未开始，按原计划等MMLU适配完成清理后串行两批原MOC，不能把本阶段完成称为完整对照完成。MMLU适配test160/570（81/79）全有效，真实进程及11453模型健康，raw3.5/1.9秒、锁/全显存驻留正常，actual1810153/2622新调用、pending20384/unknown0，窗口剩21615秒；SVAMP窗口剩22216秒，不延长。现场healthcheck_20261016_023001.json。未执行dispatch，当前仅MMLU一张GPU活动；不因新空闲卡改变授权调度或增加服务。

2026-10-16 08:45最新状态（覆盖早先运行进度，不改变授权）：MMLU新批1140/1140全有效并完成raw/成本/缓存/本地原解析器评分及资源清理审计；115文件归档SHA98b5cb78eb26766a812963aa5411253f1870e35671eeaf65dfe68b789969f4b4，completion_audit passed、实际7870810token/12846calls、pending/unknown0，永不自动重跑。SVAMP于原deadline08:40:16（1792111216.6382632）停止并08:40:24完成全部自有进程/子进程及11454/11455清理；适配600/600保持，原MOC仅428/600（0.3为215、0.7为213），总1028有效、无重复/已提交invalid，缺172，不评分部分test、不声称完整比较。实际3908073token、7444新调用，其中7443次有返回用量（含3热身）；一个截止时被中断请求无返回，原budget的10192 pending保持不改，独立closure_report以reservation_id核对started7441-settled7440后记录reconciled_unknown，审计pending0、独立unknown保守上界10192，cap合计3918265，不能再与原pending重复相加或冒充actual。126文件完整唯一归档已下载逐文件SHA核验，SHA9395abc1dfbaf27e533def227e6fa0ce621b39640c287027445634c7c3b93e46；本地SVAMP expiry_closure_20261016/closure_report.json和窗口到期收口记录.md，远端新计划expired_archives/moc_v21_svamp_train40_test_20261015/。原GSM/Pro到期未完成与已完成Pro val不变。现在没有活动生成；moc仍ACTIVE只读核查保留证据，无变化保持安静，不重复到期提醒。SVAMP和原GSM/Pro均禁止自动dispatch/恢复/延长；只有新的可信人类明确窗口授权才重新评审续未提交。原budget中该10192预留是已独立核销为unknown的同一项，后续不得当作新故障反复通知或重复收费。未达到所有五批完整实验完成审计且用户未取消，不PAUSED。

# 无时间窗口的并行续跑

2026-10-16用户明确要求立即开始、不设窗口、尽可能使用GPU跑完。管理入口保留原累计预算、STOP、冻结算法/数据/policy与已提交终态，只移除common及merger各一处实验截止条件；原文件和run_start没有改写。原请求超时、trace静默与隔离代码5秒仍保留。原MMLU1140及Pro val112完整批次不重跑。

启动前363文件全量原始归档逐文件SHA下载核验通过，SHA576c6a6d64c3e27e04bff2b9e6c01ba40afb1ba33c0652a00550dac34c9f6a5b。8项CPU检查（六模块AST只改时间条件、STOP、预算函数不变、终态续跑限制、GPU准入/不重叠/不足卡）本地及远端通过；初始运行管理本地提交c50928f。未推送。

09:37:11同时启动三批。Pro监督器3007459，GPU1主27B/4合并9B，端口11449/11456；GSM监督器3007460，GPU2/5，11448/11450。SVAMP首次监督器3007461在第二次GPU利用率准入检查失败，在启动任何模型服务前退出；自有资源清理。完整119文件归档下载SHA核验通过，SHA00d4dd4b47f272e80779be368fd66d7f2b841cb86f4c49ed5c82663c23334f00。针对同一根因仅一次换卡恢复，09:39:39启动监督器3011499，GPU6/0，11454/11455，旧记录保留，未重复核销费用。

launch_verification.json已核对六张卡不重叠、所有27B/9B的CUDA分配及短生成、digest、size_vram>=size、各批两worker实际cmdline、raw更新时间。三批phase=test/arm=moc，experiment_wall_deadline=null，当前无error，旧run_start及已有终态字节前缀和policy/source/data锁均保持。原MOC新增终态：GSM513/600，Pro16/560，SVAMP432/600，全有效无重复；适配600/560/600保留。

该快照累计预算：GSM actual7816362token/11513calls、pending20384/unknown0；Pro actual4236036/5315、pending20384/unknown0；SVAMP actual3932541/7496、pending20384/unknown10192。SVAMP旧10192已经移入unknown，当前pending是新进行中请求，须同时计cap，不能把旧独立closure初值当作当前费用。预算未重置；所有新增启动热身计入费用。

监控ACTIVE、十分钟及原线程不变，健康状态安静，重要阶段/恢复/完成/失败才通知。只使用resume_no_deadline_20261016管理入口，旧协调器与dispatch保留作历史，不重启。GPU3/7上的资源未触碰；并行三批已使用可准入的三个主卡和三个独立合并卡。
