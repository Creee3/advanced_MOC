Approved and prepared; not launched yet.

2026-10-15 19:14 Asia/Shanghai live check supersedes the preparation entry: training24/24 valid; frozen-transfer val144/240 valid (72 per density), adapted val/test pending. Supervisor2439535, workers2462838/2462839 and service2439674 have matching live cmdlines; raw updated0.3/3.1 seconds ago. Live11448 confirms locked27B digest, size_vram=size. Actual1145401 tokens, pending20384, unknown0;23684 seconds remain in original window. No STOP/error/relaunch. Evidence: ../moc_cross_dataset_plan_20261015/healthcheck_20261015_191433.json.

19:53 heartbeat: frozen-transfer val240/240 complete; adapted val58/240 committed (29 per density), all valid. Runtime workers/live model/source-data locks verified, raw2.9/6.1s old. Actual1644108 tokens, pending20384, unknown0;21354s in original window. No relaunch/tuning. Snapshot ../moc_cross_dataset_plan_20261015/healthcheck_20261015_195324.json.

20:32 heartbeat: both val arms complete240/240 each, all480 valid; automatically entered adapted author-test40/600 (20 per density). Frozen source/data hashes and actual processes/live full-VRAM model verified. Actual2177392tokens,pending20384,unknown0,remaining18986s. No test scoring/tuning/relaunch. Evidence ../moc_cross_dataset_plan_20261015/healthcheck_20261015_203252.json.

22:42 adaptive_complete: test600/600 valid, val480/480 valid, train24/24 valid, own resources cleaned. Actual3760366tokens/5211newcalls, pending0/unknown0. Coordinator phase-transition guard raced supervisor final exit and stopped before originalMOC launch. Root cause/241-file raw archive/7 passing management regressions documented in parent coordinator recovery; dataset source/model/config/data/policy locks unchanged, original deadline/budget retained.22:46:51 patched coordinator resumed baseline-only schedule; adapted terminals preserved, no test scoring.

23:10用户要求下午至今全部实验汇总：已下载GSM完成val的12个文件并核对归档/逐文件SHA、本地gold与模型冻结锁，原作者数字解析器CPU重评分一致。冻结迁移113/120(.3)、112/120(.7)；24题适配116/120、115/120，推理逻辑token−9.55%/−13.73%，McNemar p=.25/.375，未证明显著提升；不是与MOC val对照。24训练题/120标签，训练实际579378token已含原账本。val_review_20261015保存复算脚本/结果/说明。未新增生成、未读test gold或调参；当前原MOC test70/600仍运行，适配600保留。每日汇总在../daily_summary_20261015/。

# 无时间窗口的并行续跑

2026-10-16用户明确要求立即开始、不设窗口、尽可能使用GPU跑完。管理入口保留原累计预算、STOP、冻结算法/数据/policy与已提交终态，只移除common及merger各一处实验截止条件；原文件和run_start没有改写。原请求超时、trace静默与隔离代码5秒仍保留。原MMLU1140及Pro val112完整批次不重跑。

启动前363文件全量原始归档逐文件SHA下载核验通过，SHA576c6a6d64c3e27e04bff2b9e6c01ba40afb1ba33c0652a00550dac34c9f6a5b。8项CPU检查（六模块AST只改时间条件、STOP、预算函数不变、终态续跑限制、GPU准入/不重叠/不足卡）本地及远端通过；初始运行管理本地提交c50928f。未推送。

09:37:11同时启动三批。Pro监督器3007459，GPU1主27B/4合并9B，端口11449/11456；GSM监督器3007460，GPU2/5，11448/11450。SVAMP首次监督器3007461在第二次GPU利用率准入检查失败，在启动任何模型服务前退出；自有资源清理。完整119文件归档下载SHA核验通过，SHA00d4dd4b47f272e80779be368fd66d7f2b841cb86f4c49ed5c82663c23334f00。针对同一根因仅一次换卡恢复，09:39:39启动监督器3011499，GPU6/0，11454/11455，旧记录保留，未重复核销费用。

launch_verification.json已核对六张卡不重叠、所有27B/9B的CUDA分配及短生成、digest、size_vram>=size、各批两worker实际cmdline、raw更新时间。三批phase=test/arm=moc，experiment_wall_deadline=null，当前无error，旧run_start及已有终态字节前缀和policy/source/data锁均保持。原MOC新增终态：GSM513/600，Pro16/560，SVAMP432/600，全有效无重复；适配600/560/600保留。

该快照累计预算：GSM actual7816362token/11513calls、pending20384/unknown0；Pro actual4236036/5315、pending20384/unknown0；SVAMP actual3932541/7496、pending20384/unknown10192。SVAMP旧10192已经移入unknown，当前pending是新进行中请求，须同时计cap，不能把旧独立closure初值当作当前费用。预算未重置；所有新增启动热身计入费用。

监控ACTIVE、十分钟及原线程不变，健康状态安静，重要阶段/恢复/完成/失败才通知。只使用resume_no_deadline_20261016管理入口，旧协调器与dispatch保留作历史，不重启。GPU3/7上的资源未触碰；并行三批已使用可准入的三个主卡和三个独立合并卡。
