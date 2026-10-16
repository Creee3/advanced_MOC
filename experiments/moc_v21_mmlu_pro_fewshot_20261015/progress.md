Approved and prepared; not launched yet.

2026-10-15 19:14 Asia/Shanghai live check supersedes the preparation entry: training14/14 valid; frozen-transfer val112/112 and adapted val112/112 valid. Adapted author-test started:46/560 valid (density0.3:24;0.7:22). Supervisor2440203, workers2479040/2479042 and service2440361 have matching live cmdlines; active raw updated16.5/6.0 seconds ago. Live11449 confirms locked27B digest, size_vram=size. Actual1397274 tokens, pending20384, unknown0;23703 seconds remain in original window. No STOP/error/relaunch. No test scoring or tuning. Evidence: ../moc_cross_dataset_plan_20261015/healthcheck_20261015_191433.json.

2026-10-15 user requested completed MMLU-Pro val results: downloaded completed val results/traces and frozen model/locks/training report only; archive and12 file hashes verified. Original author parser rechecked all224 rows; local val gold SHA and policy/model freeze verified. Frozen-transfer32/56 at each density; adapted30/56 (.3),27/56 (.7). Logical inference modeltokens−1.44%/+0.012%. Transfer abstains all336 nonroot receivers per density due unsupported actions; adapted uses333/319 evidence sets but no observed accuracy gain. Paired95% intervals cross0; no equivalence or confirmed degradation claim. Full analysis in val_review_20261015/val结果说明.md and val_analysis.json. No test targets accessed, no policy changes, no experiment relaunch. Local preliminary trace inspection hit a Windows GBK decode error; resolved with explicit UTF-8 before scoring.

22:22 heartbeat: adaptive_complete, author adapted-test560/560 all valid (280 per density), val224/224/train14/14 valid. Own workers/services/children stopped and11449 closed; cleanup_complete true independently supported by no live registered cmdlines/connection refusal. Actual4127233tokens/5134newcalls, pending0/unknown0;12401seconds remain in original deadline. Not the full experiment finish: original-MOC author-test still awaits coordinator after GSM adaptive completion. Do not relaunch this completed adaptive phase or score partial test comparison. Snapshot ../moc_cross_dataset_plan_20261015/healthcheck_20261015_222256.json.

# 无时间窗口的并行续跑

2026-10-16用户明确要求立即开始、不设窗口、尽可能使用GPU跑完。管理入口保留原累计预算、STOP、冻结算法/数据/policy与已提交终态，只移除common及merger各一处实验截止条件；原文件和run_start没有改写。原请求超时、trace静默与隔离代码5秒仍保留。原MMLU1140及Pro val112完整批次不重跑。

启动前363文件全量原始归档逐文件SHA下载核验通过，SHA576c6a6d64c3e27e04bff2b9e6c01ba40afb1ba33c0652a00550dac34c9f6a5b。8项CPU检查（六模块AST只改时间条件、STOP、预算函数不变、终态续跑限制、GPU准入/不重叠/不足卡）本地及远端通过；初始运行管理本地提交c50928f。未推送。

09:37:11同时启动三批。Pro监督器3007459，GPU1主27B/4合并9B，端口11449/11456；GSM监督器3007460，GPU2/5，11448/11450。SVAMP首次监督器3007461在第二次GPU利用率准入检查失败，在启动任何模型服务前退出；自有资源清理。完整119文件归档下载SHA核验通过，SHA00d4dd4b47f272e80779be368fd66d7f2b841cb86f4c49ed5c82663c23334f00。针对同一根因仅一次换卡恢复，09:39:39启动监督器3011499，GPU6/0，11454/11455，旧记录保留，未重复核销费用。

launch_verification.json已核对六张卡不重叠、所有27B/9B的CUDA分配及短生成、digest、size_vram>=size、各批两worker实际cmdline、raw更新时间。三批phase=test/arm=moc，experiment_wall_deadline=null，当前无error，旧run_start及已有终态字节前缀和policy/source/data锁均保持。原MOC新增终态：GSM513/600，Pro16/560，SVAMP432/600，全有效无重复；适配600/560/600保留。

该快照累计预算：GSM actual7816362token/11513calls、pending20384/unknown0；Pro actual4236036/5315、pending20384/unknown0；SVAMP actual3932541/7496、pending20384/unknown10192。SVAMP旧10192已经移入unknown，当前pending是新进行中请求，须同时计cap，不能把旧独立closure初值当作当前费用。预算未重置；所有新增启动热身计入费用。

监控ACTIVE、十分钟及原线程不变，健康状态安静，重要阶段/恢复/完成/失败才通知。只使用resume_no_deadline_20261016管理入口，旧协调器与dispatch保留作历史，不重启。GPU3/7上的资源未触碰；并行三批已使用可准入的三个主卡和三个独立合并卡。
