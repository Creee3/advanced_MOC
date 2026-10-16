# 跨数据集实验：GSM8K 与 MMLU-Pro

## 已批准范围

用户于 2026-10-15 明确批准按以下配置执行：

- GSM8K：24 train / 120 val / 作者 test 300；
- MMLU-Pro：14 train / 56 val / 作者 test 280；
- 两个数据集最多占用两张 GPU，先各用一张 27B 主卡并行完成训练、拟合、val 与适配方法 test；两批适配阶段结束并释放主卡后，再串行运行各自原 MOC baseline（27B 主模型 + 9B 合并器）。
- 每 10 分钟检查一次，只管理这两批；旧 AQuA、HumanEval、MBPP 和已完成 v1/v2 任务不恢复。

### 2026-10-15 追加授权：原MOC val补测

用户要求补测MMLU-Pro同一val56的原MOC，并在澄清资源上限后明确回复“允许新增两张，立即独立补测”。新增目录moc_mmlu_pro_val_baseline_20261015，只跑原MOC两个密度共112条件；临时新增GPU0/4，重叠时本聊天最多四张，原两批自身仍最多两张。此例外不授权其他数据或旧任务。新增批独立7200秒/6000新调用/4000000模型token上限，旧两批预算/截止不重置。原两组val不复跑、不调参；完整输出后做三组本地配对评分。

## 研究问题

在作者固定 DAG、角色、提示词、最终汇总和解码设置下，少量本任务标签训练的 v2.1 是否能在作者 test 上维持或提升答案质量，同时减少全流程通信 token。GSM8K 使用原 MathSolver/FinalRefer 路径；MMLU-Pro 使用原 AnalyzeAgent/FinalRefer 路径，Wiki Searcher 不纳入本轮。

## 数据隔离

GSM8K 的 24/120 从官方 train7473 按 seed42 的固定哈希抽取；MMLU-Pro 没有官方 train，本轮从官方 validation70 按 14 类分层抽取每类 1 题作 train、每类其余 4 题作 val。作者仓库原始 GSM8K300 与 MMLU-Pro280 保留为 test。三 split 规范化题文交集为 0；答案只在本地 gold 文件，绝不进入云端 prompt。

## 执行约束

每批 8 小时绝对截止、20,000 新调用、16,000,000 模型 token 上限；等待时间计入截止，恢复不重置预算。先完成 CPU 预检和唯一快照，再上传并启动协调器。终端 invalid 结果保留，重复运行只从精确兼容缓存续未提交条件。适配 checkpoint 在 val/test 前冻结，不根据 test 调参。

# 无时间窗口的并行续跑

2026-10-16用户明确要求立即开始、不设窗口、尽可能使用GPU跑完。管理入口保留原累计预算、STOP、冻结算法/数据/policy与已提交终态，只移除common及merger各一处实验截止条件；原文件和run_start没有改写。原请求超时、trace静默与隔离代码5秒仍保留。原MMLU1140及Pro val112完整批次不重跑。

启动前363文件全量原始归档逐文件SHA下载核验通过，SHA576c6a6d64c3e27e04bff2b9e6c01ba40afb1ba33c0652a00550dac34c9f6a5b。8项CPU检查（六模块AST只改时间条件、STOP、预算函数不变、终态续跑限制、GPU准入/不重叠/不足卡）本地及远端通过；初始运行管理本地提交c50928f。未推送。

09:37:11同时启动三批。Pro监督器3007459，GPU1主27B/4合并9B，端口11449/11456；GSM监督器3007460，GPU2/5，11448/11450。SVAMP首次监督器3007461在第二次GPU利用率准入检查失败，在启动任何模型服务前退出；自有资源清理。完整119文件归档下载SHA核验通过，SHA00d4dd4b47f272e80779be368fd66d7f2b841cb86f4c49ed5c82663c23334f00。针对同一根因仅一次换卡恢复，09:39:39启动监督器3011499，GPU6/0，11454/11455，旧记录保留，未重复核销费用。

launch_verification.json已核对六张卡不重叠、所有27B/9B的CUDA分配及短生成、digest、size_vram>=size、各批两worker实际cmdline、raw更新时间。三批phase=test/arm=moc，experiment_wall_deadline=null，当前无error，旧run_start及已有终态字节前缀和policy/source/data锁均保持。原MOC新增终态：GSM513/600，Pro16/560，SVAMP432/600，全有效无重复；适配600/560/600保留。

该快照累计预算：GSM actual7816362token/11513calls、pending20384/unknown0；Pro actual4236036/5315、pending20384/unknown0；SVAMP actual3932541/7496、pending20384/unknown10192。SVAMP旧10192已经移入unknown，当前pending是新进行中请求，须同时计cap，不能把旧独立closure初值当作当前费用。预算未重置；所有新增启动热身计入费用。

监控ACTIVE、十分钟及原线程不变，健康状态安静，重要阶段/恢复/完成/失败才通知。只使用resume_no_deadline_20261016管理入口，旧协调器与dispatch保留作历史，不重启。GPU3/7上的资源未触碰；并行三批已使用可准入的三个主卡和三个独立合并卡。
