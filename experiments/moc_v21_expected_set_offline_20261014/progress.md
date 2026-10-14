# Progress

2026-10-14：用户要求继续更新。审计发现停止规则量混淆、每次 bootstrap 截距未正确中心化、序列状态支持不足。新建独立 v2.1 离线目录，保留所有 v2 冻结原件。不启 GPU、不重跑基准、不读取 val 进行模型修订。

protocol预注册commit219aa66。新增ExpectedSetSampler：一次评分、比较总净收益、不用bootstrap上界/token、不进入已选集合的未知状态。7项行为测试通过，包含作者原Graph.arun/FinalRefer的空/非空通信集成、预算边界、支持弃权、较高组总收益胜过单条每token收益、模型不稳定不强迫探索。期望收益学习器及35题隔离OOF由并行代理实施。

ExpectedGroupUtilityModel的9项数值/模式锁/题bootstrap重采样检查通过，既有v2的12项与v1的6项回归通过。首次锁定OOF176条均fail，随后独立源码审查发现旧collect把pair完整path传给singleton，产生训练/生产特征不一致。首次结果原样归档attempt00_stored_features；协议更正commit78faecb先于固定方案复核。只派生singleton path/connectedflag，原label、cost、其余特征和v2冻结原件不改。

更正106条singleton后按固定协议复核一次，MSE旧参照0.359065/full0.370540/compact0.350914均差于zero0.304762/prior0.303869，净收益全负，未晋级。35题、70episode、176标签，动作无组读取；新cal/val/test未评估。原首次源码与输出SHA重建核验完整，最终offline_audit检查来源/派生数据/题折/权重/净收益/实际全流程费用/checkpoint SHA通过；19新检查加18历史回归通过。报告与代码提交，本轮结束，新增模型token=0，不扩大云端批次。
