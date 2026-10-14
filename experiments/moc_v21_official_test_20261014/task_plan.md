# 冻结v2.1与原MOC官方test254对照

用户授权：官方完整test254；1张27B GPU共享两组，另1张9B GPU运行原MOC合并器。官方test有历史接触，明确披露，不称首次独立确认。

1. [x] 清单与GPU核查：dev30/val60来自官方dev，与test254零交叉；test历史已覆盖254。主GPU1和一个9B适用GPU可用。
2. [x] 新目录实现双模型、费用/断点、仅两组测试执行；保持v2.1算法冻结。
3. [x] 模型无调用预检通过；两DAG7/15edges，四次原Graph/FinalRefer检查，dense MOC25merger calls，v2.1无merger；费用/原路由通过。准备Git冻结与部署。
4. [x] GPU准入/CUDA/短生成/显存驻留后启动1016条件。23:38北京时间启动，GPU0主27B/GPU1合并9B；23:39:59核对2/1016有效（两组各1），无error，新实际响应和费用持续提交。
5. [ ] 真实日志进度核对；完成后下载raw/hash、配对分析与费用/清理审计。

原MOC有5候选合并，必须保留9B及其调用费用。两种密度共享254题。禁止调参、替换题、改变模型、重新拟合或将历史结果挑选为有利缓存。

本批新上限：8小时累计墙钟、30000新增调用、12000000新增模型token；包含两服务warmup。STOP与预算优先；原期限/账本不重置。正常结果及invalid终态条件均不重做。无自动监控创建。

## Errors
- 初次查找旧run_val.py不存在，核对后实际旧文件为run_experiment.py；无执行。
- 两次试读旧prepare_data.py路径不存在，实际为prepare_local.py；清单已直接核验。
- 首次无模型原Graph测试的fake embedding为float64，与作者GCN float32不符；改为float32，属测试替身修复，0真实调用/费用，旧staging包保留attempt00。
