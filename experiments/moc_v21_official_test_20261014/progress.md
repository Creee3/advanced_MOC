# Progress

2026-10-14：用户授权test对比并确认官方完整254，明确选择共享1main27B+独立merger9B。已核验原分割为dev30/val60来自officialdev、与test零交叉；另train48拟合/校准35/12有效。test曾全部在旧实验接触，披露历史而不伪装未见。

本批仅冻结v2.1与原MOC、两密度1016条件。GPU服务尚未启动；新目录准备中。

实现test-only双模型路由、主/merger分账与联合冲突拒绝恢复；MOC原AsyncClient请求被指向私有9B，保持原温度和默认输出长度，5候选不改。任何merger失败作invalid，未知费用预留则停止审计；无CPU回退。254原输入hash/三划分交集检查，历史4334test调用加入，9613键9188可复用/425冲突全排除；不依正确性选择。旧test工具超时0。正式test gold未读/未部署。

无模型staging原Graph4条件通过，两密度每组8main calls，dense MOC25merger、候选0；路由/原选项/费用/mock预算结算通过。首次fake embedding float64不兼容原GCN，改float32，0实际调用，失败staging原件保留attempt00。最终包39文件SHA654ff8b9b4ca3aa6537554c52440a817bba9144ef519892cb5c0f9412a26c6d0。checkpoint同e2ce9cd7...fd210；协议提交后才部署/真GPU。

协议dc7fc11提前提交并推送origin研究分支，部署新目录、验证包SHA；远端同源码无模型预检通过。23:38:01北京时间监督器1863666启动，主服务1863790/GPU0、merger服务1864318/GPU1，两个worker1864717/1864718共享11448并用11450合并。启动时GPU0/1已释放且连续满足准入，由冻结调度器选择（与准备时空闲卡不同，不改共享配置）。两服务CUDA真实分配/OK短生成/size_vram=size验证：27B16757418884byte，9B6567493631byte；warmup32token纳入费用。

23:39:59真实phase test，2/1016终态提交，两组各1有效，error为空。现场核对PIDcmdline与新响应created_at、日志raw年龄6.6/4.8秒；23次已预留新增调用，实际15083token（包含warm32），两在途预留20384正常。仅运行当前两组，无训练/其他基准/自动监控。完整test质量与费用待收口，不分析部分正确率。
