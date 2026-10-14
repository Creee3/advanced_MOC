# Progress

2026-10-14：用户授权test对比并确认官方完整254，明确选择共享1main27B+独立merger9B。已核验原分割为dev30/val60来自officialdev、与test零交叉；另train48拟合/校准35/12有效。test曾全部在旧实验接触，披露历史而不伪装未见。

本批仅冻结v2.1与原MOC、两密度1016条件。GPU服务尚未启动；新目录准备中。

实现test-only双模型路由、主/merger分账与联合冲突拒绝恢复；MOC原AsyncClient请求被指向私有9B，保持原温度和默认输出长度，5候选不改。任何merger失败作invalid，未知费用预留则停止审计；无CPU回退。254原输入hash/三划分交集检查，历史4334test调用加入，9613键9188可复用/425冲突全排除；不依正确性选择。旧test工具超时0。正式test gold未读/未部署。

无模型staging原Graph4条件通过，两密度每组8main calls，dense MOC25merger、候选0；路由/原选项/费用/mock预算结算通过。首次fake embedding float64不兼容原GCN，改float32，0实际调用，失败staging原件保留attempt00。最终包39文件SHA654ff8b9b4ca3aa6537554c52440a817bba9144ef519892cb5c0f9412a26c6d0。checkpoint同e2ce9cd7...fd210；协议提交后才部署/真GPU。
