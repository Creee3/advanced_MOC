Prepared after explicit two-additional-GPU authorization; awaiting CPU preflight and launch.

2026-10-15 19:30:18 Asia/Shanghai launched independent val/moc supervisor2497046. Uploaded43 frozen files and parent val byte identity verified. Local coverage checks accept112 terminals including invalid and reject missing/duplicate/wrong-arm conditions. CPU preflight passed6 synthetic native Graph conditions,4 separate-locals checks,4 isolation denials and timeout, with0 model calls.

19:30:59现场：stage=val,arm=moc；workers2498180/2498181、supervisor与两服务真实cmdline均匹配。新增GPU0主服务11451/PID2497248，GPU4合并器11452/PID2497656；真实CUDA分配与短生成通过，27B与9B digest正确，实时size_vram=size分别16757418884与6567493631。开始处理首批条件，尚未提交首条结果；原GSM与Pro监督器仍健康。独立7200秒窗口剩余7159秒，旧实验预算和截止未变。证据launch_inspection.json。

10分钟监控moc已通过工具更新为同时管理原两批及本补测，明确本次临时新增两张、合计最多四张；补测结束清理新服务。全三批审计完成后才暂停。

19:31:37复核：两个worker已真实生成并持续写入calls_raw，年龄0.5/1.7秒，正在完成首批条件，尚无终态提交；日志为作者原ISM流程，无运行错误。

20:03完成112/112有效终态；原MOC密度0.3正确29/56，0.7正确28/56。新增两服务/worker/子进程清理完成、端口11451/11452关闭，独立现场检查通过，无重启或重复条件。完整归档SHA1c6971a3353fbe3bb51d2fe38cfe8d3bb2cc94f176b109590e2344f88e94242f，77文件哈希核对、冻结锁/原解析器/父val输入一致通过。所有正式请求reservation逐条结算；实际802056tokens/1400newcalls含热身32tokens，logical994632tokens/1791calls含393cachehits；unknown0、pending0、输出截断0、invalid0。completion_audit.json=passed。

三组val报告已写入父实验val_review_20261015/三组val对照结果.md；adapted对MOC稀疏+1题、dense−1题，推理logicaltokens−10.26%/−52.94%，两个密度正确总数均57/112；小样本不证明非劣。14题训练实际408512tokens单列，不能用推理节省直接声称含训练总成本降低。原两批仍运行，不暂停总监控；本补测禁止再次launch。
