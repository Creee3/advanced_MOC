# 整理进度

2026-10-16：开始只读盘点。已读取旧日报和最近五批收口记录，确认需补入AQuA/HumanEval并替换旧运行进度。准备生成Excel与中文说明。

完成：提取六个test数据集和两数据集val，核对终态唯一键、正确数、已知逻辑用量、改对/改错数和精确McNemar；沿用最终审计中的bootstrap区间。

提取脚本适配了三种既有归档差异：HumanEval下载凭据字段为files、MMLU records没有run层、独立Pro val位于remote_snapshot/records。初次路径/字段断言失败已修复，属于本次提取器兼容问题，没有修改来源。

完成：使用artifact-tool生成Excel、中文报告、CSV/JSON及ZIP；没有新增模型调用。所有六张工作表均渲染检查，共8个预览范围。修复空表头、方法标签截断、行高以及原MOC训练标签。

首次导出核验发现表格默认没有autoFilter；按文档显式设置showFilterButton后重新导出，openpyxl只读检查确认全部表格筛选器存在。核对了导出公式与缓存值，测试输入变化重算并恢复，不存在Excel错误单元格。

最终verification passed：54个来源SHA、7148行唯一条件、12个test比较、24组方法用量及七批费用总和一致。ZIP逐成员SHA和CRC通过。整理包为221071字节，SHA256 4fd5c9353264527783850a64774d8b0d50d6e60df04015be320bf73f77a10325。交付目录：D:/liuxin_workspace/ICML2027/outputs/01a11c24-c60c-7b42-99e6-70e3c00dea5e/experiment_summary_20261015_16/。

所有冻结实验、test金标、原raw、policy与源文件保持原样。本次只新增汇总和生成脚本；不操作调度、不推送。
