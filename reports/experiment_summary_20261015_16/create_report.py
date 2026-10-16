"""Write the narrative from audited summary data; no model calls or experiment edits."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/01a11c24-c60c-7b42-99e6-70e3c00dea5e/experiment_summary_20261015_16'
d = json.loads((HERE / 'summary_data.json').read_text(encoding='utf-8'))

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

def method(x):
    return {'moc': '原MOC', 'adapted': '任务适配', 'frozen_transfer': 'AQuA冻结迁移'}.get(x, x)

test_rows = []
for x in d['test_comparisons']:
    n = x['questions']
    test_rows.append([x['dataset'], x['density'], n,
        f"{x['moc_correct']}/{n}（{x['moc_correct']/n:.2%}）",
        f"{x['candidate_correct']}/{n}（{x['candidate_correct']/n:.2%}）",
        f"{(x['candidate_correct']-x['moc_correct'])/n*100:+.2f}",
        f"[{x['ci_low']*100:.2f}, {x['ci_high']*100:.2f}]",
        f"{x['mcnemar_p']:.6f}", f"{1-x['candidate_tokens']/x['moc_tokens']:.2%}", x['source_id']])

cost_rows = [[x['dataset'], f"{x['actual_tokens']:,}", f"{x['unknown_upper']:,}", x['pending'], f"{x['new_calls']:,}", x['source_id']] for x in d['batches']]
t = d['totals']
cost_rows.append(['合计', f"{t['actual_tokens']:,}", f"{t['unknown_upper']:,}", t['pending'], f"{t['new_calls']:,}", ''])

val_rows = [[x['dataset'],x['density'],method(x['arm']),f"{x['correct']}/{x['questions']}",f"{x['correct']/x['questions']:.2%}",f"{x['logical_tokens']:,}",x['source_id']] for x in d['val_arms']]
pair_rows = [[x['dataset'],x['density'],method(x['candidate'])+' − '+method(x['reference']),f"{x['delta']*100:+.2f}",f"[{x['ci_low']*100:.2f}, {x['ci_high']*100:.2f}]",f"{x['mcnemar_p']:.6f}",f"{-x['token_change']:.2%}"] for x in d['val_pairs']]
route_rows = [[x['dataset'],x['density'],x['nonroot_decisions'],x['empty'],f"{x['empty']/x['nonroot_decisions']:.2%}",x['single'],x['pair'],x['unsupported'],x['nonpositive']] for x in d['routing']]

report = '''# 2026年10月15—16日 MOC实验数据汇总

本次汇总包含六个数据集的完整test、GSM8K/MMLU-Pro的val，以及独立Pro原MOC val补测。七个费用批次均已有完成记录；六个test数据集共12组配对比较。本文按北京时间整理，使用最终审计结果覆盖早先尚未跑完的日报。AQuA目录日期虽为10月14日，但跨日运行并于15日完成，仍列入本轮盘点。

**目前最稳定的观察是逻辑token下降，准确率表现有升有降。** MMLU-Pro密度0.7出现+5.00个百分点的提升；其余test差异区间均包含0。SVAMP两密度全弃选、HumanEval稠密组全弃选，因此不能用整体省token来证明内容区分、消息互补或多阶路由已学会。

## 1. 文件与阅读方式

- `实验数据汇总_20261015-16.xlsx`：六个可筛选工作表，含test对照、val对照、批次成本、test用量、路由诊断、口径与历史。
- `question_results.csv`：7,148行条件记录；6,332行test，另有816行val。保留题号、方法、密度、正确与否、状态、用量和来源ID，不包含题文、金标答案、生成代码。
- `test_comparisons.csv`、`val_arms.csv`、`cost_ledger.csv`、`routing.csv`：可直接继续分析的小表。
- `summary_data.json`：完整结构化汇总；`source_manifest.json`：54个来源文件的路径与SHA256。表中的Sxxx可在Excel证据索引或该文件查到。
- `verification.json`：提取、来源哈希、Excel公式及导出值核验记录。以上均为本地一致性核验，不宣称独立复现。

## 2. 完整test结果

v2.1一列中，AQuA使用既有冻结checkpoint，HumanEval直接迁移该checkpoint；其余四个数据集使用各自冻结的任务适配policy。每一行在同一批题目上配对比较。准确率差和CI单位均为**百分点**；token为main+merger的已知逻辑用量，包含精确缓存命中。

'''
report += table(['数据集','密度','题数','原MOC','v2.1','差值pp','95% CI pp','McNemar p','token减少','证据'],test_rows)
report += '''

统计沿用最终分析中的20,000次question bootstrap（seed42）和精确双侧McNemar，并重新核对正确数、改对/改错数和McNemar值。p值与区间没有做12组比较的多重校正。MMLU-Pro 0.7是一个具体条件的正向观察，不能据此概括跨任务普遍改善；区间包含0也不能证明等效或非劣。两密度使用同一批题，不应视为两份独立样本。

AQuA的9个invalid完整保留并计0，分母仍为每组254；9个条件涉及5个不同题目，不是9个独立题目。其余五个数据集test终态全部有效。AQuA存在历史test接触，不能称首次未见测试。HumanEval使用作者的MultiPL-E转换版本和已有代码执行评分，本次整理没有重新执行生成代码。

## 3. 训练与划分

| 数据集 | 本轮训练题 | val | 作者test题 | 训练标签 | 说明 |
| --- | --- | --- | --- | --- | --- |
| AQuA | 0（复用旧训练） | 本轮无新增 | 254 | 不计为本轮新增 | 使用AQuA冻结checkpoint；旧训练开销单列 |
| HumanEval | 0 | 无 | 164 | 无新增 | AQuA checkpoint零样本迁移，没有代码域训练 |
| GSM8K | 24 | 120 | 300 | 120 | 官方train按seed42锁定24/120；test仅作测试 |
| MMLU-Pro | 14 | 56 | 280 | 74 | 官方validation70按14类每类1 train、4 val |
| MMLU | 40 | 0 | 285 | 214 | 官方dev固定40科各1题，不覆盖全部57科 |
| SVAMP | 40 | 0 | 300 | 200 | 从full1000排除作者test、近重复和数字模板后固定抽取；研究用划分，并非官方train |

标签数是相关动作标签数，不是独立题目数。MMLU和SVAMP各80个训练episode，没有新增val或跨任务迁移组。所有policy在相应val/test前冻结，没有根据test换题、重训或调参。SVAMP仍有题族语义隔离方面的限制，不能把字符串/数字模板排除说成语义彻底隔离。

## 4. val结果

'''
report += table(['数据集','密度','方法','正确/题数','准确率','逻辑token','证据'],val_rows)
report += '\n\n配对差异如下；GSM的参照是冻结迁移，并没有原MOC val组。Pro原MOC val来自已完成的独立112条件补测。\n\n'
report += table(['数据集','密度','比较','差值pp','95% CI pp','McNemar p','token减少'],pair_rows)
report += '''

这些val比较不能与test混为一个样本量，也不能把GSM适配相对冻结迁移的结果写成相对原MOC的结果。Pro冻结迁移的普通非根节点在两密度均全弃选，不能只看val准确率就认定迁移成功学会了消息价值。

## 5. 累计调用与成本

下表是**七个批次的累计费用账本**，包含原MOC与v2.1两组以及各自训练、val、热身、中断/废弃工作；不是单一方法的部署费用，也不是严格日历两天的全项目消费。主表的条件数是6,332个test终态，加112个独立Pro val终态，共6,444；GSM/Pro适配批次内部val成本已经包含在整批费用，不能再加一遍。

'''
report += table(['批次','actual模型token','unknown保守上界','pending','新增调用','证据'],cost_rows)
report += f"\n\nactual加独立unknown上界共 **{t['actual_tokens']+t['unknown_upper']:,} token**。unknown是无返回请求的保守预留，不是已知实际消耗；pending现为0。SVAMP旧中断请求10,192已经独立核销为unknown，不能与同一旧pending重复相加；AQuA unknown上界为59,344。\n"
report += f'''
共享AQuA来源checkpoint的旧训练为{d['shared_prior_training_tokens']:,} token，仅作一次历史参考，不分别加到AQuA和HumanEval费用。更早v1训练及废弃工作{d['earlier_v1_training_waste_tokens']:,} token另列历史，不混进七批总账。AQuA已知中断/废弃部分9,274 token、21调用包含在该批actual，不能再加。

Excel“Test用量明细”另列逻辑总token、main、merger、终态新增token、调用和cache hit。**逻辑用量与实际新增不可互换**：例如SVAMP适配密度0.7终态模型请求全部命中精确缓存，新增token为0，但其923,956逻辑token仍完整计入比较。全流程actual还包含未提交工作、训练和热身，不能只用成功test行求和取代账本。没有统一货币价格和受控端到端时延数据，因此不换算金额或宣称相同比例加速。

## 6. 路由诊断与结果解释

统一分母为有效图中**有候选消息的普通非根接收节点**；排除根节点自然没有候选的决策。旧报告常列每题7个普通节点，本表按每题6个非根节点统计，两者不是计数矛盾。“双条”表示选两条消息，并不等于已经证明互补；hop列是所选消息的结构距离，不是因果效用证明。

'''
report += table(['数据集','密度','非根决策','空选择','空选择率','单条','双条','支持不足','净收益非正'],route_rows)
report += '''

**SVAMP的机制已定位。** 40题产生的200个标签中，198个为0、2个为负、正标签为0，冻结help头22个权重及截距均为0。harm预测非负，help−harm不为正，再减通信成本后无法超过空集的0分，因此可支持候选也全部被拒绝；其余由既定support门控拒绝。两种密度均是1,800次非根决策全部弃选（若含根节点则各2,100次）。普通agent继续生成，FinalRefer仍读取所有最终输出，因此不能描述成系统完全没有协作或多agent汇总。

SVAMP稠密组70.08%的token减少中，包含去掉9B合并器的1,531,488 token，也包含main输入减少；不能全部归因于学到消息效用。该冻结策略两密度均269/300，比原MOC少3题/2题，区间包含0。准确率接近不能证明所有通信无用，更不能证明多阶路由有效。关于训练为何没有正标签，局部empty条件76/80已答对、二元正确性信号饱和、接收者覆盖有限和局部训练代理偏差都是有限解释，目前没有唯一根因的因果证据。

HumanEval稠密组984次非根决策全部因支持不足弃选，主要反映迁移覆盖限制。GSM、Pro、MMLU确实出现单条/双条及不同hop的选择，但当前对照仍不能隔离“内容评分”“二元组交互”和“去掉合并器”各自的贡献。

因此，现有数据适合支持“固定候选结构上的效用门控能降低通信用量，任务效果依赖训练信号与覆盖”的描述；尚不足以支持“普遍提高准确率”“学会多阶互补”“省token且已证非劣”这些更强主张。

## 7. 历史诊断与未执行计划

'''
report += table(['项目','日期','状态','发现','证据'],[[x['experiment'],x['date'],x['status'],x['finding'],x['source_id']] for x in d['history']])
report += '''

10月14日v1/v2/离线OOF与val60只作开发历史。特别是compact离线MSE 0.350914劣于zero 0.304762，不能省略这一负结果来声称离线可学习性已证。代码域train48计划已搁置，没有实际新增HumanEval/MBPP训练、样本或费用。10月15日晚日报中的部分test计数已经过时，以本表最终完成数据为准。

## 8. 证据与核验边界

本次从现有结果、终态raw、已执行代码评分、费用账本、缓存及归档下载清单提取数据，核对每个(question_id, arm, density)键、正确数、已知逻辑token、配对改对/改错数、精确McNemar及来源SHA。最终bootstrap区间沿用原审计分析，不根据结果另选随机数或分位数实现。Excel的准确率、差值、token减少率、预算计入量和合计使用公式，并验证输入变化会触发重算、恢复后与原值一致。

下列归档SHA用于关联最终完整证据包；交付ZIP是这些结果的整理包，不是重新打包全部原始大文件。各批audit是记录与源码/费用一致性审计，不是独立复现实验，也不保证策略具有有效区分能力。

'''
report += table(['批次','最终归档SHA256','审计'],[[x['dataset'],x['archive_sha256'],x['audit']] for x in d['batches']])
report += '\n\n完整文件来源见`source_manifest.json`，原始实验文件未被修改。本次整理没有新增模型调用、训练、测试生成或远端调度。\n'

for dest in [HERE / '实验数据汇总_20261015-16.md', OUT / '实验数据汇总_20261015-16.md']:
    dest.write_text(report, encoding='utf-8')
(OUT / 'source_manifest.json').write_bytes((HERE / 'source_manifest.json').read_bytes())
print(json.dumps({'status':'written','report':str(OUT / '实验数据汇总_20261015-16.md'),'characters':len(report)},ensure_ascii=True))
