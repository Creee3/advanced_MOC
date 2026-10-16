import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url));
const data=JSON.parse(await fs.readFile(path.join(here,'summary_data.json'),'utf8'));
const out=path.resolve(here,'../../../outputs/01a11c24-c60c-7b42-99e6-70e3c00dea5e/experiment_summary_20261015_16');
await fs.mkdir(out,{recursive:true});
const wb=Workbook.create();
const names=['Test对照','Val对照','批次成本','Test用量明细','路由诊断','口径与历史'];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const color={ink:'#192A43',head:'#293E5C',line:'#D6DDE7',muted:'#53657A',light:'#F2F5F9'};
const method=x=>({moc:'原MOC',adapted:'任务适配',frozen_transfer:'冻结迁移',v21:'v2.1',v21_expected_set:'v2.1'}[x]??x);
function col(n){let x='';for(n++;n;n=Math.floor((n-1)/26))x=String.fromCharCode(65+(n-1)%26)+x;return x;}
function setup(sheet,title,context,widths){
  sheet.showGridLines=false;
  sheet.getRange(`A1:${col(widths.length-1)}160`).format.font.name='Arial';
  sheet.getRange(`A1:${col(widths.length-1)}160`).format.font.size=11;
  sheet.getRange(`A1:${col(widths.length-1)}160`).format.font.color=color.ink;
  widths.forEach((w,i)=>sheet.getRange(`${col(i)}1:${col(i)}160`).format.columnWidthPx=w);
  sheet.getRange('A2').values=[[title]];
  sheet.getRange('A2').format.font.size=16;sheet.getRange('A2').format.font.bold=true;
  sheet.getRange('A3').values=[[context]];sheet.getRange('A3').format.font.color=color.muted;
  sheet.getRange(`A4:${col(widths.length-1)}4`).format.borders={bottom:{style:'thin',color:color.line}};
  sheet.getRange(`A1:${col(widths.length-1)}160`).format.verticalAlignment='center';
}
function table(sheet,headers,rows,start,name,widthNotes=[]){
  const end=start+rows.length,last=col(headers.length-1),range=`A${start}:${last}${end}`;
  sheet.getRange(range).values=[headers,...rows];
  sheet.getRange(range).format.rowHeightPx=28;
  sheet.getRange(`A${start}:${last}${start}`).format.fill=color.head;
  sheet.getRange(`A${start}:${last}${start}`).format.font.color='#FFFFFF';
  sheet.getRange(`A${start}:${last}${start}`).format.font.bold=true;
  sheet.getRange(`A${start}:${last}${start}`).format.horizontalAlignment='center';
  sheet.getRange(`A${start}:${last}${start}`).format.wrapText=true;
  sheet.getRange(`A${start}:${last}${start}`).format.rowHeightPx=42;
  for(let i=start+1;i<=end;i++)if((i-start)%2===0)sheet.getRange(`A${i}:${last}${i}`).format.fill=color.light;
  const createdTable=sheet.tables.add(range,true,name);
  createdTable.showFilterButton=true;
  for(const [idx,formula] of widthNotes){for(let i=start+1;i<=end;i++)sheet.getRange(`${col(idx)}${i}`).formulas=[[formula(i)]];}
  if(rows.length>12)sheet.freezePanes.freezeRows(start);
  return end;
}
const s=sheets['Test对照'];
setup(s,'MOC实验结果：2026-10-15—16','完整test。准确率差与区间单位：百分点。每行对应同一批题目的配对比较。',[115,185,65,70,85,85,90,90,90,95,95,110,100,125,125,85,85,85]);
const testRows=data.test_comparisons.map(r=>[r.dataset,r.training,r.density,r.questions,r.moc_correct,r.candidate_correct,null,null,null,r.ci_low*100,r.ci_high*100,r.mcnemar_p,null,r.moc_tokens,r.candidate_tokens,r.moc_invalid,r.candidate_invalid,r.source_id]);
table(s,['数据集','v2.1训练方式','密度','题数','MOC正确','v2.1正确','MOC准确率','v2.1准确率','差值(pp)','CI下界(pp)','CI上界(pp)','McNemar p','token减少率','MOC逻辑token','v2.1逻辑token','MOC invalid','v2.1 invalid','证据ID'],testRows,6,'TestComparisons',[
 [6,r=>`=E${r}/D${r}`],[7,r=>`=F${r}/D${r}`],[8,r=>`=(H${r}-G${r})*100`],[12,r=>`=1-O${r}/N${r}`]
]);
s.getRange('G7:H18').setNumberFormat('0.00%');s.getRange('I7:K18').setNumberFormat('+0.00;-0.00;0.00');s.getRange('L7:L18').setNumberFormat('0.000000');s.getRange('M7:M18').setNumberFormat('0.00%');s.getRange('N7:O18').setNumberFormat('#,##0');
s.getRange('B7:B18').format.wrapText=true;s.getRange('A7:R18').format.rowHeightPx=40;
s.getRange('I7:I18').conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{font:{color:'#A33D35'}}});
s.getRange('P7:Q18').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#FFF0DB',font:{color:'#9B4A10',bold:true}}});
s.getRange('A21').values=[['注意：p值及区间未做多重比较校正。AQuA invalid计0。token为含缓存的逻辑用量，稠密组包含去掉9B合并器的影响。']];
s.tabColor='#293E5C';

const v=sheets['Val对照'];
setup(v,'已完成val对照','GSM8K没有原MOC val组；MMLU-Pro原MOC val来自独立补测。val与test不合并。',[115,70,175,75,85,105,145,85,85,150]);
table(v,['数据集','密度','方法','题数','正确','准确率','逻辑token','invalid','证据ID'],data.val_arms.map(x=>[x.dataset,x.density,method(x.arm),x.questions,x.correct,null,x.logical_tokens,x.invalid,x.source_id]),6,'ValArms',[[5,r=>`=E${r}/D${r}`]]);
v.getRange('F7:F16').setNumberFormat('0.00%');v.getRange('G7:G16').setNumberFormat('#,##0');
v.getRange('A20').values=[['配对差异（候选减参照）']];v.getRange('A20').format.font.bold=true;
table(v,['数据集','密度','参照','候选','差值(pp)','CI下界(pp)','CI上界(pp)','McNemar p','token减少率','证据ID'],data.val_pairs.map(x=>[x.dataset,x.density,method(x.reference),method(x.candidate),x.delta*100,x.ci_low*100,x.ci_high*100,x.mcnemar_p,-x.token_change,x.source_id]),22,'ValPairs');
v.getRange('C23:D28').format.wrapText=true;v.getRange('E23:G28').setNumberFormat('+0.00;-0.00;0.00');v.getRange('H23:H28').setNumberFormat('0.000000');v.getRange('I23:I28').setNumberFormat('0.00%');

const c=sheets['批次成本'];
setup(c,'七个已完成批次的累计用量','包含两组方法、训练、val、热身和中断成本。跨日批次累计数不等于日历两天内的消费。',[120,90,80,85,85,80,140,135,100,145,100,85]);
table(c,['批次','本轮训练题','训练标签','终态条件','有效','invalid','actual token','unknown上界','pending','预算计入token','新调用','证据ID'],data.batches.map(x=>[x.dataset,x.training_questions,x.labels??'n.a.',x.conditions,x.valid,x.invalid,x.actual_tokens,x.unknown_upper,x.pending,null,x.new_calls,x.source_id]),6,'BatchCosts',[[9,r=>`=SUM(G${r}:I${r})`]]);
c.getRange('D7:K15').setNumberFormat('#,##0');c.getRange('A15').values=[['合计']];for(const k of ['D','E','F','G','H','I','J','K'])c.getRange(`${k}15`).formulas=[[`=SUM(${k}7:${k}13)`]];
c.getRange('A15:L15').format.font.bold=true;c.getRange('A15:L15').format.borders={top:{style:'thin',color:color.line}};
c.getRange('A18').values=[['共享历史训练（参考，不并入上表）']];c.getRange('A19:C21').values=[['项目','模型token','处理方式'],['共享旧训练',data.shared_prior_training_tokens,'AQuA来源checkpoint，只列一次，不向两个test批次重复添加'],['旧v1及废弃',data.earlier_v1_training_waste_tokens,'历史开发参考，不并入两天七批总账']];c.getRange('B20:B21').setNumberFormat('#,##0');c.getRange('C20:L21').format.wrapText=false;
c.getRange('A24').values=[['SVAMP旧中断请求10,192已核销为unknown。AQuA unknown上界59,344。unknown不是actual，也不与同一旧pending重复相加。']];

const u=sheets['Test用量明细'];
setup(u,'Test逻辑用量与实际新增的区别','每行是一个数据集、方法和密度。逻辑token包含cache hit；终态新增token不含未提交中断工作。',[110,95,195,65,75,85,85,145,140,140,140,110,110,85]);
table(u,['数据集','方法','训练方式','密度','题数','正确','invalid','逻辑总token','主模型token','合并器token','终态新增token','逻辑调用','cache hit','证据ID'],data.test_arms.map(x=>[x.dataset,method(x.arm),x.arm==='moc'?'无适配训练':x.training,x.density,x.questions,x.correct,x.invalid,x.logical_tokens,x.main_tokens,x.merger_tokens,x.new_tokens,x.logical_calls,x.cache_hits,x.source_id]),6,'ArmTokenDetails');
u.getRange('E7:M30').setNumberFormat('#,##0');
u.getRange('C7:C30').format.wrapText=true;u.getRange('A7:N30').format.rowHeightPx=40;

const r=sheets['路由诊断'];
setup(r,'消息选择与覆盖诊断','统一分母：有效图中有候选消息的普通接收节点，已排除根节点。空选择仍保留最终汇总。',[110,65,85,100,85,85,85,95,100,110,80,80,80,80,115,115,85]);
table(r,['数据集','密度','有效图','非根决策','空选择','单条','双条','空选择率','支持不足','净收益非正','hop1','hop2','hop3','hop4','编码秒','路由秒','证据ID'],data.routing.map(x=>[x.dataset,x.density,x.valid_graphs,x.nonroot_decisions,x.empty,x.single,x.pair,null,x.unsupported,x.nonpositive,x.hop1,x.hop2,x.hop3,x.hop4,x.encoding_seconds,x.routing_seconds,x.source_id]),6,'RoutingDiagnostics',[[7,i=>`=E${i}/D${i}`]]);
r.getRange('H7:H18').setNumberFormat('0.00%');r.getRange('O7:P18').setNumberFormat('0.000');
r.getRange('H7:H18').conditionalFormats.add('cellIs',{operator:'equal',formula:1,format:{fill:'#FFF0DB',font:{color:'#9B4A10',bold:true}}});
r.getRange('A21').values=[['SVAMP：训练200标签中198零、2负、0正；help头全0。HumanEval稠密非根984次全部因支持不足弃选。']];

const notes=sheets['口径与历史'];
setup(notes,'结果解释、历史边界与证据索引','报告时区：Asia/Shanghai。保留所有已完成结果和失败限制，不重新生成或调参。',[180,760,560]);
const definitions=[
 ['汇总范围','10月15—16日涉及并完成的六数据集test与独立Pro val，共七批。AQuA跨日运行，目录日期为14日，完成于15日。'],
 ['题数与条件','6332个test终态来自1583个数据集内题目×2密度×2方法。密度重复使用同一题，不能把样本量翻倍。'],
 ['逐题CSV','7148行：6332个test条件，另有GSM val480与Pro三组val336。Pro补测112已包含在336中。'],
 ['失败处理','AQuA有9个invalid，完整分母计0，未删除。其余五个test数据集生成终态全部有效。'],
 ['统计口径','配对question bootstrap20000次、seed42，精确双侧McNemar。沿用原分析的随机数实现与分位数口径，不重选结果。'],
 ['多重比较','12个test对照的p值和区间为逐条件原始统计，未做多重比较校正。不能仅凭一项正向结果概括普遍改善。'],
 ['非劣与等效','区间跨零不证明非劣或等效。没有预先定义非劣界限。'],
 ['主要正向观察','MMLU-Pro密度0.7：153/280对139/280，差+5.00pp，CI[1.43,8.57]pp，原始p0.009355。'],
 ['成本归因','所有12项对照的已知逻辑token均下降。稠密原MOC含9B合并器，节省不能全部归因于学习评分或二元组。'],
 ['SVAMP限制','全弃选由缺少正收益训练信号与支持门控解释；该结果不证明学会多阶路由。没有依据test更换样本或重训。'],
 ['HumanEval限制','AQuA checkpoint直接迁移，未做代码域训练。稠密非根节点全弃选，保留最终汇总。'],
 ['数据限制','MMLU训练40科不覆盖57科；SVAMP40是研究划分而非官方train。新两批无val。历史test接触不能抹去。'],
 ['费用含义','actual、unknown与pending分列。总账为七批累计，不是单个方法部署费用或严格日历两天消费。不换算货币。'],
 ['CPU与延迟','编码/路由秒为累计组件观测，不是受控端到端加速。未完整测量的训练CPU耗时不当0。'],
 ['审计含义','结果、原始记录、缓存、源码、解析器与费用一致性核验；不宣称独立复现或多阶效用因果归因。'],
];
table(notes,['口径','说明'],definitions,6,'Definitions');notes.getRange('B7:B21').format.wrapText=true;notes.getRange('A7:B21').format.rowHeightPx=42;
notes.getRange('A24').values=[['历史开发与未执行计划（不计入主test和七批费用）']];
table(notes,['项目','状态与日期','发现'],data.history.map(x=>[x.experiment,`${x.date}；${x.status}`,x.finding]),26,'ExperimentHistory');notes.getRange('B27:C31').format.wrapText=true;notes.getRange('A27:C31').format.rowHeightPx=48;
notes.getRange('A34').values=[['证据索引（与数据表证据ID对应，路径相对工作区）']];
table(notes,['证据ID','来源文件','SHA256'],data.sources.map(x=>[x.id,x.path,x.sha256]),36,'EvidenceSources');
notes.getRange(`B37:B${36+data.sources.length}`).format.wrapText=true;notes.getRange(`A37:C${36+data.sources.length}`).format.rowHeightPx=46;
notes.getRange(`C37:C${36+data.sources.length}`).format.font.size=10;

wb.recalculate();
const headline=s.getRange('G7:O18').values;
data.test_comparisons.forEach((x,i)=>{
 const a=headline[i];
 if(Math.abs(a[0]-x.moc_correct/x.questions)>1e-10||Math.abs(a[1]-x.candidate_correct/x.questions)>1e-10||Math.abs(a[6]-(1-x.candidate_tokens/x.moc_tokens))>1e-10)throw Error('Formula mismatch '+x.dataset);
});
const total=c.getRange('G15:K15').values[0];
const expected=[data.totals.actual_tokens,data.totals.unknown_upper,data.totals.pending,data.totals.actual_tokens+data.totals.unknown_upper,data.totals.new_calls];
if(JSON.stringify(total)!==JSON.stringify(expected))throw Error('Cost total mismatch '+JSON.stringify(total));
// Check that downstream formulas respond to edited inputs, then restore the audited values.
const originalCorrect=s.getRange('E7').values[0][0];
s.getRange('E7').values=[[originalCorrect+1]];wb.recalculate();
if(Math.abs(s.getRange('G7').values[0][0]-(originalCorrect+1)/data.test_comparisons[0].questions)>1e-10)throw Error('Accuracy recalculation failed');
s.getRange('E7').values=[[originalCorrect]];
const originalActual=c.getRange('G7').values[0][0];
c.getRange('G7').values=[[originalActual+1]];wb.recalculate();
if(c.getRange('G15').values[0][0]!==expected[0]+1)throw Error('Cost recalculation failed');
c.getRange('G7').values=[[originalActual]];wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:1500});
await fs.writeFile(path.join(out,'workbook_formula_scan.ndjson'),errors.ndjson);
const check=await wb.inspect({kind:'table',range:'Test对照!A6:R18',include:'values,formulas',tableMaxRows:13,tableMaxCols:18,maxChars:3500});
await fs.writeFile(path.join(out,'workbook_inspection.ndjson'),check.ndjson);
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(out,'实验数据汇总_20261015-16.xlsx'));
const previews=[['Test对照','A1:M19','test'],['Test对照','N6:R18','test_tokens'],['Val对照','A1:J29','val'],['批次成本','A1:L24','cost'],['Test用量明细','A1:N31','usage'],['路由诊断','A1:Q22','routing'],['口径与历史','A1:B22','definitions'],['口径与历史','A24:C42','history_sources']];
for(const [sheetName,range,name] of previews){const blob=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(out,`preview_${name}.png`),new Uint8Array(await blob.arrayBuffer()));}
await fs.writeFile(path.join(out,'workbook_validation.json'),JSON.stringify({status:'passed',sheets:names.length,comparison_rows:12,formula_values_verified:true,source_cost_totals_match:true,formula_error_scan:errors.ndjson,rendered_ranges:previews.map(x=>x[2]),xlsx:'实验数据汇总_20261015-16.xlsx'},null,2));
console.log(JSON.stringify({status:'exported',output:path.join(out,'实验数据汇总_20261015-16.xlsx'),sheets:names.length,previews:previews.length}));
