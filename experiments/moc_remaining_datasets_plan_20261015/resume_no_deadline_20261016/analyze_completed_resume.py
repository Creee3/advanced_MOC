"""Local-only complete-test scoring and full-cost consistency audit after resume."""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
SPECS={'moc_v21_gsm8k_fewshot_20261015':('GSM8K',300,'GSM8KDataset','answer'),
       'moc_v21_mmlu_pro_fewshot_20261015':('MMLU-Pro',280,'MMLUProDataset','answer'),
       'moc_v21_svamp_train40_test_20261015':('SVAMP',300,'SVAMPDataset','Answer')}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def tokens(r):return r['response']['prompt_eval_count']+r['response']['eval_count']

def native_parser(source,class_name):
    tree=ast.parse(source.read_text(encoding='utf-8'));nodes=[]
    for name in (('MathDataset',class_name) if class_name!='MMLUProDataset' else (class_name,)):
        node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name))
        node.bases=[ast.Name(id='object' if name in ('MathDataset','MMLUProDataset') else 'MathDataset',ctx=ast.Load())]
        node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('_clean_numeric_result','postprocess_answer','record_to_target_answer')]
        for n in ast.walk(node):
            if isinstance(n,ast.FunctionDef):
                n.returns=None
                for arg in n.args.args:arg.annotation=None
        nodes.append(node)
    scope={'re':re};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),scope)
    cls=scope[class_name];return cls,cls.__new__(cls)

def main():
    p=argparse.ArgumentParser();p.add_argument('name',choices=SPECS);args=p.parse_args()
    title,n,class_name,gold_field=SPECS[args.name];run=BASE/args.name;out=run/'test_review_20261016';raw=out/'records'/'run'
    receipt=read(out/'archive_receipt.json');manifest=read(out/'member_sha256.json')
    assert read(out/'download_verification.json')['all_hashes_match']
    for name,meta in manifest.items():assert sha(out/'records'/name)==meta['sha256'],name
    lock=read(raw/'experiment_lock.json');assert lock==read(run/'experiment_lock.json')
    for name,h in {**lock['source_hashes'],**lock['data_hashes']}.items():assert sha(raw/name)==h,name
    policy=read(raw/'policy_lock.json');assert policy['no_val_or_test_fitting'] and policy['frozen_before_val_and_test']
    assert sha(raw/'group_utility_model.json')==policy['model_sha256']
    for name,h in policy['training_commits_sha256'].items():assert sha(raw/name)==h
    assert read(raw/'status.json')['phase']=='complete' and read(raw/'status.json')['cleanup_complete'] and read(raw/'cleanup.json')['cleanup_complete']
    assert receipt['owned_cleanup_verified'] and receipt['original_terminal_prefixes_preserved']
    inputs=rows(raw/'data/test_inputs.jsonl');assert len(inputs)==n
    results=[r for f in (raw/'test').glob('shard*/results.jsonl') for r in rows(f)]
    keys=[(r['question_id'],r['arm'],r['density']) for r in results]
    assert len(keys)==len(set(keys))==n*4
    assert set(keys)=={(q['question_id'],arm,d) for q in inputs for arm in ('adapted','moc') for d in (.3,.7)}
    # Gold access starts only after complete, unique coverage and cleanup checks.
    gold_path=run/'data/test_gold_LOCAL_ONLY.jsonl';assert sha(gold_path)==lock['local_gold_sha256']['test']
    assert not (raw/'data/test_gold_LOCAL_ONLY.jsonl').exists()
    source=out/'records/author_source/datasets/data_process.py'
    author=BASE.parent/'coding_base/MOC-official'
    assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()==lock['author_commit']
    assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain'],text=True).strip()
    git_blob=subprocess.check_output(['git','-C',str(author),'show','HEAD:datasets/data_process.py'])
    # Archive bytes must match the committed source; Windows checkout may use CRLF.
    assert source.read_bytes()==git_blob
    assert (author/'datasets/data_process.py').read_bytes().replace(b'\r\n',b'\n')==git_blob
    cls,parser=native_parser(source,class_name)
    gold={q['question_id']:cls.record_to_target_answer({gold_field:q['answer']}) for q in rows(gold_path)}
    assert set(gold)=={q['question_id'] for q in inputs}
    for r in results:
        if r['status']=='ok':
            assert parser.postprocess_answer(r['final_output'])==r['answer']
            assert r['main_calls']==8 and r['merger_calls']%5==0
        assert r['logical_model_tokens']==r['main_model_tokens']+r['merger_model_tokens']
        r['correct']=int(r['status']=='ok' and r['answer']==gold[r['question_id']])
    calls={stage:[r for f in (raw/stage).glob('shard*/*calls_raw.jsonl') for r in rows(f)] for stage in ('training','val','test')}
    warm=rows(raw/'startup_calls_raw.jsonl');allraw=sum(calls.values(),[]);known=[r for r in allraw if not r['cache_hit']]+warm
    budget=read(raw/'budget.json');assert budget['reserved_tokens']==0 and sum(map(tokens,known))==budget['actual_tokens']
    trace=[r for stage in calls for f in (raw/stage).glob('shard*/trace.jsonl') for r in rows(f)]
    started={r['reservation_id']:r for r in trace if r['event_type']=='backend_request_started'}
    settled={r['reservation_id']:r for r in trace if r['event_type']=='backend_request_settled'}
    unknown={r['reservation_id']:r for r in trace if r['event_type']=='backend_request_reconciled_unknown'}
    assert not set(settled)&set(unknown) and not set(started)-set(settled)-set(unknown)
    assert sum(r['unknown_cost_upper_bound_tokens'] for r in unknown.values())==budget['unknown_cost_upper_bound_tokens']
    assert len(known)+len(unknown)==budget['new_calls']
    assert sum(r['actual_tokens'] for r in settled.values())+sum(map(tokens,warm))==budget['actual_tokens']
    capped=[dict(stage=r.get('stage'),arm=r.get('arm'),cache_hit=r['cache_hit']) for r in allraw if r['response'].get('done_reason') in ('length','max_tokens')]
    assert not any(r['cache_hit'] for r in capped)
    summary=dict(status='completed_local_scoring',dataset=title,questions=n,conditions=n*4,arms={},paired={},budget=budget,
        training=read(raw/'training_report.json'),policy_sha256=policy['model_sha256'],
        support_metadata_keys=read(raw/'group_utility_model.json')['coverage'],process_cost={},output_cap_responses=capped)
    for stage,group in calls.items():
        terminals=([q for f in (raw/stage).glob('shard*/questions.jsonl') for q in rows(f)] if stage=='training' else
            [q for f in (raw/stage).glob('shard*/results.jsonl') for q in rows(f)])
        actual=sum(tokens(r) for r in group if not r['cache_hit']);attributed=sum(r['new_model_tokens'] for r in terminals)
        assert actual>=attributed
        summary['process_cost'][stage]=dict(actual_tokens=actual,new_calls=sum(not r['cache_hit'] for r in group),
            logical_tokens_all_process_raw=sum(map(tokens,group)),cache_hits=sum(r['cache_hit'] for r in group),
            terminal_attributed_new_tokens=attributed,interrupted_unattributed_actual_tokens=actual-attributed)
    summary['process_cost']['warmup']=dict(actual_tokens=sum(map(tokens,warm)),new_calls=len(warm))
    summary['process_cost']['unknown']=dict(calls=len(unknown),conservative_upper_bound_tokens=budget['unknown_cost_upper_bound_tokens'])
    for d in (.3,.7):
        groups={}
        for arm in ('moc','adapted'):
            group=sorted([r for r in results if r['arm']==arm and r['density']==d],key=lambda r:r['question_id']);groups[arm]=group
            selections=[s for r in group if r.get('sampling') for s in r['sampling']['selections']]
            hops=[];support=Counter()
            for s in selections:
                hop={m['source']:m['hop'] for m in s['candidates']};hops.extend(hop[x] for x in s['source_ids'])
                for layer in s['layers']:
                    for score in layer.get('scores',[]):support['supported' if score.get('supported') else 'unsupported']+=1
            info=dict(n=n,correct=sum(r['correct'] for r in group),accuracy=sum(r['correct'] for r in group)/n,invalid=sum(r['status']!='ok' for r in group))
            for key in ('logical_model_tokens','main_model_tokens','merger_model_tokens','new_model_tokens','logical_calls','main_calls','merger_calls','cache_hits'):
                info[key]=sum(r[key] for r in group)
            info.update(selection_sizes=dict(Counter(len(s['source_ids']) for s in selections)),stop_reasons=dict(Counter(s['stop_reason'] for s in selections)),
                selected_hops=dict(Counter(hops)),candidate_hops=dict(Counter(m['hop'] for s in selections for m in s['candidates'])),support_evaluations=dict(support),
                encoding_input_tokens=sum(r['sampling']['encoding_input_tokens'] for r in group if r.get('sampling')),
                encoding_seconds=sum(r['sampling']['encoding_seconds'] for r in group if r.get('sampling')),
                routing_seconds=sum(r['sampling']['routing_seconds'] for r in group if r.get('sampling')))
            summary['arms'][f'{arm}|{d}']=info
        assert [r['question_id'] for r in groups['moc']]==[r['question_id'] for r in groups['adapted']]
        diff=[b['correct']-a['correct'] for a,b in zip(groups['moc'],groups['adapted'])]
        gains,losses=diff.count(1),diff.count(-1);discordant=gains+losses;rng=random.Random(42)
        bootstrap=sorted(sum(rng.choices(diff,k=n))/n for _ in range(20000))
        summary['paired'][str(d)]=dict(gains=gains,losses=losses,accuracy_difference=sum(diff)/n,bootstrap95=[bootstrap[499],bootstrap[19499]],
            bootstrap_replicates=20000,bootstrap_unit='question',seed=42,
            mcnemar_exact_two_sided=min(1,2*sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1,
            logical_token_change=summary['arms'][f'adapted|{d}']['logical_model_tokens']/summary['arms'][f'moc|{d}']['logical_model_tokens']-1)
    audit=dict(status='passed',dataset=title,members_verified=len(manifest),archive_sha256=receipt['archive_sha256'],
        source_data_policy_verified=True,local_gold_hash_verified=True,original_parser_sha256=sha(source),original_parser_rechecked=True,
        author_commit=lock['author_commit'],author_worktree_clean=True,archived_parser_matches_git_blob=True,
        local_parser_checkout_eol_only_difference=sha(source)!=sha(author/'datasets/data_process.py'),conditions=n*4,exact_coverage=True,duplicates=0,
        invalid=sum(r['status']!='ok' for r in results),budget=budget,known_raw_actual_and_calls_match=True,
        unknown_calls=len(unknown),unknown_reconciled_once=True,pending_trace_reservations=0,cleanup_verified=True,
        original_terminal_prefixes_preserved=True,original_run_start_preserved=True,no_independent_reproduction_claim=True)
    (out/'test_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (run/'completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=[f'# {title} 完成结果与审计（2026-10-16）','',
        f'完整作者test{n}题×两密度×adapted/原MOC，共{n*4}终态，invalid{audit["invalid"]}、重复0。生成完成并清理后才读取本地gold，原作者解析器及金标SHA复核。用户新授权取消实验时间截止，原run_start历史及终态前缀保留；累计预算不重置，未重训或按test调参。','',
        '| 密度 | 原MOC正确 | adapted正确 | 差（百分点） | 配对bootstrap95%（百分点） | McNemar p | main+merger逻辑token变化 |',
        '|---|---:|---:|---:|---|---:|---:|']
    for d in (.3,.7):
        pair=summary['paired'][str(d)];old=summary['arms'][f'moc|{d}'];new=summary['arms'][f'adapted|{d}'];lo,hi=pair['bootstrap95']
        lines.append(f'| {d} | {old["correct"]}/{n}（{old["accuracy"]:.2%}） | {new["correct"]}/{n}（{new["accuracy"]:.2%}） | {100*pair["accuracy_difference"]:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] | {pair["mcnemar_exact_two_sided"]:.6f} | {pair["logical_token_change"]:+.2%} |')
    lines+=['','题目配对bootstrap20000次、seed42；精确双侧McNemar。跨零区间不证明提升或非劣；本报告不以token节省证明学会多阶路由。','',
        '| 密度/方法 | 主逻辑token | 合并逻辑token | 总逻辑token（含cachehit） | 终态归属新增token | cachehit调用 |',
        '|---|---:|---:|---:|---:|---:|']
    for key,v in summary['arms'].items():lines.append(f'| {key.replace("|", " / ")} | {v["main_model_tokens"]} | {v["merger_model_tokens"]} | {v["logical_model_tokens"]} | {v["new_model_tokens"]} | {v["cache_hits"]} |')
    lines+=['','| 全过程阶段 | 已知actual token | 新调用（有返回） | 全部raw逻辑token | cachehit | 未归入终态新增字段的已知actual |',
        '|---|---:|---:|---:|---:|---:|']
    for stage in calls:
        v=summary['process_cost'][stage];lines.append(f'| {stage} | {v["actual_tokens"]} | {v["new_calls"]} | {v["logical_tokens_all_process_raw"]} | {v["cache_hits"]} | {v["interrupted_unattributed_actual_tokens"]} |')
    lines+=['',f'预算全批实际{budget["actual_tokens"]}token/{budget["new_calls"]}新调用；热身{sum(map(tokens,warm))}token/{len(warm)}次，独立unknown上界{budget["unknown_cost_upper_bound_tokens"]}token/{len(unknown)}次，pending0。unknown不是actual。',
        '全过程raw包含截止中断前已发生、后来由精确缓存续用的前缀开销；未归入终态行的新增费用不能漏计，也不能一概解释为不可复用的废弃。部署逻辑量按完整终态计算，含cache hit；过程raw逻辑量可能包含中断重放，二者不混用。训练、val、热身、中断及unknown费用全部列出；没有单价，不换算货币。','',
        '| 密度（adapted） | 空选择 | 单条 | 双条 | 选中hop1/2/3/4消息数 | 编码输入token/秒 | 路由秒 |',
        '|---|---:|---:|---:|---|---|---:|']
    for d in (.3,.7):
        v=summary['arms'][f'adapted|{d}'];sizes=v['selection_sizes'];hops=v['selected_hops']
        lines.append(f'| {d} | {sizes.get(0,0)} | {sizes.get(1,0)} | {sizes.get(2,0)} | '+ '/'.join(str(hops.get(i,0)) for i in (1,2,3,4))+f' | {v["encoding_input_tokens"]}/{v["encoding_seconds"]:.3f} | {v["routing_seconds"]:.3f} |')
    lines+=['','空选择包含根节点无候选；supported仅表示训练中出现过角色/hop/组大小元数据层，不保证语义分布外识别或全部候选组合有监督。全部support键、候选/选中hop与stop reason详见test_analysis.json。MiniLM编码与路由为本地累计CPU调用时间，不等于GPU端到端加速；训练编码/拟合CPU耗时未独立完整记录，不当作零。',
        'FinalRefer仍读所有最终输出，普通agent数量不减少。相对原MOC的token节省包含替代9B合并调用的影响，不能单独归因于学到高阶证据互补。固定材料、Vanilla下游干预目标是代理，不提供全图最优或因果效用保证。']
    training=summary['training']
    lines += [f'有效训练{training["effective_questions"]}/{training["planned_questions"]}题，排除{len(training["excluded"])}题；{training["labels"]}个标签、{training["metadata"]["fit_episode_count"]}个episode。policy SHA256：`{summary["policy_sha256"]}`。']
    if title=='SVAMP':lines+=['SVAMP使用作者gsm8k prompt域和原SVAMP解析器。40训练是官方full1000排除作者test/近重复/数字模板后的研究划分，非官方train；题族语义隔离仍有限。无val，不能称独立验证或首次未见测试。', '本次SVAMP两种密度下的全部2100次普通节点决策均为空选择，其中各300次为根节点无候选；有候选时也未选择任何消息。两密度适配组输出及逻辑token一致，稠密组复用精确兼容缓存。这一结果体现全弃选行为，不能作为学会内容区分、消息互补或多阶路由的证据。']
    elif title=='MMLU-Pro':lines+=['14train/56val来自官方validation70按类别固定1/4分配；作者test280只作test，不能把先前开发/测试接触抹去。']
    else:lines+=['24train/120val来自官方train7473的固定seed42划分；作者test300只作test，不能称首次未见测试。']
    lines+=['',f'全归档{len(manifest)}成员逐文件SHA/字节数核验通过，包含run全部raw/trace/results/预算/服务/配置/锁、此次运行管理代码与授权、原作者Python源码。archive SHA256：`{receipt["archive_sha256"]}`。',
        'completion_audit passed；原始已知用量、热身、unknown与账本/trace逐项对齐；原冻结源码/数据/policy及训练提交匹配，自有进程/子进程/服务端口清理核验。原作者仓库commit及工作树干净复核。属于本次输出一致性审计，不宣称独立复现。','']
    (out/f'{title}完成结果与审计_20261016.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(dataset=title,paired=summary['paired'],arms={k:{t:v[t] for t in ('correct','accuracy','logical_model_tokens')} for k,v in summary['arms'].items()},process_cost=summary['process_cost'],audit='passed'),ensure_ascii=True))
if __name__=='__main__':main()
