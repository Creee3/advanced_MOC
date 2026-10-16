"""Read existing completed records only; no model calls, fitting or new scoring execution."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone, timedelta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / 'experiments'
OUT = REPO.parent / 'outputs/01a11c24-c60c-7b42-99e6-70e3c00dea5e/experiment_summary_20261015_16'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}

def source(p):
    p = Path(p).resolve()
    key = str(p.relative_to(REPO.parent)).replace('\\', '/')
    if key not in SOURCES:
        SOURCES[key] = dict(id=f'S{len(SOURCES)+1:03}', path=key, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return SOURCES[key]['id']

def read(p):
    source(p)
    return json.loads(Path(p).read_text(encoding='utf-8'))

def rows(p):
    source(p)
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def write_csv(name, data):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)

spec = importlib.util.spec_from_file_location('completed_audit', BASE/'moc_remaining_datasets_plan_20261015/resume_no_deadline_20261016/analyze_completed_resume.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
PARSER = REPO/'coding_base/MOC-official/datasets/data_process.py'
source(PARSER)

datasets = [
    ('AQuA','moc_v21_official_test_20261014',254,'AQuA冻结checkpoint','历史test接触；9个invalid计0；逻辑token为已知用量'),
    ('HumanEval','moc_v21_humaneval_test_20261015',164,'AQuA checkpoint零样本迁移','作者MultiPL-E版；稠密非根节点全弃选；使用已有代码测试评分'),
    ('GSM8K','moc_v21_gsm8k_fewshot_20261015',300,'本任务24题训练','官方train固定24/120；test不调参'),
    ('MMLU-Pro','moc_v21_mmlu_pro_fewshot_20261015',280,'本任务14题训练','官方validation70按14/56划分；统计未做多重比较校正'),
    ('MMLU','moc_v21_mmlu_train40_test_20261015',285,'本任务40题训练','dev固定40科各1题；无val；不覆盖57科'),
    ('SVAMP','moc_v21_svamp_train40_test_20261015',300,'本任务40题训练','研究划分40题；无val；两密度适配全弃选'),
]
comparisons=[]; arms=[]; routing=[]; costs=[]; question_rows=[]; val_arms=[]; val_pairs=[]

def add_questions(dataset, run_name, split, root, gold=None, score=None):
    result=[]
    for p in sorted((root/split).glob('shard*/results.jsonl')):
        sid=source(p)
        for r in rows(p):
            key=(r['question_id'],r['arm'],r['density'])
            correct=int(score[key]['passed']) if score is not None else int(r['status']=='ok' and r['answer']==gold[r['question_id']])
            q=dict(dataset=dataset,batch=run_name,split=split,question_id=r['question_id'],arm=r['arm'],density=r['density'],
                   generation_status=r['status'],correct_or_passed=correct,logical_model_tokens=r['logical_model_tokens'],
                   main_model_tokens=r['main_model_tokens'],merger_model_tokens=r['merger_model_tokens'],
                   new_model_tokens=r['new_model_tokens'],logical_calls=r['logical_calls'],new_calls=r['new_calls'],
                   cache_hits=r['cache_hits'],source_id=sid)
            assert q['logical_model_tokens']==q['main_model_tokens']+q['merger_model_tokens']
            question_rows.append(q);result.append(q)
    assert len({(x['question_id'],x['arm'],x['density']) for x in result})==len(result)
    return result

for title,name,n,train,limitation in datasets:
    run=BASE/name
    legacy=title in ('AQuA','HumanEval')
    analysis_path=run/('analysis.json' if legacy else 'test_review_20261016/test_analysis.json')
    a=read(analysis_path);audit=read(run/'completion_audit.json')
    assert audit['status'] in ('passed','record_and_cost_consistency_passed')
    if legacy:
        receipt=read(run/'analysis_receipt_20261015.json')
        for key,h in receipt.get('analysis_sources_and_outputs',receipt.get('files')).items():
            if (run/key).exists():assert hashlib.sha256((run/key).read_bytes()).hexdigest()==h,key
        root=run/'remote_snapshot'
        candidate='v21_expected_set'
        if title=='AQuA':
            gp=REPO.parent/'experiments/moc_full_graph_compression_20261013/data/test_evaluation_only.jsonl'
            assert hashlib.sha256(gp.read_bytes()).hexdigest()==read(root/'data_preparation.json')['official_source_manifest']['files']['test_evaluation_only.jsonl']
            gold={r['question_id']:r['correct'] for r in rows(gp)};score=None
        else:
            gold=None;score={(r['question_id'],r['arm'],r['density']):r for r in rows(root/'final_scores.jsonl')}
        rr=add_questions(title,name,'test',root,gold,score)
        actual=audit['known_actual_new_tokens'];calls=audit.get('new_calls',audit.get('new_calls_including_unknown'))
        budget=dict(actual_tokens=actual,new_calls=calls,reserved_tokens=0,unknown_cost_upper_bound_tokens=audit['unknown_cost_upper_bound_tokens'])
        train_n=0;labels=None
    else:
        records=run/'test_review_20261016/records'
        candidates=[p.parent for p in records.rglob('experiment_lock.json') if (p.parent/'test').is_dir() and 'attempts' not in p.parts]
        assert len(candidates)==1,(title,candidates)
        root=candidates[0];candidate='adapted'
        lock=read(root/'experiment_lock.json')
        assert hashlib.sha256((run/'data/test_gold_LOCAL_ONLY.jsonl').read_bytes()).hexdigest()==lock['local_gold_sha256']['test']
        cls,parser=module.native_parser(PARSER,{'GSM8K':'GSM8KDataset','MMLU-Pro':'MMLUProDataset','MMLU':'MMLUDataset','SVAMP':'SVAMPDataset'}[title]) if title!='MMLU' else (None,None)
        if title=='MMLU':
            # MMLU target is a single answer letter in the already audited local gold.
            gold={r['question_id']:r['answer'] for r in rows(run/'data/test_gold_LOCAL_ONLY.jsonl')}
        else:
            field='Answer' if title=='SVAMP' else 'answer'
            gold={r['question_id']:cls.record_to_target_answer({field:r['answer']}) for r in rows(run/'data/test_gold_LOCAL_ONLY.jsonl')}
        rr=add_questions(title,name,'test',root,gold)
        budget=audit['budget'];train_n=a['training'].get('effective_questions',a['training'].get('n_questions',a['training'].get('planned_questions')));labels=a['training']['labels']
    assert len(rr)==4*n
    for d in (.3,.7):
        if title=='AQuA':
            v=a['primary_full254'][str(d)];ci=v['paired_accuracy_difference_bootstrap95'];p=v['two_sided_sign_p'];g=v['improved'];l=v['degraded']
        elif title=='HumanEval':
            v=a['comparisons'][str(d)];ci=v['accuracy_difference_bootstrap95'];p=v['exact_mcnemar_two_sided_p'];g=v['improved'];l=v['degraded']
        else:
            v=a['paired'][str(d)];ci=v['bootstrap95'];p=v['mcnemar_exact_two_sided'];g=v['gains'];l=v['losses']
        groups={arm:[r for r in rr if r['arm']==arm and r['density']==d] for arm in ('moc',candidate)}
        info={}
        for arm,group in groups.items():
            assert len(group)==n
            key=f'{arm}@{d}' if legacy else f'{arm}|{d}'
            src=a['metrics'][key] if legacy else a['arms'][key]
            expected_correct=src.get('correct',src.get('passed'))
            correct=sum(r['correct_or_passed'] for r in group);tokens=sum(r['logical_model_tokens'] for r in group)
            assert correct==expected_correct
            assert tokens==src.get('logical_model_tokens',src.get('known_inference_tokens',src.get('full_inference_tokens')))
            info[arm]=dict(correct=correct,tokens=tokens,invalid=sum(r['generation_status']!='ok' for r in group))
            arms.append(dict(dataset=title,training=train,density=d,arm=arm,questions=n,correct=correct,invalid=info[arm]['invalid'],
                             logical_tokens=tokens,main_tokens=sum(r['main_model_tokens'] for r in group),merger_tokens=sum(r['merger_model_tokens'] for r in group),
                             logical_calls=sum(r['logical_calls'] for r in group),new_tokens=sum(r['new_model_tokens'] for r in group),cache_hits=sum(r['cache_hits'] for r in group),source_id=source(analysis_path)))
        before={r['question_id']:r['correct_or_passed'] for r in groups['moc']};after={r['question_id']:r['correct_or_passed'] for r in groups[candidate]}
        assert sum(after[q]>before[q] for q in before)==g and sum(after[q]<before[q] for q in before)==l
        z=g+l;exact=min(1.,2*sum(math.comb(z,k) for k in range(min(g,l)+1))/2**z) if z else 1.
        assert abs(exact-p)<1e-12
        comparisons.append(dict(dataset=title,training=train,density=d,questions=n,moc_correct=info['moc']['correct'],candidate_correct=info[candidate]['correct'],
            ci_low=ci[0],ci_high=ci[1],mcnemar_p=p,gains=g,losses=l,moc_tokens=info['moc']['tokens'],candidate_tokens=info[candidate]['tokens'],
            moc_invalid=info['moc']['invalid'],candidate_invalid=info[candidate]['invalid'],limitations=limitation,source_id=source(analysis_path)))
        # Use one denominator across datasets: decisions with at least one candidate, valid graphs only.
        raw=[r for pth in (root/'test').glob('shard*/results.jsonl') for r in rows(pth) if r['arm']==candidate and r['density']==d and r['status']=='ok']
        ss=[s for r in raw for s in r['sampling']['selections'] if s['candidates']]
        size=Counter(len(s['source_ids']) for s in ss);reason=Counter(s['stop_reason'] for s in ss)
        hs=Counter(m['hop'] for s in ss for m in s['candidates'] if m['source'] in s['source_ids'])
        routing.append(dict(dataset=title,density=d,valid_graphs=len(raw),nonroot_decisions=len(ss),empty=size[0],single=size[1],pair=size[2],
            unsupported=reason['no_supported_expected_action'],nonpositive=reason['expected_net_utility_nonpositive'],hop1=hs[1],hop2=hs[2],hop3=hs[3],hop4=hs[4],
            encoding_seconds=sum(r['sampling']['encoding_seconds'] for r in raw),routing_seconds=sum(r['sampling']['routing_seconds'] for r in raw),source_id=source(analysis_path)))
    costs.append(dict(dataset=title,batch=name,training_questions=train_n,labels=labels,conditions=4*n,valid=sum(r['generation_status']=='ok' for r in rr),invalid=sum(r['generation_status']!='ok' for r in rr),
        actual_tokens=budget['actual_tokens'],unknown_upper=budget['unknown_cost_upper_bound_tokens'],pending=budget['reserved_tokens'],new_calls=budget['new_calls'],
        archive_sha256=audit['archive_sha256'],audit=audit['status'],source_id=source(run/'completion_audit.json'),note=limitation))

run=BASE/'moc_mmlu_pro_val_baseline_20261015';a=read(run/'completion_audit.json');bud=a['budget']
costs.append(dict(dataset='Pro独立val',batch=run.name,training_questions=0,labels=None,conditions=112,valid=112,invalid=0,actual_tokens=bud['actual_tokens'],unknown_upper=bud['unknown_cost_upper_bound_tokens'],pending=bud['reserved_tokens'],new_calls=bud['new_calls'],archive_sha256='1c6971a3353fbe3bb51d2fe38cfe8d3bb2cc94f176b109590e2344f88e94242f',audit=a['status'],source_id=source(run/'completion_audit.json'),note='仅补原MOC val，原适配/迁移组不重复收费'))

for title,name,file in [('GSM8K',datasets[2][1],'val_review_20261015/val_analysis.json'),('MMLU-Pro',datasets[3][1],'val_review_20261015/three_arm_val_analysis.json')]:
    run=BASE/name;path=run/file;a=read(path);goldrows=rows(run/'data/val_gold_LOCAL_ONLY.jsonl')
    cls,parser=module.native_parser(PARSER,'GSM8KDataset' if title=='GSM8K' else 'MMLUProDataset')
    gold={r['question_id']:cls.record_to_target_answer(r) for r in goldrows}
    root=run/'test_review_20261016/records/run';rr=add_questions(title,name,'val',root,gold)
    if title=='MMLU-Pro':rr+=add_questions(title,'moc_mmlu_pro_val_baseline_20261015','val',BASE/'moc_mmlu_pro_val_baseline_20261015/remote_snapshot/records',gold)
    for key,v in a['arms'].items():
        arm,d=key.split('|');group=[r for r in rr if r['arm']==arm and r['density']==float(d)]
        observed=(len(group),sum(r['correct_or_passed'] for r in group),sum(r['logical_model_tokens'] for r in group))
        expected=(v['n'],v['correct'],v['logical_model_tokens'])
        assert observed==expected,(title,key,observed,expected)
        val_arms.append(dict(dataset=title,density=float(d),arm=arm,questions=v['n'],correct=v['correct'],invalid=v['invalid'],logical_tokens=v['logical_model_tokens'],source_id=source(path)))
    pairs=a.get('paired_adapted_vs_transfer',a.get('paired_vs_moc'))
    for key,v in pairs.items():
        arm,d=key.split('|') if '|' in key else ('adapted',key)
        ci=v.get('bootstrap95',v.get('bootstrap95_percentile'))
        val_pairs.append(dict(dataset=title,density=float(d),reference='frozen_transfer' if title=='GSM8K' else 'moc',candidate=arm,delta=v['accuracy_difference'],ci_low=ci[0],ci_high=ci[1],mcnemar_p=v['mcnemar_exact_two_sided'],token_change=v['logical_token_change'],source_id=source(path)))

assert len(comparisons)==12 and len(costs)==7 and len(question_rows)==7148
assert sum(c['invalid'] for c in costs)==9
totals={k:sum(c[k] for c in costs) for k in ['actual_tokens','unknown_upper','pending','new_calls','conditions','valid','invalid']}
assert totals['actual_tokens']==39181781 and totals['unknown_upper']==69536 and totals['pending']==0
assert totals['new_calls']==60218 and totals['conditions']==6444
history=[
 dict(experiment='v1 stateful val',date='2026-10-14',status='历史诊断，不计两天主结果',finding='稀疏19/60对原MOC24/60；稠密28/60持平，全停行为需披露',source_id=source(BASE/'moc_stateful_sampling_val_20261014/analysis.json')),
 dict(experiment='v2 bundles train/val',date='2026-10-14',status='历史诊断，不计两天主结果',finding='已有结果不代表v2.1最终test；冻结原件保留',source_id=source(BASE/'moc_v2_evidence_bundles_train_20261014/analysis.json')),
 dict(experiment='v2.1 offline OOF',date='2026-10-14',status='历史CPU诊断，无新模型调用',finding='compact MSE0.350914，劣于zero0.304762；不能主张离线可学习性已证',source_id=source(BASE/'moc_v21_expected_set_offline_20261014/oof_report.json')),
 dict(experiment='v2.1 AQuA val60',date='2026-10-14',status='历史诊断，不混入官方test',finding='稀疏29/60、稠密35/60；开发val不能并入test',source_id=source(BASE/'moc_v21_expected_set_val_20261014/analysis.json')),
 dict(experiment='代码域train48准备',date='2026-10-15',status='搁置，未选数据/未训练/无模型调用',finding='HumanEval无本轮新增48题训练结果；MBPP替代准备没有执行',source_id=source(BASE/'moc_v21_code_adaptation_train48_20261015/progress.md')),
]
data=dict(period='2026-10-15 to 2026-10-16 Asia/Shanghai',generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),test_comparisons=comparisons,test_arms=arms,val_arms=val_arms,val_pairs=val_pairs,routing=routing,batches=costs,totals=totals,history=history,
    shared_prior_training_tokens=888561,shared_prior_training_scope='历史共享训练，只列一次参考，不计本轮七批actual总账；含训练摊销是假设场景',earlier_v1_training_waste_tokens=657528,
    checks=dict(primary_test_rows=6332,val_rows=816,all_rows=7148,correct_counts_match=True,logical_tokens_match=True,gain_loss_and_exact_mcnemar_match=True,source_receipts_checked=True,no_model_calls=True),sources=list(SOURCES.values()))
(HERE/'summary_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'summary_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for file,table in [('test_comparisons.csv',comparisons),('val_arms.csv',val_arms),('cost_ledger.csv',costs),('routing.csv',routing),('question_results.csv',question_rows)]:write_csv(file,table)
(HERE/'source_manifest.json').write_text(json.dumps(list(SOURCES.values()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(comparisons=len(comparisons),batches=len(costs),question_rows=len(question_rows),totals=totals,output=str(OUT)),ensure_ascii=True))
