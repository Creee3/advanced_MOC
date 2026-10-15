"""Frozen official-test analysis after full completion. No backend calls."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'remote_snapshot'
GOLD=ROOT.parents[2]/'experiments/moc_full_graph_compression_20261013/data/test_evaluation_only.jsonl'

def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def right(r,gold):return int(r['status']=='ok' and r['answer']==gold[r['question_id']])

def paired(reference,candidate,gold,ids):
    ids=sorted(ids)
    a=np.asarray([right(reference[q],gold) for q in ids]);b=np.asarray([right(candidate[q],gold) for q in ids])
    delta=b-a
    ref=np.asarray([reference[q]['logical_model_tokens'] for q in ids],dtype=float)
    cand=np.asarray([candidate[q]['logical_model_tokens'] for q in ids],dtype=float)
    upper=np.asarray([reference[q].get('unknown_model_tokens_upper_bound',0) for q in ids],dtype=float)
    improved=int((delta>0).sum());degraded=int((delta<0).sum());n=improved+degraded
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(improved,degraded)+1))/2**n) if n else 1.
    rng=np.random.default_rng(42);draw=rng.integers(0,len(ids),size=(20000,len(ids)))
    boot=delta[draw].mean(axis=1)
    cost_boot=cand[draw].sum(axis=1)/ref[draw].sum(axis=1)
    return dict(question_pairs=len(ids),reference_correct=int(a.sum()),candidate_correct=int(b.sum()),
        reference_accuracy=float(a.mean()),candidate_accuracy=float(b.mean()),accuracy_difference=float(delta.mean()),
        improved=improved,degraded=degraded,tied=int((delta==0).sum()),two_sided_sign_p=p,
        paired_accuracy_difference_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        reference_known_tokens=int(ref.sum()),candidate_known_tokens=int(cand.sum()),
        reference_unknown_token_upper_bound=int(upper.sum()),
        candidate_to_reference_known_token_ratio=float(cand.sum()/ref.sum()),
        token_ratio_with_reference_unknown_usage_bounds=[float(cand.sum()/(ref+upper).sum()),float(cand.sum()/ref.sum())],
        paired_token_ratio_bootstrap95=np.quantile(cost_boot,[.025,.975]).tolist(),
        observed_operational_target_met=bool(b.sum()>=a.sum() and cand.sum()<ref.sum()),
        interval_scope='descriptive paired question bootstrap, seed42/20000; historically exposed test; no noninferiority guarantee')

def routing(items):
    ledger=[r['sampling'] for r in items if r['status']=='ok']
    selections=[s for l in ledger for s in l['selections']]
    nonroot=[s for s in selections if s['candidates']]
    sizes=Counter(str(len(s['source_ids'])) for s in nonroot)
    selected=Counter(str(m['hop']) for s in nonroot for m in s['candidates'] if m['source'] in s['source_ids'])
    available=Counter(str(m['hop']) for s in nonroot for m in s['candidates'])
    return dict(valid_graphs=len(ledger),ordinary_receivers=len(selections),nonroot_receivers=len(nonroot),
        selected_set_sizes=dict(sizes),read_messages_by_hop=dict(selected),available_messages_by_hop=dict(available),
        read_messages=sum(selected.values()),available_messages=sum(available.values()),
        stop_reasons=dict(Counter(s['stop_reason'] for s in nonroot)),
        communication_text_tokens=sum(s['communication_tokens'] for s in selections),
        encoding_seconds=sum(l['encoding_seconds'] for l in ledger),routing_seconds=sum(l['routing_seconds'] for l in ledger),
        sketch_bytes_generated=sum(l['sketch_bytes_generated'] for l in ledger),
        routing_sketch_bytes_read=sum(l['routing_sketch_bytes_read'] for l in ledger))

def main():
    audit=read(ROOT/'completion_audit.json')
    assert audit['status']=='record_and_cost_consistency_passed'
    assert sha(GOLD)==read(SNAP/'data_preparation.json')['official_source_manifest']['files']['test_evaluation_only.jsonl']
    gold={r['question_id']:r['correct'] for r in rows(GOLD)}
    records=[r for f in (SNAP/'test').glob('shard*/results.jsonl') for r in rows(f)]
    assert len(records)==1016 and len(gold)==254
    assert set(gold)=={r['question_id'] for r in records}
    invalid=[{k:v for k,v in r.items() if k not in ('final_output','sampling')} for r in records if r['status']=='invalid']
    excluded_all={r['question_id'] for r in invalid}
    metrics={};primary={};common={};global_common={};routes={}
    for density in (.3,.7):
        groups={arm:{r['question_id']:r for r in records if r['arm']==arm and r['density']==density} for arm in ('moc','v21_expected_set')}
        for arm,items in groups.items():
            assert set(items)==set(gold)
            metrics[f'{arm}@{density}']=dict(questions=254,valid_conditions=sum(r['status']=='ok' for r in items.values()),
                invalid_conditions=sum(r['status']=='invalid' for r in items.values()),
                correct=sum(right(r,gold) for r in items.values()),accuracy=sum(right(r,gold) for r in items.values())/254,
                known_inference_tokens=sum(r['logical_model_tokens'] for r in items.values()),
                main_model_tokens=sum(r['main_model_tokens'] for r in items.values()),
                merger_model_tokens=sum(r['merger_model_tokens'] for r in items.values()),
                unknown_inference_token_upper_bound=sum(r.get('unknown_model_tokens_upper_bound',0) for r in items.values()),
                logical_known_calls=sum(r['logical_calls'] for r in items.values()),
                main_calls=sum(r['main_calls'] for r in items.values()),merger_known_calls=sum(r['merger_calls'] for r in items.values()),
                inference_failure_semantics='invalid conditions scored as failed system attempts, not semantic model errors; partial-call costs retained')
        reference,candidate=groups['moc'],groups['v21_expected_set']
        primary[str(density)]=paired(reference,candidate,gold,set(gold))
        bad={q for q in gold if reference[q]['status']!='ok' or candidate[q]['status']!='ok'}
        common[str(density)]=dict(excluded_question_ids=sorted(bad),comparison=paired(reference,candidate,gold,set(gold)-bad))
        global_common[str(density)]=dict(excluded_question_ids=sorted(excluded_all),comparison=paired(reference,candidate,gold,set(gold)-excluded_all))
        routes[str(density)]=routing(list(candidate.values()))
    by_key={(r['question_id'],r['arm'],r['density']):r for r in records}
    qids=sorted(gold)
    # Both densities share a question; bootstrap the254 clusters, never508
    # independent samples. Each question contributes the mean of its two deltas.
    deltas=np.asarray([sum(right(by_key[q,'v21_expected_set',d],gold)-right(by_key[q,'moc',d],gold) for d in (.3,.7))/2 for q in qids])
    rng=np.random.default_rng(42);draw=rng.integers(0,254,size=(20000,254))
    aggregate=dict(question_clusters=254,density_conditions_per_arm=508,
        accuracy_difference=float(deltas.mean()),paired_cluster_accuracy_difference_bootstrap95=np.quantile(deltas[draw].mean(axis=1),[.025,.975]).tolist())
    v2=read(ROOT.parent/'moc_v2_evidence_bundles_train_20261014/analysis.json')
    training=v2['new_cost']['training_tokens_including_exclusions']
    assert training==888561
    cand=sum(metrics[f'v21_expected_set@{d}']['known_inference_tokens'] for d in (.3,.7))
    ref=sum(metrics[f'moc@{d}']['known_inference_tokens'] for d in (.3,.7))
    budget=read(SNAP/'budget.json')
    cost=dict(candidate_known_inference_tokens=cand,moc_known_inference_tokens=ref,
        candidate_to_moc_known_inference_ratio=cand/ref,shared_training_tokens_once=training,
        candidate_training_plus_this_test_tokens=training+cand,
        training_plus_candidate_to_moc_ratio=(training+cand)/ref,
        earlier_v1_training_and_waste_separate=657528,new_v21_training_tokens=0,
        actual_new_test_tokens_known=budget['actual_tokens'],actual_new_test_unknown_usage_upper_bound=budget['unknown_cost_upper_bound_tokens'],
        actual_new_test_usage_bounds=[budget['actual_tokens'],budget['actual_tokens']+budget['unknown_cost_upper_bound_tokens']],
        actual_new_calls_including_unknown=budget['new_calls'],unknown_calls=budget['unknown_call_count'],
        warmup_tokens=audit['startup_tokens'],known_discarded_partial_execution_tokens=audit['known_discarded_partial_execution_tokens'],
        expense_scope='both experimental arms, warmups and discarded work; not the deployment cost of v2.1 alone',
        logical_scope='full prompt+completion for every ordinary/final/merger call in the terminal execution, cache hits charged fully; missing usage shown separately',
        training_scope='shared v2 training counted once, not re-paid for v2.1; earlier val/development expenses not hidden inside inference savings')
    report=dict(questions=254,terminal_conditions=1016,valid_conditions=1007,invalid_conditions=9,
        metrics=metrics,primary_full254=primary,common_valid_pairs_by_density=common,
        global_common_valid_question_sensitivity=global_common,aggregate_cluster_analysis=aggregate,
        routing=routes,cost=cost,invalid_conditions_detail=invalid,
        historical_test_exposure=True,policy_tuned_on_this_test=False,
        interpretation='frozen v2.1 versus original MOC; official test historically exposed to older methods; observed comparison, not untouched independent confirmation',
        causal_limit='two-message cap and removal of generative consolidation also contribute; results do not isolate learned selector quality or temporal communication')
    (ROOT/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('primary_full254','common_valid_pairs_by_density','global_common_valid_question_sensitivity','aggregate_cluster_analysis','cost','routing')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
