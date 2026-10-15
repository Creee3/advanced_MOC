"""Predeclared paired HumanEval analysis; no model calls or test-based fitting."""
from collections import Counter
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];SNAP=ROOT/'remote_snapshot'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def key(r):return r['question_id'],r['arm'],r['density']


def paired(reference,candidate,score,ids):
    a=np.asarray([int(score[key(reference[q])]['passed']) for q in ids])
    b=np.asarray([int(score[key(candidate[q])]['passed']) for q in ids])
    x=np.asarray([reference[q]['logical_model_tokens'] for q in ids],dtype=float)
    y=np.asarray([candidate[q]['logical_model_tokens'] for q in ids],dtype=float)
    delta=b-a;gain=int((delta>0).sum());loss=int((delta<0).sum());discordant=gain+loss
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(gain,loss)+1))/2**discordant) if discordant else 1.
    rng=np.random.default_rng(42);draw=rng.integers(0,len(ids),size=(20000,len(ids)))
    ratio=y[draw].sum(axis=1)/x[draw].sum(axis=1)
    return dict(question_pairs=len(ids),moc_passed=int(a.sum()),v21_passed=int(b.sum()),
        moc_pass_at_1=float(a.mean()),v21_pass_at_1=float(b.mean()),accuracy_difference=float(delta.mean()),
        accuracy_difference_bootstrap95=np.quantile(delta[draw].mean(axis=1),[.025,.975]).tolist(),
        improved=gain,degraded=loss,tied=int((delta==0).sum()),exact_mcnemar_two_sided_p=p,
        moc_full_inference_tokens=int(x.sum()),v21_full_inference_tokens=int(y.sum()),
        token_change_fraction=float(y.sum()/x.sum()-1),token_ratio_bootstrap95=np.quantile(ratio,[.025,.975]).tolist(),
        observed_target_met=bool(b.sum()>=a.sum() and y.sum()<x.sum()),
        inference_scope='all ordinary/final/merger prompt+completion tokens, including exact-cache logical reuse')


def routing(records):
    ledgers=[r['sampling'] for r in records]
    selections=[s for ledger in ledgers for s in ledger['selections']]
    ordinary=[s for s in selections if s['candidates']]
    sizes=Counter(str(len(s['source_ids'])) for s in ordinary)
    available=Counter(str(m['hop']) for s in ordinary for m in s['candidates'])
    chosen=Counter(str(m['hop']) for s in ordinary for m in s['candidates'] if m['source'] in s['source_ids'])
    return dict(ordinary_receivers=len(selections),nonroot_receivers=len(ordinary),selected_set_sizes=dict(sizes),
        stop_reasons=dict(Counter(s['stop_reason'] for s in ordinary)),read_messages_by_hop=dict(chosen),
        available_messages_by_hop=dict(available),read_messages=sum(chosen.values()),available_messages=sum(available.values()),
        communication_text_tokens=sum(s['communication_tokens'] for s in selections),
        encoding_seconds=sum(l['encoding_seconds'] for l in ledgers),routing_seconds=sum(l['routing_seconds'] for l in ledgers),
        encoding_input_tokens=sum(l['encoding_input_tokens'] for l in ledgers),
        encoding_calls={name:sum(l['encoding_calls'][name] for l in ledgers) for name in ('sender','receiver')},
        sketch_bytes_generated=sum(l['sketch_bytes_generated'] for l in ledgers),
        routing_sketch_bytes_read=sum(l['routing_sketch_bytes_read'] for l in ledgers),
        timing_scope='observed cumulative component seconds; not a controlled end-to-end latency comparison')


def main():
    audit=read(ROOT/'completion_audit.json');assert audit['status']=='record_and_cost_consistency_passed'
    records=[r for p in sorted((SNAP/'test').glob('shard*/results.jsonl')) for r in rows(p)]
    scores={key(r):r for r in rows(SNAP/'final_scores.jsonl')}
    ids=sorted(r['question_id'] for r in rows(SNAP/'data/test_inputs.jsonl'))
    comparisons={};metrics={};routes={}
    for d in (.3,.7):
        by_arm={a:{r['question_id']:r for r in records if r['arm']==a and r['density']==d} for a in ('moc','v21_expected_set')}
        comparisons[str(d)]=paired(by_arm['moc'],by_arm['v21_expected_set'],scores,ids)
        routes[str(d)]=routing(list(by_arm['v21_expected_set'].values()))
        for a,items in by_arm.items():
            values=list(items.values())
            metrics[f'{a}@{d}']=dict(questions=len(values),passed=sum(scores[key(r)]['passed'] for r in values),
                full_inference_tokens=sum(r['logical_model_tokens'] for r in values),
                main_model_tokens=sum(r['main_model_tokens'] for r in values),merger_model_tokens=sum(r['merger_model_tokens'] for r in values),
                logical_calls=sum(r['logical_calls'] for r in values),main_calls=sum(r['main_calls'] for r in values),
                merger_calls=sum(r['merger_calls'] for r in values),early_returns=sum(r['early_return_count'] for r in values),
                final_test_failure_types=dict(Counter(scores[key(r)].get('error_type') for r in values if not scores[key(r)]['passed'])))
    reference=sum(metrics[f'moc@{d}']['full_inference_tokens'] for d in (.3,.7))
    candidate=sum(metrics[f'v21_expected_set@{d}']['full_inference_tokens'] for d in (.3,.7))
    prior=read(ROOT.parent/'moc_v2_evidence_bundles_train_20261014/analysis.json')['new_cost']['training_tokens_including_exclusions']
    assert prior==888561
    result=dict(dataset='MOC-supplied MultiPL-E transformed Python HumanEval164',
        evaluation='one frozen generated completion per task/method/density; author postprocessing and hidden tests',
        checkpoint_training='AQuA train35, 176 initial-state labels; no HumanEval fitting or code-role mapping',
        comparisons=comparisons,metrics=metrics,routing=routes,
        common_valid_sensitivity=dict(excluded_question_ids=[],identical_to_primary=True),
        costs=dict(moc_inference_tokens_both_densities=reference,v21_inference_tokens_both_densities=candidate,
            inference_token_change_fraction=candidate/reference-1,reused_training_tokens_charged_once=prior,
            v21_with_reused_training_charged_once=candidate+prior,
            token_change_with_training_charged_once=(candidate+prior)/reference-1,
            current_experiment_actual_new_tokens=audit['known_actual_new_tokens'],
            current_experiment_new_calls=audit['new_calls'],warmup_tokens=audit['startup_tokens'],
            unknown_cost_upper_bound_tokens=0,discarded_new_calls=0),
        audit_status=audit['status'],statistical_scope='paired question bootstrap seed42, 20000 replicates per density; no noninferiority claim',
        limitations=['zero-training cross-domain transfer, most code roles unsupported',
            'single decoding seed and two fixed graph topologies',
            'hard two-message cap and removal of Phi confound attribution to learned grouping',
            'final all-output aggregation remains; empty ordinary-node communication does not imply globally no communication',
            'observed routing/encoding time is not GPU or end-to-end speedup',
            'not byte-identical to original OpenAI HumanEval prompts; no paper-number equivalence claim'])
    (ROOT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'analysis/per_condition_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['question_id','arm','density','passed','logical_model_tokens','main_model_tokens','merger_model_tokens','main_calls','merger_calls','early_return_count']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:writer.writerow({f:(int(scores[key(r)]['passed']) if f=='passed' else r[f]) for f in fields})
    print(json.dumps(dict(comparisons=comparisons,metrics=metrics,routing=routes,costs=result['costs']),indent=2))


if __name__=='__main__':main()
