"""Recompute labels, validate frozen artifacts, all calls and the paid ledger."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'remote_snapshot'
def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    frozen=json.loads((SNAP/'experiment_lock.json').read_text())
    policy=json.loads((SNAP/'policy_lock.json').read_text())
    for name,digest in {**frozen['source_hashes'],**frozen['data_hashes']}.items():
        assert sha(SNAP/name)==digest,name
    assert sha(SNAP/'experiment_lock.json')==policy['experiment_lock_sha256']
    assert sha(SNAP/'group_utility_model.json')==policy['model_sha256']
    assert policy['frozen_before_val'] and not frozen['val_gold_included'] and not frozen['test_included']
    manifest=json.loads((SNAP/'final_manifest.json').read_text())
    events=[e for p in (SNAP/'val').glob('shard*/trace.jsonl') for e in rows(p)]
    first_val=min(e['time_unix'] for e in events)
    assert manifest['group_utility_model.json']['mtime_ns']/1e9 < first_val
    assert manifest['policy_lock.json']['mtime_ns']/1e9 < first_val
    config=json.loads((SNAP/'config.json').read_text())
    raw=[r for phase in ('training','val') for p in (SNAP/phase).glob('shard*/calls_raw.jsonl') for r in rows(p)]
    count=Counter(); limits=[]
    for r in raw:
        assert r['request']['model']==config['main_model']
        expected={'seed':42,'temperature':0,'num_predict':2000,'num_ctx':8192,'num_gpu':999}
        assert r['request']['options']==expected
        response=r['response']
        assert response['done'] and response['eval_count']>0
        if response.get('done_reason')=='length':
            limits.append(dict(stage=r['stage'],key=r['key'],question_id=r['question_id']))
        if not r['cache_hit']:
            count['new_calls']+=1
            count['new_tokens']+=response['prompt_eval_count']+response['eval_count']
    status=json.loads((SNAP/'status.json').read_text())
    warm=sum(s['warmup'].get('prompt_eval_count',0)+s['warmup'].get('eval_count',0) for s in status['services'])
    budget=json.loads((SNAP/'budget.json').read_text())
    assert budget['reserved_tokens']==0
    assert budget['actual_tokens']==count['new_tokens']+warm
    assert budget['new_calls']==count['new_calls']+len(status['services'])
    assert budget['actual_tokens']<=config['max_new_model_tokens'] and budget['new_calls']<=config['max_new_calls']
    gold={x['question_id']:x['correct'] for x in rows(SNAP/'data/train_gold_LOCAL_ONLY.jsonl')}
    committed=[q for p in (SNAP/'training').glob('shard*/questions.jsonl') for q in rows(p)]
    assert len({q['question_id'] for q in committed})==len(committed)
    labels=0
    for question in committed:
        episodes={e['episode_id']:e for e in question['episodes']}
        for r in question['labels']:
            variants=episodes[r['episode_id']]['variants']
            left,right=variants[r['left']],variants[r['right']]
            answer=gold[question['question_id']]
            assert r['downstream_delta']==int(right['final_answer']==answer)-int(left['final_answer']==answer)
            assert r['local_delta']==int(right['receiver_answer']==answer)-int(left['receiver_answer']==answer)
            labels+=1
    for name,digest in policy['training_commitments_sha256'].items():
        assert sha(SNAP/name)==digest
    results=[r for p in (SNAP/'val').glob('shard*/results.jsonl') for r in rows(p)]
    assert len(results)==120 and len({(r['question_id'],r['density']) for r in results})==120
    for r in results:
        matching=[x for x in raw if x['stage']=='val' and x['question_id']==r['question_id'] and x['density']==r['density']]
        actual=sum(x['response']['prompt_eval_count']+x['response']['eval_count'] for x in matching)
        assert actual==r['logical_model_tokens']
        if r['status']=='ok':
            assert r['logical_calls']==8 and r['sampling']['valid']
            assert all(s['communication_tokens']<=1024 for s in r['sampling']['selections'])
    cleanup=json.loads((SNAP/'cleanup.json').read_text())
    assert cleanup['cleanup_complete'] and status['phase']=='complete'
    report={'status':'passed' if not limits and all(r['status']=='ok' for r in results) else 'passed_with_integrity_exceptions',
            'record_and_cost_consistency':'passed','training_labels_recomputed':labels,
            'valid_training_questions':len(committed),'validation_conditions':len(results),
            'valid_val_conditions':sum(r['status']=='ok' for r in results),'output_limit_calls':limits,
            'all_new_model_tokens_including_warmup':budget['actual_tokens'],'owned_cleanup_complete':True,
            'checkpoint_and_policy_timestamps_before_first_val':True,
            'scope':'consistency audit, not proof that the learned method improves quality or cost'}
    (ROOT/'completion_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
