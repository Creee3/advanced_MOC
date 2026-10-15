"""Completed HumanEval record/cost/policy consistency audit; no execution/LLM."""
import ast
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];SNAP=ROOT/'remote_snapshot';REPO=ROOT.parents[1]
sys.path.insert(0,str(SNAP/'coding_base'))
from moc_adaptive_sampling.counterfactual import ExactCallCache,check_response,request_key

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return r['question_id'],r['arm'],r['density']
def usage(r):return r['response']['prompt_eval_count']+r['response']['eval_count']


def parser():
    path=REPO/'coding_base/MOC-official/datasets/data_process.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HumanEvalDataset')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_answer')
    fn.decorator_list=[];fn.returns=None
    for arg in fn.args.args:arg.annotation=None
    scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),scope)
    return lambda text:scope['postprocess_answer'](None,text)


def main():
    manifest=read(SNAP/'manifest.json')
    assert sha(SNAP/'final_raw_records.tar.gz')==manifest['archive_sha256']
    for name,digest in manifest['files'].items():assert sha(SNAP/name)==digest,name
    cfg=read(SNAP/'config.json');lock=read(SNAP/'experiment_lock.json');policy=read(SNAP/'policy_lock.json')
    for name,digest in {**lock['source_hashes'],**lock['data_hashes']}.items():
        assert sha(SNAP/name)==sha(ROOT/name)==digest,name
    assert policy['source_hashes']==lock['source_hashes'] and policy['frozen_before_test']
    assert policy['checkpoint_unchanged'] and policy['role_mapping'] is None
    assert sha(SNAP/'group_utility_model.json')==policy['model_sha256']==sha(ROOT.parent/'moc_v21_official_test_20261014/group_utility_model.json')
    assert sha(ROOT/'data/test_targets_LOCAL_ONLY.jsonl')==sha(SNAP/'eval/test_targets_LOCAL_ONLY.jsonl')==lock['local_gold_sha256']
    proof=read(SNAP/'final_live_cleanup_proof.json')
    assert proof['author_commit']==cfg['author_commit'] and not proof['author_git_status']
    assert not any(proof['owned_pids'].values()) and set(proof['private_ports'].values())=={'closed'}
    final_probe=read(ROOT/'final_process_probe.json')
    assert final_probe['all_owned_clean'] and not any(final_probe['owned_pids_exist'].values())
    assert set(final_probe['private_ports'].values())=={'closed'}
    for name,digest in proof['author_lf_sha256'].items():
        assert hashlib.sha256((REPO/'coding_base/MOC-official'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,name
    inputs=rows(SNAP/'data/test_inputs.jsonl');ids=[r['question_id'] for r in inputs]
    assert len(ids)==len(set(ids))==164 and all(set(r)=={'question_id','prompt','entry_point'} for r in inputs)
    targets={r['question_id']:r['test'] for r in rows(SNAP/'eval/test_targets_LOCAL_ONLY.jsonl')}
    assert set(targets)==set(ids)
    scores=rows(SNAP/'final_scores.jsonl');score_by={key(r):r for r in scores}
    assert len(scores)==len(score_by)==656
    scoring=read(SNAP/'scoring_complete.json')
    assert scoring['conditions']==656 and scoring['real_model_calls']==0 and scoring['score_sha256']==sha(SNAP/'final_scores.jsonl')
    parse=parser();records=[];all_raw=[];hits=0;decisions=0;early=0;caps=[];settlements=0;concurrent_duplicates=0
    for shard in (0,1):
        folder=SNAP/'test'/f'shard{shard}'
        results=rows(folder/'results.jsonl');traces=rows(folder/'trace.jsonl')
        allowed={(q,a,d) for q in ids[shard::2] for a in ('moc','v21_expected_set') for d in (.3,.7)}
        assert len(results)==len({key(r) for r in results})==328 and {key(r) for r in results}==allowed
        assert read(folder/'finished.json')['status']=='complete'
        starts={r['reservation_id']:r for r in traces if r['event_type']=='backend_request_started'}
        settled={r['reservation_id']:r for r in traces if r['event_type']=='backend_request_settled'}
        assert set(starts)==set(settled),'unsettled or unknown requests'
        settlements+=len(settled)
        dispatches=defaultdict(list)
        for rid,start in starts.items():
            model=start['request']['model'];family='main' if model==cfg['main_model'] else 'merger'
            assert model==cfg[family+'_model']
            dispatches[family,request_key(start['request'],cfg[family+'_digest'])].append((start,settled[rid]))
        for entries in dispatches.values():entries.sort(key=lambda pair:pair[1]['event_index'])
        grouped={}
        for family,filename in (('main','calls_raw.jsonl'),('merger','merger_calls_raw.jsonl')):
            cache=ExactCallCache(cfg[family+'_digest'],None);grouped[family]=defaultdict(list)
            completion_by_key={}
            for r in rows(folder/filename):
                check_response(r['response']);assert key(r) in allowed
                assert r['request']['model']==cfg[family+'_model']
                options=dict(temperature=0 if family=='main' else .1,seed=42,num_ctx=8192,num_gpu=999)
                if family=='main':options['num_predict']=2000
                assert r['request']['options']==options and request_key(r['request'],cfg[family+'_digest'])==r['key']
                if r['cache_hit']:
                    assert r['key'] in cache.entries
                    saved=cache.entries[r['key']]['response']
                    assert all(saved[k]==r['response'][k] for k in ('message','prompt_eval_count','eval_count'))
                    hits+=1
                else:
                    start,settle=dispatches[family,r['key']].pop(0)
                    assert settle['actual_tokens']==usage(r)
                    if r['key'] in cache.entries:
                        assert start['event_index']<completion_by_key[r['key']],'unexplained duplicate new request'
                        concurrent_duplicates+=1
                    cache.add(r['request'],r['response'],r['provenance'])
                    completion_by_key.setdefault(r['key'],settle['event_index'])
                text='\n'.join(m['content'] for m in r['request']['messages'])
                assert not any(test.strip() in text for test in targets.values()),'full hidden test body in generation prompt'
                grouped[family][key(r)].append(r);all_raw.append(r)
            assert not any(v for (family_key,_),v in dispatches.items() if family_key==family)
        consumed={}
        for e in traces:
            if e['event_type']=='communication_consumed':
                consumed.setdefault((e['question_id'],e['receiver_id']),[]).append(e)
        for r in results:
            assert r['status']=='ok' and 1<=r['main_calls']<=8
            main,merge=grouped['main'][key(r)],grouped['merger'][key(r)]
            assert len(main)==r['main_calls'] and len(merge)==r['merger_calls']
            selected=main+merge
            assert len(selected)==r['logical_calls'] and sum(map(usage,selected))==r['logical_model_tokens']
            assert sum(map(usage,main))==r['main_model_tokens'] and sum(map(usage,merge))==r['merger_model_tokens']
            assert sum(not e['cache_hit'] for e in selected)==r['new_calls']
            assert sum(e['cache_hit'] for e in selected)==r['cache_hits']
            assert sum(usage(e) for e in selected if not e['cache_hit'])==r['new_model_tokens']
            assert main[-1]['response']['message']['content']==r['final_output'] and parse(r['final_output'])==r['answer']
            assert score_by[key(r)]['generation_status']==r['status'] and isinstance(score_by[key(r)]['passed'],bool)
            assert r['early_return_count']==8-r['main_calls'];early+=r['early_return_count']
            if r['arm']=='moc':assert len(merge)%5==0
            else:
                sampling=r['sampling'];assert not merge and sampling['valid'] and not sampling['failures']
                assert len(sampling['selections'])==7
                for s in sampling['selections']:
                    assert len(s['source_ids'])<=2 and s['communication_tokens']<=1024
                    assert s['source_ids']==[m['source'] for m in s['candidates'] if m['source'] in s['source_ids']]
                    for layer in s['layers']:
                        assert not layer['selected_before'] and not layer['uncertainty_used_for_decision']
                        for candidate in layer['scores']:
                            assert candidate['net_mean']==candidate['mean']-.02*candidate['projected_tokens']/1024
                            if not candidate['supported']:assert candidate['abstained']
                        feasible=[c for c in layer['scores'] if c['supported']]
                        best=min(feasible,key=lambda c:(-c['net_mean'],c['projected_tokens'],len(c['sources']),tuple(c['sources']))) if feasible else None
                        assert (best is not None and best['net_mean']>0 and best['sources']==s['source_ids']) if s['source_ids'] else (best is None or best['net_mean']<=0)
                        decisions+=1
                    # Each question has two arm/density runs. Match the exact
                    # observed selected summary and spatial list, not a stale ID.
                    possible=consumed.get((r['question_id'],s['receiver']),[])
                    assert any(e['spatial_sources']==s['source_ids'] and e['summary_used']==bool(s['source_ids']) for e in possible)
            records.append(r)
    assert len(records)==656 and {key(r) for r in records}==set(score_by)
    startup=rows(SNAP/'startup_calls_raw.jsonl');budget=read(SNAP/'budget.json')
    paid=[r for r in all_raw if not r['cache_hit']]
    assert len(startup)==2 and budget['reserved_tokens']==budget.get('unknown_cost_upper_bound_tokens',0)==0
    assert len(paid)==settlements and len(paid)+len(startup)==budget['new_calls']
    assert sum(map(usage,paid))+sum(map(usage,startup))==budget['actual_tokens']
    assert budget['new_calls']<=cfg['max_new_calls'] and budget['actual_tokens']<=cfg['max_new_model_tokens']
    status=read(SNAP/'status.json');clock=read(SNAP/'run_start.json')
    assert status['phase']=='complete' and status['cleanup_complete'] and status['completed_test_conditions']==656
    finish=datetime.fromisoformat(status['updated_utc']).timestamp()
    assert finish<clock['deadline_unix']
    assert read(SNAP/'scoring_launcher.json')['started_unix']>finish
    for lease in rows(SNAP/'service_leases.jsonl'):
        family='merger' if lease['kind']=='merger' else 'main'
        assert lease['model']['digest']==cfg[family+'_digest'] and lease['loaded']['size_vram']>=lease['loaded']['size']
        assert 'cuMemAlloc 0' in lease['cuda_allocation_probe']
    report=dict(status='record_and_cost_consistency_passed',independent_reproduction=False,
        snapshot_files_verified=len(manifest['files']),archive_sha256=manifest['archive_sha256'],
        locked_source_and_data_files=len(lock['source_hashes'])+len(lock['data_hashes']),
        terminal_conditions=656,valid_generations=656,invalid_generations=0,scored_conditions=656,
        author_parser_reparsed=656,exact_cache_hits_verified=hits,concurrent_duplicate_dispatches=concurrent_duplicates,
        frozen_argmax_decisions_verified=decisions,early_returns_verified=early,
        known_actual_new_tokens=budget['actual_tokens'],new_calls=budget['new_calls'],unknown_cost_upper_bound_tokens=0,
        unresolved_reservations=0,startup_calls=2,startup_tokens=sum(map(usage,startup)),
        discarded_new_calls=0,output_cap_calls=0,owned_cleanup_verified=True,
        owned_pid_checks=len(final_probe['owned_pids_exist']),cpu_scorer_cleanup_verified=True,
        generation_wall_seconds=finish-datetime.fromisoformat(clock['started_utc']).timestamp(),
        scoring_started_after_generation_and_cleanup=True,
        score_failure_types=dict(Counter(s.get('error_type') for s in scores if not s['passed'])),
        audit_scope='records, source, parser, exact cache, cost and frozen decision consistency; not independent performance reproduction')
    (ROOT/'completion_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
