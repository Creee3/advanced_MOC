"""Read-only completed-test consistency audit; no model or remote calls."""
from collections import Counter, defaultdict
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'remote_snapshot'
sys.path.insert(0,str(SNAP/'coding_base'))
from moc_adaptive_sampling.counterfactual import ExactCallCache,check_response,request_key

def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def usage(e):return e['response']['prompt_eval_count']+e['response']['eval_count']
def key(e):return e['question_id'],e['arm'],e['density']

def author_parser():
    path=ROOT.parents[1]/'coding_base/MOC-official/src/prompt/aqua_prompt_set.py'
    proof=read(ROOT/'final_live_cleanup_proof.json')
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==proof['parser_lf_sha256']
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AQUAPromptSet')
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_answer')
    function.decorator_list=[];function.returns=None
    for arg in function.args.args:arg.annotation=None
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
    return lambda text:namespace['postprocess_answer'](None,text)

def replay_cache(kind,folder,raw,history,digest):
    """Replay each exact cache epoch at the two archived recovery boundaries."""
    previous=[ROOT/'interrupted_snapshot_20261015',ROOT/'repeat_error_snapshot_20261015']
    filename='calls_raw.jsonl' if kind=='main' else 'merger_calls_raw.jsonl'
    boundaries=[0]+[len(rows(s/'test'/folder.name/filename)) for s in previous]+[len(raw)]
    hits=0;rejected=0
    for epoch in range(3):
        prefix=raw[:boundaries[epoch]]
        if epoch:
            assert prefix==rows(previous[epoch-1]/'test'/folder.name/filename),'prior raw prefix changed'
        records=list(history);failed_tools=set()
        if kind=='main':
            trace=rows(previous[epoch-1]/'test'/folder.name/'trace.jsonl') if epoch else []
            tools=defaultdict(set)
            for e in trace:
                if e.get('event_type') in ('new_tool_result','cached_tool_result'):tools[e['request_key']].add(str(e['result']))
            for a,b in zip(trace,trace[1:]):
                if a.get('event_type')=='code_execution_finished' and a.get('timed_out') and b.get('event_type')=='new_tool_result':failed_tools.add(b['request_key'])
        for e in prefix:
            try:check_response(e['response'])
            except ValueError:continue
            for tool in tools.get(e['key'],{None}) if kind=='main' else {None}:
                r=dict(request=e['request'],response=e['response'],provenance={'audit':'prior_raw'})
                if tool is not None:r['tool_result']=tool
                records.append(r)
        cache=ExactCallCache(digest,None);cache.load_historical(records)
        for k in failed_tools:cache.entries.pop(k,None)
        for e in raw[boundaries[epoch]:boundaries[epoch+1]]:
            assert request_key(e['request'],digest)==e['key']
            if e['cache_hit']:
                assert e['key'] in cache.entries,('unproven cache hit',kind,folder.name,epoch,e['key'])
                saved=cache.entries[e['key']]['response']
                assert all(saved[k]==e['response'][k] for k in ('message','prompt_eval_count','eval_count'))
                hits+=1
            else:
                assert e['key'] not in cache.entries,('unexpected cache miss',kind,folder.name,epoch,e['key'])
                try:cache.add(e['request'],e['response'],e['provenance'])
                except ValueError as error:
                    assert 'output cap reached' in str(error)
                    rejected+=1
    return dict(cache_hits_verified=hits,output_caps=rejected,epochs=3,boundaries=boundaries)

def main():
    recovery_snapshots=[]
    for name,archive in [('interrupted_snapshot_20261015','interrupted_raw.tar.gz'),('repeat_error_snapshot_20261015','before_recovery.tar.gz')]:
        folder=ROOT/name;prior=read(folder/'manifest.json')
        assert sha(folder/archive)==prior['archive_sha256']
        files=prior['files'] if isinstance(prior['files'],dict) else {item['path']:item['sha256'] for item in prior['files']}
        for filename,digest in files.items():assert sha(folder/filename)==digest,(name,filename)
        recovery_snapshots.append(dict(directory=name,files_verified=len(prior['files']),archive_sha256=prior['archive_sha256']))
    manifest=read(SNAP/'manifest.json')
    assert sha(SNAP/'final_raw_records.tar.gz')==manifest['archive_sha256']
    for name,digest in manifest['files'].items():assert sha(SNAP/name)==digest,name
    frozen=read(SNAP/'experiment_lock.json');policy=read(SNAP/'policy_lock.json');cfg=read(SNAP/'config.json')
    for name,digest in {**frozen['source_hashes'],**frozen['data_hashes']}.items():
        assert sha(SNAP/name)==sha(ROOT/name)==digest,name
    assert sha(SNAP/'experiment_lock.json')==policy['experiment_lock_sha256']
    assert sha(SNAP/'group_utility_model.json')==policy['model_sha256']
    amendment=read(SNAP/'runtime_recovery_manifest_20261015.json')
    assert sha(SNAP/'runtime_recovery_manifest_20261015.json')==frozen['runtime_amendment_sha256']==policy['runtime_amendment_sha256']
    first=read(ROOT/'interrupted_snapshot_20261015/experiment_lock.json')
    for name,digest in first['source_hashes'].items():
        if name in amendment['patches']:
            assert amendment['patches'][name]['before_sha256']==digest
            assert frozen['source_hashes'][name]==amendment['patches'][name]['after_sha256']
        else:assert frozen['source_hashes'][name]==digest
    assert first['data_hashes']==frozen['data_hashes'] and policy['frozen_before_test']
    assert not frozen['test_gold_included'] and not frozen['training_labels_included']
    proof=read(ROOT/'final_live_cleanup_proof.json')
    assert proof['author_commit']==cfg['author_commit'] and not proof['author_git_status']
    assert not any(proof['owned_pids'].values()) and all(x=='closed' for x in proof['private_ports'].values())
    inputs=rows(SNAP/'data/test_inputs.jsonl');qids=[r['question_id'] for r in inputs]
    assert len(qids)==len(set(qids))==254 and all(set(r)=={'question_id','question','options'} for r in inputs)
    parse=author_parser();all_results=[];all_raw=[];cache_proofs={};consumed_new=set()
    caps=[];timeouts=[];unknown=[];grouped_by_shard={};reparsed=0;decisions=0
    dag=read(SNAP/'expected_dags.json')
    for shard in (0,1):
        folder=SNAP/'test'/f'shard{shard}'
        results=rows(folder/'results.jsonl');traces=rows(folder/'trace.jsonl')
        allowed={(q,a,d) for q in qids[shard::2] for a in ('moc','v21_expected_set') for d in (.3,.7)}
        assert len(results)==508 and {key(r) for r in results}==allowed
        assert read(folder/'finished.json')['status']=='complete'
        grouped={}
        for kind,name,family in [('main','calls_raw.jsonl','main'),('merger','merger_calls_raw.jsonl','merger')]:
            raw=rows(folder/name);group=defaultdict(list)
            history=[r for n in (('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl') if kind=='main' else ('merger_historical_calls_LOCAL_ONLY.jsonl',)) for r in rows(SNAP/'data'/n)]
            cache_proofs[f'{shard}_{kind}']=replay_cache(kind,folder,raw,history,cfg[family+'_digest'])
            for index,e in enumerate(raw):
                assert key(e) in allowed and e['stage']=='test'
                assert e['request']['model']==cfg[family+'_model']
                options={'temperature':0,'seed':42,'num_ctx':8192,'num_gpu':999,'num_predict':2000} if kind=='main' else {'temperature':.1,'seed':42,'num_ctx':8192,'num_gpu':999}
                assert e['request']['options']==options and e['response']['done']
                assert all(isinstance(e['response'][k],int) and e['response'][k]>=0 for k in ('prompt_eval_count','eval_count'))
                if e['response'].get('done_reason')=='length':caps.append(dict(shard=shard,kind=kind,question_id=e['question_id'],arm=e['arm'],density=e['density'],key=e['key'],tokens=usage(e)))
                group[key(e)].append((index,e));all_raw.append((shard,kind,index,e))
            grouped[kind]=group
        for r in results:
            chosen={k:(grouped[k][key(r)][-r[k+'_calls']:] if r[k+'_calls'] else []) for k in grouped}
            main=[e for _,e in chosen['main']];merge=[e for _,e in chosen['merger']]
            selected=main+merge
            assert len(main)==r['main_calls'] and len(merge)==r['merger_calls']
            assert len(selected)==r['logical_calls'] and sum(map(usage,selected))==r['logical_model_tokens']
            assert sum(map(usage,main))==r['main_model_tokens'] and sum(map(usage,merge))==r['merger_model_tokens']
            assert sum(not e['cache_hit'] for e in selected)==r['new_calls']
            assert sum(e['cache_hit'] for e in selected)==r['cache_hits']
            assert sum(usage(e) for e in selected if not e['cache_hit'])==r['new_model_tokens']
            for kind,events in chosen.items():
                for index,e in events:
                    if not e['cache_hit']:consumed_new.add((shard,kind,index))
            if r['status']=='ok':
                assert r['main_calls']==8 and all(e['response'].get('done_reason')!='length' for e in selected)
                assert r['final_output']==main[-1]['response']['message']['content']
                assert r['answer']==parse(r['final_output']);reparsed+=1
                if r['arm']=='moc':assert r['merger_calls']%5==0
                else:
                    assert not merge and r['sampling']['valid'] and not r['sampling']['failures']
                    assert len(r['sampling']['selections'])==7
                    for s in r['sampling']['selections']:
                        assert len(s['source_ids'])<=2 and s['communication_tokens']<=1024
                        assert s['source_ids']==[m['source'] for m in s['candidates'] if m['source'] in s['source_ids']]
                        for layer in s['layers']:
                            assert not layer['selected_before'] and not layer['uncertainty_used_for_decision']
                            scores=layer['scores']
                            for c in scores:assert c['net_mean']==c['mean']-.02*c['projected_tokens']/1024
                            feasible=[c for c in scores if c['supported']]
                            best=min(feasible,key=lambda c:(-c['net_mean'],c['projected_tokens'],len(c['sources']),tuple(c['sources']))) if feasible else None
                            if s['source_ids']:
                                assert best and best['net_mean']>0 and best['sources']==s['source_ids']
                                if len(s['source_ids'])==2:
                                    u,v=best['path'];assert best['is_connected_path'] and u in dag[str(r['density'])]['predecessors'][v]
                            else:assert best is None or best['net_mean']<=0
                            decisions+=1
            else:assert r['status']=='invalid' and r.get('reason')
        settled={e['reservation_id'] for e in traces if e.get('event_type')=='backend_request_settled'}
        reconciled={e['reservation_id'] for e in traces if e.get('event_type')=='backend_request_reconciled_unknown'}
        starts={e['reservation_id']:e for e in traces if e.get('event_type')=='backend_request_started'}
        assert not (set(starts)-settled-reconciled)
        unknown.extend(starts[k] for k in reconciled)
        for a,b in zip(traces,traces[1:]):
            if a.get('event_type')=='code_execution_finished' and a.get('timed_out'):
                assert b['event_type']=='new_tool_result'
                timeouts.append(dict(shard=shard,question_id=b['question_id'],arm=b['arm'],density=b['density'],event_index=a['event_index']))
        grouped_by_shard[shard]=grouped
        all_results+=results
    by_key={key(r):r for r in all_results}
    assert len(by_key)==1016
    assert all(by_key[key(e)]['status']=='invalid' for e in caps+timeouts)
    startup=rows(SNAP/'startup_calls_raw.jsonl');budget=read(SNAP/'budget.json')
    new=[e for _,_,_,e in all_raw if not e['cache_hit']]
    assert len(startup)==6 and budget['reserved_tokens']==0
    assert sum(map(usage,new))+sum(map(usage,startup))==budget['actual_tokens']
    assert len(new)+len(startup)+budget['unknown_call_count']==budget['new_calls']
    assert len(unknown)==3 and sum(e['reserved_tokens'] for e in unknown)+10192==budget['unknown_cost_upper_bound_tokens']==59344
    assert budget['unknown_call_count']==4 and budget['actual_tokens']+59344<=cfg['max_new_model_tokens']
    assert budget['new_calls']<=cfg['max_new_calls']
    status=read(SNAP/'status.json');cleanup=read(SNAP/'cleanup.json');clock=read(SNAP/'run_start.json')
    assert status['phase']=='complete' and status['cleanup_complete'] and cleanup['cleanup_complete']
    assert not any(cleanup[k] for k in ('remaining_ports','owned_children_remaining','owned_processes_remaining'))
    for lease in rows(SNAP/'service_leases.jsonl'):
        family='merger' if lease['kind']=='merger' else 'main'
        assert lease['model']['digest']==cfg[family+'_digest'] and lease['loaded']['size_vram']>=lease['loaded']['size']
        assert 'cuMemAlloc 0' in lease['cuda_allocation_probe']
    finish=datetime.fromisoformat(status['updated_utc']).timestamp()
    active=clock['prior_active_wall_seconds']+finish-datetime.fromisoformat(clock['extension_started_utc']).timestamp()
    assert finish<=clock['deadline_unix'] and active<=28800
    discarded=[e for s,k,i,e in all_raw if not e['cache_hit'] and (s,k,i) not in consumed_new]
    report=dict(status='record_and_cost_consistency_passed',independent_reproduction=False,
        snapshot_files_verified=len(manifest['files']),archive_sha256=manifest['archive_sha256'],
        recovery_snapshots_verified=recovery_snapshots,
        source_and_data_hashes_verified=len(frozen['source_hashes'])+len(frozen['data_hashes']),
        original_algorithm_model_data_unchanged=True,explicit_runtime_amendment=amendment['scope'],
        terminal_conditions=len(all_results),valid_conditions=reparsed,invalid_conditions=1016-reparsed,
        author_parser_reparsed_answers=reparsed,cache_proofs=cache_proofs,frozen_argmax_decisions_verified=decisions,
        output_cap_calls=caps,code_timeouts=timeouts,all_conditions_output_complete=False,
        known_actual_new_tokens=budget['actual_tokens'],new_calls_including_unknown=budget['new_calls'],
        startup_calls=len(startup),startup_tokens=sum(map(usage,startup)),
        unknown_calls=4,unknown_cost_upper_bound_tokens=59344,unresolved_reservations=0,
        known_discarded_partial_execution_calls=len(discarded),known_discarded_partial_execution_tokens=sum(map(usage,discarded)),
        discarded_scope='new known calls before interrupted conditions were replayed; already included in actual expense',
        cumulative_window_wall_seconds=active,owned_cleanup_verified=True,
        completed_beijing=datetime.fromisoformat(status['updated_utc']).astimezone(__import__('datetime').timezone(__import__('datetime').timedelta(hours=8))).isoformat(),
        historical_test_exposure=True,scope='consistency audit; invalid units and unknown usage remain explicitly disclosed')
    (ROOT/'completion_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
