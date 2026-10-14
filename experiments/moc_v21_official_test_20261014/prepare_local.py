"""Reuse original officialtest IDs and outcome-blind exact request histories."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = REPO.parent
sys.path.insert(0, str(REPO/'coding_base'))
from moc_adaptive_sampling.counterfactual import check_response, request_key, ExactCallCache


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (ROOT/'status.json').exists()
    cfg = json.loads((ROOT/'config.json').read_text())
    data = ROOT/'data'; data.mkdir(exist_ok=True)
    origin = WORK/'experiments/moc_full_graph_compression_20261013/data'
    shutil.copyfile(origin/'test_inputs.jsonl', data/'test_inputs.jsonl')
    inputs = rows(data/'test_inputs.jsonl'); ids = {r['question_id'] for r in inputs}
    assert len(inputs) == len(ids) == 254
    assert all(set(r) == {'question_id','question','options'} for r in inputs)
    old = WORK/'experiments/moc_stateful_sampling_val_20261014/data'
    overlaps = {split: len(ids & {r['question_id'] for r in rows(old/(split+'_inputs.jsonl'))})
                for split in ('train','dev','val')}
    assert not any(overlaps.values())
    history = rows(REPO/'experiments/moc_v21_expected_set_val_20261014/data/historical_calls_LOCAL_ONLY.jsonl')
    history += rows(REPO/'experiments/moc_v21_expected_set_val_20261014/data/val_historical_calls_LOCAL_ONLY.jsonl')
    # Incorporate completed v2.1 own calls with its preserved tool observations.
    for folder in (REPO/'experiments/moc_v21_expected_set_val_20261014/remote_snapshot/val').glob('shard*'):
        tools = {}
        for e in rows(folder/'trace.jsonl'):
            if e['event_type'] in ('new_tool_result','cached_tool_result'):
                tools.setdefault(e['request_key'], set()).add(str(e['result']))
        for line, call in enumerate(rows(folder/'calls_raw.jsonl'), 1):
            for result in tools.get(call['key'], {None}):
                r = dict(request=call['request'], response=call['response'], provenance={
                    'source':str(folder.relative_to(REPO)/'calls_raw.jsonl'),'source_sha256':sha(folder/'calls_raw.jsonl'),'raw_line':line})
                if result is not None:r['tool_result']=result
                history.append(r)
    archive = WORK/'experiments/moc_full_graph_compression_20261013/stopped_20261014/stopped_snapshot_20261014.tar.gz'
    archive_sha = sha(archive)
    seen = set(); admitted_old = 0; rejected = []
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            if not member.isfile() or '/test/' not in member.name:
                continue
            if member.name.endswith('results.jsonl'):
                seen |= {r['question_id'] for r in [json.loads(s) for s in t.extractfile(member).read().decode().splitlines() if s.strip()]}
            if not member.name.endswith('trace.jsonl'):
                continue
            events = [json.loads(s) for s in t.extractfile(member).read().decode().splitlines() if s.strip()]
            tool_results = {}
            for e in events:
                if e['event_type']=='code_execution_finished':
                    key = e.get('question_id'),e.get('receiver_id'),e.get('receiver_event_id')
                    tool_results.setdefault(key,set()).add(str(e['result']))
            for line, e in enumerate(events,1):
                if e['event_type']!='llm_call':continue
                try:
                    check_response(e['response'])
                    model = e['request']['model']
                    assert model in (cfg['main_model'],cfg['merger_model'])
                    request_key(e['request'],cfg['main_digest'] if model==cfg['main_model'] else cfg['merger_digest'])
                except (KeyError, ValueError, AssertionError) as error:
                    rejected.append(dict(member=member.name,event_index=line,reason=str(error)));continue
                key = e.get('question_id'),e.get('receiver_id'),e.get('receiver_event_id')
                for result in tool_results.get(key,{None}):
                    r = dict(request=e['request'], response=e['response'], provenance={
                        'source_archive_sha256':archive_sha,'member':member.name,'event_index':line,'origin':'old_test_exact_request_only'})
                    if result is not None:r['tool_result']=result
                    history.append(r);admitted_old+=1
    assert seen & ids == ids
    main_history=[r for r in history if r['request']['model']==cfg['main_model']]
    merger_history=[r for r in history if r['request']['model']==cfg['merger_model']]
    def write(name, records):
        (data/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf8')
    write('historical_calls_LOCAL_ONLY.jsonl',main_history)
    write('val_historical_calls_LOCAL_ONLY.jsonl',[])  # retained compatibility path, no outcome data
    write('merger_historical_calls_LOCAL_ONLY.jsonl',merger_history)
    cache = ExactCallCache(cfg['main_digest'],None);admission=cache.load_historical(main_history)
    report=dict(test_questions=254,conditions=1016,gold_read=False,gold_upload=False,
        original_input_sha256=sha(origin/'test_inputs.jsonl'),cross_split_overlap=overlaps,
        historically_exposed_test_questions=len(seen&ids),historically_untouched_test_questions=0,
        official_source_manifest=json.loads((origin/'manifest.json').read_text()),
        old_archive_sha256=archive_sha,old_test_call_records=admitted_old,rejected_calls=rejected,
        main_history_admission=admission,cache_conflicts=cache.historical_conflicts,
        merger_history_records=len(merger_history),no_historical_full_MOC_conditions_reused=True)
    (ROOT/'data_preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('test_questions','conditions','cross_split_overlap','old_test_call_records','main_history_admission','merger_history_records')}))


if __name__=='__main__':main()
