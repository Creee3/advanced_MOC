"""Post-generation final tests only; no GPU/model calls and no policy changes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from sandbox import run

ROOT=Path(__file__).resolve().parent


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--targets',type=Path,required=True)
    args=parser.parse_args()
    status=json.loads((ROOT/'status.json').read_text())
    assert status['phase']=='complete' and status['cleanup_complete'],'finish generation and GPU cleanup first'
    lock=json.loads((ROOT/'experiment_lock.json').read_text())
    assert hashlib.sha256(args.targets.read_bytes()).hexdigest()==lock['local_gold_sha256']
    for name,digest in {**lock['source_hashes'],**lock['data_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    import sys
    cfg=json.loads((ROOT/'config.json').read_text())
    sys.path.insert(0,cfg['author_directory'])
    from datasets.data_process import HumanEvalDataset
    targets={r['question_id']:r for r in rows(args.targets)}
    inputs=rows(ROOT/'data/test_inputs.jsonl')
    expected={(r['question_id'],arm,d) for r in inputs for arm in ('moc','v21_expected_set') for d in (.3,.7)}
    results=[r for p in sorted((ROOT/'test').glob('shard*/results.jsonl')) for r in rows(p)]
    assert len(results)==len({(r['question_id'],r['arm'],r['density']) for r in results})==656
    assert {(r['question_id'],r['arm'],r['density']) for r in results}==expected
    output=ROOT/'final_scores.jsonl'
    done={(r['question_id'],r['arm'],r['density']) for r in rows(output)} if output.exists() else set()
    for record in results:
        key=record['question_id'],record['arm'],record['density']
        if key in done:continue
        start=time.monotonic()
        if record['status']=='ok':
            code=HumanEvalDataset.postprocess_answer(None,record['final_output'])
            assert code==record['answer'],'author postprocess mismatch'
            outcome=run(code,targets[record['question_id']]['test'],cfg['final_test_timeout_seconds'])
            if outcome.get('error_type')=='SandboxFailure':raise RuntimeError(str(outcome))
        else:
            outcome=dict(passed=False,error_type='InvalidGeneration',error=record['reason'])
        item=dict(question_id=key[0],arm=key[1],density=key[2],generation_status=record['status'],
                  **outcome,seconds=time.monotonic()-start)
        with output.open('a',encoding='utf-8') as stream:stream.write(json.dumps(item,ensure_ascii=False)+'\n')
        done.add(key)
        print(json.dumps(dict(scored=len(done),target=656,question_id=key[0],arm=key[1],density=key[2])),flush=True)
    (ROOT/'scoring_complete.json').write_text(json.dumps(dict(status='complete',conditions=len(done),
        targets_sha256=hashlib.sha256(args.targets.read_bytes()).hexdigest(),
        score_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),real_model_calls=0),indent=2)+'\n')


if __name__=='__main__':main()
