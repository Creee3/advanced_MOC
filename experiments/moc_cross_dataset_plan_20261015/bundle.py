"""One maintained runtime, frozen per-dataset snapshots for auditable deployment."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    base=json.loads((ROOT.parent/'moc_v21_humaneval_test_20261015/config.json').read_text())
    for domain,counts,port in [('gsm8k',(24,120,300),11448),('mmlu_pro',(14,56,280),11449)]:
        out=ROOT.parent/f'moc_v21_{domain}_fewshot_20261015';remote='/home/liuxin/ICML2027/experiments/'+out.name
        if (out/'experiment_lock.json').exists():raise RuntimeError('already frozen')
        for p in (ROOT/'runtime').glob('*.py'):shutil.copyfile(p,out/p.name)
        ext=out/'coding_base/moc_adaptive_sampling';ext.mkdir(parents=True,exist_ok=True)
        for p in (REPO/'coding_base/moc_adaptive_sampling').glob('*.py'):shutil.copyfile(p,ext/p.name)
        shutil.copyfile(ROOT.parent/'moc_v21_official_test_20261014/group_utility_model.json',out/'frozen_aqua_model.json')
        roles=(['Math Solver','Mathematical Analyst','Programming Expert','Inspector','Math Solver','Mathematical Analyst','Programming Expert'] if domain=='gsm8k' else
               ['Knowlegable Expert','Critic','Mathematician','Psychologist','Historian','Doctor','Lawyer'])
        c=dict(base,domain=domain,extension_directory=remote+'/coding_base',main_ports=[port],roles=roles,
            phase='approved_fewshot_adaptation',train_questions=counts[0],val_questions=counts[1],test_questions=counts[2],
            validation_conditions=counts[1]*4,test_conditions=counts[2]*4,max_new_calls=20000,max_new_model_tokens=16000000,
            conditions_total=counts[1]*4+counts[2]*4,code_timeout_seconds=5)
        c['model']=dict(c['model'],user_authorized_diagnostic_val=False,training_gate_passed=None)
        save(out/'config.json',c)
        for name in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl'):
            (out/'data'/name).write_text('')
        train=counts[0];val=counts[1];test=counts[2]
        (out/'protocol.md').write_text(f'''# Approved {domain}: small-task adaptation of v2.1

User approved 2026-10-15: train{train}, val{val}, author test{test}; densities .3/.7.
User additionally authorized two GPUs, one main per dataset concurrently, and 10-minute monitoring.
Two GPUs maximum: parallel train/fit/val/adapted-test stages, then serial original-MOC stages using one27B+one9B. No third GPU. Original per-dataset8h deadline includes waiting, never reset.

Data: split_manifest.json. GSM train/val are disjoint samples of official train7473; Pro uses one official-validation question per category for fit, remaining four for val. Author test bytes/answers unchanged. Train{train} is the planned count; excluded incomplete questions are reported, not silently replaced. Val/test do not enter fitting, prompt answers or hyperparameter selection.

Algorithm unchanged: MiniLM, maxhop4, one-shot empty/singleton/direct-edge pair, serialized1024token budget, cost penalty.02, compact22 weighted ridge10, bootstrap32(seed42) descriptive only. Two outcome-blind receiver-role episodes per training problem, alternating density. Downstream signed correctness labels from fixed upstream/Vanilla downstream replay; this is a proxy, not ground-truth full-MAS utility. Same author roles, DAG seed42, task prompts, all-output FinalRefer and original parsers.

Only native-domain thin interface added for AnalyzeAgent; Wiki Searcher unsupported and absent from pinned seven roles. Math executes original separate globals/locals semantics in restricted Landlock/seccomp subprocess,5sec, with permission/resource restrictions disclosed. Both arms share this executor. No author source edit.

Val compares frozen AQuA v2.1 and task-trained v2.1; test compares task-trained v2.1 and originalMOC(K2, originalISM and five9B candidates). No new invented baseline or duplicate baseline loop. Baseline reuse audit first; only exact matched results reused. Missing MOC results generated once. All terminal invalid outcomes retained; only unfinished work resumes from exact compatible cache. Source and policy frozen before val/test; no fitting on val or parameter search.

Gemma2:27b/9b Q4_0, MiniLM, temperature0 main, original merger.1, seed42, context8192/main output2000, num_gpu999, existing CUDAv13 only. At most two GPUs globally. Each dataset cap8h absolute/20000newcalls/16000000actual+unknown-upper+pending tokens. STOP/limits never bypassed; same-root-cause recovery <=2, preserve unique archives and all costs. No Qwen/new models/drivers or other users' processes.

Report paired per-density accuracy+20000 question bootstrap seed42+McNemar; main+merger prompt/completion logicaltokens including cache hits, encoder/router time, actual training/warmup/discard/unknown costs separately; support/empty/singleton/pair/hops. Equal observed counts or CI crossing0 do not prove noninferiority. Zero-target-training transfer and few-shot adaptation must remain distinct. No test feedback changes.
''',encoding='utf-8')
        (out/'task_plan.md').write_text('1. Freeze inputs/protocol/source.\n2. CPU preflight.\n3. Parallel training, fit, val, adapted test.\n4. Serial missing original MOC baseline.\n5. Download raw/SHA, local gold scoring and cost/coverage audit.\n',encoding='utf-8')
        (out/'progress.md').write_text('Approved and prepared; not launched yet.\n',encoding='utf-8')
        sources=[*out.glob('*.py'),*ext.glob('*.py'),out/'config.json']
        data=[out/'split_manifest.json',out/'baseline_reuse.json',out/'frozen_aqua_model.json',
              *[out/'data'/f'{p}_inputs.jsonl' for p in ('train','val','test')],out/'data/train_gold_LOCAL_ONLY.jsonl']
        lock=dict(source_hashes={p.relative_to(out).as_posix():sha(p) for p in sorted(sources)},
            data_hashes={p.relative_to(out).as_posix():sha(p) for p in sorted(data)},
            local_gold_sha256={p:sha(out/'data'/f'{p}_gold_LOCAL_ONLY.jsonl') for p in ('train','val','test')},
            author_commit=c['author_commit'],author_test_sha256=json.loads((out/'split_manifest.json').read_text())['author_test_sha256'])
        save(out/'experiment_lock.json',lock)
        names=set(lock['source_hashes'])|set(lock['data_hashes'])|{'experiment_lock.json','protocol.md','task_plan.md','progress.md'}
        names|={'data/'+n for n in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl')}
        with tarfile.open(out/'upload_LOCAL_ONLY.tar.gz','w:gz') as tar:
            for n in sorted(names):tar.add(out/n,arcname=n,recursive=False)
        print(json.dumps(dict(dataset=domain,files=len(names),archive=str(out/'upload_LOCAL_ONLY.tar.gz'),sha256=sha(out/'upload_LOCAL_ONLY.tar.gz'))))

if __name__=='__main__':main()
