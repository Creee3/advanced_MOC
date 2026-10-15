"""Outcome-blind train40 preparation; native author test; no generation calls."""
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
WORKSPACE=REPO.parent
OLD=ROOT.parent/'moc_cross_dataset_plan_20261015'
AUTHOR=REPO/'coding_base/MOC-official'
NAMES={d:f'moc_v21_{d}_train40_test_20261015' for d in ('mmlu','svamp')}

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in r),encoding='utf-8')
def norm(t):return ' '.join(str(t).split()).casefold()
def rank(x):return sha(('42:'+x).encode())
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MOC-train40-audit'}),timeout=40) as f:return f.read()
def similar(a,b):
    matcher=difflib.SequenceMatcher(None,a,b,autojunk=False)
    return matcher.real_quick_ratio()>=.9 and matcher.quick_ratio()>=.9 and matcher.ratio()>=.9

def pools():
    official=[]
    url='https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=all&split=dev'
    payloads=[]
    for offset in (0,100,200):
        blob=fetch(url+f'&offset={offset}&length=100');payloads.append(sha(blob))
        official.extend(x['row'] for x in json.loads(blob)['rows'])
    assert len(official)==285
    # Match the independently downloaded local dev copy; answers checked without printing.
    local=WORKSPACE/'experiments/gdesigner_adversarial_10plus20_20261008/source/datasets/MMLU/data/dev'
    localrows=[dict(question=r[0],choices=r[1:5],answer='ABCD'.index(r[5]),subject=f.stem.removesuffix('_dev'))
        for f in sorted(local.glob('*.csv')) for r in csv.reader(f.open(encoding='utf-8-sig'))]
    canonical=lambda rs:sorted((norm(r['question']),tuple(r['choices']),r['answer'],r['subject']) for r in rs)
    assert canonical(localrows)==canonical(official)
    mmlu=[dict(question_id='mmlu:official_dev:'+r['subject']+':'+sha(norm(r['question']).encode()),
        question=r['question'],category=r['subject'],task=r['question']+'\n'+''.join(f'Option {c}: {v}\n' for c,v in zip('ABCD',r['choices'])),answer='ABCD'[r['answer']]) for r in official]
    svamp_url='https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json'
    blob=fetch(svamp_url)
    assert sha(blob)==sha((WORKSPACE/'coding_base/ARG-Designer-main/datasets/SVAMP/SVAMP.json').read_bytes())
    svamp=[svamp_wrap(r,'official_full') for r in json.loads(blob)]
    return {'mmlu':mmlu,'svamp':svamp},{'mmlu':dict(url=url,payload_sha256=payloads,local_exact_match=True,rows=285),
        'svamp':dict(url=svamp_url,raw_sha256=sha(blob),rows=1000)}

def svamp_wrap(r,source):
    q=r['Body']+' '+r['Question']
    return dict(question_id='svamp:'+source+':'+str(r['ID']),question=q,category=r['Type'],
        task=f"Context: {r['Body']}\nQuestion: {r['Question']}\nAnswer: ",answer=r['Answer'])

def prepare():
    ps,provenance=pools()
    for domain in NAMES:
        out=ROOT.parent/NAMES[domain]
        if (out/'experiment_lock.json').exists():raise RuntimeError('already frozen: '+str(out))
        source=AUTHOR/'datasets/test'/('mmlu_test_n5.csv' if domain=='mmlu' else 'svamp_test_n300.json')
        if domain=='mmlu':
            test=[dict(question_id='mmlu:author_test:'+sha(norm(r['question']).encode()),question=r['question'],category=r['subject'].removesuffix('_test'),
                task=r['question']+'\n'+''.join(f'Option {c}: {r[c]}\n' for c in 'ABCD'),answer=r['correct_answer']) for r in csv.DictReader(source.open(encoding='utf-8-sig'))]
        else:test=[svamp_wrap(r,'author_test') for r in json.loads(source.read_text(encoding='utf-8'))]
        assert len(test)==(285 if domain=='mmlu' else 300)
        texts=[norm(r['question']) for r in test]
        excluded=[];eligible=[]
        for row in ps[domain]:
            q=norm(row['question'])
            matches=[i for i,t in enumerate(texts) if similar(q,t)]
            if matches:excluded.append(dict(question_id=row['question_id'],test_ids=[test[i]['question_id'] for i in matches],exact=q in texts))
            else:eligible.append(row)
        if domain=='mmlu':
            cats=sorted({r['category'] for r in eligible},key=rank)[:40]
            candidates=[r for cat in cats for r in sorted([x for x in eligible if x['category']==cat],key=lambda r:rank(r['question_id']))]
        else:
            # Also remove the same number-abstracted wording template. This is
            # conservative but cannot guarantee recovery of every source family.
            template=lambda t:re.sub(r'\d+(?:\.\d+)?','<num>',norm(t))
            test_templates={template(r['question']) for r in test}
            eligible=[r for r in eligible if template(r['question']) not in test_templates]
            candidates=sorted(eligible,key=lambda r:rank(r['question_id']))
        train=[];seen_cats=set()
        for row in candidates:
            if domain=='mmlu' and row['category'] in seen_cats:continue
            if any(similar(norm(row['question']),norm(r['question'])) for r in train):continue
            train.append(row);seen_cats.add(row['category'])
            if len(train)==40:break
        assert len(train)==40
        assert not any(similar(norm(r['question']),t) for r in train for t in texts)
        for split,part in [('train',train),('val',[]),('test',test)]:
            lines(out/'data'/f'{split}_inputs.jsonl',[{k:v for k,v in r.items() if k!='answer'} for r in part])
            lines(out/'data'/f'{split}_gold_LOCAL_ONLY.jsonl',[dict(question_id=r['question_id'],answer=r['answer']) for r in part])
        save(out/'split_manifest.json',dict(domain=domain,seed=42,train_count=40,val_count=0,test_count=len(test),
            selection='SHA256(42:category) first40 subjects, then SHA256(42:question_id) one per subject' if domain=='mmlu' else 'SHA256(42:question_id) first40 eligible; near-duplicate and numeric-template exclusions before selection',
            train_ids=[r['question_id'] for r in train],train_categories=sorted(seen_cats),test_ids=[r['question_id'] for r in test],
            author_test_sha256=sha(source.read_bytes()),source=provenance[domain],eligible_after_exclusions=len(eligible),
            normalized_exact_train_test_overlap=0,near_duplicate_threshold=.90,near_duplicate_method='difflib.SequenceMatcher normalized question, autojunk=False',
            train_test_near_overlap=0,exclusions=excluded,source_pool_sha256=sha(json.dumps(ps[domain],sort_keys=True,ensure_ascii=False).encode()),
            limitation='SVAMP source-family identities unavailable; wording/number-template filters do not guarantee semantic-family isolation' if domain=='svamp' else '40 train subjects cannot cover all57; untouched dev topics are not a new independent test'))
        save(out/'baseline_reuse.json',dict(status='no_exact_matching_artifact_found',searched=['current repository completed AQuA/HumanEval and active GSM8K/MMLU-Pro','historical other-method MMLU artifacts do not match this author MOC configuration'],reusable_conditions=[],rule='exact input/roles/DAG/prompt/model/decoder/merger/parser only'))
        print(domain,'train',len(train),'test',len(test),'eligible',len(eligible),'near/exact exclusions',len(excluded))

def bundle():
    base=json.loads((ROOT.parent/'moc_v21_gsm8k_fewshot_20261015/config.json').read_text())
    for domain,port in [('mmlu',11453),('svamp',11454)]:
        out=ROOT.parent/NAMES[domain];remote='/home/liuxin/ICML2027/experiments/'+out.name
        if (out/'experiment_lock.json').exists():raise RuntimeError('already frozen')
        for p in (OLD/'runtime').glob('*.py'):shutil.copyfile(p,out/p.name)
        ext=out/'coding_base/moc_adaptive_sampling';ext.mkdir(parents=True,exist_ok=True)
        for p in (REPO/'coding_base/moc_adaptive_sampling').glob('*.py'):shutil.copyfile(p,ext/p.name)
        shutil.copyfile(ROOT.parent/'moc_v21_official_test_20261014/group_utility_model.json',out/'frozen_aqua_model.json')
        # Native task interfaces only; no predictor, candidate or decoder change.
        p=out/'domain_runtime.py';s=p.read_text()
        s=s.replace('from datasets.data_process import GSM8KDataset,MMLUProDataset','from datasets.data_process import MMLUDataset,SVAMPDataset')
        s=s.replace("return GSM8KDataset if CONFIG['domain']=='gsm8k' else MMLUProDataset","return {'mmlu':MMLUDataset,'svamp':SVAMPDataset}[CONFIG['domain']]")
        s=s.replace("return dataset().record_to_target_answer({'answer':value})","return dataset().record_to_target_answer({('correct_answer' if CONFIG['domain']=='mmlu' else 'Answer'):value})")
        s=s.replace("CONFIG['domain']=='gsm8k'","CONFIG['domain']=='svamp'");p.write_text(s,encoding='utf-8')
        for n in ('runner.py','common.py','preflight.py'):
            p=out/n;s=p.read_text().replace("CONFIG['domain']=='mmlu_pro'","CONFIG['domain']=='mmlu'").replace("CONFIG['domain']=='gsm8k'","CONFIG['domain']=='svamp'")
            p.write_text(s,encoding='utf-8')
        p=out/'supervise.py';s=p.read_text().replace("            for arm in ('frozen_transfer','adapted'):stage('val',arm,services,deadline)\n",'')
        s=s.replace("for phase,arms in [('val',{'adapted','frozen_transfer'}),('test',{'adapted'} if adaptive else {'adapted','moc'})]:","for phase,arms in [('test',{'adapted'} if adaptive else {'adapted','moc'})]:")
        # Pin dispatch-selected UUIDs so separate launchers cannot race onto one card.
        s=s.replace("now=gpu.gpu_snapshot();stable=", "now=gpu.gpu_snapshot();allowed=CONFIG.get('allowed_gpu_uuids',[]);now=[g for g in now if not allowed or g['uuid'] in allowed];stable=")
        p.write_text(s,encoding='utf-8')
        roles=base['roles'] if domain=='svamp' else ['Knowlegable Expert','Critic','Mathematician','Psychologist','Historian','Doctor','Lawyer']
        c=dict(base,domain=domain,extension_directory=remote+'/coding_base',main_ports=[port],merger_port=11455,
            roles=roles,phase='approved_train40_no_val',train_questions=40,val_questions=0,test_questions=285 if domain=='mmlu' else 300,
            validation_conditions=0,conditions_total=(285 if domain=='mmlu' else 300)*4,test_conditions=(285 if domain=='mmlu' else 300)*4,
            user_authorization='40 samples each; no validation; train and author test at densities .3/.7; idle GPU only',
            allowed_gpu_uuids=[])
        if domain=='svamp':
            # The author's experiment explicitly maps SVAMP to GSM8K prompts.
            c['prompt_domain']='gsm8k'
            for name in ('runner.py','common.py','preflight.py'):
                p=out/name
                p.write_text(p.read_text().replace("PaperRecipe(domain=CONFIG['domain']", "PaperRecipe(domain=CONFIG['prompt_domain']"),encoding='utf-8')
        save(out/'config.json',c)
        for n in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl'):(out/'data'/n).write_text('')
        (out/'protocol.md').write_text(f'''# {domain}: approved train40, no validation

User approved forty training problems and direct author test; densities0.3/0.7. No validation or zero-shot transfer arm. Author test{c['test_questions']} inputs and original parser locked. Train-only answers can be on cloud; test gold stays local and is never a prompt. Data source/exclusions in split_manifest.json. SVAMP is a custom split of official full1000; MMLU dev40 covers40 of57 subjects.

Frozen v2.1 algorithm: MiniLM compact22 weighted ridge10/bootstrap32 seed42, empty/single/direct-edge pair <=2 messages, maxhop4, serialized1024-token budget, penalty.02. Two outcome-blind receiver episodes per training problem, fixed upstream/Vanilla downstream labels. No new features, training samples, algorithms or hyperparameters. Fit checkpoint frozen once before test, no refit or tuning from test. Excluded training conditions are reported, not replaced.

Author seven roles/seed42 DAG/prompts/final aggregation/native parser; MMLU AnalyzeAgent, SVAMP MathSolver, FinalRefer. OriginalMOC K2/ISM/five9B candidates. Existing Gemma2:27b and9b Q4_0, main temperature0/output2000/ctx8192, merger.1, num_gpu999, existingcuda_v13. Existing Landlock/seccomp separate-globals/locals executor5sec shared across arms. No CPU model fallback or new models/drivers.

New cohort at most two GPUs; old GSM/Pro cohort at most two, overlap total at most four. New adaptive runs one27B each; after both clean, serial originalMOC uses27B+9B. Ports11453/11454/11455 only. First launch only after twice-stable10sec main util<=15/free>=21000MiB; merger free>=10000MiB. Actual CUDA allocation/short generation/digest/fullVRAM required. No other users' process control. Resource queue before first supervisor launch has no run_start; thereafter8h absolute includes phase waiting, never reset/extend. Per dataset20000newcalls/16000000 actual+unknown upper+pending tokens, STOP dominates. Errors require unique full archive, cleanup and expense audit before resume; terminal ok/invalid/excluded retained; same-root recovery at most twice.

After full generation/cleanup, download allraw/trace/results/budget/service records/source locks and SHA, score local test gold using author parser. Report per-density pairedaccuracy, bootstrap20000seed42/McNemar, full main+merger logicaltokens including cachehits, actual training/warmup/discard/unknown costs, encoding/router time and support/empty/pair/hop coverage. No partial test metric used for decisions. No validation means no independent validation claim.
''',encoding='utf-8')
        (out/'task_plan.md').write_text('1. Frozen train40/test inputs and preflight.\n2. Await eligible GPU then training→fit→adapted test, no val.\n3. Both new adaptive jobs clean, serial originalMOC test.\n4. Local scoring/fullraw SHA/cost/coverage/resource audit.\n',encoding='utf-8')
        (out/'progress.md').write_text('Prepared; no model calls or GPU launch. Await CPU preflight and idle resources.\n',encoding='utf-8')
        sources=[*out.glob('*.py'),*ext.glob('*.py'),out/'config.json']
        data=[out/'split_manifest.json',out/'baseline_reuse.json',out/'frozen_aqua_model.json',*[out/'data'/f'{p}_inputs.jsonl' for p in ('train','val','test')],out/'data/train_gold_LOCAL_ONLY.jsonl']
        save(out/'experiment_lock.json',dict(source_hashes={p.relative_to(out).as_posix():sha(p.read_bytes()) for p in sorted(sources)},
            data_hashes={p.relative_to(out).as_posix():sha(p.read_bytes()) for p in sorted(data)},author_commit=c['author_commit'],
            local_gold_sha256={p:sha((out/'data'/f'{p}_gold_LOCAL_ONLY.jsonl').read_bytes()) for p in ('train','val','test')},
            author_test_sha256=json.loads((out/'split_manifest.json').read_text())['author_test_sha256']))
        package(out)

def package(out):
    lock=json.loads((out/'experiment_lock.json').read_text())
    names=set(lock['source_hashes'])|set(lock['data_hashes'])|{'experiment_lock.json','protocol.md','task_plan.md','progress.md'}
    names|={'data/'+n for n in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl')}
    with tarfile.open(out/'upload_LOCAL_ONLY.tar.gz','w:gz') as tar:
        for n in sorted(names):tar.add(out/n,arcname=n,recursive=False)
    save(out/'upload_manifest.json',{n:sha((out/n).read_bytes()) for n in sorted(names)})
    print(out.name,'upload_files',len(names),'sha',sha((out/'upload_LOCAL_ONLY.tar.gz').read_bytes()))

if __name__=='__main__':prepare();bundle()
