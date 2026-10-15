"""Prepare approved splits and immutable inputs; no model calls or gold upload."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
AUTHOR=REPO/'coding_base/MOC-official'
NAMES={'gsm8k':(24,120,300),'mmlu_pro':(14,56,280)}

def digest(value):return hashlib.sha256(value).hexdigest()
def normalized(text):return ' '.join(str(text).split()).casefold()
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MOC-split-audit'}),timeout=45) as f:return f.read()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values),encoding='utf-8')
def rank(row):return digest(('42:'+row['question_id']).encode())

def prepare():
    commit=json.loads(fetch('https://api.github.com/repos/openai/grade-school-math/commits?path=grade_school_math/data/train.jsonl&per_page=1'))[0]['sha']
    gsm_url=f'https://raw.githubusercontent.com/openai/grade-school-math/{commit}/grade_school_math/data/train.jsonl'
    gsm_raw=fetch(gsm_url)
    gsm=[dict(json.loads(x),official_index=i) for i,x in enumerate(gsm_raw.splitlines()) if x.strip()]
    revision=json.loads(fetch('https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro'))['sha']
    pro_url='https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FMMLU-Pro&config=default&split=validation&offset=0&length=100'
    pro=[x['row'] for x in json.loads(fetch(pro_url))['rows']]
    assert len(gsm)==7473 and len(pro)==70
    for domain,(nt,nv,ne) in NAMES.items():
        out=ROOT.parent/f'moc_v21_{domain}_fewshot_20261015'
        if (out/'experiment_lock.json').exists():raise RuntimeError('already frozen: '+str(out))
        original=AUTHOR/'datasets/test'/('gsm8k_test_n300.csv' if domain=='gsm8k' else 'mmlu_pro_test_n20.csv')
        test=list(csv.DictReader(original.open(encoding='utf-8')))
        pool=gsm if domain=='gsm8k' else pro
        def wrap(row,source):
            question=row['question']
            qid=(f'gsm8k:train:{row["official_index"]}' if domain=='gsm8k' and source!='author_test' else
                 f'{domain}:{source}:'+str(row.get('question_id',digest(normalized(question).encode()))))
            if domain=='gsm8k': task='Question: '+question+'\n'
            else:
                options=row.get('options') or [row.get(c,'') for c in 'ABCDEFGHIJ']
                task=question+'\n'+''.join(f'Option {c}: {v}\n' for c,v in zip('ABCDEFGHIJ',options))
                task+=''.join(f'Option {c}: nan\n' for c in 'ABCDEFGHIJ'[len(options):])
                if source=='author_test':
                    task=question+'\n'+''.join(f'Option {c}: {row[c] or "nan"}\n' for c in 'ABCDEFGHIJ')
            return dict(question_id=qid,task=task,question=question,category=row.get('category','math'),
                        source_id=row.get('question_id',row.get('official_index')),answer=row['answer'])
        all_pool=[wrap(r,'official_train' if domain=='gsm8k' else 'official_validation') for r in pool]
        test_rows=[wrap(r,'author_test') for r in test]
        test_text={normalized(r['question']) for r in test_rows}
        assert len(test_rows)==ne and not any(normalized(r['question']) in test_text for r in all_pool)
        if domain=='gsm8k':
            selected=sorted(all_pool,key=rank)[:nt+nv]; train=selected[:nt];val=selected[nt:]
        else:
            cats=sorted({r['category'] for r in all_pool});assert len(cats)==14
            train=[min([r for r in all_pool if r['category']==c],key=rank) for c in cats]
            chosen={r['question_id'] for r in train};val=[r for r in all_pool if r['question_id'] not in chosen]
        sets=[{normalized(r['question']) for r in part} for part in (train,val,test_rows)]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        assert (len(train),len(val),len(test_rows))==(nt,nv,ne)
        for split,part in [('train',train),('val',val),('test',test_rows)]:
            public=[{k:v for k,v in r.items() if k!='answer'} for r in part]
            gold=[dict(question_id=r['question_id'],answer=r['answer']) for r in part]
            lines(out/'data'/f'{split}_inputs.jsonl',public)
            lines(out/'data'/f'{split}_gold_LOCAL_ONLY.jsonl',gold)
        save(out/'split_manifest.json',dict(domain=domain,seed=42,selection='SHA256(42:question_id); Pro one per category',
            train=[r['question_id'] for r in train],val=[r['question_id'] for r in val],test=[r['question_id'] for r in test_rows],
            author_test_sha256=digest(original.read_bytes()),normalized_exact_overlap=0,
            official_source_url=gsm_url if domain=='gsm8k' else pro_url,
            official_source_revision=commit if domain=='gsm8k' else revision,
            official_pool_canonical_sha256=digest(json.dumps(pool,sort_keys=True,ensure_ascii=False).encode()),
            official_gsm_raw_sha256=digest(gsm_raw) if domain=='gsm8k' else None,
            pro_api_revision_note='API rows not revision-addressable; payload content hash frozen, revision provenance only' if domain=='mmlu_pro' else None))
        save(out/'baseline_reuse.json',dict(status='no_matching_MOC_artifact_found_in_current_repository',
            searched=['current repo experiments: AQuA and HumanEval only','workspace historical MOC experiments inventoried; different task/protocol'],
            reusable_conditions=[],rule='question/prompt/roles/DAG/model/decoding/merger/parser must match; fill missing only'))
        print(domain,len(train),len(val),len(test_rows),digest(original.read_bytes()))

if __name__=='__main__':prepare()
