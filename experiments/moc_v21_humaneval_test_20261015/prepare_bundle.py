"""Reproducible local preparation; never launch models or inspect test outcomes."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OLD=ROOT.parent/'moc_v21_official_test_20261014'
REMOTE='/home/liuxin/ICML2027/experiments/'+ROOT.name


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    if (ROOT/'status.json').exists() or (ROOT/'policy_lock.json').exists():
        raise RuntimeError('refuse to prepare over a locked or launched batch; use the committed exact snapshot')
    for name in ('common.py','merger_backend.py','resume_cache.py','runtime_controls.py','trace_hooks.py',
                 'supervise.py','gpu_runtime.py','probe_gpu_runtime.py','launch.py','inspect_run.py',
                 'group_utility_model.json'):
        shutil.copyfile(OLD/name,ROOT/name)
    folder=ROOT/'coding_base'/'moc_adaptive_sampling';folder.mkdir(parents=True,exist_ok=True)
    for path in (REPO/'coding_base/moc_adaptive_sampling').glob('*.py'):
        shutil.copyfile(path,folder/path.name)
    cfg=json.loads((OLD/'config.json').read_text())
    cfg.update(extension_directory=REMOTE+'/coding_base',phase='frozen_v21_humaneval_transfer_test',
               validation_conditions=656,test_conditions=656,domain='humaneval',code_timeout_seconds=10,
               final_test_timeout_seconds=100,max_new_calls=20000,max_new_model_tokens=12000000)
    cfg['gpu_policy']['max_wall_seconds']=28800
    write(ROOT/'config.json',cfg)
    (ROOT/'data').mkdir(exist_ok=True)
    source=REPO/'coding_base/MOC-official/datasets/test/humaneval_test_n164.jsonl'
    records=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert len(records)==len({r['name'] for r in records})==164
    (ROOT/'data/test_inputs.jsonl').write_text(''.join(json.dumps(dict(question_id=r['name'],prompt=r['prompt'],
        entry_point=r['entry_point']),ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    (ROOT/'data/test_targets_LOCAL_ONLY.jsonl').write_text(''.join(json.dumps(dict(question_id=r['name'],
        test=r['test'],entry_point=r['entry_point']),ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    for name in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl'):
        (ROOT/'data'/name).write_text('')
    write(ROOT/'dataset_provenance.json',dict(author_commit=cfg['author_commit'],source=str(source.relative_to(REPO)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=164,
        variant='MOC-supplied MultiPL-E transformed Python HumanEval; reworded terminology / transformed doctests',
        inputs='public prompts only; final test bodies remain local until all generation finishes'))
    roles=['Project Manager','Algorithm Designer','Programming Expert','Test Analyst','Bug Fixer','Project Manager','Algorithm Designer']
    write(ROOT/'code_role_snapshot.json',dict(roles=roles,origin='first seven roles from original HumanEvalPromptSet role cycle'))
    # The default MathSolver entry remains unchanged; HumanEval opts in explicitly.
    p=ROOT/'coding_base/moc_adaptive_sampling/paper_baselines.py'
    text=p.read_text().replace('("aqua", "gsm8k", "svamp")','("aqua", "gsm8k", "svamp", "humaneval")')
    text=text.replace('agent_names=["MathSolver"] * recipe.num_agents, decision_method="FinalRefer",',
        'agent_names=["CodeWriting" if recipe.domain == "humaneval" else "MathSolver"] * recipe.num_agents,\n                      decision_method="FinalWriteCode" if recipe.domain == "humaneval" else "FinalRefer",')
    p.write_text(text,encoding='utf-8')
    p=ROOT/'common.py';text=p.read_text()
    start=text.index('    @contextmanager\n    def controls(');end=text.index('\n\nclass AuthorReplay',start)
    text=text[:start]+'''    @contextmanager
    def controls(self, code_module, final_class):
        from code_runtime import code_observers
        with code_observers(code_module, final_class, self.recorder, self.counter):
            yield
'''+text[end:]
    text=text[:text.index('\n\nclass AuthorReplay')]+'''

def imports(host,directory,stage):
    backend=Backend(host,directory,stage)
    api=load_author_api(CONFIG['author_directory'])
    code_module=importlib.import_module('src.agents.code_writing')
    final_class=importlib.import_module('src.agents.final_decision').FinalWriteCode
    return api,backend,code_module,final_class
'''
    p.write_text(text,encoding='utf-8')
    p=ROOT/'launch.py';p.write_text(p.read_text().replace('official test254x2 arms x2 densities','HumanEval164 x2 arms x2 densities; zero-shot transfer'),encoding='utf-8')
    p=ROOT/'run_test.py';text=(OLD/'run_test.py').read_text()
    text=text.replace('all254 original test IDs','HumanEval164 public prompts; gold is not deployed to generation')
    text=text.replace("if row['status'] == 'ok' and row['main_calls'] != 8:","if row['status'] == 'ok' and not 1 <= row['main_calls'] <= 8:")
    text=text.replace('== 254','== 164').replace("ROOT/'val_role_snapshot.json'","ROOT/'code_role_snapshot.json'")
    text=text.replace("task = 'Question: '+row['question']+'\\n'+''.join(option+'\\n' for option in row['options'])","task = row['prompt']")
    text=text.replace('PaperRecipe(edge_density=density, use_cot=False)','PaperRecipe(domain="humaneval", edge_density=density, use_cot=False)')
    text=text.replace('counter_name=backend.counter.name)','counter_name=backend.counter.name, code_writing=True)')
    text=text.replace("if stats['main_calls'] != 8 or", "if not 1 <= stats['main_calls'] <= 8 or")
    text=text.replace("raise InvalidUnit('expected seven ordinary calls plus FinalRefer')", "raise InvalidUnit('expected up to seven ordinary calls plus FinalWriteCode')")
    text=text.replace("prompt = graph.decision_node.prompt_set\n                            answer = prompt.postprocess_answer(prompt, output[0][0])",'''from datasets.data_process import HumanEvalDataset
                            answer = HumanEvalDataset.postprocess_answer(None, output[0][0])''')
    text=text.replace("answer=answer, final_output=output[0][0], **stats,", "answer=answer, final_output=output[0][0], early_return_count=8-stats['main_calls'], **stats,")
    p.write_text(text,encoding='utf-8')
    print('Prepared frozen checkpoint, public inputs, isolated code-domain runtime. Run freeze_bundle.py after preflight development.')


if __name__=='__main__': main()
