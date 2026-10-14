"""Verify frozen author graphs/cache/schema without a main-model call or GPU."""
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,CONFIG['extension_directory'])
from moc_adaptive_sampling.counterfactual import ExactCallCache
from moc_adaptive_sampling.paper_baselines import load_author_api,PaperRecipe,make_graph
from moc_adaptive_sampling.expected_utility import ExpectedGroupUtilityModel
from runtime_controls import stable_ids


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    lock=json.loads((ROOT/'policy_lock.json').read_text())
    model=ExpectedGroupUtilityModel.load(ROOT/'group_utility_model.json',lock['model_sha256'])
    assert model.mode=='compact' and model.metadata['fit_question_count']==35
    frozen=json.loads((ROOT/'experiment_lock.json').read_text())
    for name,digest in {**frozen['source_hashes'],**frozen['data_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    inputs=rows(ROOT/'data/val_inputs.jsonl')
    assert len(inputs)==60 and len({r['question_id'] for r in inputs})==60
    assert all(set(r)=={'question_id','question','options'} for r in inputs)
    import ollama
    ollama.list=lambda:{'models':[]}
    api=load_author_api(CONFIG['author_directory'])
    roles=json.loads((ROOT/'val_role_snapshot.json').read_text())['roles']
    expected=json.loads((ROOT/'expected_dags.json').read_text())
    verified=[]
    for density in (.3,.7):
        random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
        import torch
        torch.manual_seed(CONFIG['seed'])
        with stable_ids():
            graph=make_graph(api,PaperRecipe(edge_density=density,use_cot=False),'adaptive_sampling',roles=roles)
        assert not graph.optimized_spatial and all(float(x) in (0.,1.) for x in graph.spatial_masks)
        graph.spatial_logits=graph.spatial_masks*0
        graph.construct_spatial_connection()
        actual={i:[p.id for p in node.spatial_predecessors] for i,node in graph.nodes.items()}
        assert actual==expected[str(density)]['predecessors']
        assert [node.role for node in graph.nodes.values()]==roles
        verified.append({'density':density,'nodes':len(actual),'edges':sum(map(len,actual.values()))})
    cache=ExactCallCache(CONFIG['main_digest'],None)
    histories=[ROOT/'data'/name for name in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl')]
    admission=cache.load_historical([r for path in histories for r in rows(path)])
    report=dict(status='passed',model_schema=lock['model_schema'],fit_questions=35,
        train_gate_passed=False,user_authorized_diagnostic_val=True,graphs=verified,
        validation_conditions=120,cache_admission=admission,main_model_calls=0,gpu_allocations=0,
        val_gold_present=False,test=False)
    (ROOT/'preflight_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
