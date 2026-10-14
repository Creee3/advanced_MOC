"""Model-free true-author DAG/prompt/cache checks before occupying GPUs."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,CONFIG['extension_directory'])
from moc_adaptive_sampling.counterfactual import ExactCallCache,request_key
from moc_adaptive_sampling.paper_baselines import load_author_api,PaperRecipe,make_graph
from runtime_controls import stable_ids

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    # No backend calls: author constructors' metadata list is an offline double.
    import ollama
    ollama.list=lambda:{'models':[]}
    api=load_author_api(CONFIG['author_directory'])
    cache=ExactCallCache(CONFIG['main_digest'],None)
    admitted=cache.load_historical(rows(ROOT/'data/historical_calls_LOCAL_ONLY.jsonl'))
    materials=rows(ROOT/'data/fixed_materials_LOCAL_ONLY.jsonl')
    manifest=json.loads((ROOT/'training_manifest.json').read_text())
    assert len(materials)==47 and len(manifest['episodes'])==94
    graphs={}; matches=missing=0
    for material in materials:
        density=material['density']
        if density not in graphs:
            with stable_ids(), contextlib.redirect_stdout(io.StringIO()):
                graph=make_graph(api,PaperRecipe(edge_density=density,use_cot=False),'vanilla_mas',
                                 roles=[m['role'] for m in material['materials'].values()])
                assert not graph.optimized_spatial and all(float(x) in (0.,1.) for x in graph.spatial_masks)
                graph.spatial_logits=graph.spatial_masks*0  # unused under fixed binary masks
                graph.construct_spatial_connection()
            graphs[density]=graph
        graph=graphs[density]
        assert {i:[p.id for p in n.spatial_predecessors] for i,n in graph.nodes.items()}==material['predecessors']
        prompt_nodes=[*graph.nodes.values(),graph.decision_node]
        for node in prompt_nodes:
            if node is graph.decision_node:
                spatial={i:{'role':material['materials'][i]['role'],'output':material['materials'][i]['text']}
                         for i in material['baseline_usage']['final_order']}
            else:
                spatial={i:{'role':material['materials'][i]['role'],'output':material['materials'][i]['text']}
                         for i in material['predecessors'][node.id]}
            system,user=node._process_inputs({'task':material['task']},spatial,{},'')
            request=dict(model=CONFIG['main_model'],messages=[dict(role='system',content=system),dict(role='user',content=user)],
                         stream=False,options=dict(num_predict=2000,temperature=0,seed=42,num_ctx=8192,num_gpu=999))
            key=request_key(request,CONFIG['main_digest'])
            if key in cache.entries:
                matches+=1
            else:
                missing+=1
    assert matches+missing==47*8 and missing<=len(cache.historical_conflicts)
    report=dict(status='passed',true_author_dags_verified=len(graphs),frozen_materials=47,baseline_prompt_matches=matches,
                excluded_ambiguous_requests=missing,cache_admission=admitted,model_calls=0,gpu_allocations=0)
    (ROOT/'preflight_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    main()
