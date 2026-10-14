"""Exercise real author imports/Graph code with local MiniLM, no LLM services."""

import asyncio
from contextlib import redirect_stdout
import ctypes
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/"config.json").read_text())
sys.path.insert(0,config["extension_directory"])
from moc_adaptive_sampling import SamplingConfig,LayerDependentSampler,MOCSamplingAdapter
from moc_adaptive_sampling.stateful import StatefulConfig,StatefulSampler,MarginalUtilityModel,FEATURE_NAMES
from moc_adaptive_sampling.features import LocalTokenizerCounter
from moc_adaptive_sampling.paper_baselines import load_author_api,PaperRecipe,make_graph,run_single_llm
from runtime_controls import stable_ids


async def check():
    cuda=ctypes.CDLL("libcuda.so.1")
    cuda_init=cuda.cuInit(0)
    count=ctypes.c_int()
    count_status=cuda.cuDeviceGetCount(ctypes.byref(count))
    assert cuda_init==0 and count_status==0 and count.value>0
    api=load_author_api(config["author_directory"])
    import ollama
    from src.llm.profile_embedding import get_sentence_embedding
    counter=LocalTokenizerCounter(config["tokenizer_path"],config["tokenizer_sha256"])
    encoder=get_sentence_embedding("real local MiniLM encoder")
    assert len(encoder)==384
    calls=[]
    def fake(*args,**kwargs):
        calls.append(kwargs["model"])
        return {"message":{"content":"A"},"prompt_eval_count":10,"eval_count":1,"done":True}
    async def fake_async(client,*args,**kwargs):
        return fake(*args,**kwargs)
    ollama.chat,ollama.AsyncClient.chat,ollama.list=fake,fake_async,lambda:{"models":[]}
    task={"task":"Question: What is 1+1?\nA)2\nB)3\nC)4\nD)5\nE)6"}
    recipe=PaperRecipe(edge_density=.3)
    with redirect_stdout(io.StringIO()),stable_ids():
        graph=make_graph(api,recipe,"adaptive_sampling")
    import numpy as np
    # This readiness test uses a declared constant-positive utility double;
    # the actual learned checkpoint is fitted only from controlled train pairs.
    size=len(FEATURE_NAMES)
    model=MarginalUtilityModel(np.zeros(size),np.ones(size),np.zeros(size),.05)
    adapter=MOCSamplingAdapter(graph,StatefulSampler(StatefulConfig(**config["candidate"]),counter,model),get_sentence_embedding)
    with redirect_stdout(io.StringIO()):
        result=await graph.arun(task,num_rounds=1,max_tries=1,max_time=60)
        single=await run_single_llm(api,recipe,task)
    assert len(calls)==9 and set(calls)=={config["main_model"]}
    assert adapter.ledger["valid"] and single["extracted_answer"]=="A"
    assert all(s["communication_tokens"]<=1024 for s in adapter.ledger["selections"])
    report={"status":"passed","cuInit":cuda_init,"cuDeviceGetCount":count.value,
            "python":sys.version,"real_minilm_dimension":len(encoder),"mock_llm_calls":len(calls),
            "actual_llm_calls":0,"utility_model":"READINESS_TEST_DOUBLE","source_commit":config["author_commit"],
            "actual_author_import_and_arun":True,"counter":counter.name,
            "maximum_candidate_hop":max((m["hop"] for s in adapter.ledger["selections"] for m in s["candidates"]),default=0)}
    (ROOT/"runtime_ready.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    asyncio.run(check())
