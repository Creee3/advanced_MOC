import asyncio
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np

from moc_adaptive_sampling import MOCSamplingAdapter, Message
from moc_adaptive_sampling.stateful import (FEATURE_NAMES, MarginalUtilityModel, StatefulConfig,
    StatefulSampler, utility_features, structural_priors)
from source_harness import graph_fixture, fake_encode

def count(text):
    return len(text.split())

def messages():
    return [Message("near","Inspector",1,1,"near analysis",np.array([1.,0.])),
            Message("far","Math Solver",2,0,"independent evidence",np.array([0.,1.]))]

PREDS={"r":("near",),"near":("far",),"far":()}

class StatefulTests(unittest.TestCase):
    def test_selected_state_changes_residual_and_value(self):
        values=messages()
        priors,_=structural_priors("r",values,PREDS)
        x,_=utility_features([1.,1.],[],values[1],values,priors,"Inspector",count,100)
        y,_=utility_features([1.,1.],[values[0]],values[1],values,priors,"Inspector",count,100)
        self.assertGreater(y[FEATURE_NAMES.index("residual_similarity")],x[FEATURE_NAMES.index("residual_similarity")])
        self.assertLess(y[FEATURE_NAMES.index("remaining_budget_fraction")],x[FEATURE_NAMES.index("remaining_budget_fraction")])

    def test_early_stop_with_remaining_budget_and_no_fallback(self):
        class OneRead:
            def predict(self,x):
                return .1 if x[FEATURE_NAMES.index("is_first_read")] else -.1
        result=StatefulSampler(StatefulConfig(communication_tokens=500),count,OneRead()).select("r",[1.,0.],messages(),PREDS)
        self.assertEqual(len(result.source_ids),1)
        self.assertEqual(result.stop_reason,"predicted_net_gain_nonpositive")
        self.assertLess(result.communication_tokens,500)

    def test_remote_positive_can_be_read_without_near_body(self):
        class Remote:
            def predict(self,x):
                return .2 if x[FEATURE_NAMES.index("hop2")] else -.2
        result=StatefulSampler(StatefulConfig(),count,Remote()).select("r",[1.,0.],messages(),PREDS)
        self.assertEqual(result.source_ids,["far"])
        self.assertTrue(any(x["action"]=="expand" for x in result.layers))

    def test_budget_is_hard_and_checkpoint_tamper_fails(self):
        size=len(FEATURE_NAMES)
        model=MarginalUtilityModel(np.zeros(size),np.ones(size),np.zeros(size),.2)
        result=StatefulSampler(StatefulConfig(communication_tokens=0),count,model).select("r",[1.,0.],messages(),PREDS)
        self.assertEqual(result.communication_tokens,0)
        self.assertEqual(result.source_ids,[])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.json"
            model.write(path)
            with self.assertRaises(ValueError):
                MarginalUtilityModel.load(path,"incorrecthash")

    def test_fit_learns_negative_and_positive_effect_without_val(self):
        size=len(FEATURE_NAMES)
        x=np.zeros((100,size))
        x[:,0]=np.linspace(-1,1,100)
        y=np.where(x[:,0]>.4,1,np.where(x[:,0]<-.4,-1,0))
        model=MarginalUtilityModel.fit(x,y,10)
        self.assertGreater(model.predict(x[-1]),.4)
        self.assertLess(model.predict(x[0]),-.4)

    def test_real_author_path_all_stop_keeps_final_and_empty_context(self):
        graph,llm=graph_fixture()
        size=len(FEATURE_NAMES)
        model=MarginalUtilityModel(np.zeros(size),np.ones(size),np.zeros(size),-.1)
        adapter=MOCSamplingAdapter(graph,StatefulSampler(StatefulConfig(),count,model),fake_encode)
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(graph.arun({"task":"source-path check"},num_rounds=1))
        self.assertTrue(adapter.ledger["valid"])
        self.assertTrue(all(x["communication_tokens"]==0 for x in adapter.ledger["selections"]))
        ordinary=adapter.ledger["llm_calls"]
        self.assertEqual(sum(x["stage"]=="final" for x in ordinary),1)
        adapter.detach()

if __name__=="__main__":
    unittest.main()
