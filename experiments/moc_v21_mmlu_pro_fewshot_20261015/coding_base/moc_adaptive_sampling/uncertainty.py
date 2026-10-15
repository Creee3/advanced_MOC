"""Question-group bootstrap help/harm heads and held-out surrogate intervals.

Group-max residual calibration is intentionally conservative with few questions.
Bounds are for this fixed-material surrogate. Neither adaptive routing nor the
full MAS has a noninferiority/coverage guarantee from this calibration.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .bundles import GROUP_FEATURE_NAMES


def coverage_key(info):
    return (info['receiver_role'], info['max_hop'], info['group_size'], info['selected_count'])


def heads_to_gain(prediction):
    values = np.clip(np.asarray(prediction, dtype=float), 0., 1.)
    values /= np.maximum(1., values.sum(axis=-1, keepdims=True))
    return values, values[..., 0]-values[..., 1]


class GroupUtilityModel:
    def __init__(self, mean, scale, weights, intercepts, coverage, radius=None,
                 alpha=.1, metadata=None):
        self.mean, self.scale = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)
        self.weights, self.intercepts = np.asarray(weights, dtype=float), np.asarray(intercepts, dtype=float)
        width = len(GROUP_FEATURE_NAMES)
        if self.mean.shape!=(width,) or self.scale.shape!=(width,) or self.weights.ndim!=3 or (
            self.weights.shape[1:]!=(width,2) or self.intercepts.shape!=(len(self.weights),2)):
            raise ValueError('v2 checkpoint dimensions mismatch')
        if not all(np.isfinite(x).all() for x in (self.mean,self.scale,self.weights,self.intercepts)) or (self.scale<=0).any():
            raise ValueError('invalid checkpoint parameters')
        if not 0<alpha<1 or (radius is not None and (not np.isfinite(radius) or radius<0)):
            raise ValueError('invalid calibration interval')
        self.coverage = {tuple(x) for x in coverage}
        self.radius, self.alpha, self.metadata = radius, alpha, metadata or {}

    def _predict(self, x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(len(GROUP_FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError('invalid v2 features')
        raw=np.einsum('f,bfh->bh',(x-self.mean)/self.scale,self.weights)+self.intercepts
        heads,gains=heads_to_gain(raw)
        return heads.mean(axis=0), float(gains.mean()), float(gains.std())

    def predict_interval(self, features, info):
        heads,mean,spread=self._predict(features)
        supported = coverage_key(info) in self.coverage and self.radius is not None
        # Spread is descriptive epistemic variability, not a standard error CI.
        lower,upper = ((max(-1.,mean-self.radius),min(1.,mean+self.radius))
                        if supported else (-1.,1.))
        return {'help':float(heads[0]),'harm':float(heads[1]),'mean':mean,
                'lower':lower,'upper':upper,'supported':supported,
                'bootstrap_spread':spread,'interval_scope':'held-out fixed-material surrogate'}

    @classmethod
    def fit(cls, features, deltas, question_ids, infos, *, regularization=10.,
            bootstrap_replicates=32, seed=42):
        x,y=np.asarray(features,dtype=float),np.asarray(deltas,dtype=float)
        groups=np.asarray(question_ids)
        if x.ndim!=2 or x.shape[1]!=len(GROUP_FEATURE_NAMES) or len(x)!=len(y) or len(y)!=len(groups) or len(infos)!=len(y):
            raise ValueError('invalid training data')
        if not np.isfinite(x).all() or not np.isin(y,[-1,0,1]).all() or len(set(groups))<5:
            raise ValueError('need finite data and >=5 independent train question groups')
        if regularization<=0 or bootstrap_replicates<8:
            raise ValueError('invalid bootstrap configuration')
        mean,scale=x.mean(axis=0),x.std(axis=0)
        scale[scale<1e-6]=1.
        normalized=(x-mean)/scale
        target=np.stack((y>0,y<0),axis=1).astype(float)
        rng=np.random.default_rng(seed)
        unique=sorted(set(groups))
        weights,intercepts=[],[]
        for _ in range(bootstrap_replicates):
            drawn=rng.choice(unique,size=len(unique),replace=True)
            indices=np.concatenate([np.flatnonzero(groups==g) for g in drawn])
            xx,yy=normalized[indices],target[indices]
            intercept=yy.mean(axis=0)
            ww=np.linalg.solve(xx.T@xx+regularization*np.eye(x.shape[1]),xx.T@(yy-intercept))
            weights.append(ww); intercepts.append(intercept)
        return cls(mean,scale,weights,intercepts,[coverage_key(i) for i in infos],
                   metadata={'fit_questions':unique,'bootstrap_unit':'question','target':'downstream_correctness_change',
                             'regularization':regularization,'seed':seed})

    def calibrate(self, features, deltas, question_ids, infos, alpha=.1):
        if not 0<alpha<1 or len(features)!=len(deltas) or len(features)!=len(question_ids) or len(infos)!=len(features):
            raise ValueError('invalid calibration data')
        ids=set(question_ids)
        if ids & set(self.metadata.get('fit_questions',())):
            raise ValueError('train/calibration questions overlap')
        if not ids or not np.isin(deltas,[-1,0,1]).all():
            raise ValueError('invalid calibration labels')
        residuals={qid:[] for qid in ids}
        for x,y,qid in zip(features,deltas,question_ids):
            residuals[qid].append(abs(float(y)-self._predict(x)[1]))
        scores=sorted(max(values) for values in residuals.values())
        rank=int(np.ceil((len(scores)+1)*(1-alpha)))
        self.radius=scores[rank-1] if rank<=len(scores) else None
        self.alpha=alpha
        self.metadata.update(calibration_questions=sorted(ids), calibration_group_max_residuals=scores,
                             calibration_rank=rank, finite_calibration=self.radius is not None,
                             limitations='fixed-material grouped residual interval; no adaptive-MAS coverage claim')
        return self

    def write(self, path):
        value={'schema':'downstream_group_utility_v2','feature_names':list(GROUP_FEATURE_NAMES),
               'mean':self.mean.tolist(),'scale':self.scale.tolist(),'weights':self.weights.tolist(),
               'intercepts':self.intercepts.tolist(),'coverage':sorted(self.coverage),
               'radius':self.radius,'alpha':self.alpha,'metadata':self.metadata}
        Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def load(cls, path, expected_sha256=None):
        raw=Path(path).read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('v2 checkpoint hash mismatch')
        value=json.loads(raw)
        if value['schema']!='downstream_group_utility_v2' or value['feature_names']!=list(GROUP_FEATURE_NAMES):
            raise ValueError('v2 schema mismatch')
        return cls(value['mean'],value['scale'],value['weights'],value['intercepts'],
                   value['coverage'],value['radius'],value['alpha'],value['metadata'])
