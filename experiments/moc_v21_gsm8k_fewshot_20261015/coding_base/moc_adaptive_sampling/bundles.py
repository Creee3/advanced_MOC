"""Receiver-conditioned evidence groups; no correctness information in routing.

Singletons remain legal. Connected ancestor paths add candidate groups, without
requiring that a remote message carry every intermediate body. GNN topology is
a proposal, not a certificate of semantic complementarity.
"""
from dataclasses import dataclass
import itertools
import numpy as np

from .sampler import Message, SamplingConfig, Selection, serialize_messages, unit_vector
from .stateful import FEATURE_NAMES, ROLES, count_context, structural_priors, utility_features


GROUP_EXTRA_NAMES = (
    'added_group_size', 'group_pair_cosine_mean', 'group_pair_cosine_min',
    'group_min_hop', 'group_max_hop', 'group_hop_span', 'group_is_connected_path',
    'group_incremental_cost_fraction', 'group_projected_cost_fraction',
    'group_solver_fraction', 'group_analyst_fraction', 'group_programmer_fraction',
    'group_inspector_fraction',
)
GROUP_FEATURE_NAMES = tuple('member_mean_' + n for n in FEATURE_NAMES) + tuple(
    'member_max_' + n for n in FEATURE_NAMES) + GROUP_EXTRA_NAMES


def ordered(messages):
    messages = list(messages)
    if len({m.source for m in messages}) != len(messages):
        raise ValueError('duplicate message source')
    return sorted(messages, key=lambda m: m.author_order)


@dataclass(frozen=True)
class EvidenceGroup:
    sources: tuple[str, ...]
    # sender->...->sender provenance path; receiver need not read the whole path.
    path: tuple[str, ...]


def evidence_groups(messages, predecessors, max_group_size=3):
    """All singletons and simple directed paths among available ancestor nodes.

    Seven-node graphs allow exact enumeration. No label-dependent proposal and
    no unbounded candidate generation for larger graphs are claimed.
    """
    messages = ordered(messages)
    if not 1 <= max_group_size <= 4 or len(messages) > 32:
        raise ValueError('supported scope: <=32 ancestors and group size 1..4')
    available = {m.source for m in messages}
    ranks = {m.source:m.author_order for m in messages}
    found = {}
    def visit(path):
        key = tuple(sorted(path, key=ranks.get))
        found.setdefault(key, EvidenceGroup(key, tuple(reversed(path))))
        if len(path) == max_group_size:
            return
        for source in predecessors.get(path[-1], ()):
            if source in available and source not in path:
                visit((*path, source))
    for message in messages:
        visit((message.source,))
    return sorted(found.values(), key=lambda g:(len(g.sources), tuple(ranks[s] for s in g.sources)))


def group_features(base, selected, group, messages, priors, predecessors,
                   receiver_role, counter, budget):
    inventory = {m.source:m for m in ordered(messages)}
    selected = ordered(selected)
    selected_ids = {m.source for m in selected}
    added = [inventory[s] for s in group.sources if s not in selected_ids]
    if not added:
        raise ValueError('group adds no evidence')
    augmented = ordered([*selected, *added])
    per_member = np.stack([utility_features(base, selected, m, messages, priors,
                            receiver_role, counter, budget)[0] for m in added])
    cosines = [float(unit_vector(a.sketch) @ unit_vector(b.sketch))
               for a,b in itertools.combinations(added, 2)]
    before = count_context(counter, selected)
    after = count_context(counter, augmented)
    increment = max(1, after-before)
    path = group.path
    connected = len(path)>1 and all(u in predecessors.get(v, ()) for u,v in zip(path, path[1:]))
    hops = [m.hop for m in added]
    denom = max(1, budget)
    extra = [len(added)/4, float(np.mean(cosines)) if cosines else 0., min(cosines) if cosines else 0.,
             min(hops)/4, max(hops)/4, (max(hops)-min(hops))/4, float(connected),
             increment/denom, after/denom]
    extra += [sum(m.role==role for m in added)/len(added) for role in ROLES]
    values = np.concatenate((per_member.mean(axis=0), per_member.max(axis=0), extra))
    if values.shape != (len(GROUP_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError('invalid group features')
    return values, {'incremental_tokens':increment, 'projected_tokens':after,
                    'added_sources':[m.source for m in added], 'path':list(group.path),
                    'is_connected_path':connected, 'max_hop':max(hops),
                    'receiver_role':receiver_role, 'group_size':len(added),
                    'selected_count':len(selected)}


@dataclass(frozen=True)
class BundleConfig(SamplingConfig):
    max_group_size: int = 3
    cost_penalty: float = .02
    local_state_weight: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if self.max_hops != 4 or not 1 <= self.max_group_size <= 4:
            raise ValueError('v2 schema uses four-hop access, group size 1..4')
        if not np.isfinite(self.cost_penalty) or self.cost_penalty < 0:
            raise ValueError('invalid cost penalty')


class BundleSampler:
    """Optimistic group selection with separate help/harm and stop bounds.

    prediction intervals concern a downstream surrogate, not final correctness
    guarantees. Unknown roles/hops/states cannot certify a nonpositive utility.
    Wide bounds may spend the entire budget; that behavior must be measured.
    """
    uses_selected_state = True

    def __init__(self, config, count_tokens, utility_model):
        self.config, self.count_tokens, self.utility_model = config, count_tokens, utility_model

    def select(self, receiver_id, receiver_sketch, messages, predecessors,
               request_key='', receiver_role=''):
        messages = ordered(messages)
        if any(not 1 <= m.hop <= self.config.max_hops for m in messages):
            raise ValueError('invalid ancestor hop')
        if not messages:
            return Selection([], '', 0, [], 'no_candidates')
        priors, _ = structural_priors(receiver_id, messages, predecessors, self.config.max_hops)
        # Include all available ancestor paths. Shortest-hop frontiers can miss
        # cross-hop paths with shortcuts; LADIES mass remains a feature only.
        groups = evidence_groups(messages, predecessors, self.config.max_group_size)
        selected, layers = [], []
        exposed_hop, reason = 1, 'candidates_exhausted'
        while True:
            selected_ids = {m.source for m in selected}
            records = []
            for group in groups:
                if set(group.sources) <= selected_ids:
                    continue
                x, info = group_features(receiver_sketch, selected, group, messages, priors,
                                         predecessors, receiver_role, self.count_tokens,
                                         self.config.communication_tokens)
                if info['projected_tokens'] > self.config.communication_tokens:
                    continue
                prediction = self.utility_model.predict_interval(x, info)
                required = ('help', 'harm', 'mean', 'lower', 'upper', 'supported')
                if any(k not in prediction for k in required):
                    raise ValueError('incomplete utility prediction')
                if not all(np.isfinite(prediction[k]) for k in required[:5]) or not (
                    -1 <= prediction['lower'] <= prediction['mean'] <= prediction['upper'] <= 1):
                    raise ValueError('invalid utility interval')
                if not prediction['supported']:
                    prediction = dict(prediction, lower=-1., upper=1.)
                penalty = self.config.cost_penalty * info['incremental_tokens']/max(1, self.config.communication_tokens)
                records.append(dict(sources=list(group.sources), **info, **prediction,
                                    net_mean=prediction['mean']-penalty,
                                    net_lower=prediction['lower']-penalty,
                                    net_upper=prediction['upper']-penalty))
            state = {'selected_before':[m.source for m in selected], 'scores':records,
                     'communication_tokens_before':count_context(self.count_tokens, selected)}
            if not records:
                reason = 'no_group_fits_remaining_budget'
                break
            promising = [r for r in records if r['net_upper'] > 0]
            if not promising:
                reason = 'surrogate_upper_bound_nonpositive'
                layers.append(dict(action='stop', hop=exposed_hop, **state))
                break
            certain = [r for r in promising if r['net_lower'] > 0]
            active = certain or promising
            key = 'net_mean' if certain else 'net_upper'
            chosen = max(active, key=lambda r:(r[key]/r['incremental_tokens'], -len(r['added_sources']),
                                               tuple(r['sources'])))
            while exposed_hop < chosen['max_hop']:
                exposed_hop += 1
                layers.append(dict(action='expand', hop=exposed_hop, read=[], **state))
            added = [m for m in messages if m.source in chosen['added_sources']]
            selected = ordered([*selected, *added])
            layers.append(dict(action='read', hop=chosen['max_hop'], **state,
                               selection_mode='positive_lower_bound' if certain else 'uncertain_optimistic',
                               read=[chosen]))
            if len(selected)==len(messages):
                reason='candidates_exhausted'
                break
        return Selection([m.source for m in selected], serialize_messages(selected),
                         count_context(self.count_tokens, selected), layers, reason)
