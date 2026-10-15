"""One-shot evidence-set decisions, aligned to initial-state supervision.

The learned signed expectation is a fixed-material downstream surrogate.
Bootstrap variability describes model stability; it is not a confidence bound
on expected utility and cannot authorize optimistic exploration or safe stops.
"""
from dataclasses import dataclass
import numpy as np

from .bundles import BundleConfig, evidence_groups, group_features, ordered
from .sampler import Selection, serialize_messages
from .stateful import count_context, structural_priors


@dataclass(frozen=True)
class ExpectedSetConfig(BundleConfig):
    communication_tokens: int = 1024
    max_group_size: int = 2

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.max_group_size <= 2:
            raise ValueError('initial-state training only covers singleton/pair sets')


class ExpectedSetSampler:
    """Argmax expected total set gain minus cost, including explicit empty=0.

    Each receiver chooses once among covered singleton/connected-pair actions.
    This is deliberately not an adaptive trajectory in selected-set space.
    Metadata support is a guard, not an out-of-distribution guarantee. An
    unsupported action is abstained from, never given an optimistic upper bound.
    """
    uses_selected_state = True  # adapter passes task/role state and receiver_role

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
        groups = evidence_groups(messages, predecessors, self.config.max_group_size)
        records, feasible = [], []
        for group in groups:
            x, info = group_features(receiver_sketch, [], group, messages, priors,
                                     predecessors, receiver_role, self.count_tokens,
                                     self.config.communication_tokens)
            if info['projected_tokens'] > self.config.communication_tokens:
                continue
            prediction = self.utility_model.predict_expected(x, info)
            required = ('help', 'harm', 'mean', 'supported')
            if any(k not in prediction for k in required):
                raise ValueError('incomplete expected utility prediction')
            if not all(np.isfinite(prediction[k]) for k in required[:3]) or not (
                0 <= prediction['help'] <= 1 and 0 <= prediction['harm'] <= 1 and
                prediction['help'] + prediction['harm'] <= 1 + 1e-10 and
                abs(prediction['mean'] - (prediction['help']-prediction['harm'])) <= 1e-10
            ):
                raise ValueError('invalid expected utility heads')
            penalty = self.config.cost_penalty * info['projected_tokens']/max(1, self.config.communication_tokens)
            record = dict(sources=list(group.sources), **info, **prediction,
                          cost_penalty=penalty, net_mean=prediction['mean']-penalty,
                          abstained=not bool(prediction['supported']))
            records.append(record)
            if prediction['supported']:
                feasible.append(record)
        state = dict(selected_before=[], scores=records, communication_tokens_before=0,
                     decision_scope='one-shot initial-state expected net group surrogate',
                     empty_net_mean=0., uncertainty_used_for_decision=False)
        if not records:
            reason = 'no_group_fits_budget'
        elif not feasible:
            reason = 'no_supported_expected_action'
        else:
            chosen = min(feasible, key=lambda r:(-r['net_mean'], r['projected_tokens'],
                         len(r['sources']), tuple(r['sources'])))
            if chosen['net_mean'] > 0:
                selected = [m for m in messages if m.source in chosen['sources']]
                cost = count_context(self.count_tokens, selected)
                if cost != chosen['projected_tokens'] or cost > self.config.communication_tokens:
                    raise ValueError('selected set cost mismatch')
                layers = [dict(action='read', hop=chosen['max_hop'], **state,
                               selection_mode='expected_net_set', read=[chosen])]
                return Selection([m.source for m in selected], serialize_messages(selected),
                                 cost, layers, 'one_evidence_set_selected')
            reason = 'expected_net_utility_nonpositive'
        return Selection([], '', 0, [dict(action='stop', hop=0, **state)], reason)
