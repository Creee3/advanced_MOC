"""Text-specific adaptation: original messages, hop pools, and a hard token cap."""

from dataclasses import dataclass, field
from typing import Callable
import hashlib
import numpy as np

from .kernels import ladies_probabilities, normalize_mass, weighted_without_replacement


@dataclass(frozen=True)
class SamplingConfig:
    max_hops: int = 4
    communication_tokens: int = 2048
    frontier_fanouts: tuple[int, ...] = (4, 3, 2, 1)
    hop_token_caps: tuple[int, ...] | None = None
    semantic_temperature: float = 1.0
    redundancy_weight: float = 0.5
    min_proxy_gain_per_token: float = 0.0
    local_state_weight: float = 0.25
    seed: int = 42

    def __post_init__(self):
        if self.max_hops < 1 or self.communication_tokens < 0:
            raise ValueError("invalid hop range or token budget")
        if len(self.frontier_fanouts) != self.max_hops or any(x < 0 for x in self.frontier_fanouts):
            raise ValueError("frontier_fanouts must specify a nonnegative cap per hop")
        if self.hop_token_caps is not None and (
            len(self.hop_token_caps) != self.max_hops or any(x < 0 for x in self.hop_token_caps)
        ):
            raise ValueError("hop_token_caps must specify a nonnegative cap per hop")
        if not np.isfinite(self.semantic_temperature) or self.semantic_temperature <= 0:
            raise ValueError("semantic_temperature must be positive and finite")
        if not 0 <= self.redundancy_weight <= 1:
            raise ValueError("redundancy_weight must be in [0,1]")
        if not np.isfinite(self.min_proxy_gain_per_token) or self.min_proxy_gain_per_token < 0:
            raise ValueError("proxy threshold must be nonnegative and finite")
        if not np.isfinite(self.local_state_weight) or self.local_state_weight < 0:
            raise ValueError("local_state_weight must be nonnegative and finite")


@dataclass(frozen=True)
class Message:
    source: str
    role: str
    hop: int
    author_order: int
    text: str
    sketch: np.ndarray = field(repr=False, compare=False)


@dataclass
class Selection:
    source_ids: list[str]
    context: str
    communication_tokens: int
    layers: list[dict]
    stop_reason: str


def serialize_messages(messages):
    """Exact original MOC *unmerged* field layout, in its order subsequence."""
    return "\n\n".join(
        f"Agent {m.source},role is {m.role},output is:\n{m.text}"
        for m in sorted(messages, key=lambda m: m.author_order)
    )


def math_communication(context):
    """Exact MathSolver summary envelope; included in the candidate token cap."""
    if not context:
        return ""
    return "At the same time, there are the following responses to the same question for your reference:\n\n" + context + "\n\n"


def unit_vector(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("a sketch must be a nonempty finite vector")
    norm = np.linalg.norm(value)
    return value / norm if norm else np.zeros_like(value)


class LayerDependentSampler:
    """LADIES frontier proposal with a frozen receiver-semantic reweighting.

    U is the incoming, row-normalized DAG support, row-selected by the current
    routing frontier and column-restricted to the next shortest-hop pool.
    p_i is proportional to LADIES(U)_i * exp(cos(q_j,z_i)/temperature).

    Routing-frontier sampling and reading a message are separate: a node whose
    body does not fit can still supply its cheap sketch for deeper expansion.
    Text admission uses a cosine novelty proxy / actual incremental tokens.
    The proxy is NOT a learned or calibrated estimate of task utility.
    """

    def __init__(self, config: SamplingConfig, count_tokens: Callable[[str], int]):
        self.config = config
        self.count_tokens = count_tokens

    def _tokens(self, selected):
        count = self.count_tokens(math_communication(serialize_messages(selected)))
        if not isinstance(count, int) or count < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return count

    def select(self, receiver_id, receiver_sketch, messages, predecessors, request_key=""):
        config = self.config
        q = unit_vector(receiver_sketch)
        messages = list(messages)
        if len({m.source for m in messages}) != len(messages):
            raise ValueError("each source must occur once at its shortest hop")
        by_id = {m.source: m for m in messages}
        z = {m.source: unit_vector(m.sketch) for m in messages}
        if any(v.shape != q.shape for v in z.values()):
            raise ValueError("receiver and sender sketches must share the same feature space")
        if any(not 1 <= m.hop <= config.max_hops for m in messages):
            raise ValueError("candidate hop outside configured range")
        # The adapter supplies task hash + stable receiver index, not shortuuid.
        seed_text = repr((config.seed, request_key or receiver_id))
        seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        selected = []
        frontier = [receiver_id]
        layers = []
        stop_reason = "max_hops"

        def proxy_gain(message):
            relevance = (float(q @ z[message.source]) + 1) / 2
            redundancy = max((max(0.0, float(z[message.source] @ z[m.source])) for m in selected), default=0.0)
            return max(0.0, relevance - config.redundancy_weight * redundancy)

        def admissible(message):
            current = self._tokens(selected)
            projected = self._tokens([*selected, message])
            increment = max(1, projected - current)
            gain = proxy_gain(message)
            return (
                projected <= config.communication_tokens
                and gain > 0
                and gain / increment > config.min_proxy_gain_per_token
            ), projected, gain

        for hop in range(1, config.max_hops + 1):
            # Cheap lookahead only. Never invoke the receiver LLM to decide a hop.
            future = [m for m in messages if m.hop >= hop and (
                config.hop_token_caps is None or config.hop_token_caps[m.hop - 1] > 0
            )]
            if not any(admissible(m)[0] for m in future):
                stop_reason = "no_remaining_proxy_gain_within_budget"
                break
            if not frontier:
                stop_reason = "empty_routing_frontier"
                break
            if config.frontier_fanouts[hop - 1] == 0:
                stop_reason = "zero_frontier_cap"
                break
            pool = [m for m in messages if m.hop == hop]
            if not pool:
                stop_reason = "no_hop_candidates"
                break
            support = np.zeros((len(frontier), len(pool)), dtype=np.float64)
            for row, parent in enumerate(frontier):
                neighbors = tuple(predecessors.get(parent, ()))
                for col, candidate in enumerate(pool):
                    if candidate.source in neighbors:
                        support[row, col] = 1.0 / max(1, len(neighbors))
            structural = ladies_probabilities(support)
            similarities = np.asarray([float(q @ z[m.source]) for m in pool])
            # Subtract max to keep exp stable, while leaving the same distribution.
            semantic = np.exp((similarities - similarities.max()) / config.semantic_temperature)
            proposal = normalize_mass(structural * semantic)
            drawn = weighted_without_replacement(proposal, config.frontier_fanouts[hop - 1], rng)
            frontier = [pool[int(i)].source for i in drawn]
            layer_start = self._tokens(selected)
            layer = {"hop": hop, "pool": [m.source for m in pool],
                     "ladies_probability": structural.tolist(), "proposal": proposal.tolist(),
                     "routing_frontier": frontier.copy(), "read": [], "rejected": []}
            # Draw a frontier first, then admit its bodies by marginal proxy/cost.
            pending = [by_id[node_id] for node_id in frontier]
            while pending:
                pending.sort(key=lambda m: proxy_gain(m) / max(1, self._tokens([*selected, m]) - self._tokens(selected)), reverse=True)
                message = pending.pop(0)
                valid, projected, gain = admissible(message)
                cap = config.hop_token_caps[hop - 1] if config.hop_token_caps is not None else config.communication_tokens
                if valid and projected - layer_start <= cap:
                    selected.append(message)
                    layer["read"].append({"source": message.source, "proxy_gain": gain,
                                          "communication_tokens_after": projected})
                else:
                    layer["rejected"].append(message.source)
            layers.append(layer)
        selected.sort(key=lambda m: m.author_order)
        return Selection([m.source for m in selected], serialize_messages(selected),
                         self._tokens(selected), layers, stop_reason)
