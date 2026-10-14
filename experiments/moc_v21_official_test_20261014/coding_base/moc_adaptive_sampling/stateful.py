"""Learned, selected-set-conditioned utility; no receiver LLM routing calls.

LADIES supplies a frontier-conditioned structural proposal, not text utility.
The local receiver-answer surrogate and its ridge fit are new task adapters.
No GNN inverse-probability or accuracy guarantee is asserted.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

from .kernels import ladies_probabilities
from .sampler import SamplingConfig, Selection, serialize_messages, math_communication, unit_vector


ROLES = ("Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector")
FEATURE_NAMES = (
    "base_similarity", "residual_similarity", "maximum_selected_similarity",
    "mean_selected_similarity", "selected_count", "remaining_budget_fraction",
    "incremental_cost_fraction", "body_cost_fraction", "ladies_mass", "local_similarity",
    "selected_relevance_mean", "residual_norm", "is_first_read",
    "hop1", "hop2", "hop3", "hop4",
    "receiver_solver", "receiver_analyst", "receiver_programmer", "receiver_inspector",
    "sender_solver", "sender_analyst", "sender_programmer", "sender_inspector",
    "relevance_x_redundancy", "relevance_x_selected_count", "relevance_x_cost",
    "residual_x_cost", "same_role", "proposal_x_relevance", "remaining_x_cost",
)


def count_context(counter, messages):
    result = counter(math_communication(serialize_messages(messages)))
    if not isinstance(result, int) or result < 0:
        raise ValueError("counter must return a nonnegative integer")
    return result


def structural_priors(receiver, messages, predecessors, max_hops=4):
    """Cheap routing-only expansion. Reading a body is never required to expand.

    All positive-mass frontier entries are retained in the small seven-node
    graph, avoiding v0's fixed fanout bottleneck. Larger graphs require an
    independently evaluated proposal-size cap; this version does not claim one.
    """
    frontier = [receiver]
    priors, layers = {}, []
    for hop in range(1, max_hops + 1):
        pool = [m for m in messages if m.hop == hop]
        if not pool or not frontier:
            break
        support = np.zeros((len(frontier), len(pool)), dtype=np.float64)
        for row, parent in enumerate(frontier):
            incoming = tuple(predecessors.get(parent, ()))
            for column, m in enumerate(pool):
                if m.source in incoming:
                    support[row, column] = 1 / max(1, len(incoming))
        mass = ladies_probabilities(support)
        frontier = [m.source for m, p in zip(pool, mass) if p > 0]
        priors.update((m.source, float(p)) for m, p in zip(pool, mass))
        layers.append({"hop": hop, "pool": [m.source for m in pool],
                       "ladies_probability": mass.tolist(), "routing_frontier": frontier.copy()})
    return priors, layers


def utility_features(base, selected, candidate, messages, priors, receiver_role, counter, budget):
    """Only task/role sketch, sender sketches and metadata; never answer/gold.

    A ridge-regularized projection removes the semantic span already selected.
    This residual is a routing proxy, not the receiver LLM's hidden state.
    """
    q = unit_vector(base)
    z = unit_vector(candidate.sketch)
    if z.shape != q.shape:
        raise ValueError("receiver and candidate sketches must share feature space")
    vectors = np.asarray([unit_vector(m.sketch) for m in selected])
    if len(selected):
        coefficients = np.linalg.solve(vectors @ vectors.T + .1 * np.eye(len(selected)), vectors @ q)
        residual = q - vectors.T @ coefficients
        similarities = vectors @ z
        maximum, mean = float(similarities.max()), float(similarities.mean())
        selected_relevance = float((vectors @ q).mean())
    else:
        residual = q.copy()
        maximum = mean = selected_relevance = 0.
    residual_norm = float(np.linalg.norm(residual))
    residual_similarity = float(unit_vector(residual) @ z)
    relevance = float(q @ z)
    current = count_context(counter, selected)
    projected = count_context(counter, [*selected, candidate])
    increment = max(1, projected - current)
    denom = max(1, budget)
    cost, remaining = increment / denom, max(0, budget - current) / denom
    local = [unit_vector(m.sketch) for m in messages if m.hop == 1]
    local_similarity = float(unit_vector(np.mean(local, axis=0)) @ z) if local else 0.
    prior = priors.get(candidate.source, 0.)
    values = [relevance, residual_similarity, maximum, mean, len(selected) / 6,
              remaining, cost, counter(candidate.text) / denom, prior, local_similarity,
              selected_relevance, residual_norm, float(not selected)]
    values += [float(candidate.hop == k) for k in range(1, 5)]
    values += [float(receiver_role == role) for role in ROLES]
    values += [float(candidate.role == role) for role in ROLES]
    values += [relevance * maximum, relevance * len(selected) / 6, relevance * cost,
               residual_similarity * cost, float(candidate.role == receiver_role),
               prior * relevance, remaining * cost]
    assert len(values) == len(FEATURE_NAMES)
    features = np.asarray(values, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("utility features must be finite")
    return features, {"incremental_tokens": increment, "projected_tokens": projected,
                      "residual_norm": residual_norm, "selected_count": len(selected)}


@dataclass(frozen=True)
class StatefulConfig(SamplingConfig):
    # These inherited v0 fanout/proxy fields are unused by v1; recorded for
    # compatibility only. There is no fixed per-hop number of bodies to read.
    cost_penalty: float = .02
    projection_ridge: float = .1

    def __post_init__(self):
        super().__post_init__()
        if self.max_hops != 4:
            raise ValueError("v1 feature schema is fixed to four-hop access")
        if not np.isfinite(self.cost_penalty) or self.cost_penalty < 0:
            raise ValueError("cost_penalty must be finite and nonnegative")
        if self.projection_ridge != .1:
            raise ValueError("fixed feature projection ridge is .1")


class MarginalUtilityModel:
    """Small regularized regression on paired {-1,0,+1} local-answer effects."""
    def __init__(self, mean, scale, weights, intercept, metadata=None):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.weights = np.asarray(weights)
        self.intercept = float(intercept)
        self.metadata = metadata or {}
        if any(x.shape != (len(FEATURE_NAMES),) for x in (self.mean, self.scale, self.weights)):
            raise ValueError("checkpoint feature dimensions mismatch")
        if not all(np.isfinite(x).all() for x in (self.mean, self.scale, self.weights)) or (self.scale <= 0).any():
            raise ValueError("invalid model parameters")

    def predict(self, features):
        return float(np.clip(((features - self.mean) / self.scale) @ self.weights + self.intercept, -1, 1))

    @classmethod
    def fit(cls, features, labels, regularization=10., metadata=None):
        x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(y) or len(x) < 2:
            raise ValueError("invalid training matrix")
        if not np.isin(y, [-1, 0, 1]).all():
            raise ValueError("labels must be paired local correctness differences")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-6] = 1.
        normalized = (x - mean) / scale
        intercept = float(y.mean())
        weights = np.linalg.solve(normalized.T @ normalized + regularization * np.eye(x.shape[1]),
                                  normalized.T @ (y - intercept))
        return cls(mean, scale, weights, intercept, metadata)

    def write(self, path):
        value = {"schema": "paired_local_utility_v1", "feature_names": list(FEATURE_NAMES),
                 "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                 "weights": self.weights.tolist(), "intercept": self.intercept,
                 "metadata": self.metadata}
        Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, expected_sha256=None):
        raw = Path(path).read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("utility checkpoint hash mismatch")
        value = json.loads(raw)
        if value["schema"] != "paired_local_utility_v1" or value["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("utility checkpoint schema mismatch")
        return cls(value["mean"], value["scale"], value["weights"], value["intercept"], value["metadata"])


class StatefulSampler:
    uses_selected_state = True

    def __init__(self, config, count_tokens, utility_model):
        self.config, self.count_tokens, self.utility_model = config, count_tokens, utility_model

    def select(self, receiver_id, receiver_sketch, messages, predecessors, request_key="", receiver_role=""):
        messages = list(messages)
        if len({m.source for m in messages}) != len(messages):
            raise ValueError("each ancestor must occur once")
        if any(not 1 <= m.hop <= self.config.max_hops for m in messages):
            raise ValueError("invalid candidate hop")
        priors, route_layers = structural_priors(receiver_id, messages, predecessors, self.config.max_hops)
        selected, layers = [], []
        reason = "candidates_exhausted" if messages else "no_candidates"
        exposed_hop = 1
        while True:
            pending = [m for m in messages if m not in selected and priors.get(m.source, 0.) > 0]
            scored = []
            for m in pending:
                x, info = utility_features(receiver_sketch, selected, m, messages, priors,
                                            receiver_role, self.count_tokens, self.config.communication_tokens)
                utility = self.utility_model.predict(x)
                net = utility - self.config.cost_penalty * info["incremental_tokens"] / max(1, self.config.communication_tokens)
                scored.append((m, utility, net, info))
            fitting = [item for item in scored if item[3]["projected_tokens"] <= self.config.communication_tokens]
            if not pending:
                reason = "candidates_exhausted" if messages else "no_candidates"
                break
            if not fitting:
                reason = "no_body_fits_remaining_budget"
                break
            # Cheap lookahead across all reachable sketches allows a positive
            # remote candidate to trigger expand even when nearby bodies are skipped.
            profitable = [item for item in fitting if item[2] > 0]
            state_record = {"selected_before": [m.source for m in selected],
                            "communication_tokens_before": count_context(self.count_tokens, selected),
                            "scores": [{"source": m.source, "hop": m.hop, "predicted_local_gain": u,
                                        "net_gain": n, **info} for m, u, n, info in scored]}
            if not profitable:
                reason = "predicted_net_gain_nonpositive"
                layers.append({"action": "stop", "hop": exposed_hop, **state_record})
                break
            best = max(profitable, key=lambda item: (item[2] / item[3]["incremental_tokens"], -item[0].author_order))
            m, utility, net, info = best
            if m.hop > exposed_hop:
                exposed_hop += 1
                route = next((x for x in route_layers if x["hop"] == exposed_hop), {})
                layers.append({"action": "expand", "hop": exposed_hop, "read": [],
                               "routing_frontier": route.get("routing_frontier", []), **state_record})
                continue
            selected.append(m)
            layers.append({"action": "read", "hop": m.hop, **state_record,
                           "read": [{"source": m.source, "predicted_local_gain": utility,
                                     "net_gain": net, "communication_tokens_after": info["projected_tokens"]}]})
        selected.sort(key=lambda m: m.author_order)
        return Selection([m.source for m in selected], serialize_messages(selected),
                         count_context(self.count_tokens, selected), layers, reason)
