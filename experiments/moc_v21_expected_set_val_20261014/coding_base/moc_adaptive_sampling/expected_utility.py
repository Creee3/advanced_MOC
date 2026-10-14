"""One-shot expected downstream utility, with question-balanced ridge fitting.

The target is the recorded fixed-material downstream correctness change.  A
point estimate of that conditional expectation is distinct from a prediction
interval for a single noisy {-1, 0, 1} outcome.  Question bootstrap summaries
are descriptive model variation only; they never constitute a coverage claim,
an optimistic routing score, or a certificate that stopping is safe.
"""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .bundles import GROUP_EXTRA_NAMES, GROUP_FEATURE_NAMES
from .uncertainty import heads_to_gain


SCHEMA = "one_shot_expected_group_utility_v2_1"
COMPACT_FEATURE_NAMES = tuple(GROUP_EXTRA_NAMES) + (
    "member_mean_base_similarity", "member_mean_ladies_mass", "member_mean_same_role",
    "member_mean_receiver_solver", "member_mean_receiver_analyst",
    "member_mean_receiver_programmer", "member_mean_receiver_inspector",
    "member_max_base_similarity", "member_max_ladies_mass",
)
FEATURE_NAMES_BY_MODE = {"full": tuple(GROUP_FEATURE_NAMES), "compact": COMPACT_FEATURE_NAMES}
BOOTSTRAP_SCOPE = "descriptive question-bootstrap model variation; not a confidence or prediction interval"
WEIGHT_SCHEME = "equal question mass; equal episode mass within question; equal action mass within episode; sum=n_rows"


def _ids(values, name):
    result = list(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def hierarchical_sample_weights(question_ids, episode_ids):
    """Balance dependent actions inside episodes and dependent episodes in a question."""
    questions = _ids(question_ids, "question_ids")
    episodes = _ids(episode_ids, "episode_ids")
    if len(questions) != len(episodes):
        raise ValueError("question/episode lengths differ")
    action_counts = Counter(zip(questions, episodes))
    episodes_by_question = {}
    for question, episode in action_counts:
        episodes_by_question.setdefault(question, set()).add(episode)
    row_count, question_count = len(questions), len(episodes_by_question)
    return np.asarray([
        row_count / (question_count * len(episodes_by_question[question]) * action_counts[question, episode])
        for question, episode in zip(questions, episodes)
    ], dtype=float)


def _coverage_key(info):
    try:
        role = info["receiver_role"]
        values = [info[name] for name in ("max_hop", "group_size", "selected_count")]
    except (KeyError, TypeError) as error:
        raise ValueError("missing expected-utility support metadata") from error
    if not isinstance(role, str) or not role or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values
    ):
        raise ValueError("invalid expected-utility support metadata")
    hop, size, selected = map(int, values)
    if not 1 <= hop <= 4 or not 1 <= size <= 4 or selected < 0:
        raise ValueError("invalid expected-utility support metadata")
    return role, hop, size, selected


def _weighted_ridge(features, targets, sample_weights, regularization):
    """Solve weighted ridge with an unpenalized, jointly fitted intercept."""
    x, y, weights = (np.asarray(value, dtype=float) for value in (features, targets, sample_weights))
    if x.ndim != 2 or y.ndim != 2 or weights.shape != (len(x),) or len(x) != len(y):
        raise ValueError("invalid weighted ridge dimensions")
    if not all(np.isfinite(value).all() for value in (x, y, weights)) or (
        weights < 0
    ).any() or weights.sum() <= 0 or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("invalid weighted ridge values")
    x_mean, y_mean = np.average(x, axis=0, weights=weights), np.average(y, axis=0, weights=weights)
    centered_x, centered_y = x - x_mean, y - y_mean
    slopes = np.linalg.solve(
        centered_x.T @ (weights[:, None] * centered_x) + regularization * np.eye(x.shape[1]),
        centered_x.T @ (weights[:, None] * centered_y),
    )
    intercept = y_mean - x_mean @ slopes
    return slopes, intercept


class ExpectedGroupUtilityModel:
    """Deterministic expected-gain prediction for supported empty-selected states.

    Support is only an observed metadata stratum, not a semantic out-of-
    distribution detector or a guarantee about downstream MAS correctness.
    Callers must abstain on supported=False; no widened optimistic bounds exist.
    """

    def __init__(self, mean, scale, weights, intercepts, bootstrap_weights,
                 bootstrap_intercepts, coverage, *, mode="compact", metadata=None):
        if mode not in FEATURE_NAMES_BY_MODE:
            raise ValueError("mode must be full or compact")
        self.mode = mode
        self.feature_names = tuple(FEATURE_NAMES_BY_MODE[mode])
        self.feature_indices = tuple(GROUP_FEATURE_NAMES.index(name) for name in self.feature_names)
        self.mean, self.scale = np.array(mean, dtype=float, copy=True), np.array(scale, dtype=float, copy=True)
        self.weights, self.intercepts = np.array(weights, dtype=float, copy=True), np.array(intercepts, dtype=float, copy=True)
        self.bootstrap_weights = np.array(bootstrap_weights, dtype=float, copy=True)
        self.bootstrap_intercepts = np.array(bootstrap_intercepts, dtype=float, copy=True)
        width = len(self.feature_names)
        if self.mean.shape != (width,) or self.scale.shape != (width,) or self.weights.shape != (width, 2) or (
            self.intercepts.shape != (2,) or self.bootstrap_weights.ndim != 3 or
            self.bootstrap_weights.shape[1:] != (width, 2) or
            self.bootstrap_intercepts.shape != (len(self.bootstrap_weights), 2) or
            len(self.bootstrap_weights) < 8
        ):
            raise ValueError("expected-utility checkpoint dimensions mismatch")
        arrays = (self.mean, self.scale, self.weights, self.intercepts,
                  self.bootstrap_weights, self.bootstrap_intercepts)
        if not all(np.isfinite(value).all() for value in arrays) or (self.scale <= 0).any():
            raise ValueError("invalid expected-utility checkpoint parameters")
        self.coverage = set()
        for key in coverage:
            if len(key) != 4:
                raise ValueError("invalid checkpoint support key")
            validated = _coverage_key(dict(zip(("receiver_role", "max_hop", "group_size", "selected_count"), key)))
            if validated[-1] != 0:
                raise ValueError("one-shot checkpoint supports selected_count=0 only")
            self.coverage.add(validated)
        if not self.coverage:
            raise ValueError("empty expected-utility support metadata")
        self.metadata = copy.deepcopy(metadata or {})

    def _project(self, features):
        value = np.asarray(features, dtype=float)
        if value.shape != (len(GROUP_FEATURE_NAMES),) or not np.isfinite(value).all():
            raise ValueError("expected-utility input must contain the finite locked 77 features")
        return (value[list(self.feature_indices)] - self.mean) / self.scale

    def predict_expected(self, features, info):
        x = self._project(features)
        key = _coverage_key(info)
        supported = key[-1] == 0 and key in self.coverage
        heads, gain = heads_to_gain(x @ self.weights + self.intercepts)
        _, bootstrap_gains = heads_to_gain(
            np.einsum("f,bfh->bh", x, self.bootstrap_weights) + self.bootstrap_intercepts
        )
        quantiles = np.quantile(bootstrap_gains, [.025, .5, .975])
        return {
            "help": float(heads[0]), "harm": float(heads[1]), "mean": float(gain),
            "supported": supported, "abstain": not supported,
            "support_reason": "observed_empty_selected_metadata" if supported else "unsupported_metadata_abstain",
            "bootstrap_spread": float(bootstrap_gains.std()),
            "bootstrap_quantiles": {"q025": float(quantiles[0]), "q500": float(quantiles[1]), "q975": float(quantiles[2])},
            "bootstrap_scope": BOOTSTRAP_SCOPE,
            "prediction_scope": "expected fixed-material downstream correctness change; uncalibrated point estimate",
        }

    @classmethod
    def fit(cls, features, deltas, question_ids, infos, *, episode_ids, mode="compact",
            regularization=10., bootstrap_replicates=32, seed=42):
        if mode not in FEATURE_NAMES_BY_MODE:
            raise ValueError("mode must be full or compact")
        x, y = np.asarray(features, dtype=float), np.asarray(deltas, dtype=float)
        questions, episodes = _ids(question_ids, "question_ids"), _ids(episode_ids, "episode_ids")
        infos = list(infos)
        if x.ndim != 2 or x.shape[1] != len(GROUP_FEATURE_NAMES) or y.shape != (len(x),) or (
            len(questions) != len(x) or len(episodes) != len(x) or len(infos) != len(x)
        ):
            raise ValueError("invalid expected-utility training dimensions")
        if not np.isfinite(x).all() or not np.isin(y, [-1, 0, 1]).all() or len(set(questions)) < 5:
            raise ValueError("need finite features, ternary deltas, and >=5 training question groups")
        coverage = [_coverage_key(info) for info in infos]
        if any(key[-1] != 0 for key in coverage):
            raise ValueError("one-shot fit requires selected_count=0 for every action")
        if not np.isfinite(regularization) or regularization <= 0 or (
            isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, (int, np.integer)) or bootstrap_replicates < 8
        ):
            raise ValueError("invalid expected-utility fit configuration")
        indices = [GROUP_FEATURE_NAMES.index(name) for name in FEATURE_NAMES_BY_MODE[mode]]
        x = x[:, indices]
        row_weights = hierarchical_sample_weights(questions, episodes)
        mean = np.average(x, axis=0, weights=row_weights)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=row_weights))
        scale[scale < 1e-6] = 1.
        normalized = (x - mean) / scale
        target = np.stack((y > 0, y < 0), axis=1).astype(float)
        weights, intercepts = _weighted_ridge(normalized, target, row_weights, regularization)
        unique = sorted(set(questions))
        question_array = np.asarray(questions)
        rng = np.random.default_rng(seed)
        bootstrap_weights, bootstrap_intercepts = [], []
        for _ in range(bootstrap_replicates):
            counts = Counter(rng.choice(unique, size=len(unique), replace=True))
            replicate_weights = row_weights * np.asarray([counts[q] for q in question_array], dtype=float)
            # Each sampled occurrence retains the original equal question mass.
            # Explicit normalization locks ridge scale even if roundoff changes it.
            replicate_weights *= len(x) / replicate_weights.sum()
            slopes, intercept = _weighted_ridge(normalized, target, replicate_weights, regularization)
            bootstrap_weights.append(slopes)
            bootstrap_intercepts.append(intercept)
        metadata = {
            "fit_questions": unique, "fit_question_count": len(unique), "fit_rows": len(x),
            "fit_episode_count": len(set(zip(questions, episodes))),
            "target": "expected fixed-material downstream correctness change",
            "weight_scheme": WEIGHT_SCHEME, "sample_weight_sum": float(row_weights.sum()),
            "regularization": float(regularization), "seed": int(seed),
            "bootstrap_unit": "question", "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_scope": BOOTSTRAP_SCOPE,
            "primary_prediction": "deterministic all-fit weighted ridge; clip/renormalize help and harm",
            "standardization_scope": "weighted current fit partition only; reused by bootstrap replicates",
            "calibration_as_confidence_interval": False,
            "support_scope": "observed role/hop/group-size metadata with selected_count=0; abstain otherwise",
            "feature_mode": mode, "feature_selection": list(FEATURE_NAMES_BY_MODE[mode]),
            "limitations": "single noisy outcomes and dependent actions; no adaptive or full-MAS coverage guarantee",
        }
        return cls(mean, scale, weights, intercepts, bootstrap_weights, bootstrap_intercepts,
                   coverage, mode=mode, metadata=metadata)

    def _payload(self):
        return {
            "schema": SCHEMA, "input_feature_names": list(GROUP_FEATURE_NAMES),
            "feature_mode": self.mode, "feature_names": list(self.feature_names),
            "feature_indices": list(self.feature_indices), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "weights": self.weights.tolist(), "intercepts": self.intercepts.tolist(),
            "bootstrap_weights": self.bootstrap_weights.tolist(),
            "bootstrap_intercepts": self.bootstrap_intercepts.tolist(),
            "coverage": sorted(self.coverage), "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def _from_payload(cls, value):
        try:
            mode = value["feature_mode"]
            names = FEATURE_NAMES_BY_MODE[mode]
            indices = [GROUP_FEATURE_NAMES.index(name) for name in names]
            if value["schema"] != SCHEMA or value["input_feature_names"] != list(GROUP_FEATURE_NAMES) or (
                value["feature_names"] != list(names) or value["feature_indices"] != indices
            ):
                raise ValueError("expected-utility schema or feature selection mismatch")
            metadata = value["metadata"]
            if metadata.get("feature_mode") != mode or metadata.get("feature_selection") != list(names) or (
                metadata.get("calibration_as_confidence_interval") is not False or
                metadata.get("bootstrap_scope") != BOOTSTRAP_SCOPE
            ):
                raise ValueError("expected-utility metadata lock mismatch")
            return cls(value["mean"], value["scale"], value["weights"], value["intercepts"],
                       value["bootstrap_weights"], value["bootstrap_intercepts"], value["coverage"],
                       mode=mode, metadata=metadata)
        except (KeyError, TypeError) as error:
            raise ValueError("invalid expected-utility checkpoint") from error

    def clone(self):
        return self._from_payload(self._payload())

    def write(self, path):
        raw = (json.dumps(self._payload(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        Path(path).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path, expected_sha256=None):
        raw = Path(path).read_bytes()
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("expected-utility checkpoint hash mismatch")
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("invalid expected-utility checkpoint JSON") from error
        return cls._from_payload(value)
