import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from moc_adaptive_sampling.bundles import GROUP_EXTRA_NAMES, GROUP_FEATURE_NAMES
from moc_adaptive_sampling.expected_utility import (
    BOOTSTRAP_SCOPE, COMPACT_FEATURE_NAMES, ExpectedGroupUtilityModel,
    _weighted_ridge, hierarchical_sample_weights,
)


INFO = dict(receiver_role="Inspector", max_hop=1, group_size=1, selected_count=0)


def fixture():
    rng = np.random.default_rng(19)
    features = rng.normal(3., 2., size=(18, len(GROUP_FEATURE_NAMES)))
    questions = [f"q{i // 3}" for i in range(18)]
    episodes = [f"q{i // 3}-e{i % 3 // 2}" for i in range(18)]
    deltas = np.tile([-1, 0, 1], 6)
    return features, deltas, questions, [dict(INFO) for _ in questions], episodes


def fitted(mode="compact", seed=42):
    x, y, questions, infos, episodes = fixture()
    return ExpectedGroupUtilityModel.fit(x, y, questions, infos, episode_ids=episodes,
                                         mode=mode, bootstrap_replicates=8, seed=seed)


class ExpectedUtilityTests(unittest.TestCase):
    def test_weighted_intercept_satisfies_nonzero_design_normal_equations(self):
        x = np.array([[2., 1.], [3., 5.], [7., 2.], [8., 9.]])
        y = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 0.]])
        sample_weights = np.array([.3, 1.4, .8, 1.5])
        regularization = 10.
        slopes, intercept = _weighted_ridge(x, y, sample_weights, regularization)
        residual = y - (x @ slopes + intercept)
        np.testing.assert_allclose((sample_weights[:, None] * residual).sum(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(x.T @ (sample_weights[:, None] * residual),
                                   regularization * slopes, atol=1e-12)
        # The old intercept=weighted_y_mean formula violates these equations.
        self.assertGreater(np.linalg.norm(intercept - np.average(y, axis=0, weights=sample_weights)), .01)

    def test_questions_episodes_and_actions_receive_equal_nested_weight(self):
        questions = ["q0", "q0", "q0", "q1", "q1", "q1", "q1"]
        episodes = ["e0", "e0", "e1", "e2", "e2", "e2", "e2"]
        weights = hierarchical_sample_weights(questions, episodes)
        self.assertAlmostEqual(weights.sum(), len(questions))
        self.assertAlmostEqual(weights[:3].sum(), weights[3:].sum())
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        self.assertEqual(weights[0], weights[1])
        np.testing.assert_allclose(weights[3:], weights[3])
        # Episode identifiers need only be unique within their question.
        np.testing.assert_allclose(hierarchical_sample_weights(["q0", "q1"], ["same", "same"]), [1., 1.])

    def test_compact_feature_selection_is_locked_and_standardization_is_fit_only(self):
        model = fitted()
        expected = tuple(GROUP_EXTRA_NAMES) + (
            "member_mean_base_similarity", "member_mean_ladies_mass", "member_mean_same_role",
            "member_mean_receiver_solver", "member_mean_receiver_analyst",
            "member_mean_receiver_programmer", "member_mean_receiver_inspector",
            "member_max_base_similarity", "member_max_ladies_mass",
        )
        self.assertEqual(COMPACT_FEATURE_NAMES, expected)
        self.assertEqual(len(model.feature_names), 22)
        x, _, questions, _, episodes = fixture()
        weights = hierarchical_sample_weights(questions, episodes)
        np.testing.assert_allclose(model.mean, np.average(x[:, model.feature_indices], axis=0, weights=weights))
        before = model.mean.copy()
        model.predict_expected(np.full(len(GROUP_FEATURE_NAMES), 1e4), INFO)
        np.testing.assert_array_equal(model.mean, before)
        full = fitted("full")
        self.assertEqual(full.feature_names, tuple(GROUP_FEATURE_NAMES))
        self.assertEqual(full.weights.shape, (77, 2))

    def test_refuses_selected_states_in_training_and_invalid_labels(self):
        x, y, questions, infos, episodes = fixture()
        infos[0]["selected_count"] = 1
        with self.assertRaisesRegex(ValueError, "selected_count=0"):
            ExpectedGroupUtilityModel.fit(x, y, questions, infos, episode_ids=episodes)
        infos[0]["selected_count"] = 0
        y[0] = 2
        with self.assertRaises(ValueError):
            ExpectedGroupUtilityModel.fit(x, y, questions, infos, episode_ids=episodes)
        with self.assertRaises(ValueError):
            ExpectedGroupUtilityModel.fit(x[:9], np.zeros(9), questions[:9], infos[:9], episode_ids=episodes[:9])

    def test_unobserved_metadata_abstains_without_optimistic_bounds(self):
        model = fitted()
        x = fixture()[0][0]
        self.assertTrue(model.predict_expected(x, INFO)["supported"])
        for unsupported in (dict(INFO, selected_count=1), dict(INFO, receiver_role="Programming Expert"),
                            dict(INFO, max_hop=4), dict(INFO, group_size=2)):
            prediction = model.predict_expected(x, unsupported)
            self.assertFalse(prediction["supported"])
            self.assertTrue(prediction["abstain"])
            self.assertEqual(prediction["support_reason"], "unsupported_metadata_abstain")
            self.assertNotIn("upper", prediction)
            self.assertNotIn("lower", prediction)

    def test_bootstrap_spread_does_not_change_primary_prediction_or_claim_confidence(self):
        model = fitted()
        x = fixture()[0][0]
        before = model.predict_expected(x, INFO)
        # Alternate bootstrap draws may change descriptive variation, never the primary fit.
        alternate = fitted(seed=53).predict_expected(x, INFO)
        self.assertEqual(before["mean"], alternate["mean"])
        self.assertEqual(before["help"], alternate["help"])
        self.assertEqual(before["harm"], alternate["harm"])
        self.assertNotEqual(before["bootstrap_spread"], alternate["bootstrap_spread"])
        self.assertEqual(before["bootstrap_scope"], BOOTSTRAP_SCOPE)
        self.assertFalse(model.metadata["calibration_as_confidence_interval"])
        self.assertNotIn("interval_scope", before)
        self.assertNotIn("lower", before)
        self.assertNotIn("upper", before)
        self.assertFalse(hasattr(model, "calibrate"))

    def test_clone_checkpoint_hash_and_feature_tamper_detection(self):
        model = fitted()
        clone = model.clone()
        x = fixture()[0][0]
        self.assertEqual(clone.predict_expected(x, INFO), model.predict_expected(x, INFO))
        clone.weights[0, 0] += 1.
        self.assertFalse(np.array_equal(clone.weights, model.weights))
        clone.metadata["fit_questions"].append("different")
        self.assertNotIn("different", model.metadata["fit_questions"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            digest = model.write(path)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            restored = ExpectedGroupUtilityModel.load(path, expected_sha256=digest)
            self.assertEqual(restored.predict_expected(x, INFO), model.predict_expected(x, INFO))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                ExpectedGroupUtilityModel.load(path, expected_sha256="wrong")
            payload = json.loads(path.read_text())
            payload["feature_indices"][0] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "feature selection mismatch"):
                ExpectedGroupUtilityModel.load(path)

    def test_all_fit_primary_weighted_normal_equations(self):
        model = fitted()
        x, y, questions, _, episodes = fixture()
        xx = (x[:, model.feature_indices] - model.mean) / model.scale
        target = np.stack((y > 0, y < 0), axis=1).astype(float)
        weights = hierarchical_sample_weights(questions, episodes)
        residual = target - (xx @ model.weights + model.intercepts)
        np.testing.assert_allclose((weights[:, None] * residual).sum(axis=0), 0., atol=1e-11)
        np.testing.assert_allclose(xx.T @ (weights[:, None] * residual),
                                   10. * model.weights, atol=1e-11)
        self.assertEqual(model.metadata["fit_questions"], sorted(set(questions)))
        self.assertEqual(model.metadata["fit_episode_count"], len(set(zip(questions, episodes))))

    def test_bootstrap_question_multiplicity_matches_repeated_rows(self):
        rng = np.random.default_rng(7)
        questions = [f"q{question}" for question in range(5) for _ in range(question + 1)]
        episodes = [f"{question}-episode" for question in questions]
        x = rng.normal(4., 2., size=(len(questions), len(GROUP_FEATURE_NAMES)))
        y = np.asarray([[-1, 0, 1, 0, 1][int(question[1:])] for question in questions])
        model = ExpectedGroupUtilityModel.fit(x, y, questions, [INFO] * len(x),
                                              episode_ids=episodes, bootstrap_replicates=8, seed=42)
        # Reproduce the first question draw explicitly as repeated row indices.
        # In particular q2 and q3 are repeated, while q1 and q4 are absent.
        drawn = np.random.default_rng(42).choice(sorted(set(questions)), size=5, replace=True)
        self.assertLess(len(set(drawn)), len(drawn))
        indices = np.concatenate([np.flatnonzero(np.asarray(questions) == question) for question in drawn])
        weights = hierarchical_sample_weights(questions, episodes)[indices]
        weights *= len(x) / weights.sum()
        normalized = (x[:, model.feature_indices] - model.mean) / model.scale
        target = np.stack((y > 0, y < 0), axis=1).astype(float)
        slopes, intercept = _weighted_ridge(normalized[indices], target[indices], weights, 10.)
        np.testing.assert_allclose(model.bootstrap_weights[0], slopes, atol=1e-12)
        np.testing.assert_allclose(model.bootstrap_intercepts[0], intercept, atol=1e-12)
        # Collapsing repeated questions to a unique set would change this fit.
        unique_indices = np.concatenate([np.flatnonzero(np.asarray(questions) == question)
                                         for question in sorted(set(drawn))])
        wrong_weights = hierarchical_sample_weights(
            np.asarray(questions)[unique_indices], np.asarray(episodes)[unique_indices])
        wrong_weights *= len(x) / wrong_weights.sum()
        wrong_slopes, wrong_intercept = _weighted_ridge(
            normalized[unique_indices], target[unique_indices], wrong_weights, 10.)
        self.assertGreater(np.linalg.norm(slopes - wrong_slopes) + np.linalg.norm(intercept - wrong_intercept), .01)


if __name__ == "__main__":
    unittest.main()
