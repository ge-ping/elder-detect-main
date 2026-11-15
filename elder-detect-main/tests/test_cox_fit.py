"""Landmark Cox 拟合器与研究预测的单元测试。"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal.cox_fit import fit_landmark_cox, _partial_likelihood  # noqa: E402
from longitudinal.survival import (  # noqa: E402
    CoxModelArtifact,
    predict_cumulative_risk,
    predict_cumulative_risk_research,
)


def brute_force(times, events, covariates, beta, ridge=0.0):
    linear = covariates @ beta
    loglik = 0.0
    gradient = -ridge * beta
    for index in np.flatnonzero(events):
        risk = np.flatnonzero(times >= times[index])
        weights = np.exp(linear[risk])
        total = weights.sum()
        loglik += linear[index] - np.log(total)
        gradient += covariates[index] - (
            weights[:, None] * covariates[risk]
        ).sum(axis=0) / total
    loglik -= 0.5 * ridge * float(beta @ beta)
    return loglik, gradient


class CoxFitTest(unittest.TestCase):
    def test_partial_likelihood_matches_bruteforce_with_ties(self):
        times = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
        events = np.array([1, 1, 0, 1, 0])
        covariates = np.array([[0.0], [1.0], [0.5], [-1.0], [2.0]])
        for beta in (np.array([0.0]), np.array([0.5]), np.array([-1.2])):
            loglik, gradient, _ = _partial_likelihood(
                times, events, covariates, beta, 0.0
            )
            expected_loglik, expected_gradient = brute_force(
                times, events, covariates, beta
            )
            self.assertAlmostEqual(loglik, expected_loglik, places=10)
            np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10)

    def test_fit_matches_scipy_optimiser(self):
        rng = np.random.default_rng(3)
        n = 120
        covariate = rng.normal(size=(n, 1))
        hazard = 0.05 * np.exp(0.8 * covariate[:, 0])
        times = np.minimum(rng.exponential(1.0 / hazard), 10.0)
        events = (times < 10.0).astype(int)
        fit = fit_landmark_cox(
            times, events, covariate, ["x"], horizon_days=10.0, ridge=0.0,
        )
        self.assertEqual(fit.status, "ok")
        standardized = (covariate - covariate.mean()) / covariate.std(ddof=0)
        objective = lambda beta: -_partial_likelihood(
            times, events, standardized, np.asarray(beta), 0.0
        )[0]
        solution = minimize(objective, np.zeros(1), method="BFGS", tol=1e-12)
        expected = float(solution.x[0] / covariate.std(ddof=0))
        self.assertAlmostEqual(fit.coefficients[0], expected, places=5)
        self.assertGreater(fit.coefficients[0], 0.0)

    def test_risk_is_monotone_and_reproducible(self):
        rng = np.random.default_rng(4)
        n = 80
        x = rng.normal(size=(n, 1))
        hazard = 0.08 * np.exp(1.0 * x[:, 0])
        times = np.minimum(rng.exponential(1.0 / hazard), 30.0)
        events = (times < 30.0).astype(int)
        fit = fit_landmark_cox(times, events, x, ["x"], horizon_days=30.0)
        self.assertAlmostEqual(fit.baseline_survival, fit.baseline_survival)
        artifact = CoxModelArtifact.from_dict({
            "schema_version": 2, "model_type": "landmark_cox",
            "bundle_id": "test", "horizon_days": 30,
            "feature_names": list(fit.feature_names),
            "coefficients": list(fit.coefficients),
            "feature_centers": list(fit.centers),
            "baseline_survival": fit.baseline_survival,
            "calibrated": False, "trained_at": "2024-01-01T00:00:00+00:00",
            "evidence_ids": ["test"],
        })
        risks = [
            predict_cumulative_risk_research({"x": value}, artifact)
            for value in (-2.0, 0.0, 2.0)
        ]
        self.assertTrue(all(0.0 <= risk <= 1.0 for risk in risks))
        self.assertEqual(risks, sorted(risks))
        with self.assertRaisesRegex(ValueError, "尚未校准"):
            predict_cumulative_risk({"x": 0.0}, artifact)

    def test_penalized_objective_contains_ridge_penalty(self):
        times = np.array([1.0, 2.0, 3.0, 3.0])
        events = np.array([1, 0, 1, 0])
        covariates = np.array([[-1.0], [0.0], [0.5], [1.5]])
        beta = np.array([0.7])
        ridge = 1e-4
        penalized, _, _ = _partial_likelihood(times, events, covariates, beta, ridge)
        unpenalized, _, _ = _partial_likelihood(times, events, covariates, beta, 0.0)
        self.assertAlmostEqual(
            unpenalized - penalized, 0.5 * ridge * float(beta @ beta), places=12
        )

    def test_finite_difference_matches_penalized_gradient(self):
        times = np.array([1.0, 2.0, 3.0, 3.0])
        events = np.array([1, 0, 1, 0])
        covariates = np.array([[-1.0], [0.0], [0.5], [1.5]])
        ridge = 1e-4
        epsilon = 1e-5
        for beta in (np.array([0.7]), np.array([-0.4])):
            value = lambda b: _partial_likelihood(
                times, events, covariates, np.asarray(b), ridge
            )[0]
            finite_difference = (
                value(beta + epsilon) - value(beta - epsilon)
            ) / (2 * epsilon)
            gradient = _partial_likelihood(
                times, events, covariates, beta, ridge
            )[1][0]
            self.assertAlmostEqual(finite_difference, gradient, places=7)

    def test_fit_reports_penalized_objective(self):
        rng = np.random.default_rng(9)
        n = 40
        x = rng.normal(size=(n, 1))
        times = np.minimum(rng.exponential(8.0, n), 30.0)
        events = (times < 30.0).astype(int)
        fit = fit_landmark_cox(times, events, x, ["x"], horizon_days=30.0, ridge=1e-4)
        self.assertEqual(fit.notes["objective"], "penalized_partial_log_likelihood")
        self.assertGreaterEqual(float(fit.notes["penalty_at_solution"]), 0.0)

    def test_no_events_and_constant_covariate_are_reported(self):
        times = np.array([1.0, 2.0, 3.0])
        covariates = np.array([[0.1], [0.2], [0.3]])
        no_events = fit_landmark_cox(
            times, np.zeros(3, dtype=int), covariates, ["x"], horizon_days=3.0,
        )
        self.assertEqual(no_events.status, "no_events")
        constant = fit_landmark_cox(
            times, np.array([1, 0, 1]), np.ones((3, 1)), ["x"], horizon_days=3.0,
        )
        self.assertEqual(constant.status, "no_active_features")


if __name__ == "__main__":
    unittest.main()
