"""
test_jacobian.py
=================

pytest-compatible tests for Theorem 3: Jacobian Selectivity Bound.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))

from verify import (
    relu,
    relu_derivative,
    sample_inputs_around_hyperplane,
    compute_neuron_stats,
    check_theorem_3_1_relu,
    check_theorem_3_2_smooth,
    check_theorem_3_3_layer,
)


# ---------------------------------------------------------------------------
# Theorem 3.1: ReLU single-neuron exact bound
# ---------------------------------------------------------------------------

class TestTheorem3_1:
    def test_exact_equality(self):
        """For ReLU, max Jacobian gap must equal ||w||_2 exactly."""
        res = check_theorem_3_1_relu(d=8, n_samples=100, trials=5)
        assert res.passed, f"min_ratio={res.metric:.4f}, expected >0.99"
        assert res.metric > 0.99, f"ratio too low: {res.metric}"

    def test_hyperplane_sampling(self):
        """Samples must straddle the hyperplane."""
        rng = np.random.default_rng(42)
        w = rng.normal(size=8)
        b = rng.normal()
        X_pos, X_neg = sample_inputs_around_hyperplane(w, b, 100, seed=42)
        # Verify classification
        z_pos = X_pos @ w + b
        z_neg = X_neg @ w + b
        assert np.all(z_pos > 0), f"Positive samples misclassified"
        assert np.all(z_neg <= 0), f"Negative samples misclassified"

    def test_neuron_stats_structure(self):
        """Stats fields must be populated correctly."""
        rng = np.random.default_rng(42)
        w = rng.normal(size=8)
        b = rng.normal()
        X_pos, X_neg = sample_inputs_around_hyperplane(w, b, 50, seed=42)
        X = np.vstack([X_pos, X_neg])
        stats = compute_neuron_stats(w, b, X)
        assert stats.w_norm > 0
        assert stats.selectivity >= 0
        assert stats.jacobian_gap >= 0
        assert 0.99 <= stats.ratio <= 1.01

    def test_small_dimension(self):
        """Test with d=2 for visualizability."""
        res = check_theorem_3_1_relu(d=2, n_samples=50, trials=3)
        assert res.passed

    def test_large_dimension(self):
        """Test with d=64."""
        res = check_theorem_3_1_relu(d=64, n_samples=50, trials=3)
        assert res.passed


# ---------------------------------------------------------------------------
# Theorem 3.2: Smooth activations
# ---------------------------------------------------------------------------

class TestTheorem3_2:
    @pytest.mark.parametrize("act", ["tanh", "sigmoid"])
    def test_positive_correlation(self, act):
        """Higher selectivity must correlate with higher Jacobian gap."""
        res = check_theorem_3_2_smooth(act, d=8, n_samples=100, trials=5)
        assert res.passed, f"correlation={res.metric:.3f}, expected >0.5"
        assert res.metric > 0.5, f"correlation too weak: {res.metric}"


# ---------------------------------------------------------------------------
# Theorem 3.3: Multi-neuron layer
# ---------------------------------------------------------------------------

class TestTheorem3_3:
    def test_frobenius_positive(self):
        """Frobenius gap must be positive when neurons toggle."""
        res = check_theorem_3_3_layer(m=8, d=8, n_samples=100)
        assert res.passed
        assert res.metric > 0, f"max_frob={res.metric}, expected >0"

    def test_more_neurons(self):
        """Test with m=16 neurons."""
        res = check_theorem_3_3_layer(m=16, d=8, n_samples=50)
        assert res.passed

    def test_tall_matrix(self):
        """Test with m > d (overcomplete layer)."""
        res = check_theorem_3_3_layer(m=16, d=4, n_samples=100)
        assert res.passed


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

class TestSanity:
    def test_relu_matches_numpy(self):
        """Our ReLU matches NumPy maximum."""
        z = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(relu(z), expected)

    def test_relu_derivative_correct(self):
        """Derivative is indicator of positivity."""
        z = np.array([-1, 0, 1])
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(relu_derivative(z), expected)

    def test_sampling_balance(self):
        """Equal numbers on each side."""
        rng = np.random.default_rng(42)
        w = rng.normal(size=8)
        b = rng.normal()
        X_pos, X_neg = sample_inputs_around_hyperplane(w, b, 100, seed=42)
        assert len(X_pos) == 50
        assert len(X_neg) == 50
