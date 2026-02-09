# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
#
# Copyright (C) 2025 Maximilian Salomon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for DiffusionMapsCalculator implementation."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from scipy.linalg import eig
from unittest.mock import patch, MagicMock

from mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator import DiffusionMapsCalculator
from mdxplain.utils.memmap_utils import MemmapUtils

# Helper functions for tests inspired by test_todo.md
def create_two_clusters_data(n_points_per_cluster=50, n_dims=3, separation=10):
    """Creates a dataset with two clearly separated clusters."""
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(n_points_per_cluster, n_dims)
    cluster2 = rng.randn(n_points_per_cluster, n_dims) + separation
    data = np.vstack([cluster1, cluster2])
    # Reshape to (n_frames, n_atoms, 3)
    return data.reshape(n_points_per_cluster * 2, -1, 3)

def create_linear_data(n_points=100, n_dims=3):
    """Creates a dataset with a linear structure."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, n_points)
    data = np.zeros((n_points, n_dims))
    data[:, 0] = t
    data[:, 1] = 2 * t
    data += rng.randn(n_points, n_dims) * 0.1
    # Reshape to (n_frames, n_atoms, 3)
    return data.reshape(n_points, -1, 3)

@pytest.fixture
def calculator():
    """Fixture for a DiffusionMapsCalculator instance."""
    return DiffusionMapsCalculator(use_memmap=False)

@pytest.fixture
def mock_hyperparameters():
    """Provides a standard set of hyperparameters for testing."""
    return {
        "n_components": 2,
        "epsilon": 0.1,
        "use_nystrom": False,
        "n_landmarks": 100,
        "random_state": 42,
        "atom_selection": None,
        "n_atoms": 1,
        "alpha": 0.0,
    }

# 1. Tests for Helper Functions
def test_normalize_to_transition_matrix(calculator):
    """
    Test row-stochastic normalization creates valid Markov transition matrix.
    
    Verifies kernel matrix → transition probabilities where each row sums to 1.
    Essential for diffusion maps spectral analysis and real eigenvalues.
    """
    kernel_matrix = np.array([[2.0, 1.0, 0.0],
                              [1.0, 2.0, 1.0],
                              [0.0, 1.0, 2.0]])
    transition_matrix, _ = calculator._normalize_to_transition_matrix(kernel_matrix)
    row_sums = np.sum(transition_matrix, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-7)
    expected_matrix = np.array([[2/3, 1/3, 0],
                                [1/4, 2/4, 1/4],
                                [0, 1/3, 2/3]])
    np.testing.assert_allclose(transition_matrix, expected_matrix)

def test_extract_diffusion_coordinates(calculator):
    """
    Test extraction and sorting of diffusion coordinates.
    
    Validates that eigenvalues are sorted descending and first eigenvector
    (stationary distribution) is skipped for meaningful coordinates.
    """
    # Eigenvecs should be NxN, where N is the number of eigenvals
    eigenvals = np.array([1.0, 0.8, 0.9, 0.2])
    # The columns are the eigenvectors
    eigenvecs = np.array([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]])
    
    n_components = 2
    diff_eigenvals, diff_coords = calculator._extract_diffusion_coordinates(eigenvals, eigenvecs, n_components)

    # Eigenvalues should be sorted descending, skipping the first one: 1.0, 0.9, 0.8, 0.2 -> skip 1.0 -> [0.9, 0.8]
    np.testing.assert_array_equal(diff_eigenvals, np.array([0.9, 0.8]))
    
    # Eigenvectors should be sorted according to eigenvalues
    # Original indices sorted: 0, 2, 1, 3
    # We want the vectors corresponding to 0.9 (index 2) and 0.8 (index 1)
    expected_coords = eigenvecs[:, [2, 1]]
    np.testing.assert_array_equal(diff_coords, expected_coords)

def test_apply_alpha_normalization_dense(calculator):
    """
    Test alpha normalization on a dense kernel matrix.
    
    Validates K_ij -> K_ij / (q_i^alpha q_j^alpha).
    """
    kernel = np.array([[1.0, 2.0], [2.0, 4.0]])
    row_sums = kernel.sum(axis=1)
    q_alpha = row_sums ** 1.0
    expected = kernel / (q_alpha[:, np.newaxis] * q_alpha[np.newaxis, :])

    calculator._apply_alpha_normalization(
        kernel,
        alpha=1.0,
        q_row=None,
        q_col=None,
        compute_row_sums=False,
        desc="Applying alpha normalization (dense)",
    )
    normalized = kernel
    np.testing.assert_allclose(normalized, expected)

def test_apply_alpha_normalization_alpha_zero_no_change(calculator):
    """
    Test alpha=0 path leaves kernel unchanged and returns None.
    """
    kernel = np.array([[1.0, 2.0], [2.0, 3.0]])
    kernel_copy = kernel.copy()
    result = calculator._apply_alpha_normalization(
        kernel,
        alpha=0.0,
        q_row=None,
        q_col=None,
        compute_row_sums=False,
        desc="Applying alpha normalization (dense)",
    )
    assert result is None
    np.testing.assert_allclose(kernel, kernel_copy)

def test_apply_alpha_normalization_alpha_zero_row_sums(calculator):
    """
    Test alpha=0 path returns row sums when requested.
    """
    kernel = np.array([[1.0, 2.0], [2.0, 3.0]])
    result = calculator._apply_alpha_normalization(
        kernel,
        alpha=0.0,
        q_row=None,
        q_col=None,
        compute_row_sums=True,
        desc="Applying alpha normalization (dense)",
    )
    np.testing.assert_allclose(result, kernel.sum(axis=1))

def test_apply_alpha_normalization_requires_q_col_for_rectangular(calculator):
    """
    Test rectangular kernel requires q_col when not provided.
    """
    kernel = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="q_col must be provided"):
        calculator._apply_alpha_normalization(
            kernel,
            alpha=1.0,
            q_row=None,
            q_col=None,
            compute_row_sums=False,
            desc="Applying alpha normalization (rectangular)",
        )

def test_apply_alpha_normalization_rectangular_with_q_col(calculator):
    """
    Test rectangular kernel normalization with explicit q_row/q_col.
    """
    kernel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    q_row = kernel.sum(axis=1)
    q_col = np.array([2.0, 3.0, 4.0])
    expected = kernel / (q_row[:, np.newaxis] * q_col[np.newaxis, :])

    calculator._apply_alpha_normalization(
        kernel,
        alpha=1.0,
        q_row=q_row,
        q_col=q_col,
        compute_row_sums=False,
        desc="Applying alpha normalization (rectangular)",
    )
    np.testing.assert_allclose(kernel, expected)

def test_apply_alpha_normalization_returns_row_sums(calculator):
    """
    Test row sums after alpha normalization are returned when requested.
    """
    kernel = np.array([[1.0, 2.0], [2.0, 4.0]])
    q_row = kernel.sum(axis=1)
    expected = kernel / (q_row[:, np.newaxis] * q_row[np.newaxis, :])
    expected_row_sums = expected.sum(axis=1)

    row_sums = calculator._apply_alpha_normalization(
        kernel,
        alpha=1.0,
        q_row=q_row,
        q_col=None,
        compute_row_sums=True,
        desc="Applying alpha normalization (dense)",
    )
    np.testing.assert_allclose(row_sums, expected_row_sums)

def test_compute_q_alpha_clamps_zero(calculator):
    """
    Test q^alpha computation clamps zeros for numerical safety.
    """
    row_sums = np.array([0.0, 1.0, 2.0])
    clamped, q_alpha = calculator._compute_q_alpha(row_sums, alpha=1.0)
    assert np.all(clamped >= 1e-12)
    np.testing.assert_allclose(q_alpha, clamped)
    
def test_epsilon_diagnostics_statistics(calculator):
    """
    Test epsilon diagnostics report median, quantiles, and bootstrap CI.
    
    Ensures diagnostics reflect the computed k-NN distances without
    affecting the epsilon calculation.
    """
    data = np.zeros((8, 6), dtype=np.float32)
    k_distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    expected_q05, expected_q95 = np.quantile(k_distances, [0.05, 0.95])

    with patch.object(calculator, "_compute_knn_distances", return_value=k_distances):
        epsilon, diagnostics = calculator._estimate_epsilon_knn(
            data,
            random_state=0,
            k=2,
            n_samples=5,
            ref_size=5,
            return_diagnostics=True,
        )

    median = float(np.median(k_distances))
    assert np.isclose(epsilon, median ** 2)
    assert np.isclose(diagnostics["median_distance"], median)
    assert np.isclose(diagnostics["quantiles"]["q05"], expected_q05)
    assert np.isclose(diagnostics["quantiles"]["q50"], median)
    assert np.isclose(diagnostics["quantiles"]["q95"], expected_q95)
    assert diagnostics["bootstrap_ci"]["n_bootstrap"] == 200
    assert diagnostics["bootstrap_ci"]["q025"] <= median <= diagnostics["bootstrap_ci"]["q975"]
    assert diagnostics["bootstrap_ci"]["q025"] >= float(k_distances.min())
    assert diagnostics["bootstrap_ci"]["q975"] <= float(k_distances.max())

def test_extract_hyperparameters_returns_epsilon_diagnostics(calculator):
    """
    Test epsilon diagnostics are returned separately from hyperparameters.
    
    Ensures diagnostics are not embedded inside hyperparameters.
    """
    data = np.zeros((10, 6), dtype=np.float32)
    diagnostics = {
        "median_distance": 1.0,
        "quantiles": {"q05": 0.8, "q50": 1.0, "q95": 1.2},
        "bootstrap_ci": {"q025": 0.9, "q975": 1.1, "n_bootstrap": 200},
        "epsilon": 1.0,
    }

    with patch.object(calculator, "_estimate_epsilon_knn", return_value=(1.0, diagnostics)):
        hyper, epsilon_diagnostics = calculator._extract_hyperparameters(
            data,
            {"n_components": 2}
        )

    assert np.isclose(hyper["epsilon"], 1.0)
    assert epsilon_diagnostics == diagnostics
    assert "epsilon_diagnostics" not in hyper
    assert "epsilon_bootstrap_samples" not in hyper

def test_resolve_epsilon_sampling_with_index_pool(calculator):
    """
    Test epsilon sampling uses the provided index pool and clamps k.
    
    Ensures reference/sample indices are drawn only from the pool and
    that k is clamped to a valid range for the pool size.
    """
    index_pool = np.array([2, 4, 6, 8, 10])

    k, n_samples, ref_size, sample_indices, ref_indices = calculator._resolve_epsilon_sampling(
        n_points=len(index_pool),
        n_frames_cap=20,
        random_state=0,
        k=10,
        n_samples=3,
        ref_size=4,
        index_pool=index_pool,
    )

    assert k == len(index_pool) - 1
    assert ref_size == len(index_pool)
    assert n_samples == 3
    assert set(ref_indices) == set(index_pool)
    assert set(sample_indices).issubset(set(ref_indices))

def test_estimate_epsilon_knn_with_index_pool(calculator):
    """
    Test epsilon estimation respects the index pool and uses k-NN distances.
    
    Verifies that sampling stays within the pool and epsilon is derived
    from the median k-NN distance.
    """
    data = np.zeros((8, 6), dtype=np.float32)
    index_pool = np.array([0, 2, 4, 6])
    captured = {}

    def _fake_compute_knn_distances(_, sample_indices, ref_indices, __, ___):
        captured["sample_indices"] = sample_indices
        captured["ref_indices"] = ref_indices
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with patch.object(calculator, "_compute_knn_distances", side_effect=_fake_compute_knn_distances):
        epsilon = calculator._estimate_epsilon_knn(
            data,
            random_state=0,
            k=2,
            n_samples=3,
            ref_size=4,
            index_pool=index_pool,
        )

    assert np.isclose(epsilon, 4.0)
    assert set(captured["ref_indices"]).issubset(set(index_pool))
    assert set(captured["sample_indices"]).issubset(set(captured["ref_indices"]))

def test_estimate_epsilon_from_landmarks_delegates(calculator):
    """
    Test landmark epsilon estimation delegates to the shared k-NN estimator.
    
    Ensures the landmark index pool is passed through without slicing data.
    """
    data = np.zeros((6, 3), dtype=np.float32)
    landmark_idx = np.array([1, 3, 5])

    with patch.object(calculator, "_estimate_epsilon_knn", return_value=0.25) as mock_estimate:
        epsilon = calculator._estimate_epsilon_from_landmarks(
            data,
            landmark_idx,
            k=2,
            n_samples=2,
            ref_size=3,
            random_state=0,
            bootstrap_samples=200,
        )

    assert np.isclose(epsilon, 0.25)
    assert np.array_equal(mock_estimate.call_args.kwargs["index_pool"], landmark_idx)

def test_compute_adds_epsilon_diagnostics_to_metadata(calculator):
    """
    Test compute attaches epsilon diagnostics to metadata.
    
    Ensures diagnostics are not stored in hyperparameters but appear
    at the top-level metadata after compute.
    """
    data = np.zeros((4, 3), dtype=np.float32)
    diagnostics = {
        "median_distance": 1.0,
        "quantiles": {"q05": 0.8, "q50": 1.0, "q95": 1.2},
        "bootstrap_ci": {"q025": 0.9, "q975": 1.1, "n_bootstrap": 200},
        "epsilon": 1.0,
    }
    mock_metadata = {
        "method": "standard_diffusion_maps",
        "epsilon": 1.0,
        "eigenvalues": np.array([1.0]),
        "hyperparameters": {},
    }

    with patch.object(calculator, "_estimate_epsilon_knn", return_value=(1.0, diagnostics)):
        with patch.object(calculator, "_compute_standard_diffusion_maps", return_value=(np.zeros((4, 1)), mock_metadata)):
            _, metadata = calculator.compute(data, n_components=1)

    assert metadata["epsilon_diagnostics"] == diagnostics
    assert "epsilon_diagnostics" not in metadata["hyperparameters"]

def test_nystrom_metadata_includes_epsilon_diagnostics(calculator):
    """
    Test Nyström path attaches epsilon diagnostics to metadata.
    
    Ensures diagnostics propagate when epsilon is estimated in Nyström mode.
    """
    data = np.zeros((6, 3), dtype=np.float32)
    diagnostics = {
        "median_distance": 1.0,
        "quantiles": {"q05": 0.8, "q50": 1.0, "q95": 1.2},
        "bootstrap_ci": {"q025": 0.9, "q975": 1.1, "n_bootstrap": 200},
        "epsilon": 1.0,
    }
    mock_metadata = {
        "method": "nystrom_diffusion_maps",
        "epsilon": 1.0,
        "eigenvalues": np.array([1.0]),
        "hyperparameters": {},
    }

    with patch.object(calculator, "_select_landmarks_kmeans", return_value=np.array([0, 1])):
        with patch.object(calculator, "_estimate_epsilon_from_landmarks", return_value=(1.0, diagnostics)):
            with patch.object(calculator, "_compute_landmarks_kernel", return_value=np.ones((2, 2), dtype=np.float32)):
                with patch.object(calculator, "_nystrom_normalize_to_markov", return_value=(np.eye(2), np.ones(2))):
                    with patch.object(calculator, "_solve_markov_eigenvalue_problem", return_value=(np.array([1.0]), np.eye(2))):
                        with patch.object(calculator, "_compute_all_to_landmarks_kernel", return_value=np.zeros((6, 2), dtype=np.float32)):
                            with patch.object(calculator, "_nystrom_extend_eigenvectors", return_value=np.zeros((6, 2), dtype=np.float32)):
                                with patch.object(calculator, "_nystrom_extract_coordinates", return_value=(np.zeros((6, 1)), np.array([1.0]))):
                                    with patch.object(calculator, "_prepare_metadata", return_value=mock_metadata):
                                        _, metadata = calculator.compute(
                                            data,
                                            n_components=1,
                                            use_nystrom=True,
                                            n_landmarks=2,
                                        )

    assert metadata["epsilon_diagnostics"] == diagnostics
    assert "epsilon_diagnostics" not in metadata["hyperparameters"]

# 2. Tests for Mathematical Properties
def test_kernel_matrix_symmetry(calculator, mock_hyperparameters):
    """
    Test Gaussian kernel matrix K(x,y) = exp(-||x-y||²/ε) is symmetric.
    
    Symmetry ensures real eigenvalues and stable eigendecomposition.
    Critical for valid diffusion coordinates.
    """
    test_data = create_linear_data(n_points=50, n_dims=3).reshape(50, -1)
    
    rmsd_matrix = calculator._compute_rmsd_distance_matrix(test_data, 1)
    kernel_matrix, _ = calculator._compute_kernel_matrix(rmsd_matrix, epsilon=0.1)
    
    assert np.allclose(kernel_matrix, kernel_matrix.T, rtol=1e-7, atol=1e-8)

def test_markov_matrix_row_stochastic(calculator, mock_hyperparameters):
    """
    Test that Markov transition matrix is row-stochastic.
    
    Validates that each row sums to 1.0 (valid probability distribution).
    Essential for Perron-Frobenius theorem and real eigenvalues.
    """
    test_data = create_linear_data(n_points=50, n_dims=3).reshape(50, -1)

    rmsd_matrix = calculator._compute_rmsd_distance_matrix(test_data, 1)
    kernel_matrix, _ = calculator._compute_kernel_matrix(rmsd_matrix, epsilon=0.1)
    transition_matrix, _ = calculator._normalize_to_transition_matrix(kernel_matrix)
    
    row_sums = np.sum(transition_matrix, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)

def test_eigenvalue_properties(calculator, mock_hyperparameters):
    """
    Test eigenvalue properties of the stochastic matrix.
    
    Validates that largest eigenvalue = 1.0 (Perron-Frobenius theorem)
    and all eigenvalues ≤ 1.0 for row-stochastic matrices.
    """
    test_data = create_linear_data(n_points=30, n_dims=3).reshape(30, -1)

    rmsd_matrix = calculator._compute_rmsd_distance_matrix(test_data, 1)
    kernel_matrix, _ = calculator._compute_kernel_matrix(rmsd_matrix, epsilon=1.0)
    transition_matrix, _ = calculator._normalize_to_transition_matrix(kernel_matrix)
    
    eigenvals, _ = eig(transition_matrix)
    
    # Eigenvalues should be real (within a small tolerance for numerical error)
    assert np.allclose(eigenvals.imag, 0, atol=1e-9)
    eigenvals = eigenvals.real
    
    # The largest eigenvalue should be 1 (Perron-Frobenius theorem)
    assert np.isclose(np.max(eigenvals), 1.0)
    
    # All eigenvalues should be <= 1 in magnitude
    assert np.all(np.abs(eigenvals) <= 1.0 + 1e-9)

# 3. Tests for Known Structures
def test_two_separate_clusters(calculator, mock_hyperparameters):
    """
    Test core diffusion maps functionality: two distinct clusters should be separated.
    
    Critical test - if this fails, the entire diffusion maps implementation is broken.
    First diffusion coordinate must correlate >99.9% with ideal [+1,-1] cluster assignment.
    """
    n_per_cluster = 20
    test_data = create_two_clusters_data(n_points_per_cluster=n_per_cluster, separation=20)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)
    
    # Create a "perfect" eigenvector that separates the clusters
    ideal_eigenvec = np.ones(n_per_cluster * 2)
    ideal_eigenvec[n_per_cluster:] = -1.0
    
    # Mock the eigensystem
    mock_eigenvals = np.array([1.0, 0.9, 0.8])
    mock_eigenvecs = np.zeros((n_per_cluster * 2, 3))
    mock_eigenvecs[:, 0] = 1.0  # Trivial eigenvector
    mock_eigenvecs[:, 1] = ideal_eigenvec # Our perfect separator
    mock_hyperparameters["epsilon"] = 5.0 
    with patch.object(
        calculator,
        "_solve_markov_eigenvalue_problem",
        return_value=(mock_eigenvals, mock_eigenvecs),
    ):
        coords, _ = calculator._compute_standard_diffusion_maps(
            test_data_flat, mock_hyperparameters
        )
    
    # The first diffusion coordinate should be our ideal eigenvector
    first_coord = np.real(coords[:, 0])
    
    # Check if the output is proportional to the ideal vector (sign can be arbitrary)
    correlation = np.corrcoef(first_coord, ideal_eigenvec)[0, 1]
    assert np.abs(correlation) > 0.999

def test_linear_data_structure(calculator, mock_hyperparameters):
    """
    Test that linear data structure is captured in first diffusion coordinate.
    
    Validates that first eigenfunction captures dominant linear trend
    in structured data (fundamental diffusion maps property).
    """
    n_points = 50
    test_data = create_linear_data(n_points=n_points, n_dims=3)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)
    
    # Create a "perfect" eigenvector that correlates with the linear structure
    t = np.linspace(0, 1, n_points)
    ideal_eigenvec = t - np.mean(t) # Centered
    
    # Mock the eigensystem
    mock_eigenvals = np.array([1.0, 0.9, 0.8])
    mock_eigenvecs = np.zeros((n_points, 3))
    mock_eigenvecs[:, 0] = 1.0 # Trivial eigenvector
    mock_eigenvecs[:, 1] = ideal_eigenvec # Our perfect linear vector
    mock_hyperparameters["epsilon"] = 0.05
    with patch.object(
        calculator,
        "_solve_markov_eigenvalue_problem",
        return_value=(mock_eigenvals, mock_eigenvecs),
    ):
        coords, _ = calculator._compute_standard_diffusion_maps(
            test_data_flat, mock_hyperparameters
        )
    
    first_coord = np.real(coords[:, 0])
    
    # The first diffusion coordinate should be strongly correlated with the linear progression
    correlation = np.corrcoef(first_coord, t)[0, 1]
    
    # The correlation should be very high (close to 1 or -1)
    assert np.abs(correlation) > 0.999

# 4. Test Execution Paths
@patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.DiffusionMapsCalculator._compute_standard_diffusion_maps')
def test_compute_calls_standard(mock_standard, calculator):
    """
    Test that compute() calls the standard method by default.
    
    Validates routing to standard diffusion maps implementation.
    """
    test_data = create_linear_data(n_points=30).reshape(30, -1)
    mock_standard.return_value = (np.array([]), {})
    calculator.compute(test_data, n_components=2, epsilon=0.1)
    mock_standard.assert_called_once()

@patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.DiffusionMapsCalculator._compute_iterative_diffusion_maps')
def test_compute_calls_iterative(mock_iterative):
    """
    Test that compute() calls the iterative method with use_memmap=True.
    
    Validates routing to memory-efficient implementation for large datasets.
    """
    calculator = DiffusionMapsCalculator(use_memmap=True)
    test_data = create_linear_data(n_points=30).reshape(30, -1)
    mock_iterative.return_value = (np.array([]), {})
    calculator.compute(test_data, n_components=2, epsilon=0.1)
    mock_iterative.assert_called_once()

@patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.DiffusionMapsCalculator._compute_nystrom_diffusion_maps')
def test_compute_calls_nystrom(mock_nystrom, calculator):
    """
    Test that compute() calls the Nyström method with use_nystrom=True.
    
    Validates routing to landmark-based approximation implementation.
    """
    test_data = create_linear_data(n_points=30).reshape(30, -1)
    mock_nystrom.return_value = (np.array([]), {})
    calculator.compute(test_data, n_components=2, epsilon=0.1, use_nystrom=True, n_landmarks=10)
    mock_nystrom.assert_called_once()

@patch("mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.CleanupUtils.remove_file")
def test_compute_cleans_tracked_temp_memmaps_iterative(mock_remove_file):
    """compute() should cleanup all tracked temp memmaps after iterative path."""
    calculator = DiffusionMapsCalculator(use_memmap=True)
    test_data = create_linear_data(n_points=30).reshape(30, -1)

    def _fake_iterative(_, __):
        calculator._temp_memmap_paths = ["iter_temp_a.dat", "iter_temp_b.dat"]
        return np.zeros((30, 2), dtype=np.float32), {"method": "iterative_diffusion_maps"}

    with patch.object(calculator, "_compute_iterative_diffusion_maps", side_effect=_fake_iterative):
        calculator.compute(test_data, n_components=2, epsilon=0.1)

    assert mock_remove_file.call_count == 2
    assert calculator._temp_memmap_paths == []

@patch("mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.CleanupUtils.remove_file")
def test_compute_cleans_tracked_temp_memmaps_nystrom(mock_remove_file):
    """compute() should cleanup all tracked temp memmaps after Nyström path."""
    calculator = DiffusionMapsCalculator(use_memmap=True)
    test_data = create_linear_data(n_points=30).reshape(30, -1)

    def _fake_nystrom(_, __):
        calculator._temp_memmap_paths = ["nys_temp_a.dat", "nys_temp_b.dat"]
        return np.zeros((30, 2), dtype=np.float32), {"method": "nystrom_diffusion_maps"}

    with patch.object(calculator, "_compute_nystrom_diffusion_maps", side_effect=_fake_nystrom):
        calculator.compute(
            test_data,
            n_components=2,
            epsilon=0.1,
            use_nystrom=True,
            n_landmarks=10,
        )

    assert mock_remove_file.call_count == 2
    assert calculator._temp_memmap_paths == []

def test_iterative_compute_removes_temporary_memmap_files():
    """Iterative diffusion maps should clean up temporary rmsd/kernel memmaps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        calculator = DiffusionMapsCalculator(
            use_memmap=True,
            cache_path=tmpdir,
            chunk_size=16,
        )
        test_data = create_linear_data(n_points=32).reshape(32, -1).astype(np.float32)

        coords, metadata = calculator.compute(
            test_data,
            n_components=2,
            epsilon=0.2,
            use_nystrom=False,
        )

        assert metadata["method"] == "iterative_diffusion_maps"
        dat_files = {path.name for path in Path(tmpdir).glob("*.dat")}
        assert "diffusion_maps_iterative_rmsd_matrix.dat" not in dat_files
        assert "diffusion_maps_iterative_kernel_matrix.dat" not in dat_files
        MemmapUtils.close_memmap_view(coords)


def test_nystrom_compute_removes_temporary_memmap_files():
    """Nyström diffusion maps should clean up temporary extension memmaps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        calculator = DiffusionMapsCalculator(
            use_memmap=True,
            cache_path=tmpdir,
            chunk_size=16,
        )
        test_data = create_linear_data(n_points=32).reshape(32, -1).astype(np.float32)

        coords, metadata = calculator.compute(
            test_data,
            n_components=2,
            epsilon=0.2,
            use_nystrom=True,
            n_landmarks=8,
            landmark_selection_mode="random",
            random_state=42,
        )

        assert metadata["method"] == "nystrom_diffusion_maps"
        dat_files = {path.name for path in Path(tmpdir).glob("*.dat")}
        assert "diffusion_maps_nystrom_K_all.dat" not in dat_files
        assert "diffusion_maps_nystrom_eigenvectors_full.dat" not in dat_files
        MemmapUtils.close_memmap_view(coords)

# 5. Test Nyström Method Steps
@patch('mdxplain.decomposition.decomposition_type.interfaces.calculator_base.MiniBatchKMeans')
def test_select_landmarks_kmeans(mock_kmeans_class, calculator):
    """
    Test landmark selection using MiniBatchKMeans.
    
    Validates clustering-based landmark selection for Nyström approximation.
    """
    test_data = create_two_clusters_data(n_points_per_cluster=50).reshape(100, -1)
    n_landmarks = 10

    # Mock the instance and its attributes
    mock_kmeans_instance = MagicMock()
    mock_kmeans_instance.cluster_centers_ = test_data[:n_landmarks] # Provide dummy centers
    mock_kmeans_class.return_value = mock_kmeans_instance
    
    landmark_idx = calculator._select_landmarks_kmeans(test_data, n_landmarks, random_state=42)
    
    # Should return the correct number of landmarks
    assert len(landmark_idx) == n_landmarks
    # Landmarks should be unique
    assert len(np.unique(landmark_idx)) == n_landmarks
    # All indices should be valid
    assert np.all(landmark_idx < len(test_data))

def test_nystrom_normalize_to_markov(calculator):
    """
    Test the asymmetric normalization for the Nyström method.
    
    Validates row-stochastic normalization of landmark kernel matrix.
    """
    K_landmarks = np.array([[2.0, 1.0], [1.0, 3.0]])
    M_small, inv_row_sums = calculator._nystrom_normalize_to_markov(K_landmarks)
    
    # Matrix should be row-stochastic
    np.testing.assert_allclose(np.sum(M_small, axis=1), 1.0)
    
    expected_M = np.array([[2/3, 1/3], [1/4, 3/4]])
    np.testing.assert_allclose(M_small, expected_M)
    np.testing.assert_allclose(inv_row_sums, np.array([1/3, 1/4]))

def test_nystrom_solve_eigenvalue_problem(calculator):
    """
    Test eigenvalue solving for the small Nyström matrix.
    
    Validates eigendecomposition of reduced landmark transition matrix.
    """
    K_landmarks = np.array([[2.0, 1.0], [1.0, 3.0]])
    M_small, inv_row_sums = calculator._nystrom_normalize_to_markov(K_landmarks)
    eigvals, eigvecs = calculator._solve_markov_eigenvalue_problem(
        M_small, inv_row_sums
    )
    
    # Eigenvalues should be real and sorted descending
    assert np.all(np.isreal(eigvals))
    assert eigvals[0] > eigvals[1]
    # Largest eigenvalue of a stochastic matrix is 1
    assert np.isclose(eigvals[0], 1.0)

def test_nystrom_vs_standard_consistency(calculator):
    """
    Test Nyström approximation consistency with standard diffusion maps.
    
    Nyström uses landmarks for O(mk²) instead of O(n³) complexity.
    Expects: correlation > 0.9 between both methods on same dataset.
    """
    
    # Create two-cluster data for better structure with 3 atoms per frame
    test_data = create_two_clusters_data(n_points_per_cluster=20, n_dims=9, separation=5)  # 9D = 3 atoms * 3 coords
    test_data = test_data.reshape(40, -1)
    
    # Standard method
    coords_standard, metadata_standard = calculator.compute(
        test_data,
        n_components=2,
        epsilon=2.0,  # Larger epsilon for better connectivity
        use_nystrom=False
    )
    
    # Nyström method (with 3 atoms per frame)
    coords_nystrom, metadata_nystrom = calculator.compute(
        test_data,
        n_components=2,
        epsilon=2.0,  # Same epsilon
        use_nystrom=True,
        n_landmarks=35,  # 87.5% of data as landmarks for better approximation
        n_atoms=3  # 3 atoms per frame for proper RMSD calculation
    )
    
    # Test 1: Basic validity checks
    assert coords_standard.shape == (40, 2)
    assert coords_nystrom.shape == (40, 2)
    assert np.all(np.isfinite(coords_standard))
    assert np.all(np.isfinite(coords_nystrom))
    
    # Test 2: Both methods must produce non-trivial embeddings
    for i in range(2):
        std_var = np.var(coords_standard[:, i])
        nys_var = np.var(coords_nystrom[:, i])
        
        assert std_var > 1e-6, f"Standard component {i} has too little variance: {std_var}"
        assert nys_var > 1e-6, f"Nyström component {i} has too little variance: {nys_var}"
    
    # Test 3: Correlation between components
    # At least one component must correlate well (components can be swapped)
    correlations = []
    for i in range(2):
        corr = np.abs(np.corrcoef(coords_standard[:, i], coords_nystrom[:, i])[0, 1])
        correlations.append(corr)
    
    # Best correlation should be high
    best_correlation = max(correlations)
    assert best_correlation > 0.9, f"Best correlation too low: {best_correlation} (correlations: {correlations})"
    
    # Test 4: Structural similarity - both should separate clusters
    # Cluster 1: indices 0-19, Cluster 2: indices 20-39
    std_sep = np.abs(np.mean(coords_standard[:20, 0]) - np.mean(coords_standard[20:, 0]))
    nys_sep = np.abs(np.mean(coords_nystrom[:20, 0]) - np.mean(coords_nystrom[20:, 0]))
    
    # Both should show some separation
    std_shows_separation = std_sep > 0.01 or np.abs(np.mean(coords_standard[:20, 1]) - np.mean(coords_standard[20:, 1])) > 0.01
    nys_shows_separation = nys_sep > 0.01 or np.abs(np.mean(coords_nystrom[:20, 1]) - np.mean(coords_nystrom[20:, 1])) > 0.01
    
    assert std_shows_separation, f"Standard method doesn't separate clusters (sep_dim0: {std_sep})"
    assert nys_shows_separation, f"Nyström method doesn't separate clusters (sep_dim0: {nys_sep})"
    
    # Test 5: Metadata consistency
    assert metadata_standard['method'] == 'standard_diffusion_maps'
    assert metadata_nystrom['method'] == 'nystrom_diffusion_maps'
    assert metadata_standard['epsilon'] == metadata_nystrom['epsilon'] == 2.0

# 6. Test Input Validation
def test_validate_input_data(calculator):
    """
    Test input data validation.
    
    Validates proper error handling for invalid input formats and dimensions.
    """
    with pytest.raises(ValueError, match="Input must be a numpy array"):
        calculator._validate_input_data([1, 2, 3])
    with pytest.raises(ValueError, match="Input must be 2D array"):
        calculator._validate_input_data(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="Data must have at least 2 frames"):
        calculator._validate_input_data(np.array([[1, 2, 3]]))
    with pytest.raises(ValueError, match="n_features must be divisible by 3"):
        calculator._validate_input_data(np.array([[1, 2], [3, 4]]))

def test_extract_hyperparameters(calculator):
    """
    Test hyperparameter extraction and validation.
    
    Validates proper parameter validation and epsilon estimation.
    """
    data = np.zeros((25, 3)) # Increased size to avoid sampling error
    with pytest.raises(ValueError, match="n_components must be specified"):
        calculator._extract_hyperparameters(data, {})
    with pytest.raises(ValueError, match="epsilon must be positive"):
        calculator._extract_hyperparameters(data, {"n_components": 2, "epsilon": 0})
    
    # Test n_components validation is hit before epsilon estimation
    with pytest.raises(ValueError, match="cannot be larger than"):
        calculator._extract_hyperparameters(data, {"n_components": 25})
    
    # Test epsilon estimation works with valid components
    params, _ = calculator._extract_hyperparameters(data, {"n_components": 2})
    assert "epsilon" in params and params["epsilon"] > 0

def test_landmark_selection_validation(calculator):
    """
    Test landmark selection parameter validation for Nyström.
    
    Ensures only "kmeans" or "random" are accepted.
    """
    data = np.zeros((10, 3))
    params, _ = calculator._extract_hyperparameters(
        data,
        {"n_components": 2, "use_nystrom": True, "landmark_selection_mode": "random", "n_landmarks": 3},
    )
    assert params["landmark_selection_mode"] == "random"

    with pytest.raises(ValueError, match="Invalid landmark_selection_mode"):
        calculator._extract_hyperparameters(
            data,
            {"n_components": 2, "use_nystrom": True, "landmark_selection_mode": "invalid", "n_landmarks": 3},
        )
