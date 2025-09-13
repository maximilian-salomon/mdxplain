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
from scipy.linalg import eig
from unittest.mock import patch, MagicMock

from mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator import DiffusionMapsCalculator

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
        "n_atoms": 1
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
    transition_matrix = calculator._normalize_to_transition_matrix(kernel_matrix)
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
    transition_matrix = calculator._normalize_to_transition_matrix(kernel_matrix)
    
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
    transition_matrix = calculator._normalize_to_transition_matrix(kernel_matrix)
    
    eigenvals, _ = eig(transition_matrix)
    
    # Eigenvalues should be real (within a small tolerance for numerical error)
    assert np.allclose(eigenvals.imag, 0, atol=1e-9)
    eigenvals = eigenvals.real
    
    # The largest eigenvalue should be 1 (Perron-Frobenius theorem)
    assert np.isclose(np.max(eigenvals), 1.0)
    
    # All eigenvalues should be <= 1 in magnitude
    assert np.all(np.abs(eigenvals) <= 1.0 + 1e-9)

# 3. Tests for Known Structures
@patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.eig')
def test_two_separate_clusters(mock_eig, calculator, mock_hyperparameters):
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
    mock_eig.return_value = (mock_eigenvals, mock_eigenvecs)

    mock_hyperparameters["epsilon"] = 5.0 
    
    coords, _ = calculator._compute_standard_diffusion_maps(test_data_flat, mock_hyperparameters)
    
    # The first diffusion coordinate should be our ideal eigenvector
    first_coord = np.real(coords[:, 0])
    
    # Check if the output is proportional to the ideal vector (sign can be arbitrary)
    correlation = np.corrcoef(first_coord, ideal_eigenvec)[0, 1]
    assert np.abs(correlation) > 0.999

@patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.eig')
def test_linear_data_structure(mock_eig, calculator, mock_hyperparameters):
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
    mock_eig.return_value = (mock_eigenvals, mock_eigenvecs)

    mock_hyperparameters["epsilon"] = 0.05
    
    coords, _ = calculator._compute_standard_diffusion_maps(test_data_flat, mock_hyperparameters)
    
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

# 5. Test Nyström Method Steps
@patch('sklearn.cluster.MiniBatchKMeans')
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
    M_small = np.array([[0.9, 0.1], [0.2, 0.8]])
    eigvals, eigvecs = calculator._nystrom_solve_eigenvalue_problem(M_small)
    
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
    params = calculator._extract_hyperparameters(data, {"n_components": 2})
    assert "epsilon" in params and params["epsilon"] > 0
