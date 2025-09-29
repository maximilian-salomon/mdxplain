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

"""Integration tests for decomposition algorithms."""

import numpy as np
import os
from unittest.mock import patch, MagicMock
from scipy.sparse.linalg import LinearOperator

from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.decomposition.decomposition_type.pca import PCA
from mdxplain.decomposition.decomposition_type.kernel_pca import KernelPCA
from mdxplain.decomposition.decomposition_type.diffusion_maps import DiffusionMaps
from mdxplain.decomposition.decomposition_type.contact_kernel_pca import ContactKernelPCA
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestDecompositionIntegration:
    """Test decomposition integration with pipeline."""
    
    def test_pca_decomposition_integration(self):
        """
        Test PCA decomposition integration with actual computation.

        Validates that PCA-Decomposition with linear coordinates mock
        correct Principal Components and explained variance computed.
        """
        
        # Setup pipeline
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=20, n_atoms=3, seed=42
        )
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add features for decomposition
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("pca_input")
        pipeline.feature_selector.add_selection("pca_input", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("pca_input")
        
        # Apply PCA with specific parameters
        test_n_components = 3
        test_random_state = 42
        pipeline.decomposition.add.pca(
            selection_name="pca_input",
            n_components=test_n_components, 
            random_state=test_random_state
        )
        
        # Verify decomposition data was computed and stored correctly
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]  # Latest decomposition
        
        # Check shape and data validity
        assert decomp_data.data is not None
        assert decomp_data.data.shape == (20, test_n_components)
        assert np.all(np.isfinite(decomp_data.data))
        
        # Verify metadata contains correct parameters and results
        assert decomp_data.decomposition_type == "pca"
        assert "decomposition_name" in decomp_data.metadata
        # Check parameters in hyperparameters section
        assert "hyperparameters" in decomp_data.metadata
        hyperparams = decomp_data.metadata["hyperparameters"]
        assert hyperparams["n_components"] == test_n_components
        assert hyperparams["random_state"] == test_random_state
        # Check explained variance data exists and is valid
        assert "explained_variance_ratio" in decomp_data.metadata
        explained_var = decomp_data.metadata["explained_variance_ratio"]
        assert len(explained_var) == test_n_components
        assert all(ratio >= 0 for ratio in explained_var)
        assert all(ratio <= 1.0 + 1e-10 for ratio in explained_var)
    
    def test_kernel_pca_decomposition_integration(self):
        """
        Test Kernel PCA decomposition integration with actual computation.

        Validates that Kernel PCA with RBF kernel and gamma Parameter
        correct nonlinear dimensionality reduction performs.
        """
        
        # Setup pipeline
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=15, n_atoms=4, seed=123
        )
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"},
            {"seqid": 3, "full_name": "RES_3"}
        ]}
        
        # Add coordinates features for kernel PCA
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("kpca_input")
        pipeline.feature_selector.add_selection("kpca_input", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("kpca_input")
        
        # Apply Kernel PCA with specific parameters
        test_n_components = 2
        test_gamma = 0.1
        test_random_state = 42

        pipeline.decomposition.add.kernel_pca(
            selection_name="kpca_input",
            n_components=test_n_components,
            gamma=test_gamma,
            random_state=test_random_state,
        )
        
        # Verify decomposition data was computed and stored correctly
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]  # Latest decomposition
        
        # Check shape and data validity
        assert decomp_data.data is not None
        assert decomp_data.data.shape == (15, test_n_components)
        assert np.all(np.isfinite(decomp_data.data))
        
        # Verify metadata contains correct parameters
        assert decomp_data.decomposition_type == "kernel_pca"
        assert "decomposition_name" in decomp_data.metadata
        # Check parameters in hyperparameters section
        assert "hyperparameters" in decomp_data.metadata
        hyperparams = decomp_data.metadata["hyperparameters"]
        assert hyperparams["n_components"] == test_n_components
        assert hyperparams["gamma"] == test_gamma
        assert hyperparams["random_state"] == test_random_state
        
    def test_diffusion_maps_decomposition_integration(self):
        """
        Test Diffusion Maps decomposition integration with concrete parameter validation.

        Validates that Diffusion Maps with epsilon Parameter and memmap
        correct spectral embedding with eigenvalues/eigenvectors creates.
        """
        # Create cache directory for DiffusionMaps
        os.makedirs("./cache", exist_ok=True)
        os.makedirs("./cache/diffmap_input_diffusion_maps", exist_ok=True)
        
        # Setup pipeline with linear coordinates for diffusion structure
        pipeline = PipelineManager()
        # Set cache directory and enable memmap for DiffusionMaps
        pipeline._decomposition_manager.cache_dir = "./cache"
        pipeline._decomposition_manager.use_memmap = True
        
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=16, n_atoms=3, seed=456
        )
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add coordinates features for DiffusionMaps requirement
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("diffmap_input")
        pipeline.feature_selector.add_selection("diffmap_input", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("diffmap_input")
        
        # Apply Diffusion Maps with specific parameters
        test_n_components = 2
        test_epsilon = 0.1
        
        pipeline.decomposition.add.diffusion_maps(
            selection_name="diffmap_input",
            n_components=test_n_components,
            epsilon=test_epsilon,
        )
        
        # Verify results
        decomp_data = pipeline._data.decomposition_data["diffmap_input_diffusion_maps"]
        
        # Check output shape matches requested components
        assert decomp_data.data.shape == (16, test_n_components), f"Expected shape (16, {test_n_components}), got {decomp_data.data.shape}"
        
        # Verify data is finite
        assert np.all(np.isfinite(decomp_data.data))
        
        # For linear coordinates, expect structured embedding with some variation
        # Test that components have reasonable ranges (not all zeros)
        component_ranges = []
        for i in range(test_n_components):
            comp_range = np.max(decomp_data.data[:, i]) - np.min(decomp_data.data[:, i])
            component_ranges.append(comp_range)
            assert comp_range >= 0, f"Component {i} has negative range: {comp_range}"
        
        # At least one component should have significant variation (not all identical)
        max_range = max(component_ranges)
        assert max_range > 1e-10, f"All components have too little variation, max range: {max_range}"
        
        # Verify metadata with concrete parameter values
        assert decomp_data.decomposition_type == "diffusion_maps" 
        assert "decomposition_name" in decomp_data.metadata
        assert "eigenvalues" in decomp_data.metadata
        assert len(decomp_data.metadata["eigenvalues"]) == test_n_components
        assert decomp_data.metadata["epsilon"] == test_epsilon
        
    def test_decomposition_components_validation(self):
        """
        Test that decomposition components have correct shapes and properties.

        Validates that different n_components values produce correct output shapes
        and PCA variance for structured data is plausible.
        """
        # Setup pipeline
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=12, n_atoms=4, seed=789
        )
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} for i in range(4)
        ]}
        
        # Add coordinates feature
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("validation_input")
        pipeline.feature_selector.add_selection("validation_input", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("validation_input")
        
        # Test different n_components values with concrete validation
        test_components = [1, 2, 3]
        for n_comp in test_components:
            pca = PCA(n_components=n_comp, random_state=42)
            # Use correct decomposition API with force=True to allow multiple decompositions
            pipeline.decomposition.add_decomposition(
                selection_name="validation_input",
                decomposition_type=pca,
                force=True
            )
            
            # Verify shape matches requested components exactly
            decomp_data = list(pipeline._data.decomposition_data.values())[-1]  # Latest decomposition
            expected_shape = (12, n_comp)
            assert decomp_data.data.shape == expected_shape, f"Expected shape {expected_shape}, got {decomp_data.data.shape}"
            
            # Verify all data is finite
            assert np.all(np.isfinite(decomp_data.data))
            
            # For linear coordinates, PCA should capture the linear trend
            if n_comp >= 1:
                # First component should have significant variance (linear data has structure)
                pc1_var = np.var(decomp_data.data[:, 0])
                assert pc1_var > 0.5, f"PC1 variance too small: {pc1_var} (expected > 0.5 for linear data)"
            
            if n_comp >= 2:
                # Second component should also have some variance
                pc2_var = np.var(decomp_data.data[:, 1])
                assert pc2_var > 0, f"PC2 variance should be positive: {pc2_var}"
            
            # Verify metadata contains hyperparameters
            assert "hyperparameters" in decomp_data.metadata
            hyperparams = decomp_data.metadata["hyperparameters"]
            assert hyperparams["n_components"] == n_comp
            assert len(decomp_data.metadata["explained_variance_ratio"]) == n_comp
    
    def test_decomposition_explained_variance(self):
        """
        Test explained variance calculation for PCA.

        Validates that PCA explained variance ratios are computed correctly,
        are sorted in descending order and sum to ≤ 1.0.
        """
        # Setup pipeline with structured data for variance analysis
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=25, n_atoms=3, seed=111
        )
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add coordinates
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("variance_test")
        pipeline.feature_selector.add_selection("variance_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("variance_test")
        
        # Apply PCA with 3 components
        pca = PCA(n_components=3, random_state=42)
        # Use correct decomposition API
        pipeline.decomposition.add_decomposition(
            selection_name="variance_test",
            decomposition_type=pca
        )
        
        # Verify results
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]  # Latest decomposition
        
        # Check that explained variance ratio exists with concrete expectations
        explained_var_ratio = decomp_data.metadata["explained_variance_ratio"]
        assert len(explained_var_ratio) == 3, f"Expected 3 variance ratios, got {len(explained_var_ratio)}"
        assert all(ratio >= 0 for ratio in explained_var_ratio), f"All variance ratios should be non-negative: {explained_var_ratio}"
        assert all(ratio <= 1.0 + 1e-10 for ratio in explained_var_ratio), f"All variance ratios should be <= 1.0: {explained_var_ratio}"
        
        # For linear coordinates data, first component should explain most variance
        assert explained_var_ratio[0] >= explained_var_ratio[1], f"PC1 should explain more variance than PC2: {explained_var_ratio[0]} vs {explained_var_ratio[1]}"
        assert explained_var_ratio[1] >= explained_var_ratio[2], f"PC2 should explain more variance than PC3: {explained_var_ratio[1]} vs {explained_var_ratio[2]}"
        
        # For linear data, first component should explain significant variance
        assert explained_var_ratio[0] > 0.5, f"PC1 should explain >50% variance for linear data, got {explained_var_ratio[0]}"
        
        # Sum of explained variance should be <= 1.0 (with floating point tolerance)
        total_variance = sum(explained_var_ratio)
        assert total_variance <= 1.0 + 1e-10, f"Total explained variance should be <= 1.0, got {total_variance}"
        
    def test_decomposition_metadata_validation(self):
        """
        Test that decomposition metadata contains all expected fields.

        Validates that PCA and Kernel PCA metadata contains the correct
        decomposition_name, cache_path and method information.
        """
        # Setup pipeline
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_fixed_distances(
            n_frames=10, n_atoms=4, distance=3.5, seed=222
        )
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"},
            {"seqid": 3, "full_name": "RES_3"}
        ]}
        
        # Add coordinates features
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("metadata_test")
        pipeline.feature_selector.add_selection("metadata_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("metadata_test")
        
        # Test PCA metadata
        pca = PCA(n_components=1, random_state=42)
        # Use correct decomposition API
        pipeline.decomposition.add_decomposition(
            selection_name="metadata_test",
            decomposition_type=pca
        )
        
        # The decomposition gets stored with a generated name (selection_name + decomposition_type)
        decomp_keys = list(pipeline._data.decomposition_data.keys())
        print(f"Available decomposition keys: {decomp_keys}")
        decomp_data = list(pipeline._data.decomposition_data.values())[0]  # First and only decomposition
        
        # Verify core metadata fields
        assert decomp_data.decomposition_type == "pca"
        assert "explained_variance" in decomp_data.metadata
        assert "decomposition_name" in decomp_data.metadata
        assert "cache_path" in decomp_data.metadata
        
        # Verify metadata values
        assert decomp_data.metadata["decomposition_name"] == "metadata_test_pca"
        
        # Test Kernel PCA metadata with varied data to avoid ARPACK singularity
        # Create new trajectory with linear coordinates for variation
        varied_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=10, n_atoms=4, seed=333
        )
        
        # Replace trajectory data
        pipeline._data.trajectory_data.trajectories = [varied_traj]
        pipeline._data.trajectory_data.n_frames = varied_traj.n_frames
        
        # Re-compute coordinates features for varied trajectory to avoid distance issues
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        
        # Apply KernelPCA with varied data
        kpca = KernelPCA(n_components=1, random_state=42)
        pipeline.decomposition.add_decomposition(
            selection_name="metadata_test",
            decomposition_type=kpca
        )
        
        kpca_data = list(pipeline._data.decomposition_data.values())[1]  # Second decomposition
        
        # Verify Kernel PCA specific metadata
        assert kpca_data.decomposition_type == "kernel_pca"
        assert "decomposition_name" in kpca_data.metadata
        assert "cache_path" in kpca_data.metadata

    # === METHOD PATH TESTS ===
        
    @patch('mdxplain.decomposition.decomposition_type.pca.pca_calculator.PCA')
    def test_pca_standard_method_integration(self, mock_sklearn_pca):
        """
        Test that PCA standard method works via pipeline API.

        Validates that sklearn.PCA is called with correct parameters
        and mock results are stored correctly in pipeline data.
        """
        # Configure mock to return realistic data
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.6, 0.4])
        mock_pca_instance.explained_variance_ = np.array([3.6, 2.4])
        mock_sklearn_pca.return_value = mock_pca_instance
        
        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=False)
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=10, n_atoms=3, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("pca_method_test")
        pipeline.feature_selector.add_selection("pca_method_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("pca_method_test")
        
        # Use pipeline API to trigger standard method
        pipeline.decomposition.add.pca(
            selection_name="pca_method_test",
            n_components=2,
            random_state=42
        )
        
        # Verify sklearn.PCA was called with correct parameters
        mock_sklearn_pca.assert_called_once_with(
            n_components=2,
            random_state=42,
            copy=False,
            whiten=True
        )
        mock_pca_instance.fit_transform.assert_called_once()
        
        # Verify results were computed correctly 
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "pca"
        assert np.array_equal(decomp_data.data, mock_pca_instance.fit_transform.return_value)
        
        # Verify metadata indicates standard method
        assert decomp_data.metadata["method"] == "standard_pca"
        assert np.array_equal(decomp_data.metadata["explained_variance_ratio"], mock_pca_instance.explained_variance_ratio_)
        
    @patch('mdxplain.decomposition.decomposition_type.pca.pca_calculator.IncrementalPCA')
    def test_pca_incremental_method_integration(self, mock_sklearn_ipca):
        """
        Test that PCA incremental method works via pipeline API.

        Validates that sklearn.IncrementalPCA is called when use_memmap=True
        and batch processing works correctly for large datasets.
        """
        # Configure mock to return realistic data
        mock_ipca_instance = MagicMock()
        mock_ipca_instance.fit_transform.return_value = np.random.rand(10, 2)
        mock_ipca_instance.explained_variance_ratio_ = np.array([0.7, 0.3])
        mock_ipca_instance.explained_variance_ = np.array([4.2, 1.8])
        mock_sklearn_ipca.return_value = mock_ipca_instance
        
        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=True, chunk_size=2000)
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=10, n_atoms=3, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("pca_incremental_test")
        pipeline.feature_selector.add_selection("pca_incremental_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("pca_incremental_test")
        
        # Enable memmap to trigger incremental method
        pipeline._decomposition_manager.use_memmap = True
        
        # Use pipeline API to trigger incremental method
        pipeline.decomposition.add.pca(
            selection_name="pca_incremental_test",
            n_components=2,
            random_state=42
        )
        
        # Verify sklearn.IncrementalPCA was called with correct parameters
        mock_sklearn_ipca.assert_called_once_with(
            n_components=2,
            batch_size=2000,
            whiten=True,
            copy=False
        )
        mock_ipca_instance.fit_transform.assert_called_once()
        
        # Verify results were computed correctly 
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "pca"
        assert np.array_equal(decomp_data.data, mock_ipca_instance.fit_transform.return_value)
        
        # Verify metadata indicates standard method
        assert decomp_data.metadata["method"] == "incremental_pca"
        assert np.array_equal(decomp_data.metadata["explained_variance_ratio"], mock_ipca_instance.explained_variance_ratio_)

    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.KernelPCA')
    def test_kernel_pca_standard_method_integration(self, mock_sklearn_kpca):
        """
        Test that KernelPCA standard method works via pipeline API.

        Validates that sklearn.KernelPCA with RBF kernel and gamma parameter
        is called correctly and hyperparameters are stored.
        """
        # Configure mock
        mock_kpca_instance = MagicMock()
        mock_kpca_instance.fit_transform.return_value = np.random.rand(8, 2)
        mock_sklearn_kpca.return_value = mock_kpca_instance
        
        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=False)
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=8, n_atoms=3, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("kpca_method_test")
        pipeline.feature_selector.add_selection("kpca_method_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("kpca_method_test")
        
        # Use pipeline API to trigger standard method
        pipeline.decomposition.add.kernel_pca(
            selection_name="kpca_method_test",
            n_components=2,
            gamma=0.5,
            random_state=42
        )
        
        # Verify sklearn.KernelPCA was called with correct parameters
        mock_sklearn_kpca.assert_called_once_with(
            n_components=2,
            kernel="rbf",
            gamma=0.5,
            random_state=42,
            copy_X=False,
            n_jobs=-1
        )
        mock_kpca_instance.fit_transform.assert_called_once()
        
        # Verify results were computed correctly
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "kernel_pca"
        assert np.array_equal(decomp_data.data, mock_kpca_instance.fit_transform.return_value)
        
        # Verify hyperparameters were passed correctly
        hyperparams = decomp_data.metadata["hyperparameters"]
        assert hyperparams["n_components"] == 2
        assert hyperparams["gamma"] == 0.5
        assert hyperparams["random_state"] == 42
        assert hyperparams["kernel"] == "rbf"
        
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.eigs')
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.KernelPCACalculator._center_kernel_inplace')
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.KernelPCACalculator._compute_chunk_wise_rbf_kernel')
    def test_kernel_pca_incremental_method_integration(self, mock_build_kernel, mock_center_kernel, mock_eigs):
        """
        Test that KernelPCA incremental method works when use_memmap=True.

        Validates that chunk-wise kernel matrix building, centering and
        eigendecomposition are called correctly for memory-efficient processing.
        """
        # Configure mock
        mock_eigenvectors = np.random.rand(8, 2)
        mock_eigenvalues = np.array([1.0, 0.8])
        expected_data = mock_eigenvectors * np.sqrt(mock_eigenvalues)
        mock_eigs.return_value = (mock_eigenvalues, mock_eigenvectors)
        mock_build_kernel.return_value = np.random.rand(8, 8)

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=True, cache_dir="./cache")
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=8, n_atoms=3, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("kpca_incremental_test")
        pipeline.feature_selector.add_selection("kpca_incremental_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("kpca_incremental_test")
        
        # Use pipeline API to trigger incremental method
        pipeline.decomposition.add.kernel_pca(
            selection_name="kpca_incremental_test",
            n_components=2,
            gamma=0.5,
            random_state=42
        )
        
        # Verify that kernel matrix was built, centered, and eigs called
        mock_build_kernel.assert_called_once()
        mock_center_kernel.assert_called_once()
        mock_eigs.assert_called_once()

        # Verify results were computed correctly
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "kernel_pca"
        assert np.allclose(decomp_data.data, expected_data)
        
        # Verify method was incremental
        assert decomp_data.metadata["method"] == "iterative_kernel_pca"
        hyperparams = decomp_data.metadata["hyperparameters"]
        assert hyperparams["n_components"] == 2
        assert hyperparams["gamma"] == 0.5
        
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.IncrementalPCA')
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.Nystroem')
    def test_kernel_pca_nystrom_method_integration(self, mock_nystroem, mock_ipca):
        """
        Test that KernelPCA Nyström method works when use_nystrom=True.

        Validates that Nyström approximation with landmarks and IncrementalPCA
        are used correctly.
        """
        # Configure mock for Nystroem
        mock_nystroem_instance = MagicMock()
        mock_nystroem_instance.transform.return_value = np.random.rand(15, 10) # n_landmarks
        mock_nystroem.return_value = mock_nystroem_instance
        
        # Configure mock for IncrementalPCA
        mock_ipca_instance = MagicMock()
        expected_data = np.random.rand(15, 2)
        mock_ipca_instance.transform.return_value = expected_data
        mock_ipca.return_value = mock_ipca_instance

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=False)
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=15, n_atoms=3, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("kpca_nystrom_test")
        pipeline.feature_selector.add_selection("kpca_nystrom_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("kpca_nystrom_test")
        
        # Use pipeline API to trigger Nyström method
        pipeline.decomposition.add.kernel_pca(
            selection_name="kpca_nystrom_test",
            n_components=2,
            gamma=0.5,
            use_nystrom=True,
            n_landmarks=10,
            random_state=42
        )
        
        # Verify Nystroem and IncrementalPCA were called
        mock_nystroem.assert_called_once()
        mock_nystroem_instance.fit.assert_called_once()
        mock_ipca.assert_called_once()
        assert mock_ipca_instance.partial_fit.call_count > 0
        assert mock_ipca_instance.transform.call_count > 0

        # Verify results were computed correctly
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "kernel_pca"
        assert np.allclose(decomp_data.data, expected_data)
        
        # Verify Nyström parameters were passed correctly
        assert decomp_data.metadata["method"] == "nystrom_kernel_pca"
        hyperparams = decomp_data.metadata["hyperparameters"]
        assert hyperparams["n_components"] == 2
        assert hyperparams["use_nystrom"] is True
        assert hyperparams["n_landmarks"] == 10
        
    @patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.DiffusionMapsCalculator._compute_standard_diffusion_maps')
    def test_diffusion_maps_standard_method_integration(self, mock_compute_standard):
        """
        Test that DiffusionMaps standard method is called via pipeline API.

        Validates that standard diffusion maps computation with epsilon parameter
        and eigenvalue-based spectral embedding works.
        """
        # Configure mock
        mock_compute_standard.return_value = (np.random.rand(8, 2), {"method": "standard_diffusion_maps", "eigenvalues": [0.9, 0.8], "epsilon": 0.1})

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=False)
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=8, n_atoms=2, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("diffmap_standard_test")
        pipeline.feature_selector.add_selection("diffmap_standard_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("diffmap_standard_test")
        
        # Use pipeline API to trigger standard method
        pipeline.decomposition.add.diffusion_maps(
            selection_name="diffmap_standard_test",
            n_components=2,
            epsilon=0.1
        )
        
        # Verify standard method was called
        mock_compute_standard.assert_called_once()

        # Verify results were stored
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "diffusion_maps"
        assert np.array_equal(decomp_data.data, mock_compute_standard.return_value[0])
        assert decomp_data.metadata["method"] == "standard_diffusion_maps"
        
    @patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.DiffusionMapsCalculator._compute_iterative_diffusion_maps')
    def test_diffusion_maps_iterative_method_integration(self, mock_compute_iterative):
        """
        Test that DiffusionMaps iterative method is called when use_memmap=True.

        Validates that memory-efficient iterative diffusion maps computation
        for large datasets with cache directory works.
        """
        # Configure mock
        mock_compute_iterative.return_value = (np.random.rand(8, 2), {"method": "iterative_diffusion_maps", "eigenvalues": [0.9, 0.8], "epsilon": 0.1})

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=True, cache_dir="./cache")
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=8, n_atoms=2, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("diffmap_iterative_test")
        pipeline.feature_selector.add_selection("diffmap_iterative_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("diffmap_iterative_test")
        
        # Use pipeline API to trigger iterative method
        pipeline.decomposition.add.diffusion_maps(
            selection_name="diffmap_iterative_test",
            n_components=2,
            epsilon=0.1
        )
        
        # Verify iterative method was called
        mock_compute_iterative.assert_called_once()

        # Verify results were stored
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "diffusion_maps"
        assert np.array_equal(decomp_data.data, mock_compute_iterative.return_value[0])
        assert decomp_data.metadata["method"] == "iterative_diffusion_maps"
        
    @patch('mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator.DiffusionMapsCalculator._compute_nystrom_diffusion_maps')
    def test_diffusion_maps_nystrom_method_integration(self, mock_compute_nystrom):
        """
        Test that DiffusionMaps Nyström method is called when use_nystrom=True.

        Validates that Nyström approximation for diffusion maps with landmarks.
        """
        # Configure mock
        mock_compute_nystrom.return_value = (np.random.rand(12, 2), {"method": "nystrom_diffusion_maps", "eigenvalues": [0.9, 0.8], "epsilon": 0.1})

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=False)
        mock_traj = MockTrajectoryFactory.create_linear_coordinates(
            n_frames=12, n_atoms=2, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"}
        ]}
        
        # Add features and create selector
        pipeline.feature.add.coordinates(atom_selection="all", force=True)
        pipeline.feature_selector.create("diffmap_nystrom_test")
        pipeline.feature_selector.add_selection("diffmap_nystrom_test", "coordinates", "all", use_reduced=False)
        pipeline.feature_selector.select("diffmap_nystrom_test")
        
        # Use pipeline API to trigger Nyström method
        pipeline.decomposition.add.diffusion_maps(
            selection_name="diffmap_nystrom_test",
            n_components=2,
            epsilon=0.1,
            use_nystrom=True,
            n_landmarks=8
        )
        
        # Verify nystrom method was called
        mock_compute_nystrom.assert_called_once()

        # Verify results were stored
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "diffusion_maps"
        assert np.array_equal(decomp_data.data, mock_compute_nystrom.return_value[0])
        assert decomp_data.metadata["method"] == "nystrom_diffusion_maps"
        
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.KernelPCACalculator._compute_standard_kernel_pca')
    def test_contact_kernel_pca_standard_method_integration(self, mock_compute_standard):
        """
        Test that ContactKernelPCA standard method is called via pipeline API.

        Validates that contact-specific kernel PCA with binary contact features
        and RBF kernel works for molecular interaction analysis.
        """
        # Configure mock
        expected_data = np.random.rand(8, 2)
        mock_compute_standard.return_value = (expected_data, {"method": "standard_kernel_pca"})

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=False)
        mock_traj = MockTrajectoryFactory.create_two_state(
            n_frames=8, n_atoms=4, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": 0, "full_name": "RES_0"},
            {"seqid": 1, "full_name": "RES_1"},
            {"seqid": 2, "full_name": "RES_2"},
            {"seqid": 3, "full_name": "RES_3"}
        ]}
        
        # Add contacts features
        pipeline.feature.add.distances(force=True)
        pipeline.feature.add.contacts(force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("contact_kpca_standard_test")
        pipeline.feature_selector.add_selection("contact_kpca_standard_test", "contacts", "all", use_reduced=False)
        pipeline.feature_selector.select("contact_kpca_standard_test")
        
        # Use pipeline API to trigger standard method
        pipeline.decomposition.add.contact_kernel_pca(
            selection_name="contact_kpca_standard_test",
            n_components=2,
            gamma=0.5,
            random_state=42
        )
        
        # Verify standard method was called
        mock_compute_standard.assert_called_once()

        # Verify results were stored
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "contact_kernel_pca"
        assert np.array_equal(decomp_data.data, expected_data)
        assert decomp_data.metadata["method"] == "standard_kernel_pca"
        
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.KernelPCACalculator._compute_incremental_kernel_pca')
    def test_contact_kernel_pca_incremental_method_integration(self, mock_compute_iterative):
        """
        Test that ContactKernelPCA incremental method works when use_memmap=True.

        Validates that memory-efficient contact kernel PCA for large trajectories
        with many contact features and chunk-wise processing works.
        """
        # Configure mock
        expected_data = np.random.rand(50, 2)
        mock_compute_iterative.return_value = (expected_data, {"method": "iterative_kernel_pca"})

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=True, cache_dir="./cache")
        mock_traj = MockTrajectoryFactory.create_two_state(
            n_frames=50, n_atoms=8, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} for i in range(8)
        ]}
        
        # Add contacts features
        pipeline.feature.add.distances(force=True)
        pipeline.feature.add.contacts(force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("contact_kpca_incremental_test")
        pipeline.feature_selector.add_selection("contact_kpca_incremental_test", "contacts", "all", use_reduced=False)
        pipeline.feature_selector.select("contact_kpca_incremental_test")
        
        # Use pipeline API to trigger incremental method
        pipeline.decomposition.add.contact_kernel_pca(
            selection_name="contact_kpca_incremental_test",
            n_components=2,
            gamma=0.3
        )
        
        # Verify iterative method was called
        mock_compute_iterative.assert_called_once()

        # Verify results were stored
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "contact_kernel_pca"
        assert np.array_equal(decomp_data.data, expected_data)
        assert decomp_data.metadata["method"] == "iterative_kernel_pca"
        
    @patch('mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator.KernelPCACalculator._compute_nystrom_kernel_pca')
    def test_contact_kernel_pca_nystrom_method_integration(self, mock_compute_nystrom):
        """
        Test that ContactKernelPCA Nyström method works when use_nystrom=True.

        Validates that landmark-based approximation for contact kernel PCA
        with n_landmarks parameter provides efficient dimensionality reduction.
        """
        # Configure mock
        expected_data = np.random.rand(10, 2)
        mock_compute_nystrom.return_value = (expected_data, {"method": "nystrom_kernel_pca"})

        # Setup pipeline with test data
        pipeline = PipelineManager(use_memmap=False)
        mock_traj = MockTrajectoryFactory.create_two_state(
            n_frames=10, n_atoms=4, seed=42
        )
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} for i in range(4)
        ]}
        
        # Add contacts features
        pipeline.feature.add.distances(force=True)
        pipeline.feature.add.contacts(force=True)
        
        # Create feature selector
        pipeline.feature_selector.create("contact_kpca_nystrom_test")
        pipeline.feature_selector.add_selection("contact_kpca_nystrom_test", "contacts", "all", use_reduced=False)
        pipeline.feature_selector.select("contact_kpca_nystrom_test")
        
        # Use pipeline API to trigger Nyström method
        pipeline.decomposition.add.contact_kernel_pca(
            selection_name="contact_kpca_nystrom_test",
            n_components=2,
            gamma=0.5,
            use_nystrom=True,
            n_landmarks=6,
            random_state=42
        )
        
        # Verify nystrom method was called
        mock_compute_nystrom.assert_called_once()

        # Verify results were stored
        decomp_data = list(pipeline._data.decomposition_data.values())[-1]
        assert decomp_data.decomposition_type == "contact_kernel_pca"
        assert np.array_equal(decomp_data.data, expected_data)
        assert decomp_data.metadata["method"] == "nystrom_kernel_pca"
