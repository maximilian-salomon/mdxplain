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

"""Integration tests for pipeline save and load functionality."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances import Distances
from mdxplain.feature.feature_type.contacts import Contacts
from mdxplain.clustering.cluster_type.dbscan import DBSCAN
from mdxplain.decomposition.decomposition_type.pca import PCA
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestPipelineSaveLoad:
    """Test pipeline save and load functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_save_load_empty_pipeline(self, temp_dir):
        """
        Test that empty pipeline save/load preserves structure.

        Validates that empty PipelineManager is correctly saved and loaded
        with all standard managers (trajectory, feature, clustering, decomposition).
        """
        # Create empty pipeline
        pipeline = PipelineManager()
        
        # Save pipeline
        save_path = temp_dir / "empty_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Verify file exists
        assert save_path.exists()
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify basic structure
        assert loaded is not None
        assert hasattr(loaded, 'trajectory')
        assert hasattr(loaded, 'feature')
        assert hasattr(loaded, 'clustering')
        assert hasattr(loaded, 'decomposition')
        
    def test_save_load_pipeline_with_trajectory(self, temp_dir):
        """
        Test that pipeline with trajectory data saves and loads correctly.

        Validates that pipeline with mock trajectory (n_frames, n_atoms, xyz)
        is identically reconstructed after save/load cycle.
        """
        # Create pipeline with mock trajectory
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=50, n_atoms=20, seed=42)
        
        # Manually set trajectory data (bypassing file loading)
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        
        # Save pipeline
        save_path = temp_dir / "trajectory_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify trajectory data
        assert loaded._data.trajectory_data.n_frames == 50
        assert loaded._data.trajectory_data.n_atoms == 20
        assert len(loaded._data.trajectory_data.trajectories) == 1
        
        # Verify coordinates are identical
        original_xyz = pipeline._data.trajectory_data.trajectories[0].xyz
        loaded_xyz = loaded._data.trajectory_data.trajectories[0].xyz
        np.testing.assert_array_equal(original_xyz, loaded_xyz)
        
    def test_save_load_pipeline_with_features(self, temp_dir):
        """
        Test that pipeline with feature data preserves arrays and metadata.

        Validates that pipeline with distances features (data + metadata)
        is correctly saved and loaded with identical feature arrays.
        """
        # Create pipeline with trajectory and features
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=30, n_atoms=10, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distance feature - minimal for save/load testing
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Save original data for comparison
        original_distances = pipeline._data.feature_data["distances"][0].data.copy()
        
        # Save pipeline
        save_path = temp_dir / "features_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify feature data exists
        assert "distances" in loaded._data.feature_data
        
        # Verify distances data
        np.testing.assert_array_equal(loaded._data.feature_data["distances"][0].data, original_distances)
        assert loaded._data.feature_data["distances"][0].data.shape == (30, 36)  # All distance pairs for 10 atoms
        
        # Verify feature metadata
        assert loaded._data.feature_data["distances"][0].feature_metadata is not None
        
    def test_save_load_pipeline_with_clustering(self, temp_dir):
        """
        Test that pipeline with clustering results preserves assignments.

        Validates that DBSCAN clustering (labels + metadata) is correctly
        saved and loaded with identical cluster assignments.
        """
        # Create pipeline with features and clustering
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_two_state(n_atoms=20, n_frames=50, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distances and create selection
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Create feature selection
        pipeline.feature_selector.create("test_selection")
        pipeline.feature_selector.add_selection( "test_selection", "distances", "all")
        pipeline.feature_selector.select("test_selection")
        
        # Add clustering
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        pipeline.clustering.add_clustering("test_selection", dbscan, use_decomposed=False, cluster_name="test_cluster")
        
        # Save original clustering results
        original_labels = pipeline._data.cluster_data['test_cluster'].labels.copy()
        original_metadata = pipeline._data.cluster_data['test_cluster'].metadata.copy()
        
        # Save pipeline
        save_path = temp_dir / "clustering_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify clustering data
        assert 'test_cluster' in loaded._data.cluster_data
        np.testing.assert_array_equal(loaded._data.cluster_data['test_cluster'].labels, original_labels)
        
        # Verify clustering metadata
        assert loaded._data.cluster_data['test_cluster'].metadata['algorithm'] == original_metadata['algorithm']
        assert loaded._data.cluster_data['test_cluster'].metadata['n_clusters'] == original_metadata['n_clusters']
        assert loaded._data.cluster_data['test_cluster'].metadata['n_noise'] == original_metadata['n_noise']
        
    def test_save_load_pipeline_with_decomposition(self, temp_dir):
        """
        Test that pipeline with decomposition results preserves components.

        Validates that PCA decomposition (components + explained variance)
        is correctly saved and loaded with identical principal components.
        """
        # Create pipeline with features and decomposition
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=40, n_atoms=15, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distances
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Create feature selection
        pipeline.feature_selector.create("decomp_selection")
        pipeline.feature_selector.add_selection( "decomp_selection", "distances", "all")
        pipeline.feature_selector.select("decomp_selection")
        
        # Add PCA decomposition
        pca = PCA(n_components=2)
        pipeline.decomposition.add_decomposition("decomp_selection", pca, decomposition_name="test_pca")
        
        # Save original decomposition results
        original_components = pipeline._data.decomposition_data['test_pca'].data.copy()
        original_metadata = pipeline._data.decomposition_data['test_pca'].metadata.copy()
        
        # Save pipeline
        save_path = temp_dir / "decomposition_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify decomposition data
        assert 'test_pca' in loaded._data.decomposition_data
        np.testing.assert_array_equal(loaded._data.decomposition_data['test_pca'].data, original_components)
        assert loaded._data.decomposition_data['test_pca'].data.shape == (40, 2)
        
        # Verify decomposition metadata
        assert loaded._data.decomposition_data['test_pca'].metadata['method'] == original_metadata['method']
        assert loaded._data.decomposition_data['test_pca'].metadata['hyperparameters']['n_components'] == original_metadata['hyperparameters']['n_components']
        assert 'explained_variance_ratio' in loaded._data.decomposition_data['test_pca'].metadata
        
    def test_save_load_complete_pipeline(self, temp_dir):
        """
        Test that complete pipeline saves and loads all components correctly.

        Validates that complete pipeline (trajectory, features, clustering,
        decomposition, feature selectors) is correctly reconstructed.
        """
        # Create complete pipeline
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_atoms=10, n_frames=25, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add multiple features
        distances = Distances()
        contacts = Contacts(cutoff=5.0)
        pipeline.feature.add_feature(distances, force=True)
        pipeline.feature.add_feature(contacts, force=True)
        
        # Reduce some features
        pipeline.feature.reduce_data(Distances(), "mean", threshold_min=0.0)
        
        # Create feature selections
        pipeline.feature_selector.create("all_features")
        pipeline.feature_selector.add_selection( "all_features", "distances", "all", use_reduced=True)
        pipeline.feature_selector.add_selection( "all_features", "contacts", "all", use_reduced=False)
        pipeline.feature_selector.select("all_features")
        
        # Add clustering and decomposition
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        pca = PCA(n_components=3)
        pipeline.clustering.add_clustering("all_features", dbscan, use_decomposed=False, cluster_name="complete_cluster")
        pipeline.decomposition.add_decomposition("all_features", pca, decomposition_name="complete_pca")
        
        # Save all original data for comparison
        original_dist_data = pipeline._data.feature_data['distances'][0].data.copy()
        original_cont_data = pipeline._data.feature_data['contacts'][0].data.copy()
        original_reduced_dist = pipeline._data.feature_data['distances'][0].reduced_data.copy()
        original_cluster_labels = pipeline._data.cluster_data['complete_cluster'].labels.copy()
        original_pca_components = pipeline._data.decomposition_data['complete_pca'].data.copy()
        original_pca_metadata = pipeline._data.decomposition_data['complete_pca'].metadata.copy()
        
        # Save pipeline
        save_path = temp_dir / "complete_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        
        # Verify file exists and has reasonable size
        assert save_path.exists()
        assert save_path.stat().st_size > 1000  # Should be substantial
        
        # Load pipeline
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Verify trajectory
        assert loaded._data.trajectory_data.n_frames == 25
        assert loaded._data.trajectory_data.n_atoms == 10
        
        # Verify features
        np.testing.assert_array_equal(loaded._data.feature_data['distances'][0].data, original_dist_data)
        np.testing.assert_array_equal(loaded._data.feature_data['contacts'][0].data, original_cont_data)
        np.testing.assert_array_equal(loaded._data.feature_data['distances'][0].reduced_data, original_reduced_dist)
        
        # Verify clustering
        np.testing.assert_array_equal(loaded._data.cluster_data['complete_cluster'].labels, original_cluster_labels)
        assert loaded._data.cluster_data['complete_cluster'].metadata['algorithm'] == 'dbscan'
        
        # Verify decomposition
        np.testing.assert_array_equal(loaded._data.decomposition_data['complete_pca'].data, original_pca_components)
        assert loaded._data.decomposition_data['complete_pca'].metadata['method'] == original_pca_metadata['method']
        
        # Verify feature selectors exist
        assert "all_features" in loaded._data.selected_feature_data
        
    def test_save_load_preserves_bound_methods(self, temp_dir):
        """
        Test that save/load preserves bound analysis method functionality.

        Validates that feature.analysis methods still work after save/load
        and return identical results.
        """
        # Create pipeline with feature
        pipeline = PipelineManager()
        mock_traj = MockTrajectoryFactory.create_simple(n_frames=20, n_atoms=8, seed=42)
        
        # Set trajectory data
        pipeline._data.trajectory_data.trajectories = [mock_traj]
        pipeline._data.trajectory_data.trajectory_names = ["mock_trajectory"]
        pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
        pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
        # Add required labels for distance calculations
        pipeline._data.trajectory_data.res_label_data = {0: [
            {"seqid": i, "full_name": f"RES_{i}"} 
            for i in range(mock_traj.n_atoms)
        ]}
        
        # Add distance feature
        distances = Distances()
        pipeline.feature.add_feature(distances, force=True)
        
        # Test analysis method before save
        original_mean = pipeline._data.feature_data['distances'][0].analysis.compute_mean()
        
        # Save and load pipeline
        save_path = temp_dir / "bound_methods_pipeline.pkl"
        pipeline.save_to_single_file(str(save_path))
        loaded = PipelineManager.load_from_single_file(str(save_path))
        
        # Test analysis method after load
        loaded_mean = loaded._data.feature_data['distances'][0].analysis.compute_mean()
        
        # Verify methods work and give same results
        np.testing.assert_almost_equal(loaded_mean, original_mean, decimal=10)
        
        # Verify other analysis methods are available
        assert hasattr(loaded._data.feature_data['distances'][0].analysis, 'compute_std')
        assert hasattr(loaded._data.feature_data['distances'][0].analysis, 'compute_min')
        assert hasattr(loaded._data.feature_data['distances'][0].analysis, 'compute_max')
        
        # Test another method
        original_std = pipeline._data.feature_data['distances'][0].analysis.compute_std()
        loaded_std = loaded._data.feature_data['distances'][0].analysis.compute_std()
        np.testing.assert_almost_equal(loaded_std, original_std, decimal=10)
        
    def test_save_load_error_handling(self, temp_dir):
        """
        Test error handling in save/load operations.

        Validates that non-existent files raise FileNotFoundError and
        invalid paths raise OSError/PermissionError.
        """
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            PipelineManager.load_from_single_file(str(temp_dir / "nonexistent.pkl"))
            
        # Test saving to invalid path
        pipeline = PipelineManager()
        invalid_path = "/invalid/path/pipeline.pkl"
        
        with pytest.raises((OSError, PermissionError)):
            pipeline.save_to_single_file(invalid_path)