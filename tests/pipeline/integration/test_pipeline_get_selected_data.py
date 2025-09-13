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

"""Tests for PipelineData.get_selected_data() with proper data verification."""

import numpy as np
import pytest
import warnings
import mdtraj as md
from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances


class TestGetSelectedDataVerification:
    """Test get_selected_data() with proper data content verification."""
    
    def setup_method(self):
        """Create synthetic trajectory with predictable data."""
        # Create topology: ALA(1), GLY(2), VAL(3), ALA(4), GLY(5)
        topology = md.Topology()
        chain = topology.add_chain()
        
        residue_names = ["ALA", "GLY", "VAL", "ALA", "GLY"]
        residues = []
        
        for name in residue_names:
            residues.append(topology.add_residue(name, chain))
        
        # Add CA atoms
        for residue in residues:
            topology.add_atom("CA", md.element.carbon, residue)
        
        # Create PREDICTABLE coordinates for testing
        coordinates = []
        for frame in range(50):  # Smaller for easier testing
            frame_coords = []
            for atom_idx in range(5):
                # Simple predictable pattern
                x = atom_idx * 2.0  # Atoms spread apart
                y = frame * 0.1     # Change over time
                z = 0.0
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        self.test_traj = md.Trajectory(xyz, topology)
        
        # Setup pipeline
        self.pipeline = PipelineManager()
        self.pipeline.data.trajectory_data.trajectories = [self.test_traj]
        self.pipeline.data.trajectory_data.trajectory_names = ["test_traj"]
        
        # Create residue metadata
        residue_metadata = []
        for i, res in enumerate(self.test_traj.topology.residues):
            residue_metadata.append({
                "resid": res.resSeq + 1,
                "seqid": res.index + 1, 
                "resname": res.name,
                "aaa_code": res.name,
                "a_code": res.name[0] if res.name else "X",
                "consensus": None,
                "full_name": f"{res.name}{res.index + 1}",
                "index": res.index
            })
        
        self.pipeline.data.trajectory_data.res_label_data = {0: residue_metadata}
        
        # Add features: distances with excluded_neighbors=0 (all 10 pairs)
        # Pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
        
        # Store reference to original data for verification
        self.original_data = self.pipeline.data.feature_data["distances"][0].data.copy()
        
    def _create_and_select_selector(self, name: str, feature_type: str, selection: str, **kwargs):
        """
        Create, configure, and execute a feature selector for testing.
        
        Convenience method that combines selector creation, selection addition,
        and selection execution in a single call. Cleans up any existing selector
        with the same name before creating a new one.
        
        Parameters
        ----------
        name : str
            Unique identifier for the feature selector
        feature_type : str
            Type of features to select (e.g., 'distances')
        selection : str
            Selection expression defining which features to include
        **kwargs : dict
            Additional parameters passed to add() method (e.g., use_reduced)
            
        Returns
        -------
        None
            Stores configured and executed selector in pipeline data
        """
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, feature_type, selection, **kwargs)
        self.pipeline.feature_selector.select(name, reference_traj=0)
    
    def _get_expected_feature_indices(self, selection: str):
        """
        Determine expected feature indices for a selection expression.
        
        Maps selection expressions to the corresponding column indices in the
        distance feature matrix. For 5 residues, distance pairs are ordered as:
        (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        at indices 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 respectively.
        
        Parameters
        ----------
        selection : str
            Selection expression (e.g., 'res ALA', 'seqid 1-3', 'all')
            
        Returns
        -------
        list of int
            Column indices in the distance feature matrix that should be selected
        """
        # Map known selections to their expected indices
        # Pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        #  Idx:   0      1      2      3      4      5      6      7      8      9
        
        if selection == "all":
            return list(range(10))  # All 10 pairs
        elif selection == "res ALA":
            # ALA residues at indices 0 and 3
            # Pairs with ALA: (0,1), (0,2), (0,3), (0,4), (1,3), (2,3), (3,4)
            return [0, 1, 2, 3, 5, 7, 9]  # 7 pairs
        elif selection == "res GLY":
            # GLY residues at indices 1 and 4
            # Pairs with GLY: (0,1), (0,4), (1,2), (1,3), (1,4), (2,4), (3,4)
            return [0, 3, 4, 5, 6, 8, 9]  # 7 pairs
        elif selection == "res VAL":
            # VAL residue at index 2
            # Pairs with VAL: (0,2), (1,2), (2,3), (2,4)
            return [1, 4, 7, 8]  # 4 pairs
        elif selection == "seqid 1-3":
            # Residues 0, 1, 2 (seqid 1, 2, 3)
            # All pairs within seqid 1-3: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4)
            return [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 9 pairs
        else:
            raise ValueError(f"Unknown selection: {selection}")
    
    def test_all_features_selection(self):
        """
        Test that all features selection returns complete original data.

        Validates that "all" selection returns identical data to original
        feature data (50 frames, 10 distance pairs).
        """
        self._create_and_select_selector("test", "distances", "all")
        
        selected_data = self.pipeline.data.get_selected_data("test")
        
        # Should return complete original data
        np.testing.assert_array_almost_equal(selected_data, self.original_data)
        assert selected_data.shape == (50, 10)
    
    def test_ala_features_selection(self):
        """
        Test that ALA features selection returns correct columns.

        Validates that "res ALA" selection returns only distance pairs with
        ALA residues (indices 0,3) - 7 out of 10 pairs.
        """
        self._create_and_select_selector("test", "distances", "res ALA")
        
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_indices = self._get_expected_feature_indices("res ALA")
        expected_data = self.original_data[:, expected_indices]
        
        # Should return exactly the ALA feature columns
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (50, 7)
    
    def test_gly_features_selection(self):
        """
        Test that GLY features selection returns correct columns.

        Validates that "res GLY" selection returns only distance pairs with
        GLY residues (indices 1,4) - 7 out of 10 pairs.
        """
        self._create_and_select_selector("test", "distances", "res GLY")
        
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_indices = self._get_expected_feature_indices("res GLY")
        expected_data = self.original_data[:, expected_indices]
        
        # Should return exactly the GLY feature columns
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (50, 7)
    
    def test_val_features_selection(self):
        """
        Test that VAL features selection returns correct columns.

        Validates that "res VAL" selection returns only distance pairs with
        VAL residue (index 2) - 4 out of 10 pairs.
        """
        self._create_and_select_selector("test", "distances", "res VAL")
        
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_indices = self._get_expected_feature_indices("res VAL")
        expected_data = self.original_data[:, expected_indices]
        
        # Should return exactly the VAL feature columns
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (50, 4)
    
    def test_seqid_range_selection(self):
        """
        Test that sequence ID range selection works correctly.

        Validates that "seqid 1-3" selection returns distance pairs for
        residues 1-3 (9 out of 10 pairs, excluding only 4-4).
        """
        self._create_and_select_selector("test", "distances", "seqid 1-3")
        
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_indices = self._get_expected_feature_indices("seqid 1-3")
        expected_data = self.original_data[:, expected_indices]
        
        # Should return exactly the seqid 1-3 feature columns
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (50, 9)


class TestGetSelectedDataWithDataSelector:
    """Test get_selected_data() with data selectors (frame selection)."""
    
    def setup_method(self):
        """
        Setup with data selector capabilities.

        Creates a simple synthetic trajectory with 5 residues and 100 frames.
        """
        # Reuse the same setup as above
        test_instance = TestGetSelectedDataVerification()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        self.original_data = test_instance.original_data
        self._create_and_select_selector = test_instance._create_and_select_selector
        self._get_expected_feature_indices = test_instance._get_expected_feature_indices
    
    def test_specific_frames_selection(self):
        """
        Test that specific frame selection with data selector works.

        Validates that data selector returns only selected frames (5,15,25,35,45)
        with correct frame mapping (5 frames, 10 features).
        """
        # Create data selector for specific frames
        self.pipeline.data_selector.create("specific_frames")
        selected_frames = [5, 15, 25, 35, 45]
        self.pipeline.data_selector.select_by_indices("specific_frames", {0: selected_frames})
        
        # Create feature selector
        self._create_and_select_selector("test", "distances", "all")
        
        # Get data with data selector
        selected_data, frame_mapping = self.pipeline.data.get_selected_data(
            "test", data_selector="specific_frames", return_frame_mapping=True
        )
        
        # Should return exactly the selected frames
        expected_data = self.original_data[selected_frames, :]
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (5, 10)
        
        # Verify frame mapping
        assert len(frame_mapping) == 5
        for i, global_idx in enumerate(frame_mapping.keys()):
            traj_idx, frame_idx = frame_mapping[global_idx]
            assert traj_idx == 0
            assert frame_idx == selected_frames[i]
    
    def test_frame_range_selection(self):
        """
        Test selecting frame range with data selector.

        Validates that data selector with frame range (10-20) returns correct frames
        and correct feature dimensions (11 frames, 10 features).
        """
        # Create data selector for frame range
        self.pipeline.data_selector.create("frame_range")
        self.pipeline.data_selector.select_by_indices("frame_range", {0: "10-20"})
        
        # Create feature selector
        self._create_and_select_selector("test", "distances", "all")
        
        # Get data with data selector
        selected_data = self.pipeline.data.get_selected_data("test", data_selector="frame_range")
        
        # Should return frames 10-20 (11 frames total)
        expected_data = self.original_data[10:21, :]
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (11, 10)
    
    def test_combined_feature_and_frame_selection(self):
        """
        Test combining feature selection and frame selection.

        Validates that combining feature selector (ALA features) with data selector
        (frames 5-15) returns the correct data.
        """
        # Create data selector for frames 5-15
        self.pipeline.data_selector.create("some_frames")
        self.pipeline.data_selector.select_by_indices("some_frames", {0: "5-15"})
        
        # Create feature selector for ALA residues
        self._create_and_select_selector("test", "distances", "res ALA")
        
        # Get data with both selectors
        selected_data = self.pipeline.data.get_selected_data("test", data_selector="some_frames")
        
        # Should return ALA features for frames 5-15
        frame_indices = list(range(5, 16))  # frames 5-15 (11 frames)
        feature_indices = self._get_expected_feature_indices("res ALA")  # 7 ALA features
        expected_data = self.original_data[np.ix_(frame_indices, feature_indices)]
        
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (11, 7)
    
    def test_stride_frame_selection(self):
        """
        Test that stride-based frame selection works correctly.

        Validates that data selector with stride returns correct frames
        and correct feature dimensions.
        0, 10, 20, 30, 40 (5 frames) with VAL features (4 features).
        """
        # Create data selector with stride
        self.pipeline.data_selector.create("stride_frames")
        self.pipeline.data_selector.select_by_indices("stride_frames", {0: {"frames": "0-40", "stride": 10}})
        
        # Create feature selector for VAL
        self._create_and_select_selector("test", "distances", "res VAL")
        
        # Get data with stride selection
        selected_data = self.pipeline.data.get_selected_data("test", data_selector="stride_frames")
        
        # Should return VAL features for frames 0, 10, 20, 30, 40
        frame_indices = [0, 10, 20, 30, 40]  # 5 frames
        feature_indices = self._get_expected_feature_indices("res VAL")  # 4 VAL features
        expected_data = self.original_data[np.ix_(frame_indices, feature_indices)]
        
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (5, 4)
    
    def test_empty_frame_selection(self):
        """
        Test with data selector that selects no frames.

        Validates that data selector with no frames returns empty result
        with appropriate handling and correct feature dimensions.
        """
        # Create empty data selector (this might cause issues, test robustness)
        self.pipeline.data_selector.create("empty")
        self.pipeline.data_selector.select_by_indices("empty", {0: []})
        
        # Create feature selector
        self._create_and_select_selector("test", "distances", "all")
        
        # Get data with empty frame selection
        selected_data = self.pipeline.data.get_selected_data("test", data_selector="empty")
        
        # Should return empty data with correct feature dimension
        assert selected_data.shape == (0, 10)
        assert isinstance(selected_data, np.ndarray)


class TestGetSelectedDataMultipleTrajectories:
    """Test get_selected_data() with multiple trajectories."""
    
    def setup_method(self):
        """
        Setup with multiple trajectories.

        Creates two synthetic trajectories with different frame counts
        to test multi-trajectory handling.
        """
        # Create first trajectory (same as before)
        test_instance = TestGetSelectedDataVerification()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        self._create_and_select_selector = test_instance._create_and_select_selector
        
        # Add second trajectory with different frame count
        topology = self.pipeline.data.trajectory_data.trajectories[0].topology
        coordinates = []
        for frame in range(30):  # Different frame count
            frame_coords = []
            for atom_idx in range(5):
                # Slightly different pattern to distinguish trajectories
                x = atom_idx * 2.0 + 1.0  # Offset by 1.0
                y = frame * 0.1 + 10.0    # Offset by 10.0
                z = 1.0                   # Different z value
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz2 = np.array(coordinates)
        test_traj2 = md.Trajectory(xyz2, topology)
        
        # Add second trajectory
        self.pipeline.data.trajectory_data.trajectories.append(test_traj2)
        self.pipeline.data.trajectory_data.trajectory_names.append("test_traj_2")
        
        # Copy residue metadata
        import copy
        self.pipeline.data.trajectory_data.res_label_data[1] = copy.deepcopy(self.pipeline.data.trajectory_data.res_label_data[0])
        
        # Recompute features for both trajectories  
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0), force=True)
        
        # Store reference to both trajectory data
        self.original_data_traj0 = self.pipeline.data.feature_data["distances"][0].data.copy()
        self.original_data_traj1 = self.pipeline.data.feature_data["distances"][1].data.copy()
    
    def test_all_trajectories_data(self):
        """
        Test that data from all trajectories is correctly concatenated.

        Validates that multi-trajectory pipeline correctly combines all trajectory data
        (total frames = sum of individual trajectory frames).
        """
        self._create_and_select_selector("test", "distances", "all")
        
        selected_data, frame_mapping = self.pipeline.data.get_selected_data("test", return_frame_mapping=True)
        
        # Should concatenate both trajectories: 50 + 30 = 80 frames
        expected_data = np.vstack([self.original_data_traj0, self.original_data_traj1])
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (80, 10)
        
        # Verify frame mapping
        assert len(frame_mapping) == 80
        
        # First 50 frames should map to trajectory 0
        for i in range(50):
            traj_idx, frame_idx = frame_mapping[i]
            assert traj_idx == 0
            assert frame_idx == i
        
        # Next 30 frames should map to trajectory 1  
        for i in range(50, 80):
            traj_idx, frame_idx = frame_mapping[i]
            assert traj_idx == 1
            assert frame_idx == i - 50
    
    def test_specific_trajectory_with_data_selector(self):
        """
        Test that data selector applies to specific trajectory correctly.

        Validates that data selector filters only frames from selected
        trajectory without affecting other trajectories.
        """
        # Create data selector for trajectory 1 only
        self.pipeline.data_selector.create("traj1_subset") 
        self.pipeline.data_selector.select_by_indices("traj1_subset", {1: [5, 10, 15, 20]})
        
        # Create feature selector
        self._create_and_select_selector("test", "distances", "res ALA")
        
        # Get data with data selector
        selected_data, frame_mapping = self.pipeline.data.get_selected_data(
            "test", data_selector="traj1_subset", return_frame_mapping=True
        )
        
        # Should return only selected frames from trajectory 1
        expected_frames = [5, 10, 15, 20]
        # ALA features: indices [0, 1, 2, 3, 5, 7, 9]
        ala_indices = [0, 1, 2, 3, 5, 7, 9]
        expected_data = self.original_data_traj1[np.ix_(expected_frames, ala_indices)]
        
        np.testing.assert_array_almost_equal(selected_data, expected_data)
        assert selected_data.shape == (4, 7)
        
        # Verify frame mapping points to trajectory 1
        assert len(frame_mapping) == 4
        for i, (_, (traj_idx, frame_idx)) in enumerate(frame_mapping.items()):
            assert traj_idx == 1
            assert frame_idx == expected_frames[i]


class TestGetSelectedDataComplexScenarios:
    """Test complex multi-trajectory scenarios with tags, clusters, and advanced selections."""
    
    def setup_method(self):
        """
        Setup 3 trajectories with different characteristics.
        
        Creates three synthetic trajectories with distinct residue sequences,
        """
        # Create three different trajectory topologies
        self.pipeline = PipelineManager()
        
        # Trajectory 1: VAL-GLY-ALA-ALA-GLY (tagged as "system_A")
        traj1 = self._create_trajectory(["VAL", "GLY", "ALA", "ALA", "GLY"], 100)
        self.pipeline.data.trajectory_data.trajectories.append(traj1)
        self.pipeline.data.trajectory_data.trajectory_names.append("system_A")
        
        # Trajectory 2: VAL-ALA-ALA-VAL-ALA (tagged as "system_B")
        traj2 = self._create_trajectory(["VAL", "ALA", "ALA", "VAL", "ALA"], 100)
        self.pipeline.data.trajectory_data.trajectories.append(traj2)
        self.pipeline.data.trajectory_data.trajectory_names.append("system_B")
        
        # Trajectory 3: VAL-VAL-ALA-GLY-VAL (tagged as "system_C")
        traj3 = self._create_trajectory(["VAL", "VAL", "ALA", "GLY", "VAL"], 120)
        self.pipeline.data.trajectory_data.trajectories.append(traj3)
        self.pipeline.data.trajectory_data.trajectory_names.append("system_C")
        
        # Add tags
        self.pipeline.trajectory.add_tags(0, ["system_A", "folded", "wildtype"])
        self.pipeline.trajectory.add_tags(1, ["system_B", "unfolded", "mutant"]) 
        self.pipeline.trajectory.add_tags(2, ["system_C", "intermediate", "wildtype"])
        
        # Create residue metadata with HARDCODED consensus labels (nachvollziehbar)
        consensus_labels = [
            ["1.10", "2.10", "3.50", "4.10", "5.50"],  # Trajectory 0: ALA-GLY-VAL-ALA-GLY
            ["1.50", "2.50", "3.50", "4.50", "5.50"],  # Trajectory 1: VAL-ALA-GLY-VAL-ALA  
            ["1.90", "2.90", "3.90", "4.90", "5.90"]   # Trajectory 2: GLY-VAL-ALA-GLY-VAL
        ]
        
        for traj_idx in range(3):
            traj = self.pipeline.data.trajectory_data.trajectories[traj_idx]
            residue_metadata = []
            for i, res in enumerate(traj.topology.residues):
                residue_metadata.append({
                    "resid": res.resSeq,
                    "seqid": res.index,
                    "resname": res.name,
                    "aaa_code": res.name,
                    "a_code": res.name[0],
                    "consensus": consensus_labels[traj_idx][i],
                    "full_name": f"{res.name}{res.index}",
                    "index": res.index
                })
            self.pipeline.data.trajectory_data.res_label_data[traj_idx] = residue_metadata
        
        # Add features
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
        
        # Create mock clustering for trajectory 1 (frames 20-40 as cluster 0) - CORRECTED
        from mdxplain.clustering.entities.cluster_data import ClusterData
        
        # Create cluster assignments: cluster 0 = frames 20-40, cluster 1 = all other frames
        labels = []
        for frame in range(100):
            if 20 <= frame <= 40:
                labels.append(0)  # Cluster 0 for frames 20-40
            else:
                labels.append(1)  # Cluster 1 for other frames
        
        cluster_data = ClusterData("test_clusters")
        cluster_data.labels = np.array(labels)  # Direct numpy array assignment like working tests
        cluster_data.n_clusters = 2
        
        # Create frame mapping: global_frame_index -> (trajectory_index, local_frame_index)
        frame_mapping = {}
        for frame_idx in range(100):
            frame_mapping[frame_idx] = (1, frame_idx)  # All frames map to trajectory 1
        cluster_data.set_frame_mapping(frame_mapping)  # Use set_frame_mapping() like working tests
        
        self.pipeline.data.cluster_data["test_clusters"] = cluster_data
        
    def _create_trajectory(self, residue_names, n_frames):
        """
        Create an MDTraj trajectory with specified residues and frame count.
        
        Constructs a synthetic trajectory with predictable coordinate patterns
        for testing purposes. Each residue gets a CA atom positioned at regular
        intervals with slight trajectory-specific offsets for differentiation.
        
        Parameters
        ----------
        residue_names : list of str
            List of residue names (3-letter codes) for the topology
        n_frames : int
            Number of frames in the trajectory
            
        Returns
        -------
        mdtraj.Trajectory
            Synthetic trajectory with specified topology and predictable coordinates
        """
        topology = md.Topology()
        chain = topology.add_chain()
        
        residues = []
        for name in residue_names:
            residues.append(topology.add_residue(name, chain))
        
        # Add CA atoms
        for residue in residues:
            topology.add_atom("CA", md.element.carbon, residue)
        
        # Create coordinates with distinct patterns per trajectory
        coordinates = []
        for frame in range(n_frames):
            frame_coords = []
            for atom_idx in range(len(residue_names)):
                x = atom_idx * 2.0 + len(residue_names) * 0.1  # Slight offset per trajectory
                y = frame * 0.1
                z = len(residue_names) * 0.5  # Z offset per trajectory
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        return md.Trajectory(xyz, topology)
    
    def test_complex_multi_trajectory_selection(self):
        """
        Test complex scenario: one DataSelector with 3 operations, one FeatureSelector with 4 adds.

        Validates that complex multi-trajectory selection with various
        data/feature selector combinations works correctly.
        """
        # === ONE DATA SELECTOR WITH 3 OPERATIONS ===
        
        self.pipeline.data_selector.create("complex_selector")
        
        # 1. First: select by indices for trajectory 0 (frames 10-50)
        self.pipeline.data_selector.select_by_indices("complex_selector", {0: "10-50"})
        
        # 2. Then: add frames by tags trajectory 2
        self.pipeline.data_selector.select_by_tags("complex_selector", ["wildtype", "intermediate"], match_all=True, mode="add")
        
        # 3. Finally: intersect with cluster 0 frames (only trajectory 1, frames 20-40)
        self.pipeline.data_selector.select_by_cluster("complex_selector", "test_clusters", cluster_ids=[0], mode="add")

        # Traj 0 frames 10-50, Traj 1 frames 20-40, Traj 2 all frames      
        
        # === ONE FEATURE SELECTOR WITH 4 COMPLEX ADDS ===
        
        self.pipeline.feature_selector.create("complex_features")
        
        # 1. ALA residues with require_all_partners=False
        # For all Trajs there is a common ALA at pos 2
        self.pipeline.feature_selector.add(
            "complex_features", "distances", "res ALA2",
            use_reduced=False, common_denominator=False, require_all_partners=False, traj_selection=[0]
        )
        
        # 2. Consensus selection with specific range
        # Traj 0: 2,3,4, Traj 1: 2,3,4 Traj 2: none
        self.pipeline.feature_selector.add(
            "complex_features", "distances", "consensus 3.50-5.50",
            use_reduced=False, common_denominator=False, require_all_partners=True
        )
        
        # 3. Seqid range with trajectory selection
        # Traj 2 -> positions 3,4
        self.pipeline.feature_selector.add(
            "complex_features", "distances", "seqid 3-4",
            use_reduced=False, common_denominator=False, traj_selection=[2]
        )
        
        # 4. VAL residues with require_all_partners=True for traj 2 only
        # For all Trajs there is a common VAL at pos 0
        self.pipeline.feature_selector.add(
            "complex_features", "distances", "res VAL0",
            use_reduced=False, common_denominator=False, require_all_partners=False, traj_selection=[1]
        )
        
        # Apply with reference trajectory 2 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Expected empty selections in consensus ADD
            self.pipeline.feature_selector.select("complex_features", reference_traj=2)
        
        # === ONE get_selected_data CALL ===
        
        selected_data, frame_mapping = self.pipeline.data.get_selected_data(
            "complex_features", data_selector="complex_selector", return_frame_mapping=True
        )
        
        # Test frame mapping independently (not using it to build expected matrix!)
        expected_frame_mapping = {}
        
        # Trajectory 0: frames 10-50 (41 frames) -> global indices 0-40
        for i, local_frame in enumerate(range(10, 51)):
            expected_frame_mapping[i] = (0, local_frame)
        
        # Trajectory 1: frames 20-40 (21 frames) -> global indices 41-61  
        for i, local_frame in enumerate(range(20, 41)):
            expected_frame_mapping[41 + i] = (1, local_frame)
        
        # Trajectory 2: frames 0-119 (120 frames) -> global indices 62-181
        for i, local_frame in enumerate(range(0, 120)):
            expected_frame_mapping[62 + i] = (2, local_frame)
        
        # Test frame mapping
        assert frame_mapping == expected_frame_mapping, f"Frame mapping mismatch"
        
        # Build expected matrix with independent logic (not using frame_mapping output!)
        expected_rows = []
        
        # Get original feature data
        original_data_0 = self.pipeline.data.feature_data["distances"][0].data
        original_data_1 = self.pipeline.data.feature_data["distances"][1].data
        original_data_2 = self.pipeline.data.feature_data["distances"][2].data
        
        # Expected features (now SORTED due to fix above)
        # Trajectory 0: frames 10-50, features [1, 2, 3, 4, 7, 8, 9] (SORTED)
        traj_0_indices = [1, 2, 3, 4, 7, 8, 9]
        for frame in range(10, 51):
            row = original_data_0[frame, traj_0_indices]
            expected_rows.append(row)
        
        # Trajectory 1: frames 20-40, features [0, 1, 2, 3, 7, 8, 9] (SORTED)
        traj_1_indices = [0, 1, 2, 3, 7, 8, 9]
        for frame in range(20, 41):
            row = original_data_1[frame, traj_1_indices]
            expected_rows.append(row)
        
        # Trajectory 2: frames 0-119, features [2, 3, 5, 6, 7, 8, 9] (SORTED)
        traj_2_indices = [2, 3, 5, 6, 7, 8, 9]
        for frame in range(0, 120):
            row = original_data_2[frame, traj_2_indices]
            expected_rows.append(row)
        
        expected_matrix = np.array(expected_rows)
        np.testing.assert_array_almost_equal(selected_data, expected_matrix)


class TestGetSelectedDataEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """
        Setup for edge case testing.

        Reuse the same setup as above.d
        """
        test_instance = TestGetSelectedDataVerification()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        self._create_and_select_selector = test_instance._create_and_select_selector
        self._get_expected_feature_indices = test_instance._get_expected_feature_indices
    
    def test_nonexistent_feature_selector(self):
        """
        Test error when feature selector does not exist.

        Validates that get_selected_data() with non-existent
        feature selector raises appropriate KeyError.
        """
        with pytest.raises(ValueError):
            self.pipeline.data.get_selected_data("nonexistent")
    
    def test_nonexistent_data_selector(self):
        """
        Test error when data selector does not exist.

        Validates that get_selected_data() with non-existent
        data selector raises appropriate KeyError.
        """
        self._create_and_select_selector("test", "distances", "all")
        
        with pytest.raises(KeyError):
            self.pipeline.data.get_selected_data("test", data_selector="nonexistent")
    
    def test_empty_feature_selection(self):
        """
        Test that behavior with empty feature selection raises ValueError.

        Validates that feature selector without selected features
        causes ValueError with descriptive error message.
        """
        # Empty selection should raise ValueError due to no reference trajectory
        with pytest.raises(ValueError, match="Reference trajectory 0 not found in selection results"):
            self._create_and_select_selector("test", "distances", "res XYZ")  # Non-existent residue
    
    def test_data_consistency_after_multiple_selections(self):
        """
        Test that data remains consistent after multiple selector operations.

        Validates that repeated get_selected_data() calls with
        identical parameters return identical results.
        """
        # Create multiple selectors
        self._create_and_select_selector("ala_sel", "distances", "res ALA")
        self._create_and_select_selector("gly_sel", "distances", "res GLY") 
        self._create_and_select_selector("all_sel", "distances", "all")
        
        # Each should return consistent data
        ala_data = self.pipeline.data.get_selected_data("ala_sel")
        gly_data = self.pipeline.data.get_selected_data("gly_sel")
        all_data = self.pipeline.data.get_selected_data("all_sel")
        
        # Basic consistency checks
        assert ala_data.shape == (50, 7)
        assert gly_data.shape == (50, 7) 
        assert all_data.shape == (50, 10)
        
        # ACTUAL CONSISTENCY TEST: Verify data comes from correct columns
        ala_indices = self._get_expected_feature_indices("res ALA")
        gly_indices = self._get_expected_feature_indices("res GLY")
        
        # ALA data should match corresponding columns in all_data
        expected_ala_from_all = all_data[:, ala_indices]
        np.testing.assert_array_almost_equal(ala_data, expected_ala_from_all)
        
        # GLY data should match corresponding columns in all_data
        expected_gly_from_all = all_data[:, gly_indices]
        np.testing.assert_array_almost_equal(gly_data, expected_gly_from_all)
        
        # Data should not be corrupted
        assert not np.any(np.isnan(ala_data))
        assert not np.any(np.isnan(gly_data))
        assert not np.any(np.isnan(all_data))
    