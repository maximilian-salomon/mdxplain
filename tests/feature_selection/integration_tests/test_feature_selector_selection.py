"""Comprehensive tests for FeatureSelector selection logic and parameters."""

import numpy as np
import pytest
import warnings
import mdtraj as md
from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances
from mdxplain.feature.feature_type.contacts.contacts import Contacts


class TestSelectionStrings:
    """Test various selection string patterns and categories."""
    
    def setup_method(self):
        """Create synthetic trajectory with controlled properties."""
        # Create topology: ALA(1), GLY(2), VAL(3), ALA(4), GLY(5), LEU(6), PHE(7)
        topology = md.Topology()
        chain = topology.add_chain()
        
        residue_names = ["ALA", "GLY", "VAL", "ALA", "GLY", "LEU", "PHE"]
        residues = []
        
        for name in residue_names:
            residues.append(topology.add_residue(name, chain))
        
        # Add CA atoms
        for residue in residues:
            topology.add_atom("CA", md.element.carbon, residue)
        
        # Create coordinates (7 atoms, 20 frames)
        coordinates = []
        for frame in range(20):
            frame_coords = []
            for atom_idx in range(7):
                x = atom_idx * 1.0
                y = frame * 0.1
                z = 0.0
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        self.test_traj = md.Trajectory(xyz, topology)
        
        # Setup pipeline
        self.pipeline = PipelineManager()
        self.pipeline.data.trajectory_data.trajectories = [self.test_traj]
        self.pipeline.data.trajectory_data.trajectory_names = ["synthetic"]
        
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
        
        # Add features: distances with excluded_neighbors=0 (all pairs)
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
    
    def _create_selector_and_add(self, name: str, selection: str, **kwargs):
        """Helper to create selector and add selection."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, "distances", selection, **kwargs)
        return name
    
    def _get_selected_indices(self, selector_name: str):
        """Helper to get selected indices from completed selection."""
        self.pipeline.feature_selector.select(selector_name, reference_traj=0)
        return self.pipeline.data.selected_feature_data[selector_name].get_results("distances")["trajectory_indices"][0]["indices"]
    
    def test_explicit_seqid_single(self):
        """Test explicit 'seqid 3' selection."""
        name = self._create_selector_and_add("test", "seqid 3")
        indices = self._get_selected_indices(name)
        
        # seqid 3 is residue index 2 (VAL)
        # Should return pairs containing residue 2: (0,2), (1,2), (2,3), (2,4), (2,5), (2,6)
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i == 2 or j == 2:  # Residue 2 (seqid 3)
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_explicit_seqid_range(self):
        """Test explicit 'seqid 2-4' selection."""
        name = self._create_selector_and_add("test", "seqid 2-4")
        indices = self._get_selected_indices(name)
        
        # seqid 2-4 are residue indices 1,2,3
        # Should return pairs containing any of these residues
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [1, 2, 3] or j in [1, 2, 3]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_explicit_seqid_multiple(self):
        """Test explicit 'seqid 1 3 5' selection."""
        name = self._create_selector_and_add("test", "seqid 1 3 5")
        indices = self._get_selected_indices(name)
        
        # seqid 1,3,5 are residue indices 0,2,4
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 2, 4] or j in [0, 2, 4]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_explicit_resid_single(self):
        """Test explicit 'resid 3' selection."""
        name = self._create_selector_and_add("test", "resid 3")
        indices = self._get_selected_indices(name)
        
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i == 3 or j == 3:  # Residue 3 (index=3, ALA)
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_explicit_res_single(self):
        """Test explicit 'res ALA' selection."""
        name = self._create_selector_and_add("test", "res ALA")
        indices = self._get_selected_indices(name)
        
        # ALA residues are at indices 0 and 3
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_explicit_res_multiple(self):
        """Test explicit 'res ALA GLY' selection."""
        name = self._create_selector_and_add("test", "res ALA GLY")
        indices = self._get_selected_indices(name)
        
        # ALA residues: 0,3; GLY residues: 1,4
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 1, 3, 4] or j in [0, 1, 3, 4]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    
    def test_smart_detection_numeric(self):
        """Test smart pattern detection for '3' → 'seqid 3'."""
        name = self._create_selector_and_add("test", "3")
        indices = self._get_selected_indices(name)
        
        # Should be same as explicit seqid 3
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i == 2 or j == 2:  # seqid 3 = index 2
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_smart_detection_range(self):
        """Test smart pattern detection for '2-4' → 'seqid 2-4'."""
        name = self._create_selector_and_add("test", "2-4")
        indices = self._get_selected_indices(name)
        
        # Should be same as explicit seqid 2-4
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [1, 2, 3] or j in [1, 2, 3]:  # seqid 2-4 = indices 1,2,3
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_smart_detection_amino_acid(self):
        """Test smart pattern detection for 'ALA' → 'res ALA'."""
        name = self._create_selector_and_add("test", "ALA")
        indices = self._get_selected_indices(name)
        
        # Should be same as explicit res ALA
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:  # ALA at indices 0,3
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_smart_detection_single_letter(self):
        """Test smart pattern detection for 'A' → 'res A'."""
        name = self._create_selector_and_add("test", "A")
        indices = self._get_selected_indices(name)
        
        # A should match ALA residues
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_special_keyword_all_lowercase(self):
        """Test 'all' keyword."""
        name = self._create_selector_and_add("test", "all")
        indices = self._get_selected_indices(name)
        
        # Should return all pairs
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected = list(range(total_pairs))
        
        assert sorted(indices) == sorted(expected)
    
    def test_special_keyword_all_uppercase(self):
        """Test 'ALL' keyword."""
        name = self._create_selector_and_add("test", "ALL")
        indices = self._get_selected_indices(name)
        
        # Should return all pairs
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected = list(range(total_pairs))
        
        assert sorted(indices) == sorted(expected)
    
    def test_special_keyword_all_mixed_case(self):
        """Test 'All' keyword."""
        name = self._create_selector_and_add("test", "All")
        indices = self._get_selected_indices(name)
        
        # Should return all pairs
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected = list(range(total_pairs))
        
        assert sorted(indices) == sorted(expected)
    
    def test_combination_and(self):
        """Test 'res ALA and seqid 3-5' combination (UNION operation)."""
        name = self._create_selector_and_add("test", "res ALA and seqid 3-5")
        indices = self._get_selected_indices(name)
        
        # AND creates UNION of selections:
        # res ALA = residues 0,3 (ALA at positions 0 and 3)  
        # seqid 3-5 = residues with seqid 3,4,5 = indices 2,3,4 (because seqid = index + 1)
        # UNION = residues 0,2,3,4
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 2, 3, 4] or j in [0, 2, 3, 4]:  # Union of both selections
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_combination_not(self):
        """Test 'not res VAL' selection."""
        name = self._create_selector_and_add("test", "not res VAL")
        indices = self._get_selected_indices(name)
        
        # VAL is at index 2, so exclude pairs containing index 2
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i != 2 and j != 2:  # Exclude residue 2 (VAL)
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)


class TestParameters:
    """Test different parameter combinations."""
    
    def setup_method(self):
        """Setup with synthetic trajectory."""
        # Reuse setup from TestSelectionStrings
        test_instance = TestSelectionStrings()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        self.test_traj = test_instance.test_traj
    
    def _create_selector_and_add(self, name: str, selection: str, **kwargs):
        """Helper to create selector and add selection."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, "distances", selection, **kwargs)
        return name
    
    def _get_selected_indices(self, selector_name: str):
        """Helper to get selected indices."""
        self.pipeline.feature_selector.select(selector_name, reference_traj=0)
        return self.pipeline.data.selected_feature_data[selector_name].get_results("distances")["trajectory_indices"][0]["indices"]
    
    def test_use_reduced_true_with_data(self):
        """Test use_reduced=True when reduced data exists."""
        # First create reduced data
        feature_data_obj = self.pipeline.data.feature_data["distances"][0]
        
        # Create reduced data with first 5 features
        reduced_data = feature_data_obj.data[:, :5]
        original_features = feature_data_obj.feature_metadata["features"]
        reduced_features = original_features[:5]
        
        feature_data_obj.reduced_data = reduced_data
        feature_data_obj.reduced_feature_metadata = {
            "is_pair": True,
            "features": reduced_features
        }
        
        # Test with use_reduced=True
        name = self._create_selector_and_add("test", "all", use_reduced=True)
        indices = self._get_selected_indices(name)
        
        # Should return indices 0-4 (all reduced features)
        expected = [0, 1, 2, 3, 4]
        assert sorted(indices) == sorted(expected)
    
    def test_use_reduced_false(self):
        """Test use_reduced=False uses full data."""
        name = self._create_selector_and_add("test", "all", use_reduced=False)
        indices = self._get_selected_indices(name)
        
        # Should return all pairs from full data
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected = list(range(total_pairs))
        assert sorted(indices) == sorted(expected)
    
    def test_require_all_partners_true(self):
        """Test require_all_partners=True for pairwise features."""
        name = self._create_selector_and_add("test", "res ALA", require_all_partners=True)
        indices = self._get_selected_indices(name)
        
        # With require_all_partners=True, only pairs where BOTH partners are ALA
        # ALA residues: 0, 3
        # Only pair (0,3) has both partners as ALA
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] and j in [0, 3]:  # Both partners must be ALA
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_require_all_partners_false(self):
        """Test require_all_partners=False (default) for pairwise features."""
        name = self._create_selector_and_add("test", "res ALA", require_all_partners=False)
        indices = self._get_selected_indices(name)
        
        # With require_all_partners=False, pairs where ANY partner is ALA
        # ALA residues: 0, 3
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:  # Any partner is ALA
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)


class TestFeatureSelectorReferenceTrajectoryLabels:
    """Test that feature selector uses labels from the correct reference trajectory."""
    
    def setup_method(self):
        """Setup with 2 trajectories having different residue labels."""
        self.pipeline = PipelineManager()
        
        # Trajectory 0: ALA-GLY-VAL with consensus labels 1.50, 2.50, 3.50
        traj0 = self._create_trajectory(["ALA", "GLY", "VAL"], 50)
        self.pipeline.data.trajectory_data.trajectories.append(traj0)
        self.pipeline.data.trajectory_data.trajectory_names.append("traj0")
        
        # Trajectory 1: VAL-ALA-GLY with DIFFERENT consensus labels 4.50, 5.50, 6.50
        traj1 = self._create_trajectory(["VAL", "ALA", "GLY"], 40)
        self.pipeline.data.trajectory_data.trajectories.append(traj1)
        self.pipeline.data.trajectory_data.trajectory_names.append("traj1")
        
        # Create DIFFERENT residue metadata for each trajectory
        # Trajectory 0 metadata
        residue_metadata_0 = [
            {"resid": 1, "seqid": 1, "resname": "ALA", "consensus": "1.50", "full_name": "ALA1", "index": 0},
            {"resid": 2, "seqid": 2, "resname": "GLY", "consensus": "2.50", "full_name": "GLY2", "index": 1},
            {"resid": 3, "seqid": 3, "resname": "VAL", "consensus": "3.50", "full_name": "VAL3", "index": 2}
        ]
        
        # Trajectory 1 metadata (DIFFERENT consensus numbers)
        residue_metadata_1 = [
            {"resid": 1, "seqid": 1, "resname": "VAL", "consensus": "4.50", "full_name": "VAL1", "index": 0},
            {"resid": 2, "seqid": 2, "resname": "ALA", "consensus": "5.50", "full_name": "ALA2", "index": 1},
            {"resid": 3, "seqid": 3, "resname": "GLY", "consensus": "6.50", "full_name": "GLY3", "index": 2}
        ]
        
        self.pipeline.data.trajectory_data.res_label_data = {
            0: residue_metadata_0,
            1: residue_metadata_1
        }
        
        # Add features
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
    
    def _create_trajectory(self, residue_names, n_frames):
        """Helper to create trajectory."""
        topology = md.Topology()
        chain = topology.add_chain()
        
        for name in residue_names:
            residue = topology.add_residue(name, chain)
            topology.add_atom("CA", md.element.carbon, residue)
        
        # Create simple coordinates
        coordinates = []
        for frame in range(n_frames):
            frame_coords = [[i * 2.0, frame * 0.1, 0.0] for i in range(len(residue_names))]
            coordinates.append(frame_coords)
        
        return md.Trajectory(np.array(coordinates), topology)
    
    def test_seqid_1_with_reference_trajectory_0_gives_exact_ala_labels(self):
        """Test: seqid 1 with reference trajectory 0 gives exactly ALA-based labels."""
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add("test", "distances", "seqid 1", common_denominator=False)
        self.pipeline.feature_selector.select("test", reference_traj=0)
        
        metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Must find exactly 2 features
        assert len(metadata) == 2
        
        # Feature 0 must be ALA1-GLY2 pair
        feature0 = metadata[0]["features"]
        assert feature0[0]["residue"]["resname"] == "ALA"
        assert feature0[0]["residue"]["seqid"] == 1
        assert feature0[0]["residue"]["consensus"] == "1.50"
        assert feature0[1]["residue"]["resname"] == "GLY"
        assert feature0[1]["residue"]["seqid"] == 2
        assert feature0[1]["residue"]["consensus"] == "2.50"
        
        # Feature 1 must be ALA1-VAL3 pair
        feature1 = metadata[1]["features"]
        assert feature1[0]["residue"]["resname"] == "ALA"
        assert feature1[0]["residue"]["seqid"] == 1
        assert feature1[0]["residue"]["consensus"] == "1.50"
        assert feature1[1]["residue"]["resname"] == "VAL"
        assert feature1[1]["residue"]["seqid"] == 3
        assert feature1[1]["residue"]["consensus"] == "3.50"
    
    def test_seqid_1_with_reference_trajectory_1_gives_exact_val_labels(self):
        """Test: seqid 1 with reference trajectory 1 gives exactly VAL-based labels."""  
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add("test", "distances", "seqid 1", common_denominator=False)
        self.pipeline.feature_selector.select("test", reference_traj=1)
        
        metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Must find exactly 2 features
        assert len(metadata) == 2
        
        # Feature 0 must be VAL1-ALA2 pair (from trajectory 1 labels)
        feature0 = metadata[0]["features"]
        assert feature0[0]["residue"]["resname"] == "VAL"
        assert feature0[0]["residue"]["seqid"] == 1
        assert feature0[0]["residue"]["consensus"] == "4.50"
        assert feature0[1]["residue"]["resname"] == "ALA"
        assert feature0[1]["residue"]["seqid"] == 2
        assert feature0[1]["residue"]["consensus"] == "5.50"
        
        # Feature 1 must be VAL1-GLY3 pair (from trajectory 1 labels)  
        feature1 = metadata[1]["features"]
        assert feature1[0]["residue"]["resname"] == "VAL"
        assert feature1[0]["residue"]["seqid"] == 1
        assert feature1[0]["residue"]["consensus"] == "4.50"
        assert feature1[1]["residue"]["resname"] == "GLY"
        assert feature1[1]["residue"]["seqid"] == 3
        assert feature1[1]["residue"]["consensus"] == "6.50"
    
    def test_consensus_4_50_with_reference_trajectory_0_finds_features_but_metadata_fails(self):
        """Test that consensus 4.50 with reference trajectory 0 finds features but metadata access fails."""
        self.pipeline.feature_selector.create("test_wrong")
        
        # Select trajectory 1 consensus (4.50) but use trajectory 0 as reference
        self.pipeline.feature_selector.add("test_wrong", "distances", "consensus 4.50")
        
        # Selection succeeds (finds features in trajectory 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Expected empty selection for traj 0
            self.pipeline.feature_selector.select("test_wrong", reference_traj=0)
        
        # Features are found in trajectory 1 
        results = self.pipeline.data.selected_feature_data["test_wrong"].get_results("distances")
        assert "trajectory_indices" in results
        assert 1 in results["trajectory_indices"]
        assert len(results["trajectory_indices"][1]["indices"]) == 2
        
        # But metadata access fails because reference trajectory 0 has no features
        with pytest.raises(ValueError, match="Reference trajectory 0 not found in selection"):
            self.pipeline.data.get_selected_metadata("test_wrong")
    
    def test_consensus_1_50_with_reference_trajectory_0_finds_exact_ala_position(self):
        """Test: consensus 1.50 with reference trajectory 0 finds exactly ALA position."""
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add("test", "distances", "consensus 1.50", common_denominator=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Expected empty selection in some trajectories
            self.pipeline.feature_selector.select("test", reference_traj=0)
        
        metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Must find exactly 2 features (ALA at position 0 paired with others)
        assert len(metadata) == 2
        
        # Feature 0 must be ALA1-GLY2 pair
        feature0 = metadata[0]["features"]
        assert feature0[0]["residue"]["consensus"] == "1.50"
        assert feature0[0]["residue"]["resname"] == "ALA"
        assert feature0[1]["residue"]["consensus"] == "2.50"
        assert feature0[1]["residue"]["resname"] == "GLY"
        
        # Feature 1 must be ALA1-VAL3 pair
        feature1 = metadata[1]["features"]
        assert feature1[0]["residue"]["consensus"] == "1.50"
        assert feature1[0]["residue"]["resname"] == "ALA"
        assert feature1[1]["residue"]["consensus"] == "3.50"
        assert feature1[1]["residue"]["resname"] == "VAL"
    
    def test_consensus_1_50_with_reference_trajectory_1_finds_features_but_metadata_fails(self):
        """Test: consensus 1.50 with reference trajectory 1 finds features but metadata access fails."""
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add("test", "distances", "consensus 1.50", common_denominator=False)
        
        # Selection succeeds (finds features in trajectory 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Expected empty selection for traj 1
            self.pipeline.feature_selector.select("test", reference_traj=1)
        
        # Features are found in trajectory 0 
        results = self.pipeline.data.selected_feature_data["test"].get_results("distances")
        assert "trajectory_indices" in results
        assert 0 in results["trajectory_indices"] 
        assert len(results["trajectory_indices"][0]["indices"]) == 2
        
        # But metadata access fails because reference trajectory 1 has no features
        with pytest.raises(ValueError, match="Reference trajectory 1 not found in selection"):
            self.pipeline.data.get_selected_metadata("test")


class TestFeatureSelectorMatrixConsistency:
    """Test matrix consistency warnings when feature counts differ between trajectories."""
    
    def setup_method(self):
        """Setup with 2 trajectories having different residue counts."""
        self.pipeline = PipelineManager()
        
        # Trajectory 0: 3 residues (ALA-GLY-VAL) -> 3 pairs
        traj0 = self._create_trajectory(["ALA", "GLY", "VAL"], 30)
        self.pipeline.data.trajectory_data.trajectories.append(traj0)
        self.pipeline.data.trajectory_data.trajectory_names.append("short")
        
        # Trajectory 1: 5 residues (ALA-GLY-VAL-ALA-GLY) -> 10 pairs
        traj1 = self._create_trajectory(["ALA", "GLY", "VAL", "ALA", "GLY"], 40)
        self.pipeline.data.trajectory_data.trajectories.append(traj1)
        self.pipeline.data.trajectory_data.trajectory_names.append("long")
        
        # Create metadata
        for traj_idx in range(2):
            traj = self.pipeline.data.trajectory_data.trajectories[traj_idx]
            residue_metadata = []
            for i, res in enumerate(traj.topology.residues):
                residue_metadata.append({
                    "resid": res.resSeq + 1,
                    "seqid": res.index + 1,
                    "resname": res.name,
                    "consensus": f"{i+1}.50",
                    "full_name": f"{res.name}{res.index + 1}",
                    "index": res.index
                })
            self.pipeline.data.trajectory_data.res_label_data[traj_idx] = residue_metadata
        
        # Add features - this creates different numbers of features per trajectory
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
    
    def _create_trajectory(self, residue_names, n_frames):
        """Helper to create trajectory."""
        topology = md.Topology()
        chain = topology.add_chain()
        
        for name in residue_names:
            residue = topology.add_residue(name, chain)
            topology.add_atom("CA", md.element.carbon, residue)
        
        coordinates = []
        for frame in range(n_frames):
            frame_coords = [[i * 2.0, frame * 0.1, 0.0] for i in range(len(residue_names))]
            coordinates.append(frame_coords)
        
        return md.Trajectory(np.array(coordinates), topology)
    
    def test_matrix_consistency_warning_on_select(self):
        """Test that inconsistent feature counts trigger warning during select()."""
        self.pipeline.feature_selector.create("inconsistent")
        
        # Select "all" features with common_denominator=False - this will have different counts per trajectory
        # Trajectory 0: 3 residues -> 3 pairs
        # Trajectory 1: 5 residues -> 10 pairs
        self.pipeline.feature_selector.add("inconsistent", "distances", "all", common_denominator=False)
        
        # Should trigger ValueError about inconsistent matrix dimensions
        with pytest.raises(ValueError, match="Feature 'distances' has inconsistent column counts"):
            self.pipeline.feature_selector.select("inconsistent", reference_traj=0)
    
    def test_matrix_consistency_with_common_denominator(self):
        """Test that common_denominator=True resolves inconsistency."""
        self.pipeline.feature_selector.create("consistent")
        
        # Use common_denominator=True to resolve inconsistency
        self.pipeline.feature_selector.add(
            "consistent", "distances", "all", 
            common_denominator=True
        )
        
        # Should NOT trigger warning because common_denominator handles it
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.pipeline.feature_selector.select("consistent", reference_traj=0)
            
            # No warnings should be triggered
            assert len(w) == 0
    
    def test_matrix_consistency_specific_selection(self):
        """Test that specific selections with common_denominator finds intersection."""
        self.pipeline.feature_selector.create("specific")
        
        # Select only ALA residues - both trajectories have ALA
        self.pipeline.feature_selector.add("specific", "distances", "res ALA")
        
        # Selection processes but finds no common features
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.pipeline.feature_selector.select("specific", reference_traj=0)
            
            # Warnings are expected when no common features found
            assert len(w) == 2  # One warning per trajectory with no matches
            for warning in w:
                assert "Feature selection resulted in no matches" in str(warning.message)


class TestMultipleFeatureTypes:
    """Test multiple feature types and multiple adds."""
    
    def setup_method(self):
        """Setup with multiple feature types."""
        test_instance = TestSelectionStrings()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def _create_selector(self, name: str):
        """Helper to create empty selector."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
    
    def test_single_feature_type_multiple_selections(self):
        """Test same feature type with multiple different selections."""
        self._create_selector("test")
        
        # Add multiple selections for distances
        self.pipeline.feature_selector.add("test", "distances", "res ALA")
        self.pipeline.feature_selector.add("test", "distances", "res GLY")
        
        self.pipeline.feature_selector.select("test", reference_traj=0)
        indices = self.pipeline.data.selected_feature_data["test"].get_results("distances")["trajectory_indices"][0]["indices"]
        
        # Should be union of ALA and GLY selections
        # ALA: indices 0,3; GLY: indices 1,4
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 1, 3, 4] or j in [0, 1, 3, 4]:  # ALA or GLY
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
        # Ensure no duplicates
        assert len(indices) == len(set(indices))


class TestTrajectorySelections:
    """Test trajectory selection parameters."""
    
    def setup_method(self):
        """Setup with multiple trajectories."""
        test_instance = TestSelectionStrings()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        
        # Add second identical trajectory
        self.pipeline.data.trajectory_data.trajectories.append(self.pipeline.data.trajectory_data.trajectories[0])
        self.pipeline.data.trajectory_data.trajectory_names.append("synthetic_2")
        
        # Copy residue metadata for second trajectory
        self.pipeline.data.trajectory_data.res_label_data[1] = self.pipeline.data.trajectory_data.res_label_data[0].copy()
        
        # Add tags for trajectory selection
        self.pipeline.data.trajectory_data.trajectory_tags = {
            0: ["system_A", "folded"],
            1: ["system_B", "unfolded"]
        }
        
        # Recompute features for both trajectories
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0), force=True)
    
    def _create_selector_and_add(self, name: str, selection: str, **kwargs):
        """Helper to create selector and add selection."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, "distances", selection, **kwargs)
        return name
    
    def test_traj_selection_single_index(self):
        """Test traj_selection=0 (single trajectory)."""
        name = self._create_selector_and_add("test", "all", traj_selection=0)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should only have trajectory 0
        assert len(results["trajectory_indices"]) == 1
        assert 0 in results["trajectory_indices"]
        
        # Should have all 21 pairs
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected_indices = list(range(total_pairs))
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected_indices)
    
    def test_traj_selection_multiple_indices(self):
        """Test traj_selection=[0, 1] (multiple trajectories)."""
        # Use common_denominator=False to avoid reduced metadata issues
        name = self._create_selector_and_add("test", "all", traj_selection=[0, 1], common_denominator=False)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should have both trajectories
        assert len(results["trajectory_indices"]) == 2
        assert 0 in results["trajectory_indices"]
        assert 1 in results["trajectory_indices"]
        
        # Both trajectories should have all 21 pairs
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected_indices = list(range(total_pairs))
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected_indices)
        assert sorted(results["trajectory_indices"][1]["indices"]) == sorted(expected_indices)
    
    def test_traj_selection_all(self):
        """Test traj_selection="all"."""
        # Use common_denominator=False to avoid reduced metadata issues
        name = self._create_selector_and_add("test", "all", traj_selection="all", common_denominator=False)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should have both trajectories
        assert len(results["trajectory_indices"]) == 2
        assert 0 in results["trajectory_indices"]
        assert 1 in results["trajectory_indices"]
        
        # Both trajectories should have all 21 pairs
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected_indices = list(range(total_pairs))
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected_indices)
        assert sorted(results["trajectory_indices"][1]["indices"]) == sorted(expected_indices)
    
    def test_traj_selection_by_index_system_a(self):
        """Test traj_selection=[0] for single trajectory."""
        name = self._create_selector_and_add("test", "all", traj_selection=[0], common_denominator=False)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should only have trajectory 0
        assert len(results["trajectory_indices"]) == 1
        assert 0 in results["trajectory_indices"]
        
        # Should have all 21 pairs
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected_indices = list(range(total_pairs))
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected_indices)
    
    def test_mixed_feature_types_different_trajectories(self):
        """Test ALA from distances for trajectory 0, GLY from contacts for trajectory 1."""
        # Add contacts feature
        self.pipeline.feature.add_feature(Contacts(cutoff=4.0))
        
        # Create selector with mixed feature types and trajectory selection
        self.pipeline.feature_selector.create("mixed_test")
        
        # ALA from distances for trajectory 0
        self.pipeline.feature_selector.add("mixed_test", "distances", "res ALA", 
                                         traj_selection=[0], common_denominator=False)
        
        # GLY from contacts for trajectory 1  
        self.pipeline.feature_selector.add("mixed_test", "contacts", "res GLY",
                                         traj_selection=[1], common_denominator=False)
        
        self.pipeline.feature_selector.select("mixed_test", reference_traj=0)
        
        # Check distances results
        distances_results = self.pipeline.data.selected_feature_data["mixed_test"].get_results("distances")
        assert len(distances_results["trajectory_indices"]) == 1
        assert 0 in distances_results["trajectory_indices"]
        
        # Check contacts results
        contacts_results = self.pipeline.data.selected_feature_data["mixed_test"].get_results("contacts")
        assert len(contacts_results["trajectory_indices"]) == 1
        assert 1 in contacts_results["trajectory_indices"]
        
        # Calculate exact ALA pairs for distances (residues 0, 3)
        expected_ala_pairs = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:  # ALA residues
                    expected_ala_pairs.append(pair_idx)
                pair_idx += 1
        
        # Calculate exact GLY pairs for contacts (residues 1, 4)
        expected_gly_pairs = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [1, 4] or j in [1, 4]:  # GLY residues
                    expected_gly_pairs.append(pair_idx)
                pair_idx += 1
        
        # Verify exact indices
        assert sorted(distances_results["trajectory_indices"][0]["indices"]) == sorted(expected_ala_pairs)
        assert sorted(contacts_results["trajectory_indices"][1]["indices"]) == sorted(expected_gly_pairs)


class TestNameManagement:
    """Test selector name management."""
    
    def setup_method(self):
        """Setup with basic trajectory."""
        test_instance = TestSelectionStrings()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def test_create_selector_correct_name(self):
        """Test selector is created with correct name."""
        self.pipeline.feature_selector.create("my_selector")
        
        assert "my_selector" in self.pipeline.data.selected_feature_data
        assert self.pipeline.data.selected_feature_data["my_selector"].name == "my_selector"
    
    def test_multiple_selectors(self):
        """Test multiple selectors can coexist."""
        self.pipeline.feature_selector.create("selector_1")
        self.pipeline.feature_selector.create("selector_2")
        
        self.pipeline.feature_selector.add("selector_1", "distances", "res ALA")
        self.pipeline.feature_selector.add("selector_2", "distances", "res GLY")
        
        assert "selector_1" in self.pipeline.data.selected_feature_data
        assert "selector_2" in self.pipeline.data.selected_feature_data
        
        # Verify they have different selections
        self.pipeline.feature_selector.select("selector_1", reference_traj=0)
        self.pipeline.feature_selector.select("selector_2", reference_traj=0)
        
        results_1 = self.pipeline.data.selected_feature_data["selector_1"].get_results("distances")["trajectory_indices"][0]["indices"]
        results_2 = self.pipeline.data.selected_feature_data["selector_2"].get_results("distances")["trajectory_indices"][0]["indices"]
        
        # Verify ALA selection (residues 0,3) vs GLY selection (residues 1,4)
        # ALA pairs: expect pairs containing residue 0 or 3
        ala_expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:  # ALA residues
                    ala_expected.append(pair_idx)
                pair_idx += 1
        
        # GLY pairs: expect pairs containing residue 1 or 4
        gly_expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [1, 4] or j in [1, 4]:  # GLY residues
                    gly_expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(results_1) == sorted(ala_expected)
        assert sorted(results_2) == sorted(gly_expected)
        assert results_1 != results_2  # Sanity check they're different
    
    def test_create_existing_selector_raises_error(self):
        """Test creating selector with existing name raises ValueError."""
        self.pipeline.feature_selector.create("test")
        
        # Create again - should raise ValueError
        with pytest.raises(ValueError, match="Feature selector 'test' already exists"):
            self.pipeline.feature_selector.create("test")


class TestCornerCases:
    """Test corner cases and error handling."""
    
    def setup_method(self):
        """Setup with basic trajectory."""
        test_instance = TestSelectionStrings()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def _create_selector_and_add(self, name: str, selection: str, **kwargs):
        """Helper to create selector and add selection."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, "distances", selection, **kwargs)
        return name
    
    def test_empty_selection_string(self):
        """Test empty selection string raises ValueError during select."""
        name = self._create_selector_and_add("test", "")
        with pytest.raises(ValueError, match="Invalid instruction"):
            self.pipeline.feature_selector.select(name, reference_traj=0)
    
    def test_whitespace_only_selection(self):
        """Test whitespace-only selection string raises ValueError during select."""
        name = self._create_selector_and_add("test", "   ")
        with pytest.raises(ValueError, match="Invalid instruction"):
            self.pipeline.feature_selector.select(name, reference_traj=0)
    
    def test_unknown_residue_name(self):
        """Test selection with unknown residue name raises UserWarning."""
        name = self._create_selector_and_add("test", "res XYZ")
        with pytest.warns(UserWarning, match="Feature selection resulted in no matches"):
            self.pipeline.feature_selector.select(name, reference_traj=0)
        
        # Should result in empty trajectory_indices
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        assert results["trajectory_indices"] == {}  # Empty dict for no matches
    
    def test_invalid_range_reverse(self):
        """Test invalid range where start > end throws ValueError."""
        name = self._create_selector_and_add("test", "seqid 5-2")
        with pytest.raises(ValueError, match="Invalid seqid range: 5-2. Start ID cannot be greater than End ID"):
            self.pipeline.feature_selector.select(name, reference_traj=0)
    
    def test_out_of_range_seqid(self):
        """Test seqid beyond available range raises UserWarning."""
        name = self._create_selector_and_add("test", "seqid 100")
        with pytest.warns(UserWarning, match="Feature selection resulted in no matches"):
            self.pipeline.feature_selector.select(name, reference_traj=0)
        
        # Should result in empty trajectory_indices
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        assert results["trajectory_indices"] == {}  # Empty dict for no matches
    
    def test_large_range(self):
        """Test very large range that exceeds available residues raises UserWarning."""
        name = self._create_selector_and_add("test", "seqid 1000-2000")
        with pytest.warns(UserWarning, match="Feature selection resulted in no matches"):
            self.pipeline.feature_selector.select(name, reference_traj=0)
        
        # Should result in empty trajectory_indices
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        assert results["trajectory_indices"] == {}  # Empty dict for no matches


class TestConsensusSelection:
    """Test consensus-based feature selection."""
    
    def setup_method(self):
        """Setup with consensus labels in residue metadata."""
        # Create topology: ALA(1), GLY(2), VAL(3), ALA(4), GLY(5), LEU(6), PHE(7)
        topology = md.Topology()
        chain = topology.add_chain()
        
        residue_names = ["ALA", "GLY", "VAL", "ALA", "GLY", "LEU", "PHE"]
        residues = []
        
        for name in residue_names:
            residues.append(topology.add_residue(name, chain))
        
        # Add CA atoms
        for residue in residues:
            topology.add_atom("CA", md.element.carbon, residue)
        
        # Create coordinates (7 atoms, 20 frames)
        coordinates = []
        for frame in range(20):
            frame_coords = []
            for atom_idx in range(7):
                x = atom_idx * 1.0
                y = frame * 0.1
                z = 0.0
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        self.test_traj = md.Trajectory(xyz, topology)
        
        # Setup pipeline
        self.pipeline = PipelineManager()
        self.pipeline.data.trajectory_data.trajectories = [self.test_traj]
        self.pipeline.data.trajectory_data.trajectory_names = ["synthetic"]
        
        # Create residue metadata with consensus labels (including None values)
        residue_metadata = []
        consensus_labels = ["3.50", "3x51", "4.50", None, "5.50", "6x50", None]
        
        for i, res in enumerate(self.test_traj.topology.residues):
            residue_metadata.append({
                "resid": res.resSeq + 1,
                "seqid": res.index + 1,
                "resname": res.name,
                "aaa_code": res.name,
                "a_code": res.name[0] if res.name else "X",
                "consensus": consensus_labels[i],
                "full_name": f"{res.name}{res.index + 1}",
                "index": res.index
            })
        
        self.pipeline.data.trajectory_data.res_label_data = {0: residue_metadata}
        
        # Add features: distances with excluded_neighbors=0 (all pairs)
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
    
    def _create_selector_and_add(self, name: str, selection: str, **kwargs):
        """Helper to create selector and add selection."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, "distances", selection, **kwargs)
        return name
    
    def _get_selected_indices(self, selector_name: str):
        """Helper to get selected indices from completed selection."""
        self.pipeline.feature_selector.select(selector_name, reference_traj=0)
        return self.pipeline.data.selected_feature_data[selector_name].get_results("distances")["trajectory_indices"][0]["indices"]
    
    def test_consensus_exact_match(self):
        """Test exact consensus match 'consensus 3.50'."""
        name = self._create_selector_and_add("test", "consensus 3.50")
        indices = self._get_selected_indices(name)
        
        # consensus "3.50" is at residue index 0
        # Should return pairs containing residue 0
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i == 0 or j == 0:  # Residue 0 has consensus "3.50"
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_consensus_wildcard_match(self):
        """Test wildcard consensus match 'consensus *50'."""
        name = self._create_selector_and_add("test", "consensus *50")
        indices = self._get_selected_indices(name)

        # consensus patterns containing "x50": "3.50", "5.50", "6x50"
        # Residues: 0, 2, 4, 5
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 2, 4, 5] or j in [0, 2, 4, 5]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_consensus_range_exact(self):
        """Test exact consensus range 'consensus 3.50-4.50'."""
        name = self._create_selector_and_add("test", "consensus 3.50-4.50")
        indices = self._get_selected_indices(name)
        
        # Should find residues between consensus 3.50 and 4.50 (indices 0, 1, 2)
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 1, 2] or j in [0, 1, 2]:  # Residues 0 (3.50), 1 (2x51), 2 (4.50)
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_consensus_range_wildcard(self):
        """Test wildcard consensus range 'consensus *51-*50'."""
        name = self._create_selector_and_add("test", "consensus *51-*50")
        indices = self._get_selected_indices(name)

        # Should find residues with patterns matching range from *51 to *50
        # consensus_labels = ["3.50", "3x51", "4.50", None, "5.50", "6x50", None]
        # Range logic: *51 (idx 1) to *50 (idx 2), then orphan ends 5.50 (idx 4) and 6x50 (idx 5)
        # Final indices: 1,2,4,5
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [1, 2, 4, 5] or j in [1, 2, 4, 5]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_consensus_all_prefix(self):
        """Test 'consensus all 3.50-5.50' including None values."""
        name = self._create_selector_and_add("test", "consensus all 3.50-5.50") 
        indices = self._get_selected_indices(name)
        
        # Should include residues with seqid 1-5 (indices 0-4), including None
        # "all" prefix includes residues with consensus=None in seqid range
        # Residues: "3.50"(0), "3x51"(1), "4.50"(2), None(3), "5.50"(4)
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 1, 2, 3, 4] or j in [0, 1, 2, 3, 4]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)

    def test_consensus_not_all_prefix(self):
        """Test 'consensus 3.50-5.50' including None values."""
        name = self._create_selector_and_add("test", "consensus 3.50-5.50") 
        indices = self._get_selected_indices(name)
        
        # wihtout "all" prefix excludes residues with consensus=None in seqid range
        # Residues: "3.50"(0), "3x51"(1), "4.50"(2), "5.50"(4)
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 1, 2, 4] or j in [0, 1, 2, 4]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
    
    def test_consensus_and_combination(self):
        """Test 'consensus 3.50 and res ALA' combination."""
        name = self._create_selector_and_add("test", "consensus 3.50 and res ALA")
        indices = self._get_selected_indices(name)
        
        # UNION of: 
        # consensus 3.50: residue 0
        # res ALA: residues 0, 3
        # Union: residues 0, 3
        expected = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i in [0, 3] or j in [0, 3]:
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)


class TestCommonDenominator:
    """Test common_denominator functionality."""
    
    def setup_method(self):
        """Setup with multiple trajectories having different features."""
        # Create basic trajectory setup
        test_instance = TestSelectionStrings()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        
        # Add second trajectory with slightly different residue composition
        # Change one residue type to create different features
        traj2 = self.pipeline.data.trajectory_data.trajectories[0]
        self.pipeline.data.trajectory_data.trajectories.append(traj2)
        self.pipeline.data.trajectory_data.trajectory_names.append("synthetic_2")
        
        # Copy and modify residue metadata for second trajectory
        import copy
        metadata2 = copy.deepcopy(self.pipeline.data.trajectory_data.res_label_data[0])
        # Change residue 2 from VAL to TYR in trajectory 1 to create difference
        metadata2[2]["resname"] = "TYR"
        metadata2[2]["aaa_code"] = "TYR"
        metadata2[2]["a_code"] = "Y"
        self.pipeline.data.trajectory_data.res_label_data[1] = metadata2
        
        # Add consensus labels for both trajectories
        for traj_idx in [0, 1]:
            for i, metadata in enumerate(self.pipeline.data.trajectory_data.res_label_data[traj_idx]):
                if i == 0:
                    metadata["consensus"] = "1x50"  # Common consensus
                elif i == 1:
                    metadata["consensus"] = "2x50"  # Common consensus
                elif i == 2:
                    # Different consensus to test filtering
                    metadata["consensus"] = "3x50" if traj_idx == 0 else "3x51"
                elif i == 3:
                    metadata["consensus"] = "4x50"  # Common consensus
                else:
                    metadata["consensus"] = None
        
        # Recompute features for both trajectories
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0), force=True)
    
    def _create_selector_and_add(self, name: str, selection: str, **kwargs):
        """Helper to create selector and add selection."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, "distances", selection, **kwargs)
        return name
    
    def test_common_denominator_single_trajectory(self):
        """Test common_denominator=True with single trajectory has no effect."""
        name = self._create_selector_and_add("test", "all", traj_selection=0, common_denominator=True)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should have all features for trajectory 0
        assert len(results["trajectory_indices"]) == 1
        assert 0 in results["trajectory_indices"]
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected = list(range(total_pairs))
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected)

    def test_common_denominator_multiple_trajectories_true(self):
        """Test common_denominator=True filters to common features only."""
        name = self._create_selector_and_add("test", "all", traj_selection="all", common_denominator=True)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should have both trajectories but only common features
        # Trajectory 0: ALA,GLY,VAL,ALA,GLY,LEU,PHE (indices 0,1,2,3,4,5,6)
        # Trajectory 1: ALA,GLY,TYR,ALA,GLY,LEU,PHE (indices 0,1,2,3,4,5,6) 
        # Different residues at index 2: VAL vs TYR
        # Common features exclude pairs involving residue 2
        assert len(results["trajectory_indices"]) == 2
        assert 0 in results["trajectory_indices"]
        assert 1 in results["trajectory_indices"]
        
        # Calculate exact common features (all pairs EXCEPT those involving residue 2)
        expected_common_indices = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if i != 2 and j != 2:  # Exclude pairs with residue 2 (different VAL/TYR)
                    expected_common_indices.append(pair_idx)
                pair_idx += 1
        
        # Both trajectories should have exact same indices
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected_common_indices)
        assert sorted(results["trajectory_indices"][1]["indices"]) == sorted(expected_common_indices)
    
    def test_common_denominator_multiple_trajectories_false(self):
        """Test common_denominator=False keeps all features per trajectory."""
        name = self._create_selector_and_add("test", "all", traj_selection="all", common_denominator=False)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should have both trajectories with all their features
        assert len(results["trajectory_indices"]) == 2
        assert 0 in results["trajectory_indices"]
        assert 1 in results["trajectory_indices"]
        
        # Both should have all features
        total_pairs = (7 * 6) // 2  # 21 pairs
        expected = list(range(total_pairs))

        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected)
        assert sorted(results["trajectory_indices"][1]["indices"]) == sorted(expected)
    
    def test_common_denominator_with_specific_selection(self):
        """Test common_denominator with specific residue selection."""
        # Select ALA residues with common_denominator=True
        name = self._create_selector_and_add("test", "res ALA", traj_selection="all", common_denominator=True)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # ALA residues are same in both trajectories (indices 0,3)
        # Should have both trajectories with same ALA-containing pairs
        assert len(results["trajectory_indices"]) == 2
        assert 0 in results["trajectory_indices"]
        assert 1 in results["trajectory_indices"]
        
        # Calculate exact ALA pairs (residues 0, 3) but exclude pairs involving residue 2 (different VAL/TYR)
        expected_ala_indices = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                if (i in [0, 3] or j in [0, 3]) and (i != 2 and j != 2):  # ALA residues, excluding pairs with residue 2
                    expected_ala_indices.append(pair_idx)
                pair_idx += 1
        
        # Both trajectories should have exact same ALA pairs
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected_ala_indices)
        assert sorted(results["trajectory_indices"][1]["indices"]) == sorted(expected_ala_indices)
    
    def test_common_denominator_consensus_selection(self):
        """Test common_denominator=True with consensus selection filters properly."""
        # Select consensus positions that exist in both trajectories with common_denominator=True
        # Positions 1x50, 2x50, 4x50 are common; 3x50 vs 3x51 are different
        name = self._create_selector_and_add("test", "consensus 1x50-3x50", traj_selection="all", common_denominator=True)
        self.pipeline.feature_selector.select(name, reference_traj=0)
        
        results = self.pipeline.data.selected_feature_data[name].get_results("distances")
        
        # Should have both trajectories
        assert len(results["trajectory_indices"]) == 2
        assert 0 in results["trajectory_indices"]
        assert 1 in results["trajectory_indices"]
        
        # Calculate pairs for consensus range 1x50-3x50 but exclude pairs involving residue 2 (different consensus 3x50/3x51)
        # Consensus setup has: traj0=[1x50, 2x50, 3x50, 4x50, None, None, None], traj1=[1x50, 2x50, 3x51, 4x50, None, None, None]
        # Common consensus: residues 0(1x50), 1(2x50), 3(4x50) - residue 2 has different consensus
        # Range 1x50-3x50 includes all between 1x50 and 3x50, which is 1x50, 2x50, but 3x50 vs 3x51 differs
        expected_consensus_indices = []
        pair_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                # Residues 0,1 have common consensus 1x50,2x50; exclude pairs with residue 2 (different consensus)
                if (i in [0, 1] or j in [0, 1]) and (i != 2 and j != 2):
                    expected_consensus_indices.append(pair_idx)
                pair_idx += 1
        
        # Both trajectories should have same consensus pairs
        assert sorted(results["trajectory_indices"][0]["indices"]) == sorted(expected_consensus_indices)
        assert sorted(results["trajectory_indices"][1]["indices"]) == sorted(expected_consensus_indices)


class TestRequireAllPartners:
    """Test require_all_partners functionality with concrete feature verification."""
    
    def setup_method(self):
        """Setup with trajectory having specific consensus labels."""
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
        
        # Create coordinates (5 atoms, 20 frames)
        coordinates = []
        for frame in range(20):
            frame_coords = []
            for atom_idx in range(5):
                x = atom_idx * 1.0
                y = frame * 0.1
                z = 0.0
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        self.test_traj = md.Trajectory(xyz, topology)
        
        # Setup pipeline
        self.pipeline = PipelineManager()
        self.pipeline.data.trajectory_data.trajectories = [self.test_traj]
        self.pipeline.data.trajectory_data.trajectory_names = ["test"]
        
        # Create residue metadata with consensus labels
        residue_metadata = []
        consensus_labels = ["1.50", "2.50", "3.50", "4.50", "5.50"]
        
        for i, res in enumerate(self.test_traj.topology.residues):
            residue_metadata.append({
                "resid": res.resSeq + 1,
                "seqid": res.index + 1,
                "resname": res.name,
                "aaa_code": res.name,
                "a_code": res.name[0] if res.name else "X",
                "consensus": consensus_labels[i],
                "full_name": f"{res.name}{res.index + 1}",
                "index": res.index
            })
        
        self.pipeline.data.trajectory_data.res_label_data = {0: residue_metadata}
        
        # Add features: distances with excluded_neighbors=0 (all pairs)
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
    
    def _create_selector_and_add(self, name: str, selection: str, **kwargs):
        """Helper to create selector and add selection."""
        if name in self.pipeline.data.selected_feature_data:
            del self.pipeline.data.selected_feature_data[name]
        self.pipeline.feature_selector.create(name)
        self.pipeline.feature_selector.add(name, "distances", selection, **kwargs)
        return name
    
    def _get_selected_indices(self, selector_name: str):
        """Helper to get selected indices from completed selection."""
        self.pipeline.feature_selector.select(selector_name, reference_traj=0)
        return self.pipeline.data.selected_feature_data[selector_name].get_results("distances")["trajectory_indices"][0]["indices"]
    
    def test_consensus_range_require_all_partners_false(self):
        """Test consensus range with require_all_partners=False (any partner in range)."""
        name = self._create_selector_and_add("test", "consensus 3.50-5.50", require_all_partners=False)
        indices = self._get_selected_indices(name)
        
        # Range 3.50-5.50 includes residues with seqid 3,4,5 (indices 2,3,4)
        # With require_all_partners=False: ANY partner in range
        # Expected pairs: all containing residue 2, 3, or 4
        expected = []
        pair_idx = 0
        for i in range(5):
            for j in range(i+1, 5):
                if i in [2, 3, 4] or j in [2, 3, 4]:  # Any partner in range
                    expected.append(pair_idx)
                pair_idx += 1
        
        assert sorted(indices) == sorted(expected)
        
        # Verify we got the expected features count
        # Should be: (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4) = 9 features
        assert len(indices) == 9
    
    def test_consensus_range_require_all_partners_true(self):
        """Test consensus range with require_all_partners=True (both partners in range)."""
        name = self._create_selector_and_add("test", "consensus 3.50-5.50", require_all_partners=True)
        indices = self._get_selected_indices(name)
        
        # Range 3.50-5.50 includes residues with seqid 3,4,5 (indices 2,3,4)
        # With require_all_partners=True: BOTH partners must be in range
        # Expected pairs: only (2,3), (2,4), (3,4)
        expected_both_in_range = []
        pair_idx = 0
        for i in range(5):
            for j in range(i+1, 5):
                if i in [2, 3, 4] and j in [2, 3, 4]:  # Both partners in range
                    expected_both_in_range.append(pair_idx)
                pair_idx += 1
        
        # This test will FAIL until the bug is fixed
        # Currently returns all 9 indices instead of just [7, 8, 9]
        assert sorted(indices) == sorted(expected_both_in_range)
        assert len(indices) == 3
    
    def test_consensus_wildcard_require_all_partners_true(self):
        """Test consensus wildcard with require_all_partners=True."""
        # Update consensus labels to have different patterns
        # Change one to not match the wildcard pattern
        self.pipeline.data.trajectory_data.res_label_data[0][1]["consensus"] = "2x40"  # GLY2 now has 2x40 instead of 2.50
        
        name = self._create_selector_and_add("test", "consensus *50", require_all_partners=True)
        indices = self._get_selected_indices(name)
        
        # Wildcard *50 now matches: 1.50, 3.50, 4.50, 5.50 (residues 0,2,3,4)
        # GLY2 (residue 1) has 2x40 which doesn't match *50
        # With require_all_partners=True: BOTH partners must have *50
        # Expected pairs: (0,2), (0,3), (0,4), (2,3), (2,4), (3,4)
        expected_both_match = []
        pair_idx = 0
        for i in range(5):
            for j in range(i+1, 5):
                if i in [0, 2, 3, 4] and j in [0, 2, 3, 4]:  # Both have *50
                    expected_both_match.append(pair_idx)
                pair_idx += 1
            
        assert sorted(indices) == sorted(expected_both_match)
        assert len(indices) == 6
    
    def test_consensus_single_require_all_partners_works(self):
        """Test that single consensus patterns work correctly with require_all_partners."""
        # Setup with duplicate consensus to test functionality
        self.pipeline.data.trajectory_data.res_label_data[0][3]["consensus"] = "3.50"  # Make position 4 also 3.50
        
        name = self._create_selector_and_add("test", "consensus 3.50", require_all_partners=True)
        indices = self._get_selected_indices(name)
        
        # Now positions 2 and 3 both have consensus 3.50
        # Only pair (2,3) should match with require_all_partners=True
        expected_indices = [7]  # Only pair (2,3)
        
        assert sorted(indices) == sorted(expected_indices)
        assert len(indices) == 1
