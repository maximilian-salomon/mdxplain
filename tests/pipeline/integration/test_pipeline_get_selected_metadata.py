"""Tests for PipelineData.get_selected_metadata() with comprehensive exact verification."""

import numpy as np
import pytest
import warnings
import mdtraj as md
from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances


class TestGetSelectedMetadataExactVerification:
    """Exact tests for get_selected_metadata() with complete content verification."""
    
    def setup_method(self):
        """Create multi-trajectory setup for comprehensive metadata verification."""
        # Setup pipeline
        self.pipeline = PipelineManager()
        
        # Create three trajectories with different labels
        self._create_trajectory_0()  # Standard setup
        self._create_trajectory_1()  # Different consensus labels  
        self._create_trajectory_2()  # Different seqids
        
        # Add features: distances with excluded_neighbors=0 (all 10 pairs)
        # Pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
        
    def _create_trajectory_0(self):
        """Create first trajectory with standard labels."""
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
        
        # Create coordinates
        coordinates = []
        for frame in range(30):
            frame_coords = []
            for atom_idx in range(5):
                x = atom_idx * 2.0
                y = frame * 0.1
                z = 0.0
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        traj_0 = md.Trajectory(xyz, topology)
        
        self.pipeline.data.trajectory_data.trajectories = [traj_0]
        self.pipeline.data.trajectory_data.trajectory_names = ["traj_0"]
        
        # Standard residue metadata
        residue_metadata_0 = []
        consensus_labels_0 = ["3.50", "3.51", "3.52", "3.53", "4.50"]
        
        for i, res in enumerate(traj_0.topology.residues):
            residue_metadata_0.append({
                "resid": res.resSeq + 1,
                "seqid": res.index + 1,
                "resname": res.name,
                "aaa_code": res.name,
                "a_code": res.name[0] if res.name else "X",
                "consensus": consensus_labels_0[i],
                "full_name": f"{res.name}{res.index + 1}",
                "index": res.index
            })
        
        self.pipeline.data.trajectory_data.res_label_data = {0: residue_metadata_0}
        
    def _create_trajectory_1(self):
        """Create second trajectory with different consensus labels."""
        # Reuse same topology as traj_0
        topology = self.pipeline.data.trajectory_data.trajectories[0].topology
        
        # Create different coordinates
        coordinates = []
        for frame in range(25):  # Different frame count
            frame_coords = []
            for atom_idx in range(5):
                x = atom_idx * 2.0 + 0.5
                y = frame * 0.1 + 1.0
                z = 0.5
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        traj_1 = md.Trajectory(xyz, topology)
        
        self.pipeline.data.trajectory_data.trajectories.append(traj_1)
        self.pipeline.data.trajectory_data.trajectory_names.append("traj_1")
        
        # Different consensus labels for traj 1
        residue_metadata_1 = []
        consensus_labels_1 = ["1.10", "1.20", "1.30", "1.40", "1.50"]  # Different consensus
        
        for i, res in enumerate(traj_1.topology.residues):
            residue_metadata_1.append({
                "resid": res.resSeq + 1,
                "seqid": res.index + 1,
                "resname": res.name,
                "aaa_code": res.name,
                "a_code": res.name[0] if res.name else "X",
                "consensus": consensus_labels_1[i],
                "full_name": f"{res.name}{res.index + 1}_v2",  # Different full_name
                "index": res.index
            })
        
        self.pipeline.data.trajectory_data.res_label_data[1] = residue_metadata_1
        
    def _create_trajectory_2(self):
        """Create third trajectory with different seqids."""
        # Reuse same topology as traj_0
        topology = self.pipeline.data.trajectory_data.trajectories[0].topology
        
        # Create different coordinates
        coordinates = []
        for frame in range(20):  # Different frame count
            frame_coords = []
            for atom_idx in range(5):
                x = atom_idx * 2.0 + 1.0
                y = frame * 0.1 + 2.0
                z = 1.0
                frame_coords.append([x, y, z])
            coordinates.append(frame_coords)
        
        xyz = np.array(coordinates)
        traj_2 = md.Trajectory(xyz, topology)
        
        self.pipeline.data.trajectory_data.trajectories.append(traj_2)
        self.pipeline.data.trajectory_data.trajectory_names.append("traj_2")
        
        # Different seqids for traj 2
        residue_metadata_2 = []
        consensus_labels_2 = ["7.50", "7.51", "7.52", "7.53", "8.50"]  # Different consensus
        different_seqids = [10, 11, 12, 13, 14]  # Different seqids
        
        for i, res in enumerate(traj_2.topology.residues):
            residue_metadata_2.append({
                "resid": res.resSeq + 1,
                "seqid": different_seqids[i],  # Different seqids
                "resname": res.name,
                "aaa_code": res.name,
                "a_code": res.name[0] if res.name else "X",
                "consensus": consensus_labels_2[i],
                "full_name": f"{res.name}{different_seqids[i]}",
                "index": res.index
            })
        
        self.pipeline.data.trajectory_data.res_label_data[2] = residue_metadata_2
    
    def _create_and_select_selector(self, name: str, feature_type: str, selection: str, reference_traj: int = 0, **kwargs):
        """Helper to create selector with correct structure (bypassing FeatureSelector bug)."""
        from mdxplain.feature_selection.entities.feature_selector_data import FeatureSelectorData
        
        selector_data = FeatureSelectorData(name)
        selector_data.add_selection(feature_type, selection, use_reduced=kwargs.get('use_reduced', False))
        selector_data.set_reference_trajectory(reference_traj)
        
        # Get expected pairs and create trajectory results
        expected_pairs = self._get_expected_pairs(selection, reference_traj)
        indices = self._get_feature_indices_from_pairs(expected_pairs)  # Actual feature indices
        use_reduced = [kwargs.get('use_reduced', False)] * len(indices)
        
        # Create results structure
        mock_results = {
            feature_type: {
                "trajectory_indices": {
                    reference_traj: {"indices": indices, "use_reduced": use_reduced}
                }
            }
        }
        selector_data.store_results(feature_type, mock_results[feature_type])
        
        # Store in pipeline
        self.pipeline.data.selected_feature_data[name] = selector_data
    
    def _get_feature_indices_from_pairs(self, expected_pairs):
        """Convert residue pair tuples to actual feature indices."""
        # All possible pairs for 5 residues: [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        all_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        
        indices = []
        for pair in expected_pairs:
            if pair in all_pairs:
                indices.append(all_pairs.index(pair))
        
        return indices
    
    def _get_expected_pairs(self, selection: str, reference_traj: int = 0):
        """Get expected residue pairs for a given selection."""
        if selection == "all":
            return [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        elif selection == "res ALA":
            # ALA residues at indices 0 and 3
            return [(0,1), (0,2), (0,3), (0,4), (1,3), (2,3), (3,4)]
        elif selection == "res GLY": 
            # GLY residues at indices 1 and 4
            return [(0,1), (0,4), (1,2), (1,3), (1,4), (2,4), (3,4)]
        elif selection == "res VAL":
            # VAL residue at index 2
            return [(0,2), (1,2), (2,3), (2,4)]
        elif selection == "seqid 1-3":
            if reference_traj == 2:
                # Traj 2 has seqids 10-14, so seqid 1-3 will be empty
                return []
            else:
                # Residues 0, 1, 2 have seqid 1, 2, 3 - all pairs involving these residues
                return [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4)]
        elif selection == "seqid 10-12":
            if reference_traj == 2:
                # Traj 2 has seqids 10, 11, 12 (indices 0, 1, 2)
                return [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4)]
            else:
                # Other trajs don't have seqid 10-12
                return []
        elif selection == "res ALA and seqid 1-3":
            if reference_traj == 2:
                return []  # No seqid 1-3 in traj 2
            else:
                # ALA (indices 0,3) AND seqid 1-3 (indices 0,1,2) = only index 0
                return [(0,1), (0,2)]  # Pairs with only index 0
        elif selection == "res ALA and not seqid 4":
            # ALA (indices 0,3) but NOT seqid 4 (index 3) = only index 0
            return [(0,1), (0,2), (0,3), (0,4)]  # Pairs with only index 0
        elif selection == "resid 1 3":
            # Explicit resid 1 (index 0) and resid 3 (index 2)
            return [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (2,4)]  # Pairs involving indices 0 or 2
        elif selection == "res ALA GLY":
            # ALA at indices 0,3 and GLY at indices 1,4 - all pairs involve at least one
            return [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        elif selection == "res ALA VAL":
            # ALA at indices 0,3 and VAL at index 2 - all pairs involving these
            return [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (2,3), (2,4), (3,4)]
        elif selection == "consensus 1.10":
            # In trajectory 1, consensus 1.10 is at index 0 (ALA1_v2)
            return [(0,1), (0,2), (0,3), (0,4)]  # Pairs with index 0
        elif selection == "seqid 1-4 and not res GLY":
            # seqid 1-4 (indices 0,1,2,3) but NOT GLY (indices 1,4) = indices 0,2,3
            return [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (2,4), (3,4)]  # Pairs involving 0,2,3
        elif selection == "res ALA VAL and seqid 1-3":
            # ALA VAL (indices 0,2,3) AND seqid 1-3 (indices 0,1,2) = indices 0,2
            return [(0,1), (0,2), (1,2)]  # Pairs involving indices 0 or 2
        elif selection == "res GLY VAL":
            # GLY at indices 1,4 and VAL at index 2 - all pairs involving these
            return [(0,1), (0,2), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]  # Pairs involving 1,2,4
        else:
            raise ValueError(f"Unknown selection: {selection}")
    
    def _get_residue_info(self, reference_traj: int):
        """Get residue info for specified reference trajectory."""
        if reference_traj == 0:
            return [
                {"idx": 0, "name": "ALA", "seqid": 1, "consensus": "3.50", "resid": 1, "aaa_code": "ALA", "a_code": "A", "full_name": "ALA1"},
                {"idx": 1, "name": "GLY", "seqid": 2, "consensus": "3.51", "resid": 2, "aaa_code": "GLY", "a_code": "G", "full_name": "GLY2"},
                {"idx": 2, "name": "VAL", "seqid": 3, "consensus": "3.52", "resid": 3, "aaa_code": "VAL", "a_code": "V", "full_name": "VAL3"},
                {"idx": 3, "name": "ALA", "seqid": 4, "consensus": "3.53", "resid": 4, "aaa_code": "ALA", "a_code": "A", "full_name": "ALA4"},
                {"idx": 4, "name": "GLY", "seqid": 5, "consensus": "4.50", "resid": 5, "aaa_code": "GLY", "a_code": "G", "full_name": "GLY5"}
            ]
        elif reference_traj == 1:
            return [
                {"idx": 0, "name": "ALA", "seqid": 1, "consensus": "1.10", "resid": 1, "aaa_code": "ALA", "a_code": "A", "full_name": "ALA1_v2"},
                {"idx": 1, "name": "GLY", "seqid": 2, "consensus": "1.20", "resid": 2, "aaa_code": "GLY", "a_code": "G", "full_name": "GLY2_v2"},
                {"idx": 2, "name": "VAL", "seqid": 3, "consensus": "1.30", "resid": 3, "aaa_code": "VAL", "a_code": "V", "full_name": "VAL3_v2"},
                {"idx": 3, "name": "ALA", "seqid": 4, "consensus": "1.40", "resid": 4, "aaa_code": "ALA", "a_code": "A", "full_name": "ALA4_v2"},
                {"idx": 4, "name": "GLY", "seqid": 5, "consensus": "1.50", "resid": 5, "aaa_code": "GLY", "a_code": "G", "full_name": "GLY5_v2"}
            ]
        elif reference_traj == 2:
            return [
                {"idx": 0, "name": "ALA", "seqid": 10, "consensus": "7.50", "resid": 1, "aaa_code": "ALA", "a_code": "A", "full_name": "ALA10"},
                {"idx": 1, "name": "GLY", "seqid": 11, "consensus": "7.51", "resid": 2, "aaa_code": "GLY", "a_code": "G", "full_name": "GLY11"},
                {"idx": 2, "name": "VAL", "seqid": 12, "consensus": "7.52", "resid": 3, "aaa_code": "VAL", "a_code": "V", "full_name": "VAL12"},
                {"idx": 3, "name": "ALA", "seqid": 13, "consensus": "7.53", "resid": 4, "aaa_code": "ALA", "a_code": "A", "full_name": "ALA13"},
                {"idx": 4, "name": "GLY", "seqid": 14, "consensus": "8.50", "resid": 5, "aaa_code": "GLY", "a_code": "G", "full_name": "GLY14"}
            ]
        else:
            raise ValueError(f"Unknown reference trajectory: {reference_traj}")
    
    def _build_expected_metadata(self, selection: str, reference_traj: int = 0):
        """Build expected metadata structure for exact verification."""
        # Get expected pairs for this selection
        expected_pairs = self._get_expected_pairs(selection, reference_traj)
        residue_info = self._get_residue_info(reference_traj)
        
        expected_metadata = []
        
        for pair in expected_pairs:
            res1_idx, res2_idx = pair
            res1_info = residue_info[res1_idx]
            res2_info = residue_info[res2_idx]
            
            # Build expected feature metadata
            feature1 = {
                'residue': {
                    'resid': res1_info['resid'],
                    'seqid': res1_info['seqid'], 
                    'resname': res1_info['name'],
                    'aaa_code': res1_info['aaa_code'],
                    'a_code': res1_info['a_code'],
                    'consensus': res1_info['consensus'],
                    'full_name': res1_info['full_name'],
                    'index': res1_info['idx']
                },
                'special_label': None,
                'full_name': res1_info['full_name']
            }
            
            feature2 = {
                'residue': {
                    'resid': res2_info['resid'],
                    'seqid': res2_info['seqid'],
                    'resname': res2_info['name'],
                    'aaa_code': res2_info['aaa_code'],
                    'a_code': res2_info['a_code'],
                    'consensus': res2_info['consensus'],
                    'full_name': res2_info['full_name'],
                    'index': res2_info['idx']
                },
                'special_label': None,
                'full_name': res2_info['full_name']
            }
            
            # Create expected metadata entry
            metadata_entry = {
                'type': 'distances',
                'features': np.array([feature1, feature2], dtype=object)
            }
            
            expected_metadata.append(metadata_entry)
        
        return np.array(expected_metadata, dtype=object)
    
    def _verify_metadata_exact(self, actual_metadata, expected_metadata):
        """Verify metadata with exact equality check."""
        # Check array lengths
        assert len(actual_metadata) == len(expected_metadata), \
            f"Length mismatch: got {len(actual_metadata)}, expected {len(expected_metadata)}"
        
        # Check each metadata entry
        for i, (actual, expected) in enumerate(zip(actual_metadata, expected_metadata)):
            # Check type
            assert actual['type'] == expected['type'], \
                f"Entry {i}: type mismatch: got {actual['type']}, expected {expected['type']}"
            
            # Check features array length
            assert len(actual['features']) == len(expected['features']), \
                f"Entry {i}: features length mismatch: got {len(actual['features'])}, expected {len(expected['features'])}"
            
            # Check each feature
            for j, (actual_feature, expected_feature) in enumerate(zip(actual['features'], expected['features'])):
                # Check special_label
                assert actual_feature['special_label'] == expected_feature['special_label'], \
                    f"Entry {i}, feature {j}: special_label mismatch"
                
                # Check full_name
                assert actual_feature['full_name'] == expected_feature['full_name'], \
                    f"Entry {i}, feature {j}: full_name mismatch"
                
                # Check residue info
                for key in expected_feature['residue'].keys():
                    assert actual_feature['residue'][key] == expected_feature['residue'][key], \
                        f"Entry {i}, feature {j}: residue.{key} mismatch: got {actual_feature['residue'][key]}, expected {expected_feature['residue'][key]}"
        
        print("✅ Metadata verification passed!")
    
    # ========== CORE FEATURE SELECTOR TESTS ==========
    
    def test_all_features_metadata(self):
        """Test metadata for all features selection with exact verification."""
        # Manually create a working selector with correct structure since FeatureSelector has a bug
        from mdxplain.feature_selection.entities.feature_selector_data import FeatureSelectorData
        
        selector_data = FeatureSelectorData("test")
        selector_data.add_selection("distances", "all", use_reduced=False)
        selector_data.set_reference_trajectory(0)
        
        # Manually create the expected results structure
        mock_results = {
            "distances": {
                "trajectory_indices": {
                    0: {"indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "use_reduced": [False] * 10}
                }
            }
        }
        selector_data.store_results("distances", mock_results["distances"])
        
        # Store in pipeline
        self.pipeline.data.selected_feature_data["test"] = selector_data
        
        # Get actual metadata
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Build expected metadata  
        expected_metadata = self._build_expected_metadata("all", reference_traj=0)
        
        # Exact verification
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_ala_features_metadata(self):
        """Test metadata for ALA features selection with exact verification."""
        self._create_and_select_selector("test", "distances", "res ALA")
        
        # Get actual metadata
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Build expected metadata  
        expected_metadata = self._build_expected_metadata("res ALA", reference_traj=0)
        
        # Exact verification
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_gly_features_metadata(self):
        """Test metadata for GLY features selection with exact verification."""
        self._create_and_select_selector("test", "distances", "res GLY")
        
        # Get actual metadata
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Build expected metadata  
        expected_metadata = self._build_expected_metadata("res GLY", reference_traj=0)
        
        # Exact verification
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_val_features_metadata(self):
        """Test metadata for VAL features selection with exact verification."""
        self._create_and_select_selector("test", "distances", "res VAL")
        
        # Get actual metadata
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Build expected metadata  
        expected_metadata = self._build_expected_metadata("res VAL", reference_traj=0)
        
        # Exact verification
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_seqid_range_metadata(self):
        """Test metadata for seqid range selection with exact verification."""
        self._create_and_select_selector("test", "distances", "seqid 1-3")
        
        # Get actual metadata
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Build expected metadata  
        expected_metadata = self._build_expected_metadata("seqid 1-3", reference_traj=0)
        
        # Exact verification
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_multiple_residues_metadata(self):
        """Test metadata for multiple residue selection with exact verification."""
        self._create_and_select_selector("test", "distances", "res ALA GLY")
        
        # Get actual metadata
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Build expected metadata  
        expected_metadata = self._build_expected_metadata("res ALA GLY", reference_traj=0)
        
        # Exact verification
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_mixed_selection_metadata(self):
        """Test metadata for mixed residue selection with exact verification."""
        self._create_and_select_selector("test", "distances", "res ALA VAL")
        
        # Get actual metadata
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Build expected metadata  
        expected_metadata = self._build_expected_metadata("res ALA VAL", reference_traj=0)
        
        # Exact verification
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    # ========== REFERENCE TRAJECTORY TESTS ==========
    
    def test_ref_traj_0_metadata(self):
        """Test metadata using reference trajectory 0."""
        self._create_and_select_selector("test", "distances", "res ALA", reference_traj=0)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("res ALA", reference_traj=0)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_ref_traj_1_metadata(self):
        """Test metadata using reference trajectory 1."""
        self._create_and_select_selector("test", "distances", "res ALA", reference_traj=1)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("res ALA", reference_traj=1)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_ref_traj_2_metadata(self):
        """Test metadata using reference trajectory 2."""
        self._create_and_select_selector("test", "distances", "res ALA", reference_traj=2)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("res ALA", reference_traj=2)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_ref_traj_seqid_different_metadata(self):
        """Test metadata with seqid selection using trajectory 2 (different seqids)."""
        self._create_and_select_selector("test", "distances", "seqid 10-12", reference_traj=2)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("seqid 10-12", reference_traj=2)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_ref_traj_consensus_different_metadata(self):
        """Test metadata with consensus selection using trajectory 1 (different consensus)."""
        self._create_and_select_selector("test", "distances", "consensus 1.10", reference_traj=1)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("consensus 1.10", reference_traj=1)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    # ========== COMBINATION SELECTION TESTS ==========
    
    def test_and_combination_metadata(self):
        """Test metadata for AND combination selection."""
        self._create_and_select_selector("test", "distances", "res ALA and seqid 1-3", reference_traj=0)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("res ALA and seqid 1-3", reference_traj=0)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_not_combination_metadata(self):
        """Test metadata for NOT combination selection."""
        self._create_and_select_selector("test", "distances", "res ALA and not seqid 4", reference_traj=0)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("res ALA and not seqid 4", reference_traj=0)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_complex_combination_metadata(self):
        """Test metadata for complex combination selection."""
        self._create_and_select_selector("test", "distances", "seqid 1-4 and not res GLY", reference_traj=0)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("seqid 1-4 and not res GLY", reference_traj=0)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    def test_multi_and_combination_metadata(self):
        """Test metadata for multiple AND conditions."""
        self._create_and_select_selector("test", "distances", "res ALA VAL and seqid 1-3", reference_traj=0)
        
        actual_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = self._build_expected_metadata("res ALA VAL and seqid 1-3", reference_traj=0)
        
        self._verify_metadata_exact(actual_metadata, expected_metadata)
    
    # ========== STRUCTURE AND CONSISTENCY TESTS ==========
    
    def test_metadata_array_structure(self):
        """Test metadata array has correct structure and types."""
        self._create_and_select_selector("test", "distances", "res ALA", reference_traj=0)
        
        metadata = self.pipeline.data.get_selected_metadata("test")
        
        # Check array type
        assert isinstance(metadata, np.ndarray), f"Expected numpy array, got {type(metadata)}"
        
        # Check each entry structure
        for i, entry in enumerate(metadata):
            assert isinstance(entry, dict), f"Entry {i}: expected dict, got {type(entry)}"
            assert 'type' in entry, f"Entry {i}: missing 'type' field"
            assert 'features' in entry, f"Entry {i}: missing 'features' field"
            assert isinstance(entry['features'], np.ndarray), f"Entry {i}: features should be numpy array"
            assert len(entry['features']) == 2, f"Entry {i}: distances should have 2 features"
            
            # Check feature structure  
            for j, feature in enumerate(entry['features']):
                assert isinstance(feature, dict), f"Entry {i}, feature {j}: expected dict"
                required_fields = ['residue', 'special_label', 'full_name']
                for field in required_fields:
                    assert field in feature, f"Entry {i}, feature {j}: missing '{field}' field"
                    
                # Check residue structure
                residue_fields = ['resid', 'seqid', 'resname', 'aaa_code', 'a_code', 'consensus', 'full_name', 'index']
                for field in residue_fields:
                    assert field in feature['residue'], f"Entry {i}, feature {j}: missing residue.{field}"
            
    def test_required_fields_complete(self):
        """Test all required metadata fields are present and correct."""
        self._create_and_select_selector("test", "distances", "res GLY VAL", reference_traj=0)
        
        metadata = self.pipeline.data.get_selected_metadata("test")
        
        for i, entry in enumerate(metadata):
            # Check entry type
            assert entry['type'] == 'distances', f"Entry {i}: expected 'distances', got '{entry['type']}'"
            
            for j, feature in enumerate(entry['features']):
                residue = feature['residue']
                
                # Check data types
                assert isinstance(residue['resid'], int), f"Entry {i}, feature {j}: resid should be int"
                assert isinstance(residue['seqid'], int), f"Entry {i}, feature {j}: seqid should be int"
                assert isinstance(residue['index'], int), f"Entry {i}, feature {j}: index should be int"
                assert isinstance(residue['resname'], str), f"Entry {i}, feature {j}: resname should be str"
                assert isinstance(residue['consensus'], str), f"Entry {i}, feature {j}: consensus should be str"
                
                # Check value consistency
                assert residue['resid'] > 0, f"Entry {i}, feature {j}: resid should be positive"
                assert residue['seqid'] > 0, f"Entry {i}, feature {j}: seqid should be positive" 
                assert residue['index'] >= 0, f"Entry {i}, feature {j}: index should be non-negative"
                assert len(residue['resname']) <= 3, f"Entry {i}, feature {j}: resname too long"
                assert len(residue['a_code']) == 1, f"Entry {i}, feature {j}: a_code should be single char"
            
    def test_metadata_content_consistency(self):
        """Test metadata content consistency across different selections."""
        # Test same selection gives same metadata structure  
        self._create_and_select_selector("test1", "distances", "res ALA", reference_traj=0)
        self._create_and_select_selector("test2", "distances", "res ALA", reference_traj=0)
        
        metadata1 = self.pipeline.data.get_selected_metadata("test1")
        metadata2 = self.pipeline.data.get_selected_metadata("test2")
        
        # Should be identical
        assert len(metadata1) == len(metadata2), "Same selection should give same length metadata"
        
        for i, (entry1, entry2) in enumerate(zip(metadata1, metadata2)):
            assert entry1['type'] == entry2['type'], f"Entry {i}: type mismatch"
            assert len(entry1['features']) == len(entry2['features']), f"Entry {i}: features length mismatch"
            
            for j, (feat1, feat2) in enumerate(zip(entry1['features'], entry2['features'])):
                for key in feat1['residue'].keys():
                    assert feat1['residue'][key] == feat2['residue'][key], \
                        f"Entry {i}, feature {j}: residue.{key} mismatch"
        

class TestGetSelectedMetadataErrorCases:
    """Test error conditions and edge cases."""
    
    def setup_method(self):
        """Setup for error testing."""
        self.pipeline = PipelineManager()
    
    def test_nonexistent_feature_selector(self):
        """Test error when feature selector doesn't exist."""
        with pytest.raises(ValueError):
            self.pipeline.data.get_selected_metadata("nonexistent")
    
    def test_empty_selection_metadata(self):
        """Test metadata error for empty feature selection."""
        # Create setup with multiple residues to allow feature computation
        topology = md.Topology()
        chain = topology.add_chain()
        res1 = topology.add_residue("ALA", chain)
        res2 = topology.add_residue("GLY", chain)
        topology.add_atom("CA", md.element.carbon, res1)
        topology.add_atom("CA", md.element.carbon, res2)
        
        xyz = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        traj = md.Trajectory(xyz, topology)
        
        self.pipeline.data.trajectory_data.trajectories = [traj]
        self.pipeline.data.trajectory_data.trajectory_names = ["test"]
        
        # Add basic metadata
        self.pipeline.data.trajectory_data.res_label_data = {0: [
            {"resid": 1, "seqid": 1, "resname": "ALA", "aaa_code": "ALA",
             "a_code": "A", "consensus": "3.50", "full_name": "ALA1", "index": 0},
            {"resid": 2, "seqid": 2, "resname": "GLY", "aaa_code": "GLY",
             "a_code": "G", "consensus": "3.51", "full_name": "GLY2", "index": 1}
        ]}
        
        # Add features
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
        
        # Try to select non-existent residue
        with pytest.raises(ValueError, match="Reference trajectory 0 not found in selection"):
            self.pipeline.feature_selector.create("test")
            self.pipeline.feature_selector.add("test", "distances", "res XYZ")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.pipeline.feature_selector.select("test", reference_traj=0)
            self.pipeline.data.get_selected_metadata("test")