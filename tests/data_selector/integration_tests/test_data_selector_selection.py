"""Comprehensive tests for DataSelector functionality."""

import numpy as np
import pytest
import mdtraj as md
from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances


class TestDataSelectorBasics:
    """Test basic DataSelector functionality."""
    
    def setup_method(self):
        """Setup with synthetic trajectory."""
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
        
        # Create coordinates (5 atoms, 100 frames)
        coordinates = []
        for frame in range(100):
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
        
        # Add features: distances with excluded_neighbors=0
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0))
    
    def test_create_data_selector_basic(self):
        """Test creating basic data selector."""
        self.pipeline.data_selector.create("test")
        
        assert "test" in self.pipeline.data.data_selector_data
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.name == "test"
        
        # Add frame selection
        self.pipeline.data_selector.select_by_indices("test", {0: list(range(10, 20))})
        assert 0 in selector.trajectory_frames
        assert selector.trajectory_frames[0] == list(range(10, 20))
    
    def test_frame_range_selection(self):
        """Test frame range selection."""
        self.pipeline.data_selector.create("test")
        self.pipeline.data_selector.select_by_indices("test", {0: list(range(20, 30))})
        
        # Test that data selector correctly selects frame range
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == list(range(20, 30))
        
        # Verify expected frame count
        expected_frames = 30 - 20  # 10 frames
        assert selector.n_selected_frames == expected_frames
    
    def test_stride_selection(self):
        """Test stride selection."""
        self.pipeline.data_selector.create("test")
        
        # Create stride selection manually (every 5th frame)
        stride_frames = list(range(0, 100, 5))
        self.pipeline.data_selector.select_by_indices("test", {0: stride_frames})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == stride_frames
        
        # With 100 frames and stride=5, should get 20 frames
        expected_frames = 100 // 5  # 20 frames
        assert selector.n_selected_frames == expected_frames
    
    def test_combined_frame_range_and_stride(self):
        """Test combination of frame range and stride."""
        self.pipeline.data_selector.create("test")
        
        # Create combined range and stride selection manually (frames 10-50 with stride 4)
        combined_frames = list(range(10, 50, 4))
        self.pipeline.data_selector.select_by_indices("test", {0: combined_frames})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == combined_frames
        
        # 40 frames with stride 4 = 10 frames
        expected_frames = len(combined_frames)  # 10 frames
        assert selector.n_selected_frames == expected_frames


class TestMultipleDataSelectors:
    """Test multiple data selectors coexisting."""
    
    def setup_method(self):
        """Setup with synthetic trajectory."""
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def test_multiple_selectors_coexist(self):
        """Test multiple data selectors can coexist."""
        self.pipeline.data_selector.create("early")
        self.pipeline.data_selector.create("late")  
        self.pipeline.data_selector.create("sparse")
        
        # Add different frame selections
        self.pipeline.data_selector.select_by_indices("early", {0: list(range(0, 25))})
        self.pipeline.data_selector.select_by_indices("late", {0: list(range(75, 100))})
        self.pipeline.data_selector.select_by_indices("sparse", {0: list(range(0, 100, 10))})
        
        # Verify all selectors exist
        assert "early" in self.pipeline.data.data_selector_data
        assert "late" in self.pipeline.data.data_selector_data
        assert "sparse" in self.pipeline.data.data_selector_data
        
        # Verify they have different settings
        early = self.pipeline.data.data_selector_data["early"]
        late = self.pipeline.data.data_selector_data["late"]
        sparse = self.pipeline.data.data_selector_data["sparse"]
        
        assert early.trajectory_frames[0] == list(range(0, 25))
        assert late.trajectory_frames[0] == list(range(75, 100))
        assert sparse.trajectory_frames[0] == list(range(0, 100, 10))
        
        # Verify different frame counts
        assert early.n_selected_frames == 25
        assert late.n_selected_frames == 25
        assert sparse.n_selected_frames == 10
    
    def test_selector_name_uniqueness(self):
        """Test that data selector names must be unique."""
        self.pipeline.data_selector.create("test")
        
        # Creating selector with same name should raise error
        with pytest.raises(ValueError, match="Data selector 'test' already exists"):
            self.pipeline.data_selector.create("test")



class TestDataSelectorMultiTrajectory:
    """Test data selector functionality with multiple trajectories."""
    
    def setup_method(self):
        """Setup with multiple trajectories for multi-trajectory selection testing."""
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        
        # Add second trajectory (50 frames)
        coordinates_2 = []
        for frame in range(50):
            frame_coords = []
            for atom_idx in range(5):
                x = atom_idx * 1.0 + 0.5  # Slightly different positions
                y = frame * 0.1 + 5.0
                z = 0.0
                frame_coords.append([x, y, z])
            coordinates_2.append(frame_coords)
        
        xyz_2 = np.array(coordinates_2)
        test_traj_2 = md.Trajectory(xyz_2, self.pipeline.data.trajectory_data.trajectories[0].topology)
        
        self.pipeline.data.trajectory_data.trajectories.append(test_traj_2)
        self.pipeline.data.trajectory_data.trajectory_names.append("synthetic_2")
        
        # Copy residue metadata
        import copy
        self.pipeline.data.trajectory_data.res_label_data[1] = copy.deepcopy(self.pipeline.data.trajectory_data.res_label_data[0])
        
        # Recompute features
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0), force=True)
    
    def test_multi_trajectory_all_frames(self):
        """Test data selector with all frames from multiple trajectories."""
        self.pipeline.data_selector.create("all_frames")
        
        # Select all frames from both trajectories
        self.pipeline.data_selector.select_by_indices("all_frames", {
            0: list(range(100)),  # All frames from trajectory 0
            1: list(range(50))    # All frames from trajectory 1
        })
        
        selector = self.pipeline.data.data_selector_data["all_frames"]
        
        # Check frames per trajectory
        assert selector.trajectory_frames[0] == list(range(100))
        assert selector.trajectory_frames[1] == list(range(50))
        
        # Total frames = 100 + 50 = 150
        assert selector.n_selected_frames == 150
    
    def test_multi_trajectory_partial_frames(self):
        """Test data selector with partial frame ranges from multiple trajectories."""
        self.pipeline.data_selector.create("partial_frames")
        
        # Select specific frame ranges from each trajectory
        # Trajectory 0: frames 50-99, Trajectory 1: frames 0-20
        self.pipeline.data_selector.select_by_indices("partial_frames", {
            0: list(range(50, 100)),  # Last 50 frames from traj 0
            1: list(range(0, 21))     # First 21 frames from traj 1
        })
        
        selector = self.pipeline.data.data_selector_data["partial_frames"]
        
        # Check frames per trajectory
        assert selector.trajectory_frames[0] == list(range(50, 100))
        assert selector.trajectory_frames[1] == list(range(0, 21))
        
        # Total frames = 50 + 21 = 71
        assert selector.n_selected_frames == 71
    
    def test_multi_trajectory_stride_frames(self):
        """Test data selector with stride selection across multiple trajectories."""
        self.pipeline.data_selector.create("stride_frames")
        
        # Apply stride=5 to both trajectories
        self.pipeline.data_selector.select_by_indices("stride_frames", {
            0: list(range(0, 100, 5)),  # Every 5th frame from traj 0
            1: list(range(0, 50, 5))    # Every 5th frame from traj 1
        })
        
        selector = self.pipeline.data.data_selector_data["stride_frames"]
        
        # Check frames per trajectory
        expected_traj_0 = list(range(0, 100, 5))  # [0, 5, 10, ..., 95]
        expected_traj_1 = list(range(0, 50, 5))   # [0, 5, 10, ..., 45]
        
        assert selector.trajectory_frames[0] == expected_traj_0
        assert selector.trajectory_frames[1] == expected_traj_1
        
        # Total frames = 20 + 10 = 30
        assert selector.n_selected_frames == 30


class TestDataSelectorErrorHandling:
    """Test error handling in data selector."""
    
    def setup_method(self):
        """Setup with synthetic trajectory."""
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def test_invalid_frame_indices(self):
        """Test error handling for invalid frame indices."""
        self.pipeline.data_selector.create("test")
        
        # Out of bounds frame indices
        with pytest.raises((ValueError, IndexError)):
            self.pipeline.data_selector.select_by_indices("test", {0: [200]})
    
    def test_invalid_trajectory_index(self):
        """Test error handling for invalid trajectory indices."""
        self.pipeline.data_selector.create("test")
        
        # Non-existent trajectory index
        with pytest.raises((ValueError, KeyError, IndexError)):
            self.pipeline.data_selector.select_by_indices("test", {999: [0, 1, 2]})
    
    def test_nonexistent_selector(self):
        """Test accessing non-existent data selector."""
        with pytest.raises(KeyError):
            _ = self.pipeline.data.data_selector_data["nonexistent"]


class TestDataSelectorIndicesAdvanced:
    """Test advanced select_by_indices functionality."""
    
    def setup_method(self):
        """Setup with multiple trajectories and tags."""
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        
        # Add second trajectory
        self.pipeline.data.trajectory_data.trajectories.append(self.pipeline.data.trajectory_data.trajectories[0])
        self.pipeline.data.trajectory_data.trajectory_names.append("synthetic_2")
        
        # Add trajectory tags
        self.pipeline.data.trajectory_data.trajectory_tags = {
            0: ["folded", "system_A"],
            1: ["unfolded", "system_B"]
        }
        
        # Copy residue metadata
        import copy
        self.pipeline.data.trajectory_data.res_label_data[1] = copy.deepcopy(self.pipeline.data.trajectory_data.res_label_data[0])
        
        # Recompute features
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0), force=True)
    
    def test_trajectory_selection_by_name(self):
        """Test trajectory selection by name."""
        self.pipeline.data_selector.create("test")
        
        # Select by trajectory name
        self.pipeline.data_selector.select_by_indices("test", {"synthetic": [10, 20, 30]})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert 0 in selector.trajectory_frames
        assert selector.trajectory_frames[0] == [10, 20, 30]
    
    def test_trajectory_selection_by_tag(self):
        """Test trajectory selection by tag."""
        self.pipeline.data_selector.create("test")
        
        # Select by tag - should apply to trajectory 0 (has "folded" tag)
        self.pipeline.data_selector.select_by_indices("test", {"tag:folded": [5, 15, 25]})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert 0 in selector.trajectory_frames
        assert selector.trajectory_frames[0] == [5, 15, 25]
    
    def test_frame_selection_single_int(self):
        """Test frame selection with single integer."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: 42})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == [42]
    
    def test_frame_selection_string_single(self):
        """Test frame selection with string single frame."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "42"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == [42]
    
    def test_frame_selection_string_range(self):
        """Test frame selection with string range."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "10-15"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == [10, 11, 12, 13, 14, 15]
    
    def test_frame_selection_string_comma_list(self):
        """Test frame selection with comma-separated list."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "10,20,30"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert sorted(selector.trajectory_frames[0]) == [10, 20, 30]
    
    def test_frame_selection_string_combined(self):
        """Test frame selection with combined ranges and singles."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "10-12,20,30-32"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        expected = [10, 11, 12, 20, 30, 31, 32]
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_frame_selection_string_all(self):
        """Test frame selection with 'all' keyword."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "all"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == list(range(100))  # All 100 frames
    
    def test_frame_selection_dict_with_stride(self):
        """Test frame selection with stride dictionary."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: {"frames": "0-50", "stride": 10}})
        
        selector = self.pipeline.data.data_selector_data["test"]
        expected = [0, 10, 20, 30, 40, 50]
        assert selector.trajectory_frames[0] == expected
    
    def test_stride_minimum_distance(self):
        """Test stride as minimum distance filter on sparse frame selection."""
        self.pipeline.data_selector.create("test")
        
        # Select sparse frames with stride applied directly
        # This tests stride on a sparse selection pattern
        sparse_frames = "0,1,2,10,11,12,20,21,50,51"
        self.pipeline.data_selector.select_by_indices("test", {0: {"frames": sparse_frames, "stride": 5}})
        
        selector = self.pipeline.data.data_selector_data["test"]
        # From [0, 1, 2, 10, 11, 12, 20, 21, 50, 51] with stride=5 (minimum distance)
        # Expected: [0, 10, 20, 50] (each frame has minimum 5 distance from next)
        expected = [0, 10, 20, 50]
        assert selector.trajectory_frames[0] == expected
    
    def test_stride_on_union_pattern(self):
        """Test stride on frames that simulate union operation result."""
        self.pipeline.data_selector.create("test")
        
        # Select frames that would result from union: [10, 15, 20, 25, 30, 35]
        # Apply stride=10 (minimum distance) to this pattern
        union_frames = "10,15,20,25,30,35"
        self.pipeline.data_selector.select_by_indices("test", {0: {"frames": union_frames, "stride": 10}})
        
        selector = self.pipeline.data.data_selector_data["test"]
        # From [10, 15, 20, 25, 30, 35] with stride=10 (min distance)
        # Expected: [10, 20, 30] (each frame minimum 10 apart)
        expected = [10, 20, 30]
        assert selector.trajectory_frames[0] == expected
    
    def test_stride_on_intersection_pattern(self):
        """Test stride on frames that simulate intersection operation result."""
        self.pipeline.data_selector.create("test")
        
        # Select frames that would result from intersection: [20, 25, 30, 35, 40, 45]
        # Apply stride=15 (minimum distance) to this pattern
        intersection_frames = "20,25,30,35,40,45"
        self.pipeline.data_selector.select_by_indices("test", {0: {"frames": intersection_frames, "stride": 15}})
        
        selector = self.pipeline.data.data_selector_data["test"]
        # From [20, 25, 30, 35, 40, 45] with stride=15 (min distance)
        # Expected: [20, 35] (minimum 15 frames apart)
        expected = [20, 35]
        assert selector.trajectory_frames[0] == expected


class TestDataSelectorModes:
    """Test different selection modes (add, subtract, intersect)."""
    
    def setup_method(self):
        """Setup with basic trajectory."""
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def test_mode_add_default(self):
        """Test default add mode (union)."""
        self.pipeline.data_selector.create("test")
        
        # First selection
        self.pipeline.data_selector.select_by_indices("test", {0: [10, 20, 30]})
        
        # Second selection with default mode (add)
        self.pipeline.data_selector.select_by_indices("test", {0: [25, 35, 45]})
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should be union: [10, 20, 30] + [25, 35, 45] = [10, 20, 25, 30, 35, 45]
        expected = [10, 20, 25, 30, 35, 45]
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_mode_add_explicit(self):
        """Test explicit add mode (union)."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: [10, 20, 30]})
        self.pipeline.data_selector.select_by_indices("test", {0: [25, 35, 45]}, mode="add")
        
        selector = self.pipeline.data.data_selector_data["test"]
        expected = [10, 20, 25, 30, 35, 45]
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_mode_subtract(self):
        """Test subtract mode (difference)."""
        self.pipeline.data_selector.create("test")
        
        # First selection: large range
        self.pipeline.data_selector.select_by_indices("test", {0: "10-50"})
        
        # Subtract some frames
        self.pipeline.data_selector.select_by_indices("test", {0: [15, 25, 35, 45]}, mode="subtract")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should be: [10-50] - [15, 25, 35, 45]
        all_frames = set(range(10, 51))
        subtract_frames = {15, 25, 35, 45}
        expected = sorted(all_frames - subtract_frames)
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_mode_intersect(self):
        """Test intersect mode (intersection)."""
        self.pipeline.data_selector.create("test")
        
        # First selection: range 10-30
        self.pipeline.data_selector.select_by_indices("test", {0: "10-30"})
        
        # Intersect with range 20-40
        self.pipeline.data_selector.select_by_indices("test", {0: "20-40"}, mode="intersect")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should be intersection: [10-30] ∩ [20-40] = [20-30]
        expected = list(range(20, 31))
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_multiple_mode_operations(self):
        """Test multiple operations with different modes."""
        self.pipeline.data_selector.create("test")
        
        # Start with base selection
        self.pipeline.data_selector.select_by_indices("test", {0: "0-50"})
        
        # Add more frames
        self.pipeline.data_selector.select_by_indices("test", {0: "60-70"}, mode="add")
        
        # Subtract some frames  
        self.pipeline.data_selector.select_by_indices("test", {0: [10, 20, 30, 65]}, mode="subtract")
        
        # Intersect with final range
        self.pipeline.data_selector.select_by_indices("test", {0: "5-80"}, mode="intersect")
        
        selector = self.pipeline.data.data_selector_data["test"]
        
        # Manual calculation:
        # 1. Start: [0-50]
        # 2. Add [60-70]: [0-50] + [60-70] 
        # 3. Subtract [10,20,30,65]: remove 10,20,30,65
        # 4. Intersect [5-80]: keep only frames in range 5-80
        initial = set(range(0, 51)) | set(range(60, 71))
        after_subtract = initial - {10, 20, 30, 65}
        final = after_subtract & set(range(5, 81))
        expected = sorted(final)
        
        assert sorted(selector.trajectory_frames[0]) == expected


class TestDataSelectorTags:
    """Test select_by_tags functionality."""
    
    def setup_method(self):
        """Setup with multiple trajectories with different tags."""
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        
        # Add second trajectory
        coordinates_2 = []
        for frame in range(80):  # Different number of frames
            frame_coords = []
            for atom_idx in range(5):
                x = atom_idx * 1.0 + 0.5  # Different positions
                y = frame * 0.1 + 10.0
                z = 1.0
                frame_coords.append([x, y, z])
            coordinates_2.append(frame_coords)
        
        xyz_2 = np.array(coordinates_2)
        test_traj_2 = md.Trajectory(xyz_2, self.pipeline.data.trajectory_data.trajectories[0].topology)
        
        self.pipeline.data.trajectory_data.trajectories.append(test_traj_2)
        self.pipeline.data.trajectory_data.trajectory_names.append("synthetic_2")
        
        # Copy residue metadata
        import copy
        self.pipeline.data.trajectory_data.res_label_data[1] = copy.deepcopy(self.pipeline.data.trajectory_data.res_label_data[0])
        
        # Add trajectory-level tags (not frame tags)
        self.pipeline.data.trajectory_data.trajectory_tags = {
            0: ["folded", "stable", "system_A"],
            1: ["unfolded", "dynamic", "system_B"]
        }
        
        # Recompute features
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0), force=True)
    
    def test_select_by_single_tag(self):
        """Test selection by single tag."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_tags("test", ["folded"])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Trajectory 0 has "folded" tag, so all 100 frames from trajectory 0
        expected = list(range(0, 100))
        assert sorted(selector.trajectory_frames[0]) == expected
        # Trajectory 1 doesn't have "folded" tag, so no frames
        assert 1 not in selector.trajectory_frames
    
    def test_select_by_multiple_tags_match_all_true(self):
        """Test selection by multiple tags with match_all=True (AND)."""
        self.pipeline.data_selector.create("test")
        
        # Select trajectories that have BOTH "folded" AND "stable" tags
        self.pipeline.data_selector.select_by_tags("test", ["folded", "stable"], match_all=True)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Only trajectory 0 has both "folded" AND "stable" tags
        expected = list(range(0, 100))
        assert sorted(selector.trajectory_frames[0]) == expected
        # Trajectory 1 doesn't have both tags
        assert 1 not in selector.trajectory_frames
    
    def test_select_by_multiple_tags_match_all_false(self):
        """Test selection by multiple tags with match_all=False (OR)."""
        self.pipeline.data_selector.create("test")
        
        # Select trajectories that have "folded" OR "unfolded" tags
        self.pipeline.data_selector.select_by_tags("test", ["folded", "unfolded"], match_all=False)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Both trajectories have either "folded" OR "unfolded" tags
        # Trajectory 0: 100 frames, Trajectory 1: 80 frames
        assert sorted(selector.trajectory_frames[0]) == list(range(0, 100))
        assert sorted(selector.trajectory_frames[1]) == list(range(0, 80))
    
    def test_select_by_tags_with_stride(self):
        """Test tag selection with stride parameter."""
        self.pipeline.data_selector.create("test")
        
        # Select trajectories with "stable" tag, with stride=5
        self.pipeline.data_selector.select_by_tags("test", ["stable"], stride=5)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Trajectory 0 has "stable" tag, with stride=5: 0, 5, 10, 15, ..., 95
        expected = list(range(0, 100, 5))
        assert sorted(selector.trajectory_frames[0]) == expected
        # Trajectory 1 doesn't have "stable" tag
        assert 1 not in selector.trajectory_frames
    
    def test_select_by_tags_with_modes(self):
        """Test tag selection with different modes."""
        self.pipeline.data_selector.create("test")
        
        # First select trajectories with "folded" tag
        self.pipeline.data_selector.select_by_tags("test", ["folded"])
        
        # Add trajectories with "unfolded" tag
        self.pipeline.data_selector.select_by_tags("test", ["unfolded"], mode="add")
        
        # Subtract trajectories with "system_B" tag
        self.pipeline.data_selector.select_by_tags("test", ["system_B"], mode="subtract")
        
        selector = self.pipeline.data.data_selector_data["test"]
        
        # After folded + unfolded - system_B:
        # Start: trajectory 0 (has "folded")
        # Add: trajectory 1 (has "unfolded") 
        # Subtract: trajectory 1 (has "system_B")
        # Result: only trajectory 0 remains
        assert sorted(selector.trajectory_frames[0]) == list(range(0, 100))
        assert 1 not in selector.trajectory_frames
    
    def test_select_by_nonexistent_tag(self):
        """Test selection by tag that doesn't exist."""
        self.pipeline.data_selector.create("test")
        
        # This should result in empty selection
        self.pipeline.data_selector.select_by_tags("test", ["nonexistent_tag"])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # No trajectories have this tag, so no frames should be selected
        assert len(selector.trajectory_frames) == 0
    
    def test_contradictory_tags_stable_unstable(self):
        """Test match_all=True with stable AND unstable tags."""
        # Add contradictory tag to trajectory 0
        self.pipeline.data.trajectory_data.trajectory_tags[0].extend(["unstable"])
        
        self.pipeline.data_selector.create("test")
        self.pipeline.data_selector.select_by_tags("test", ["stable", "unstable"], match_all=True)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Only trajectory 0 has BOTH "stable" AND "unstable" tags
        assert 0 in selector.trajectory_frames
        assert sorted(selector.trajectory_frames[0]) == list(range(0, 100))
        assert 1 not in selector.trajectory_frames
    
    def test_contradictory_tags_folded_unfolded(self):
        """Test match_all=True with folded AND unfolded tags."""
        # Add contradictory tags to trajectory 1
        self.pipeline.data_selector.create("test")
        self.pipeline.data_selector.select_by_tags("test", ["folded", "unfolded"], match_all=True)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # No trajectory has BOTH "folded" AND "unfolded" tags
        assert selector.trajectory_frames == {}
    
    def test_impossible_tag_combination(self):
        """Test match_all=True with impossible 4-tag combination."""
        # Add tags to make one trajectory have some but not all
        self.pipeline.data.trajectory_data.trajectory_tags[0].extend(["unstable"])
        self.pipeline.data.trajectory_data.trajectory_tags[1].extend(["folded", "stable"])
        
        self.pipeline.data_selector.create("test")
        self.pipeline.data_selector.select_by_tags("test", 
                                                   ["folded", "unfolded", "stable", "unstable"], 
                                                   match_all=True)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # No trajectory has all 4 contradictory tags
        assert len(selector.trajectory_frames) == 0


class TestDataSelectorCluster:
    """Test select_by_cluster functionality."""
    
    def setup_method(self):
        """Setup with trajectory and mock clustering results."""
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        
        # Mock some clustering results
        # Simulate cluster assignments for 100 frames
        from mdxplain.clustering.entities.cluster_data import ClusterData
        
        # Create mock cluster assignments
        # Cluster 0: frames 0-30, Cluster 1: frames 31-70, Cluster 2: frames 71-99
        labels = []
        for frame in range(100):
            if frame <= 30:
                labels.append(0)
            elif frame <= 70:
                labels.append(1)
            else:
                labels.append(2)
        
        cluster_data = ClusterData("test_clustering")
        cluster_data.labels = np.array(labels)  # Convert to numpy array
        
        # Create frame mapping: global_frame_index -> (trajectory_index, local_frame_index)
        frame_mapping = {}
        for global_frame in range(100):
            frame_mapping[global_frame] = (0, global_frame)  # All frames from trajectory 0
        cluster_data.set_frame_mapping(frame_mapping)
        
        # Add to pipeline data
        self.pipeline.data.cluster_data = {"test_clustering": cluster_data}
    
    def test_select_by_single_cluster(self):
        """Test selection by single cluster ID."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [0])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Cluster 0 contains frames 0-30
        expected = list(range(0, 31))
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_select_by_multiple_clusters(self):
        """Test selection by multiple cluster IDs."""
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [0, 2])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Cluster 0 (frames 0-30) + Cluster 2 (frames 71-99)
        expected = list(range(0, 31)) + list(range(71, 100))
        assert sorted(selector.trajectory_frames[0]) == sorted(expected)
    
    def test_select_by_cluster_with_stride(self):
        """Test cluster selection with stride parameter."""
        self.pipeline.data_selector.create("test")
        
        # Select cluster 1 with stride=5
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [1], stride=5)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Cluster 1 frames 31-70, with stride=5: 31, 36, 41, 46, 51, 56, 61, 66
        cluster_1_frames = list(range(31, 71))
        expected = cluster_1_frames[::5]  # Every 5th frame starting from first
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_select_by_cluster_with_modes(self):
        """Test cluster selection with different modes."""
        self.pipeline.data_selector.create("test")
        
        # Start with cluster 0
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [0])
        
        # Add cluster 2
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [2], mode="add")
        
        # Subtract some frames from cluster 1 (should not affect since cluster 1 wasn't selected)
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [1], mode="subtract")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should still have cluster 0 + cluster 2 (subtract of cluster 1 has no effect)
        expected = list(range(0, 31)) + list(range(71, 100))
        assert sorted(selector.trajectory_frames[0]) == sorted(expected)
    
    def test_select_by_nonexistent_cluster(self):
        """Test selection by cluster that doesn't exist."""
        self.pipeline.data_selector.create("test")
        
        # Select cluster ID 999 which doesn't exist - should result in empty selection
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [999])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # No frames should be selected since cluster 999 doesn't exist
        assert selector.n_selected_frames == 0
        assert len(selector.trajectory_frames) == 0
    
    def test_select_by_nonexistent_clustering(self):
        """Test selection by clustering that doesn't exist."""
        self.pipeline.data_selector.create("test")
        
        # Try to select from clustering that doesn't exist
        with pytest.raises((ValueError, KeyError)):
            self.pipeline.data_selector.select_by_cluster("test", "nonexistent_clustering", [0])


class TestDataSelectorMixedModes:
    """Test mixed mode operations with simplified setup."""
    
    def setup_method(self):
        """Setup with 2 trajectories, basic tags and simple clustering."""
        # Reuse existing basic setup
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
        
        # Add second trajectory (80 frames)
        coords = [[[atom_idx * 1.0 + 0.5, frame * 0.1 + 5.0, 1.0] for atom_idx in range(5)] 
                 for frame in range(80)]
        traj_2 = md.Trajectory(np.array(coords), self.pipeline.data.trajectory_data.trajectories[0].topology)
        self.pipeline.data.trajectory_data.trajectories.append(traj_2)
        self.pipeline.data.trajectory_data.trajectory_names.append("traj_B")
        
        # Simple tags
        self.pipeline.data.trajectory_data.trajectory_tags = {
            0: ["system_A", "stable"],
            1: ["system_B", "test"]
        }
        
        # Simple clustering: Traj 0 has cluster 0 (0-49) and 1 (50-99), Traj 1 has cluster 1 (0-39) and 0 (40-79)
        from mdxplain.clustering.entities.cluster_data import ClusterData
        labels = [0] * 50 + [1] * 50 + [1] * 40 + [0] * 40  # 180 total frames
        frame_mapping = {i: (0, i) for i in range(100)} | {i: (1, i-100) for i in range(100, 180)}
        
        cluster_data = ClusterData("simple_clustering")
        cluster_data.labels = np.array(labels)
        cluster_data.set_frame_mapping(frame_mapping)
        self.pipeline.data.cluster_data = {"simple_clustering": cluster_data}
        
        # Copy residue metadata
        import copy
        self.pipeline.data.trajectory_data.res_label_data[1] = copy.deepcopy(self.pipeline.data.trajectory_data.res_label_data[0])
        self.pipeline.feature.add_feature(Distances(excluded_neighbors=0), force=True)
    
    def test_mixed_cluster_and_tags(self):
        """Test combining cluster selection with tag selection."""
        self.pipeline.data_selector.create("test")
        
        # Start with cluster 0
        self.pipeline.data_selector.select_by_cluster("test", "simple_clustering", [0])
        # Add frames from "system_A" trajectories
        self.pipeline.data_selector.select_by_tags("test", ["system_A"], mode="add")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should have all frames from trajectory 0 (cluster 0 + system_A tag)
        assert 0 in selector.trajectory_frames
        assert sorted(selector.trajectory_frames[0]) == list(range(100))
    
    def test_mixed_modes_add_subtract(self):
        """Test add and subtract modes with different selection types."""
        self.pipeline.data_selector.create("test")
        
        # Start with all frames from trajectory 0
        self.pipeline.data_selector.select_by_indices("test", {0: "0-99"})
        # Subtract cluster 1 frames
        self.pipeline.data_selector.select_by_cluster("test", "simple_clustering", [1], mode="subtract")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should have only cluster 0 frames from trajectory 0 (frames 0-49)
        assert sorted(selector.trajectory_frames[0]) == list(range(50))
    
    def test_mixed_modes_intersect(self):
        """Test intersect mode with tag and index selections."""
        self.pipeline.data_selector.create("test")
        
        # Start with all "system_A" frames
        self.pipeline.data_selector.select_by_tags("test", ["system_A"])
        # Intersect with specific frame range
        self.pipeline.data_selector.select_by_indices("test", {0: "25-75"}, mode="intersect")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should have frames 25-75 from trajectory 0 only
        assert sorted(selector.trajectory_frames[0]) == list(range(25, 76))
        assert 1 not in selector.trajectory_frames
    
    def test_three_way_combination(self):
        """Test simple 3-operation combination."""
        self.pipeline.data_selector.create("test")
        
        # 1. Start with cluster 1
        self.pipeline.data_selector.select_by_cluster("test", "simple_clustering", [1])
        # 2. Add frames from "test" tag
        self.pipeline.data_selector.select_by_tags("test", ["test"], mode="add")
        # 3. Intersect with frame range 20-60
        self.pipeline.data_selector.select_by_indices("test", {0: "20-60", 1: "20-60"}, mode="intersect")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should have frames from both trajectories in range 20-60
        # Traj 0: cluster 1 (frames 50-99) intersect with 20-60 = frames 50-60 (11 frames)
        # Traj 1: "test" tag (all 80 frames) intersect with 20-60 = frames 20-60 (41 frames)
        expected_traj_0 = list(range(50, 61))  # 11 frames
        expected_traj_1 = list(range(20, 61))  # 41 frames
        
        assert sorted(selector.trajectory_frames[0]) == expected_traj_0
        assert sorted(selector.trajectory_frames[1]) == expected_traj_1
        assert selector.n_selected_frames == 52  # 11 + 41