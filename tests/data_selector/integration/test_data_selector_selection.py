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

"""Comprehensive tests for DataSelector functionality."""

import numpy as np
import pytest
import mdtraj as md
from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances


class TestDataSelectorBasics:
    """Test basic DataSelector functionality."""
    
    def setup_method(self):
        """
        Setup with synthetic trajectory.

        Creates a simple synthetic trajectory with 5 residues and 100 frames.
        """
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
        """
        Test basic data selector creation and frame selection.
        Validates that data selector is created correctly with proper name
        and can select frame ranges from specific trajectories.
        """
        self.pipeline.data_selector.create("test")
        
        assert "test" in self.pipeline.data.data_selector_data
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.name == "test"
        
        # Add frame selection
        self.pipeline.data_selector.select_by_indices("test", {0: list(range(10, 20))})
        assert 0 in selector.trajectory_frames
        assert selector.trajectory_frames[0] == list(range(10, 20))
    
    def test_frame_range_selection(self):
        """
        Test data selector frame range functionality.
        Validates that frame ranges are correctly selected and counted
        from trajectory data with proper bounds checking.
        """
        self.pipeline.data_selector.create("test")
        self.pipeline.data_selector.select_by_indices("test", {0: list(range(20, 30))})
        
        # Test that data selector correctly selects frame range
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == list(range(20, 30))
        
        # Verify expected frame count
        expected_frames = 30 - 20  # 10 frames
        assert selector.n_selected_frames == expected_frames
    
    def test_stride_selection(self):
        """
        Test data selector stride functionality.
        Validates that every nth frame is correctly selected using stride
        and total frame count matches expected stride calculation.
        """
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
        """
        Test combined frame range and stride selection.
        Validates that frame range combined with stride produces correct
        subset of frames with proper counting and indexing.
        """
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
        """
        Test that multiple data selectors can coexist independently.
        Validates that different selectors maintain separate frame selections
        and settings without interfering with each other.
        """
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
        """
        Test data selector name uniqueness enforcement.
        Validates that attempting to create a selector with existing name
        raises appropriate ValueError with informative message.
        """
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
        """
        Test data selector with all frames from multiple trajectories.
        Validates that selector correctly handles different trajectory lengths
        and counts total frames across all selected trajectories.
        """
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
        """
        Test data selector with partial frame ranges from multiple trajectories.
        Validates that different frame ranges can be selected per trajectory
        and total frame count reflects selected ranges only.
        """
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
        """
        Test data selector with stride selection across multiple trajectories.
        Validates that stride is correctly applied per trajectory independently
        and produces expected frame counts for each trajectory.
        """
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
        """
        Setup with synthetic trajectory.
        
        Creates a simple synthetic trajectory with 5 residues and 100 frames.
        """
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def test_invalid_frame_indices(self):
        """
        Test error handling for invalid frame indices.
        Validates that out-of-bounds frame indices raise appropriate
        ValueError or IndexError with clear error messages.
        """
        self.pipeline.data_selector.create("test")
        
        # Out of bounds frame indices
        with pytest.raises((ValueError, IndexError)):
            self.pipeline.data_selector.select_by_indices("test", {0: [200]})
    
    def test_invalid_trajectory_index(self):
        """
        Test error handling for invalid trajectory indices.
        Validates that non-existent trajectory indices raise appropriate
        ValueError, KeyError, or IndexError.
        """
        self.pipeline.data_selector.create("test")
        
        # Non-existent trajectory index
        with pytest.raises((ValueError, KeyError, IndexError)):
            self.pipeline.data_selector.select_by_indices("test", {999: [0, 1, 2]})
    
    def test_nonexistent_selector(self):
        """
        Test accessing non-existent data selector.
        Validates that accessing undefined selector raises KeyError
        for proper error handling in client code.
        """
        with pytest.raises(KeyError):
            _ = self.pipeline.data.data_selector_data["nonexistent"]


class TestDataSelectorIndicesAdvanced:
    """Test advanced select_by_indices functionality."""
    
    def setup_method(self):
        """
        Setup with multiple trajectories and tags.

        Creates two synthetic trajectories with tags for advanced selection testing.
        """
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
        """
        Test trajectory selection by name string instead of index.
        Validates that trajectory names are correctly resolved to indices
        and proper frame selection is applied to the correct trajectory.
        """
        self.pipeline.data_selector.create("test")
        
        # Select by trajectory name
        self.pipeline.data_selector.select_by_indices("test", {"synthetic": [10, 20, 30]})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert 0 in selector.trajectory_frames
        assert selector.trajectory_frames[0] == [10, 20, 30]
    
    def test_trajectory_selection_by_tag(self):
        """
        Test trajectory selection using tag prefix "tag:tagname".
        Validates that tag-based trajectory resolution works correctly
        and applies frame selection to trajectories matching the specified tag.
        """
        self.pipeline.data_selector.create("test")
        
        # Select by tag - should apply to trajectory 0 (has "folded" tag)
        self.pipeline.data_selector.select_by_indices("test", {"tag:folded": [5, 15, 25]})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert 0 in selector.trajectory_frames
        assert selector.trajectory_frames[0] == [5, 15, 25]
    
    def test_frame_selection_single_int(self):
        """
        Test frame selection with single integer value.
        Validates that a single integer is converted to a single-item list
        and properly stored in trajectory frame selection.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: 42})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == [42]
    
    def test_frame_selection_string_single(self):
        """
        Test frame selection with string representation of single frame.
        Validates that string-to-integer conversion works correctly
        and produces the same result as direct integer input.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "42"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == [42]
    
    def test_frame_selection_string_range(self):
        """
        Test frame selection using "start-end" range string format.
        Validates that hyphen-separated ranges are correctly parsed
        and expanded into inclusive frame index lists.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "10-15"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == [10, 11, 12, 13, 14, 15]
    
    def test_frame_selection_string_comma_list(self):
        """
        Test frame selection with comma-separated list of frame indices.
        Validates that comma-delimited strings are correctly parsed
        into individual frame indices with proper ordering.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "10,20,30"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert sorted(selector.trajectory_frames[0]) == [10, 20, 30]
    
    def test_frame_selection_string_combined(self):
        """
        Test frame selection combining ranges and single values.
        Validates that mixed "10-12,20,30-32" format is correctly parsed
        and produces union of all specified ranges and individual frames.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "10-12,20,30-32"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        expected = [10, 11, 12, 20, 30, 31, 32]
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_frame_selection_string_all(self):
        """
        Test frame selection using "all" keyword for complete trajectory.
        Validates that "all" string expands to full frame range
        and selects every frame from the specified trajectory.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: "all"})
        
        selector = self.pipeline.data.data_selector_data["test"]
        assert selector.trajectory_frames[0] == list(range(100))  # All 100 frames
    
    def test_frame_selection_dict_with_stride(self):
        """
        Test frame selection using dictionary with frames and stride keys.
        Validates that stride parameter correctly samples frames at regular intervals
        from the specified frame range with proper step size.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: {"frames": "0-50", "stride": 10}})
        
        selector = self.pipeline.data.data_selector_data["test"]
        expected = [0, 10, 20, 30, 40, 50]
        assert selector.trajectory_frames[0] == expected
    
    def test_stride_minimum_distance(self):
        """
        Test stride parameter as minimum distance filter on sparse selections.
        Validates that stride enforces minimum frame separation on irregular
        frame lists by filtering out frames that are too close together.
        """
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
        """
        Test stride parameter applied to frame pattern simulating union results.
        Validates that stride filtering works correctly on densely packed
        frame sequences by maintaining minimum distance requirements.
        """
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
        """
        Test stride parameter on frame pattern simulating intersection results.
        Validates that stride effectively reduces frame density in closely spaced
        sequences while preserving temporal distribution characteristics.
        """
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
        """
        Setup with basic trajectory.
        
        Creates a simple synthetic trajectory with 5 residues and 100 frames.
        """
        test_instance = TestDataSelectorBasics()
        test_instance.setup_method()
        self.pipeline = test_instance.pipeline
    
    def test_mode_add_default(self):
        """
        Test default selection mode which should be "add" (union operation).
        Validates that multiple selections without explicit mode parameter
        combine frames using union logic to expand the selection.
        """
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
        """
        Test explicitly specified "add" mode for union operations.
        Validates that mode="add" produces identical results to default behavior
        and correctly combines frame selections from multiple operations.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_indices("test", {0: [10, 20, 30]})
        self.pipeline.data_selector.select_by_indices("test", {0: [25, 35, 45]}, mode="add")
        
        selector = self.pipeline.data.data_selector_data["test"]
        expected = [10, 20, 25, 30, 35, 45]
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_mode_subtract(self):
        """
        Test "subtract" mode for difference operations on frame selections.
        Validates that mode="subtract" removes specified frames from existing
        selection and produces correct set difference results.
        """
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
        """
        Test "intersect" mode for intersection operations on frame selections.
        Validates that mode="intersect" keeps only frames present in both
        existing selection and new selection criteria.
        """
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
        """
        Test sequential application of multiple selection modes.
        Validates that complex sequences of add, subtract, and intersect operations
        produce correct final frame selections through proper mode composition.
        """
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
        """
        Setup with multiple trajectories with different tags.
        
        Creates two synthetic trajectories with distinct tags for tag-based selection testing.
        """
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
        """
        Test trajectory selection using single tag criterion.
        Validates that trajectories with matching tag are selected completely
        while trajectories without the tag are excluded from selection.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_tags("test", ["folded"])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Trajectory 0 has "folded" tag, so all 100 frames from trajectory 0
        expected = list(range(0, 100))
        assert sorted(selector.trajectory_frames[0]) == expected
        # Trajectory 1 doesn't have "folded" tag, so no frames
        assert 1 not in selector.trajectory_frames
    
    def test_select_by_multiple_tags_match_all_true(self):
        """
        Test multi-tag selection with match_all=True requiring AND logic.
        Validates that only trajectories containing all specified tags
        are selected when match_all=True is used for strict filtering.
        """
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
        """
        Test multi-tag selection with match_all=False using OR logic.
        Validates that trajectories containing any of the specified tags
        are selected when match_all=False allows inclusive filtering.
        """
        self.pipeline.data_selector.create("test")
        
        # Select trajectories that have "folded" OR "unfolded" tags
        self.pipeline.data_selector.select_by_tags("test", ["folded", "unfolded"], match_all=False)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Both trajectories have either "folded" OR "unfolded" tags
        # Trajectory 0: 100 frames, Trajectory 1: 80 frames
        assert sorted(selector.trajectory_frames[0]) == list(range(0, 100))
        assert sorted(selector.trajectory_frames[1]) == list(range(0, 80))
    
    def test_select_by_tags_with_stride(self):
        """
        Test tag-based selection combined with stride parameter.
        Validates that stride is correctly applied to trajectories selected by tags
        producing regular frame sampling from matched trajectories only.
        """
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
        """
        Test tag-based selection using different operation modes.
        Validates that add, subtract modes work correctly when applied
        to tag-based trajectory selection with proper frame handling.
        """
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
        """
        Test selection behavior when using non-existent tag.
        Validates that selecting by non-existent tag results in empty selection
        without raising errors, providing graceful handling of missing tags.
        """
        self.pipeline.data_selector.create("test")
        
        # This should result in empty selection
        self.pipeline.data_selector.select_by_tags("test", ["nonexistent_tag"])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # No trajectories have this tag, so no frames should be selected
        assert len(selector.trajectory_frames) == 0
    
    def test_contradictory_tags_stable_unstable(self):
        """
        Test match_all=True with contradictory stable AND unstable tags.
        Validates that trajectories can have contradictory tags simultaneously
        and match_all=True correctly finds trajectories with both tags.
        """
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
        """
        Test match_all=True with contradictory folded AND unfolded tags.
        Validates that when no trajectories have both contradictory tags,
        the selection results in empty frame sets as expected.
        """
        # Add contradictory tags to trajectory 1
        self.pipeline.data_selector.create("test")
        self.pipeline.data_selector.select_by_tags("test", ["folded", "unfolded"], match_all=True)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # No trajectory has BOTH "folded" AND "unfolded" tags
        assert selector.trajectory_frames == {}
    
    def test_impossible_tag_combination(self):
        """
        Test match_all=True with impossible 4-tag combination requirement.
        Validates that when no trajectory can satisfy all contradictory tag requirements,
        the selection correctly results in empty frame selection.
        """
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
        """
        Setup with trajectory and mock clustering results.
        
        Creates a simple synthetic trajectory with 5 residues and 100 frames.
        Create with mock clustering data for testing cluster-based selection.
        3 clusters are simulated with different frame assignments.
        1. Cluster 0: frames 0-30
        2. Cluster 1: frames 31-70
        3. Cluster 2: frames 71-99.
        """
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
        """
        Test frame selection based on single cluster ID.
        Validates that all frames assigned to specified cluster are selected
        and cluster-to-frame mapping works correctly for data selection.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [0])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Cluster 0 contains frames 0-30
        expected = list(range(0, 31))
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_select_by_multiple_clusters(self):
        """
        Test frame selection using multiple cluster IDs simultaneously.
        Validates that frames from all specified clusters are combined
        into unified selection with proper union of cluster memberships.
        """
        self.pipeline.data_selector.create("test")
        
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [0, 2])
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Cluster 0 (frames 0-30) + Cluster 2 (frames 71-99)
        expected = list(range(0, 31)) + list(range(71, 100))
        assert sorted(selector.trajectory_frames[0]) == sorted(expected)
    
    def test_select_by_cluster_with_stride(self):
        """
        Test cluster-based selection combined with stride parameter.
        Validates that stride is applied to frames within selected clusters
        producing regular sampling from cluster-defined frame subsets.
        """
        self.pipeline.data_selector.create("test")
        
        # Select cluster 1 with stride=5
        self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [1], stride=5)
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Cluster 1 frames 31-70, with stride=5: 31, 36, 41, 46, 51, 56, 61, 66
        cluster_1_frames = list(range(31, 71))
        expected = cluster_1_frames[::5]  # Every 5th frame starting from first
        assert sorted(selector.trajectory_frames[0]) == expected
    
    def test_select_by_cluster_with_modes(self):
        """
        Test cluster-based selection using different operation modes.
        Validates that add, subtract modes work correctly with cluster selections
        and produce expected frame set operations on cluster memberships.
        """
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
        """
        Test error handling when selecting non-existent cluster ID.
        Validates that attempting to select undefined cluster raises appropriate
        ValueError with informative message about missing cluster.
        """
        self.pipeline.data_selector.create("test")
        
        # Select cluster ID 999 which doesn't exist - should raise ValueError
        with pytest.raises(ValueError, match="Cluster ID 999 not found"):
            self.pipeline.data_selector.select_by_cluster("test", "test_clustering", [999])
    
    def test_select_by_nonexistent_clustering(self):
        """
        Test error handling when selecting from non-existent clustering.
        Validates that attempting to use undefined clustering name raises
        appropriate ValueError or KeyError with clear error indication.
        """
        self.pipeline.data_selector.create("test")
        
        # Try to select from clustering that doesn't exist
        with pytest.raises((ValueError, KeyError)):
            self.pipeline.data_selector.select_by_cluster("test", "nonexistent_clustering", [0])


class TestDataSelectorMixedModes:
    """Test mixed mode operations with simplified setup."""
    
    def setup_method(self):
        """
        Setup with 2 trajectories, basic tags and simple clustering.

        Creates two synthetic trajectories with tags and mock clustering for mixed mode testing.
        """
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
        """
        Test combination of cluster-based and tag-based selections.
        Validates that cluster and tag selection criteria can be combined
        using different modes to create complex frame selection patterns.
        """
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
        """
        Test add and subtract modes across different selection types.
        Validates that index-based and cluster-based selections can be
        combined using subtract mode to create refined frame selections.
        """
        self.pipeline.data_selector.create("test")
        
        # Start with all frames from trajectory 0
        self.pipeline.data_selector.select_by_indices("test", {0: "0-99"})
        # Subtract cluster 1 frames
        self.pipeline.data_selector.select_by_cluster("test", "simple_clustering", [1], mode="subtract")
        
        selector = self.pipeline.data.data_selector_data["test"]
        # Should have only cluster 0 frames from trajectory 0 (frames 0-49)
        assert sorted(selector.trajectory_frames[0]) == list(range(50))
    
    def test_mixed_modes_intersect(self):
        """
        Test intersect mode combining tag-based and index-based selections.
        Validates that intersection of tag selection with specific frame ranges
        produces correct overlap of criteria from different selection types.
        """
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
        """
        Test complex 3-operation combination using different selection types.
        Validates that cluster, tag, and index selections can be sequentially
        combined using different modes to produce sophisticated frame filtering.
        """
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
        