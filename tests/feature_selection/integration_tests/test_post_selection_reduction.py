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

"""
Integration tests for post-selection reduction feature.

Tests the complete workflow of feature selection with statistical reduction
using deterministic synthetic data with max/min/mean metrics.
"""
import pytest
import numpy as np
import mdtraj as md
from typing import List

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.feature.feature_type.distances.distances import Distances
from mdxplain.feature.entities.feature_data import FeatureData

class TestPostSelectionReduction:
    """Test post-selection reduction with deterministic synthetic data."""

    def setup_method(self):
        """
        Setup with fixed, controlled distance values.

        5 residues: ALA, GLY, VAL, LEU, PHE
        5 frames: 0, 1, 2, 3, 4
        10 distance pairs with exact statistical properties:

        Pair 0: min=1.0, max=1.4, mean=1.2
        Pair 1: min=2.0, max=2.4, mean=2.2
        Pair 2: min=3.0, max=3.4, mean=3.2
        Pair 3: min=4.0, max=4.4, mean=4.2
        Pair 4: min=5.0, max=5.4, mean=5.2
        Pair 5: min=6.0, max=6.4, mean=6.2
        Pair 6: min=7.0, max=7.4, mean=7.2
        Pair 7: min=8.0, max=8.4, mean=8.2
        Pair 8: min=9.0, max=9.4, mean=9.2
        Pair 9: min=10.0, max=10.4, mean=10.2
        """
        self._create_fixed_distance_data()
        self._setup_pipeline()

    def _create_fixed_distance_data(self):
        """Create fixed distance matrix with known statistical properties."""
        # 5 frames x 10 pairs with controlled values
        # Each pair has min, max, mean as defined in docstring
        self.distances = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # Frame 0
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],  # Frame 1
            [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2],  # Frame 2
            [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3],  # Frame 3
            [1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4]   # Frame 4
        ])

        # Metadata for 10 distance pairs (0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-4, 2-3, 2-4, 3-4)
        residue_names = ["ALA", "GLY", "VAL", "LEU", "PHE"]
        pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]  # 10 pairs total

        features = []
        for pair_idx, (res1, res2) in enumerate(pairs):
            features.append([
                {
                    "residue": {
                        "aaa_code": residue_names[res1],
                        "index": res1,
                        "seqid": res1,
                        "full_name": f"{residue_names[res1]}{res1}"
                    },
                    "full_name": f"{residue_names[res1]}{res1}"
                },
                {
                    "residue": {
                        "aaa_code": residue_names[res2],
                        "index": res2,
                        "seqid": res2,
                        "full_name": f"{residue_names[res2]}{res2}"
                    },
                    "full_name": f"{residue_names[res2]}{res2}"
                }
            ])

        self.metadata = {"features": features}

    def _setup_pipeline(self):
        """Setup pipeline with fixed distance data."""
        self.pipeline = PipelineManager(use_memmap=False)

        # Create minimal trajectory for n_frames access
        topology = md.Topology()
        chain = topology.add_chain()
        for name in ["ALA", "GLY", "VAL", "LEU", "PHE"]:
            residue = topology.add_residue(name, chain)
            topology.add_atom('CA', element=md.element.carbon, residue=residue)

        # Create minimal coordinates (5 frames, 5 atoms, 3 coordinates)
        xyz = np.zeros((5, 5, 3))
        trajectory = md.Trajectory(xyz, topology)

        # Set up trajectory data
        self.pipeline.data.trajectory_data.n_trajectories = 1
        self.pipeline.data.trajectory_data.trajectories = {0: trajectory}
        self.pipeline.data.trajectory_data.res_label_data = {0: [
            {"index": i, "full_name": f"{name}{i}"}
            for i, name in enumerate(["ALA", "GLY", "VAL", "LEU", "PHE"])
        ]}

        # Create FeatureData and set our fixed distances directly
        feature_data = FeatureData(Distances(), use_memmap=False)
        feature_data.data = self.distances
        feature_data.feature_metadata = self.metadata

        # Set the feature data directly in pipeline
        self.pipeline.data.feature_data["distances"] = {0: feature_data}

    def _get_selected_indices(self, selector_name: str) -> List[int]:
        """Execute selection and get indices."""
        self.pipeline.feature_selector.select(selector_name, reference_traj=0)
        results = self.pipeline.data.selected_feature_data[selector_name]
        return results.get_results("distances")["trajectory_indices"][0]["indices"]

    def _assert_exact_indices(self, actual: List[int], expected: List[int], message: str):
        """Detailed assertion shows which indices are removed."""
        # Use the actual max number of pairs based on distances data
        distances_data = self.pipeline.data.feature_data["distances"][0]
        max_pairs = distances_data.data.shape[1]
        all_indices = list(range(max_pairs))
        removed = sorted(set(all_indices) - set(actual))

        assert sorted(actual) == sorted(expected), f"""
        {message}
        Expected: {sorted(expected)}
        Actual: {sorted(actual)}
        Missing: {sorted(set(expected) - set(actual))}
        Extra: {sorted(set(actual) - set(expected))}
        Removed by reduction: {removed}
        Total pairs available: {max_pairs}
        """

    # Tests 1-11: Single Trajectory Tests

    def test_single_selection_with_max_reduction(self):
        """
        Test single selection with max reduction. 
        
        Verifies that only distance pairs with max > 8.0 are kept. 
        Expected indices are [7, 8, 9].
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=8.0  # Keep pairs with max > 8.0
        )

        # Expected: Pairs with max > 8.0 are pairs 7, 8, 9 (max values: 8.4, 9.4, 10.4)
        indices = self._get_selected_indices("test")
        expected_indices = [7, 8, 9]
        self._assert_exact_indices(indices, expected_indices,
            "Max reduction should keep only pairs with max > 8.0")

        # Validate reduction happened
        assert len(indices) < 10, f"Expected reduction, got all {len(indices)} indices"

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_min_reduction_exact_validation(self):
        """
        Test min reduction with exact index validation.

        Ensures pairs with min <= 3.0 are selected.
        Expected indices are [0, 1, 2].
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_min_reduction(
            "test", "all", threshold_max=3.0  # Keep pairs with min <= 3.0
        )

        # Expected: Pairs with min <= 3.0 are pairs 0, 1, 2 (min values: 1.0, 2.0, 3.0)
        indices = self._get_selected_indices("test")
        expected_indices = [0, 1, 2]
        self._assert_exact_indices(indices, expected_indices,
            "Min reduction should keep only pairs with min <= 3.0")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_mean_reduction_exact_validation(self):
        """
        Test mean reduction with exact index validation.

        Verifies pairs with mean >= 7.0 are retained.
        Expected indices are [6, 7, 8, 9].
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_mean_reduction(
            "test", "all", threshold_min=7.0  # Keep pairs with mean >= 7.0
        )

        # Expected: Pairs with mean >= 7.0 are pairs 6, 7, 8, 9 (mean values: 7.2, 8.2, 9.2, 10.2)
        indices = self._get_selected_indices("test")
        expected_indices = [6, 7, 8, 9]
        self._assert_exact_indices(indices, expected_indices,
            "Mean reduction should keep only pairs with mean >= 7.0")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_threshold_range_reduction(self):
        """
        Test reduction with both min and max thresholds.

        Validates pairs within range 6.0 <= max <= 12.0.
        Expected indices are [5, 6, 7, 8, 9].
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=6.0, threshold_max=12.0
        )

        indices = self._get_selected_indices("test")
        # Pairs with 6.0 <= max <= 12.0: [5, 6, 7, 8, 9] (max values: 6.4, 7.4, 8.4, 9.4, 10.4)
        expected_indices = [5, 6, 7, 8, 9]
        self._assert_exact_indices(indices, expected_indices,
            "Range reduction should keep only pairs with 6.0 <= max <= 12.0")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_multiple_selections_with_reduction(self):
        """
        Test multiple selections with different reductions.

        Verifies union combines max > 10.0 and min <= 3.0 criteria.
        Expected union is [0, 1, 2, 9].
        """
        self.pipeline.feature_selector.create("test")

        # Selection 1: Only large distances (max > 10.0)
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=10.0
        )

        # Selection 2: Only small distances (min < 4.0)
        self.pipeline.feature_selector.add.distances.with_min_reduction(
            "test", "all", threshold_max=3.0
        )

        indices = self._get_selected_indices("test")

        # Expect union of:
        # - max > 10.0: [9] (max=10.4)
        # - min <= 3.0: [0, 1, 2] (min=1.0, 2.0, 3.0)
        # Union: [0, 1, 2, 9]
        expected_indices = [0, 1, 2, 9]
        removed_indices = [3, 4, 5, 6, 7, 8]  # These are removed by both reductions

        self._assert_exact_indices(indices, expected_indices,
            f"Multiple selections should combine both reductions. Removed: {removed_indices}")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_no_features_pass_threshold(self):
        """
        Test when no features pass the threshold.

        Validates empty result when threshold exceeds all values.
        Should return empty list with warning.
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=20.0  # No pairs have max >= 20.0
        )

        indices = self._get_selected_indices("test")
        expected_indices = []  # No pairs should pass
        self._assert_exact_indices(indices, expected_indices,
            "No features should pass unrealistic threshold")

        # Validate selected data (should be empty)
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = np.empty((5, 0))
        np.testing.assert_array_equal(selected_data, expected_data,
            err_msg="Selected data should be empty array")

        # Validate selected metadata (should be empty - raises ValueError)
        with pytest.raises(ValueError, match="No valid metadata found for selection 'test'"):
            self.pipeline.data.get_selected_metadata("test")

    def test_all_features_pass_threshold(self):
        """
        Test when all features pass the threshold.

        Verifies all 10 pairs selected with min >= 0.0.
        All indices [0-9] should be selected.
        """
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_min_reduction(
            "test", "all", threshold_min=0.0  # All pairs have min >= 0.0
        )

        indices = self._get_selected_indices("test")
        expected_indices = list(range(10))  # All pairs should pass
        self._assert_exact_indices(indices, expected_indices,
            "All features should pass minimal threshold")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_residue_selection_with_reduction(self):
        """Test reduction on residue-specific selection."""
        # This test would require implementing residue selection parsing
        # For now, we use "all" and validate the reduction logic
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_std_reduction(
            "test", "all", threshold_min=0.05  # Lower threshold for our linear data
        )

        indices = self._get_selected_indices("test")
        # All pairs have std ~0.1, so all should pass threshold_min=0.05
        expected_indices = list(range(10))  # All pairs
        self._assert_exact_indices(indices, expected_indices,
            "STD reduction with low threshold should keep all pairs")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_cross_trajectory_true_single_trajectory(self):
        """Test cross_trajectory=True with single trajectory (should work normally)."""
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=10.0, cross_trajectory=True
        )

        indices = self._get_selected_indices("test")
        expected_indices = [9]  # Max > 10.0: only index 9 (max=10.4)
        self._assert_exact_indices(indices, expected_indices,
            "cross_trajectory=True with single trajectory should work normally")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_cross_trajectory_false_single_trajectory(self):
        """Test cross_trajectory=False with single trajectory (should work normally)."""
        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=10.0, cross_trajectory=False
        )

        indices = self._get_selected_indices("test")
        expected_indices = [9]  # Max > 10.0: only index 9 (max=10.4)
        self._assert_exact_indices(indices, expected_indices,
            "cross_trajectory=False with single trajectory should work normally")

        # Validate selected data
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    # Multi-Trajectory Tests (Tests 12-15)

    def _add_multiple_trajectories_with_different_stats(self):
        """Add 3 trajectories with different statistical properties."""
        self._create_multi_trajectory_data()
        self._setup_multi_trajectory_pipeline()

    def _create_multi_trajectory_data(self):
        """Create 3 trajectories with different but predictable distances."""
        # Trajectory 0: Original (already exists in self.distances)
        self.distances_traj0 = self.distances

        # Trajectory 1: Shifted by +0.5 (1.5-10.9)
        self.distances_traj1 = self.distances + 0.5

        # Trajectory 2: Different pattern - some higher, some lower
        # Designed so different features pass thresholds
        self.distances_traj2 = np.array([
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.5, 9.5, 10.5, 11.5],  # Frame 0
            [0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 8.6, 9.6, 10.6, 11.6],  # Frame 1
            [0.7, 1.7, 2.7, 3.7, 4.7, 5.7, 8.7, 9.7, 10.7, 11.7],  # Frame 2
            [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 8.8, 9.8, 10.8, 11.8],  # Frame 3
            [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 8.9, 9.9, 10.9, 11.9]   # Frame 4
        ])

        # Statistical properties for debugging:
        # Traj 0: max values = [1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4]
        # Traj 1: max values = [1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9]
        # Traj 2: max values = [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 8.9, 9.9, 10.9, 11.9]

    def _setup_multi_trajectory_pipeline(self):
        """Setup pipeline with 3 trajectories having different distance data."""
        self.pipeline = PipelineManager(use_memmap=False)

        # Create topology
        topology = md.Topology()
        chain = topology.add_chain()
        for name in ["ALA", "GLY", "VAL", "LEU", "PHE"]:
            residue = topology.add_residue(name, chain)
            topology.add_atom('CA', element=md.element.carbon, residue=residue)

        # Create 3 trajectories
        xyz = np.zeros((5, 5, 3))
        traj0 = md.Trajectory(xyz, topology)
        traj1 = md.Trajectory(xyz, topology)
        traj2 = md.Trajectory(xyz, topology)

        # Setup trajectory data
        self.pipeline.data.trajectory_data.n_trajectories = 3
        self.pipeline.data.trajectory_data.trajectories = {0: traj0, 1: traj1, 2: traj2}
        self.pipeline.data.trajectory_data.res_label_data = {
            0: [{"index": i, "full_name": f"{name}{i}"} for i, name in enumerate(["ALA", "GLY", "VAL", "LEU", "PHE"])],
            1: [{"index": i, "full_name": f"{name}{i}"} for i, name in enumerate(["ALA", "GLY", "VAL", "LEU", "PHE"])],
            2: [{"index": i, "full_name": f"{name}{i}"} for i, name in enumerate(["ALA", "GLY", "VAL", "LEU", "PHE"])]
        }

        # Create FeatureData for each trajectory with different distances
        feature_data_0 = FeatureData(Distances(), use_memmap=False)
        feature_data_0.data = self.distances_traj0
        feature_data_0.feature_metadata = self.metadata

        feature_data_1 = FeatureData(Distances(), use_memmap=False)
        feature_data_1.data = self.distances_traj1
        feature_data_1.feature_metadata = self.metadata

        feature_data_2 = FeatureData(Distances(), use_memmap=False)
        feature_data_2.data = self.distances_traj2
        feature_data_2.feature_metadata = self.metadata

        # Set feature data for all trajectories
        self.pipeline.data.feature_data["distances"] = {
            0: feature_data_0,
            1: feature_data_1,
            2: feature_data_2
        }

    def test_multi_trajectory_cross_trajectory_true(self):
        """
        Test multi-trajectory with cross_trajectory=True.

        Verifies common denominator across trajectories.
        With max > 8.5 threshold:
        - Traj 0: features [8, 9] pass (max 9.4, 10.4)
        - Traj 1: features [7, 8, 9] pass (max 8.9, 9.9, 10.9)
        - Traj 2: features [6, 7, 8, 9] pass (max 8.9, 9.9, 10.9, 11.9)
        - Intersection: [8, 9]
        """
        self._add_multiple_trajectories_with_different_stats()

        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=8.5, cross_trajectory=True
        )

        indices = self._get_selected_indices("test")
        expected_indices = [8, 9]  # Common features across all 3 trajectories
        self._assert_exact_indices(indices, expected_indices,
            "Cross-trajectory=True should keep only common features across all trajectories")

        # Validate selected data (all trajectories concatenated)
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = np.vstack([
            self.distances_traj0[:, expected_indices],
            self.distances_traj1[:, expected_indices],
            self.distances_traj2[:, expected_indices]
        ])
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values from all trajectories")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_multi_trajectory_cross_trajectory_false(self):
        """
        Test multi-trajectory with cross_trajectory=False.

        Validates exact indices per trajectory before ValueError.
        With max > 8.5 threshold:
        - Traj 0: keeps [8, 9] (max 9.4, 10.4) - 2 features
        - Traj 1: keeps [7, 8, 9] (max 8.9, 9.9, 10.9) - 3 features
        - Traj 2: keeps [6, 7, 8, 9] (max 8.9, 9.9, 10.9, 11.9) - 4 features
        Expected: ValueError about inconsistent column counts
        """
        self._add_multiple_trajectories_with_different_stats()

        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=8.5, cross_trajectory=False
        )

        # Execute select() and catch the expected ValueError
        try:
            self.pipeline.feature_selector.select("test", reference_traj=0)
            assert False, "Should have raised ValueError for inconsistent column counts"
        except ValueError as e:
            # Verify the error is about inconsistent column counts
            assert "inconsistent column counts" in str(e)

        # The trajectory indices are still set in FeatureSelectorData despite the error
        selector_data = self.pipeline.data.selected_feature_data["test"]
        all_results = selector_data.get_all_results()
        distance_results = all_results["distances"]["trajectory_indices"]

        # Verify exact indices per trajectory that caused the inconsistency
        assert 0 in distance_results, "Trajectory 0 should be in results"
        assert 1 in distance_results, "Trajectory 1 should be in results"
        assert 2 in distance_results, "Trajectory 2 should be in results"

        traj_0_indices = distance_results[0]["indices"]
        traj_1_indices = distance_results[1]["indices"]
        traj_2_indices = distance_results[2]["indices"]

        # Verify the exact indices that cause the inconsistency
        assert traj_0_indices == [8, 9], f"Traj 0 should have [8,9], got {traj_0_indices}"
        assert traj_1_indices == [7, 8, 9], f"Traj 1 should have [7,8,9], got {traj_1_indices}"
        assert traj_2_indices == [6, 7, 8, 9], f"Traj 2 should have [6,7,8,9], got {traj_2_indices}"

    def test_partial_trajectory_selection_cross_trajectory(self):
        """
        Test reduction on subset of trajectories.

        Validates reduction applies only to selected trajectories.
        With traj_selection=[0, 1] and max > 8.5:
        - Only Traj 0 and Traj 1 participate in reduction
        - Traj 0: features [8, 9] pass (max 9.4, 10.4)
        - Traj 1: features [7, 8, 9] pass (max 8.9, 9.9, 10.9)
        - Intersection: [8, 9]
        - Traj 2 not affected by reduction
        """
        self._add_multiple_trajectories_with_different_stats()

        self.pipeline.feature_selector.create("test")
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=8.5,
            traj_selection=[0, 1],  # Only trajectories 0 and 1
            cross_trajectory=True
        )

        indices = self._get_selected_indices("test")
        expected_indices = [8, 9]  # Common between traj 0 and 1
        self._assert_exact_indices(indices, expected_indices,
            "Partial trajectory selection should find intersection of selected trajectories only")

        # Validate selected data (only selected trajectories)
        selected_data = self.pipeline.data.get_selected_data("test")
        expected_data = np.vstack([
            self.distances_traj0[:, expected_indices],
            self.distances_traj1[:, expected_indices]
        ])  # Only trajectories 0 and 1 were selected
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_mixed_reductions_different_cross_trajectory(self):
        """
        Test multiple selections with different cross_trajectory settings.

        Uses different selection strategies but consistent feature counts.
        Selection 1: "resid 0" on traj_selection=[0] (no reduction)
        - Traj 0: indices [0, 1, 2, 3] (4 features - pairs involving residue 0)

        Selection 2: "all" with min <= 4.5, cross_trajectory=False on traj_selection=[1]
        - Traj 1: indices [0, 1, 2, 3] (4 features)

        Selection 3: "all" with max >= 8.9, cross_trajectory=False on traj_selection=[2]
        - Traj 2: indices [6, 7, 8, 9] (4 features)

        Result: All trajectories have exactly 4 features each
        """
        self._add_multiple_trajectories_with_different_stats()

        self.pipeline.feature_selector.create("test")

        # Selection 1: Residue-based selection (no reduction)
        self.pipeline.feature_selector.add_selection(
            "test", "distances", "resid 0", traj_selection=[0]
        )

        # Selection 2: Min reduction on trajectory 1
        self.pipeline.feature_selector.add.distances.with_min_reduction(
            "test", "all", threshold_max=4.5, cross_trajectory=False, traj_selection=[1]
        )

        # Selection 3: Max reduction on trajectory 2
        self.pipeline.feature_selector.add.distances.with_max_reduction(
            "test", "all", threshold_min=8.9, cross_trajectory=False, traj_selection=[2]
        )

        # Execute selection and verify trajectory-specific indices
        self.pipeline.feature_selector.select("test", reference_traj=0)

        # Check the trajectory-specific indices in the results
        results = self.pipeline.data.selected_feature_data["test"]
        trajectory_results = results.get_results("distances")["trajectory_indices"]

        # Verify each trajectory has exactly the expected indices
        assert 0 in trajectory_results, "Trajectory 0 should be in results"
        assert 1 in trajectory_results, "Trajectory 1 should be in results"
        assert 2 in trajectory_results, "Trajectory 2 should be in results"

        traj_0_indices = trajectory_results[0]["indices"]
        traj_1_indices = trajectory_results[1]["indices"]
        traj_2_indices = trajectory_results[2]["indices"]

        # Verify exact indices per trajectory
        assert traj_0_indices == [0, 1, 2, 3], f"Traj 0 should have [0,1,2,3], got {traj_0_indices}"
        assert traj_1_indices == [0, 1, 2, 3], f"Traj 1 should have [0,1,2,3], got {traj_1_indices}"
        assert traj_2_indices == [6, 7, 8, 9], f"Traj 2 should have [6,7,8,9], got {traj_2_indices}"

        # The overall selected indices (for reference trajectory)
        expected_indices = [0, 1, 2, 3]
        self._assert_exact_indices(traj_0_indices, expected_indices,
            "Reference trajectory should have expected indices")

        # Validate selected data (each trajectory uses its own feature indices)
        selected_data = self.pipeline.data.get_selected_data("test")
        # Traj 0 uses indices [0,1,2,3], Traj 1 uses [0,1,2,3], Traj 2 uses [6,7,8,9]
        expected_data = np.vstack([
            self.distances_traj0[:, [0, 1, 2, 3]],  # Traj 0: resid 0
            self.distances_traj1[:, [0, 1, 2, 3]],  # Traj 1: min reduction
            self.distances_traj2[:, [6, 7, 8, 9]]   # Traj 2: max reduction
        ])
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values from each trajectory")

        # Validate selected metadata (uses features from expected_indices)
        selected_metadata = self.pipeline.data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"
    