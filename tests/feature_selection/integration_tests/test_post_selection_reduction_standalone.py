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
Standalone mode tests for post-selection reduction.

Tests post-selection reduction using direct FeatureSelectorManager calls
without PipelineManager wrapper. Uses deterministic synthetic data for
exact assertions and validation.
"""
import pytest
import numpy as np
import mdtraj as md
from typing import List

from mdxplain.pipeline.entities.pipeline_data import PipelineData
from mdxplain.feature_selection.managers.feature_selector_manager import FeatureSelectorManager
from mdxplain.feature.entities.feature_data import FeatureData
from mdxplain.feature.feature_type.distances.distances import Distances


class TestPostSelectionReductionStandalone:
    """Test post-selection reduction in standalone mode."""

    def setup_method(self):
        """
        Setup standalone mode with deterministic synthetic data.

        Creates PipelineData and FeatureSelectorManager directly without
        PipelineManager wrapper. Uses same fixed distance data as main tests.
        """
        self._create_fixed_distance_data()
        self._setup_standalone_pipeline()

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

    def _setup_standalone_pipeline(self):
        """Setup standalone pipeline without PipelineManager."""
        self.pipeline_data = PipelineData()
        self.manager = FeatureSelectorManager()

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
        self.pipeline_data.trajectory_data.n_trajectories = 1
        self.pipeline_data.trajectory_data.trajectories = {0: trajectory}
        self.pipeline_data.trajectory_data.res_label_data = {0: [
            {"index": i, "full_name": f"{name}{i}"}
            for i, name in enumerate(["ALA", "GLY", "VAL", "LEU", "PHE"])
        ]}

        # Create FeatureData and set our fixed distances directly
        feature_data = FeatureData(Distances(), use_memmap=False)
        feature_data.data = self.distances
        feature_data.feature_metadata = self.metadata

        # Set the feature data directly in pipeline
        self.pipeline_data.feature_data["distances"] = {0: feature_data}

    def _get_selected_indices(self, selector_name: str) -> List[int]:
        """Get selected indices from standalone selector."""
        self.manager.select(self.pipeline_data, selector_name, reference_traj=0)
        results = self.pipeline_data.selected_feature_data[selector_name]
        return results.get_results("distances")["trajectory_indices"][0]["indices"]

    def _assert_exact_indices(self, actual: List[int], expected: List[int], message: str):
        """Detailed assertion shows which indices are removed."""
        all_indices = list(range(10))  # All possible distance pair indices
        removed = sorted(set(all_indices) - set(actual))

        assert sorted(actual) == sorted(expected), f"""
        {message}
        Expected: {sorted(expected)}
        Actual: {sorted(actual)}
        Missing: {sorted(set(expected) - set(actual))}
        Extra: {sorted(set(actual) - set(expected))}
        Removed by reduction: {removed}
        """

    def test_standalone_mode_basic_functionality(self):
        """
        Test basic standalone mode without reduction.

        Verifies all 10 distance pairs selected without reduction.
        Expected indices are [0-9].
        """
        # Create selector without pipeline wrapper
        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all"
        )

        # Execute selection
        indices = self._get_selected_indices("test")

        # Should select all 10 distance pairs
        expected_indices = list(range(10))
        self._assert_exact_indices(indices, expected_indices,
            "Basic standalone mode should select all distance pairs")

        # Validate selected data
        selected_data = self.pipeline_data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_standalone_mode_with_max_reduction(self):
        """
        Test standalone mode with max reduction.

        Validates max > 10.0 keeps only pair 9.
        Expected indices are [9].
        """
        # Setup with explicit reduction config
        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            reduction={
                "metric": "max",
                "threshold_min": 10.0
            }
        )

        # Execute selection with reduction
        indices = self._get_selected_indices("test")

        # Pairs with max > 10.0: [9] (max=10.4)
        expected_indices = [9]
        self._assert_exact_indices(indices, expected_indices,
            "Standalone max reduction should keep only pairs with max > 10.0")

        # Validate selected data
        selected_data = self.pipeline_data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_standalone_mode_with_mean_reduction(self):
        """
        Test standalone mode with mean reduction.

        Verifies pairs with mean >= 7.0 are selected.
        Expected indices are [6, 7, 8, 9].
        """
        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            reduction={
                "metric": "mean",
                "threshold_min": 7.0,
                "cross_trajectory": True
            }
        )

        indices = self._get_selected_indices("test")

        # Pairs with mean >= 7.0: [6, 7, 8, 9] (mean: 7.2, 8.2, 9.2, 10.2)
        expected_indices = [6, 7, 8, 9]
        self._assert_exact_indices(indices, expected_indices,
            "Standalone mean reduction should keep only pairs with mean >= 7.0")

        # Validate selected data
        selected_data = self.pipeline_data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_standalone_mode_with_min_reduction(self):
        """
        Test standalone mode with min reduction.

        Validates pairs with min <= 5.0 are kept.
        Expected indices are [0, 1, 2, 3, 4].
        """
        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            reduction={
                "metric": "min",
                "threshold_max": 5.0
            }
        )

        indices = self._get_selected_indices("test")

        # Pairs with min <= 5.0: [0, 1, 2, 3, 4] (min: 1.0, 2.0, 3.0, 4.0, 5.0)
        expected_indices = [0, 1, 2, 3, 4]
        self._assert_exact_indices(indices, expected_indices,
            "Standalone min reduction should keep only pairs with min <= 5.0")

        # Validate selected data
        selected_data = self.pipeline_data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_standalone_mode_multiple_selections(self):
        """
        Test multiple selections with different reductions.

        Verifies union of max > 10.0 and min <= 3.0.
        Expected indices are [0, 1, 2, 9].
        """
        self.manager.create(self.pipeline_data, "test")

        # Selection 1: Only large distances (max > 10.0)
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            reduction={"metric": "max", "threshold_min": 10.0}
        )

        # Selection 2: Only small distances (min < 4.0)
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            reduction={"metric": "min", "threshold_max": 3.0}
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
        selected_data = self.pipeline_data.get_selected_data("test")
        expected_data = self.distances[:, expected_indices]
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    # Multi-Trajectory Tests for Standalone Mode

    def _add_multiple_trajectories_with_different_stats(self):
        """Add 3 trajectories with different statistical properties."""
        self._create_multi_trajectory_data()
        self._setup_multi_trajectory_pipeline_standalone()

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

    def _setup_multi_trajectory_pipeline_standalone(self):
        """Setup standalone pipeline with 3 trajectories having different distance data."""
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
        self.pipeline_data.trajectory_data.n_trajectories = 3
        self.pipeline_data.trajectory_data.trajectories = {0: traj0, 1: traj1, 2: traj2}
        self.pipeline_data.trajectory_data.res_label_data = {
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
        self.pipeline_data.feature_data["distances"] = {
            0: feature_data_0,
            1: feature_data_1,
            2: feature_data_2
        }

    def test_standalone_multi_trajectory_cross_trajectory_true(self):
        """
        Test standalone multi-trajectory with cross_trajectory=True.

        Verifies common denominator across trajectories.
        With max > 8.5 threshold:
        - Traj 0: features [8, 9] pass (max 9.4, 10.4)
        - Traj 1: features [7, 8, 9] pass (max 8.9, 9.9, 10.9)
        - Traj 2: features [6, 7, 8, 9] pass (max 8.9, 9.9, 10.9, 11.9)
        - Intersection: [8, 9]
        """
        self._add_multiple_trajectories_with_different_stats()

        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            reduction={"metric": "max", "threshold_min": 8.5, "cross_trajectory": True}
        )

        indices = self._get_selected_indices("test")
        expected_indices = [8, 9]  # Common features across all 3 trajectories
        self._assert_exact_indices(indices, expected_indices,
            "Standalone cross-trajectory=True should keep only common features across all trajectories")

        # Validate selected data (all trajectories concatenated)
        selected_data = self.pipeline_data.get_selected_data("test")
        expected_data = np.vstack([
            self.distances_traj0[:, expected_indices],
            self.distances_traj1[:, expected_indices],
            self.distances_traj2[:, expected_indices]
        ])
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values from all trajectories")

        # Validate selected metadata
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_standalone_multi_trajectory_cross_trajectory_false(self):
        """
        Test standalone multi-trajectory with cross_trajectory=False.

        Validates exact indices per trajectory before ValueError.
        With max > 8.5 threshold:
        - Traj 0: keeps [8, 9] (max 9.4, 10.4) - 2 features
        - Traj 1: keeps [7, 8, 9] (max 8.9, 9.9, 10.9) - 3 features
        - Traj 2: keeps [6, 7, 8, 9] (max 8.9, 9.9, 10.9, 11.9) - 4 features
        Expected: ValueError about inconsistent column counts
        """
        self._add_multiple_trajectories_with_different_stats()

        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            reduction={"metric": "max", "threshold_min": 8.5, "cross_trajectory": False}
        )

        # Execute select() and catch the expected ValueError
        try:
            self.manager.select(self.pipeline_data, "test", reference_traj=0)
            assert False, "Should have raised ValueError for inconsistent column counts"
        except ValueError as e:
            # Verify the error is about inconsistent column counts
            assert "inconsistent column counts" in str(e)

        # The trajectory indices are still set in FeatureSelectorData despite the error
        selector_data = self.pipeline_data.selected_feature_data["test"]
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

    def test_standalone_partial_trajectory_selection_cross_trajectory(self):
        """
        Test standalone reduction on subset of trajectories.

        Validates reduction applies only to selected trajectories.
        With traj_selection=[0, 1] and max > 8.5:
        - Only Traj 0 and Traj 1 participate in reduction
        - Traj 0: features [8, 9] pass (max 9.4, 10.4)
        - Traj 1: features [7, 8, 9] pass (max 8.9, 9.9, 10.9)
        - Intersection: [8, 9]
        - Traj 2 not affected by reduction
        """
        self._add_multiple_trajectories_with_different_stats()

        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all",
            traj_selection=[0, 1],  # Only trajectories 0 and 1
            reduction={"metric": "max", "threshold_min": 8.5, "cross_trajectory": True}
        )

        indices = self._get_selected_indices("test")
        expected_indices = [8, 9]  # Common between traj 0 and 1
        self._assert_exact_indices(indices, expected_indices,
            "Standalone partial trajectory selection should find intersection of selected trajectories only")

        # Validate selected data (only selected trajectories)
        selected_data = self.pipeline_data.get_selected_data("test")
        expected_data = np.vstack([
            self.distances_traj0[:, expected_indices],
            self.distances_traj1[:, expected_indices]
        ])  # Only trajectories 0 and 1 were selected
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values")

        # Validate selected metadata
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_standalone_mixed_reductions_different_cross_trajectory(self):
        """
        Test standalone multiple selections with different cross_trajectory settings.

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

        self.manager.create(self.pipeline_data, "test")

        # Selection 1: Residue-based selection (no reduction)
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "resid 0", traj_selection=[0]
        )

        # Selection 2: Min reduction on trajectory 1
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all", traj_selection=[1],
            reduction={"metric": "min", "threshold_max": 4.5, "cross_trajectory": False}
        )

        # Selection 3: Max reduction on trajectory 2
        self.manager.add_selection(
            self.pipeline_data, "test", "distances", "all", traj_selection=[2],
            reduction={"metric": "max", "threshold_min": 8.9, "cross_trajectory": False}
        )

        # Execute selection and verify trajectory-specific indices
        self.manager.select(self.pipeline_data, "test", reference_traj=0)

        # Check the trajectory-specific indices in the results
        results = self.pipeline_data.selected_feature_data["test"]
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
        selected_data = self.pipeline_data.get_selected_data("test")
        # Traj 0 uses indices [0,1,2,3], Traj 1 uses [0,1,2,3], Traj 2 uses [6,7,8,9]
        expected_data = np.vstack([
            self.distances_traj0[:, [0, 1, 2, 3]],  # Traj 0: resid 0
            self.distances_traj1[:, [0, 1, 2, 3]],  # Traj 1: min reduction
            self.distances_traj2[:, [6, 7, 8, 9]]   # Traj 2: max reduction
        ])
        np.testing.assert_array_almost_equal(selected_data, expected_data,
            err_msg="Selected data should match expected distance values from each trajectory")

        # Validate selected metadata (uses features from expected_indices)
        selected_metadata = self.pipeline_data.get_selected_metadata("test")
        expected_metadata = [self.metadata["features"][i] for i in expected_indices]
        selected_features = [item["features"] for item in selected_metadata]
        assert selected_features == expected_metadata, \
            f"Metadata mismatch: Expected metadata for indices {expected_indices}"

    def test_standalone_mode_error_handling(self):
        """
        Test error handling in standalone mode.

        Validates proper exceptions for non-existent selectors.
        Ensures ValueError raised for invalid operations.
        """
        # Test adding selection to non-existent selector
        with pytest.raises(ValueError, match="Feature selector 'nonexistent' does not exist"):
            self.manager.add_selection(
                self.pipeline_data, "nonexistent", "distances", "all"
            )

        # Test selecting from non-existent selector
        with pytest.raises(ValueError, match="Feature selector 'nonexistent' does not exist"):
            self.manager.select(self.pipeline_data, "nonexistent")

        # Test with invalid feature type
        self.manager.create(self.pipeline_data, "test")
        self.manager.add_selection(
            self.pipeline_data, "test", "invalid_feature", "all"
        )
        with pytest.raises(ValueError, match="Feature 'invalid_feature' not found in pipeline data"):
            self.manager.select(self.pipeline_data, "test")
