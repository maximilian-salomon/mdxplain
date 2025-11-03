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

"""Integration tests for cross-trajectory feature reduction with EXACT data comparisons."""

import numpy as np
import pytest
import mdtraj as md
from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from mdxplain.feature.entities.feature_data import FeatureData
from mdxplain.feature.feature_type.distances.distances import Distances
from mdxplain.feature.feature_type.contacts.contacts import Contacts


class MockFeatureData:
    """Mock FeatureData for testing with exact data arrays."""

    def __init__(self, data_array):
        """
        Initialize mock feature data.

        Parameters
        ----------
        data_array : np.ndarray
            Exact data array to use as feature data
        """
        self.data = data_array
        self.reduced_data = None
        self.feature_metadata = {
            "features": np.array([f"feature_{i}" for i in range(data_array.shape[1])])
        }
        self.reduced_feature_metadata = None
        self.reduction_info = None
        self.use_memmap = False
        self.chunk_size = 1000
        self.reduced_cache_path = "/tmp/test_reduced.dat"

        # Mock feature type with calculator
        self.feature_type = self._create_mock_feature_type()

    def _create_mock_feature_type(self):
        """Create mock feature type with calculator."""
        from unittest.mock import MagicMock
        mock_feature_type = MagicMock()
        mock_calculator = MagicMock()

        # Mock the compute_dynamic_values method
        def mock_compute_dynamic_values(input_data, metric, threshold_min=None, threshold_max=None, **kwargs):
            # Calculate actual metric values for the test data
            if metric == "cv":
                means = np.mean(input_data, axis=0)
                stds = np.std(input_data, axis=0)
                metric_values = stds / means
            elif metric == "frequency":
                # For binary data, frequency is just the mean
                metric_values = np.mean(input_data, axis=0)
            elif metric == "max":
                metric_values = np.max(input_data, axis=0)
            elif metric == "min":
                metric_values = np.min(input_data, axis=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            # Apply thresholds
            mask = np.ones(len(metric_values), dtype=bool)
            if threshold_min is not None:
                mask &= (metric_values >= threshold_min)
            if threshold_max is not None:
                mask &= (metric_values <= threshold_max)

            # Extract data using mask
            indices = np.where(mask)[0]
            dynamic_data = input_data[:, indices]
            feature_names = [f"feature_{i}" for i in indices]

            return {
                "indices": (indices,),
                "values": metric_values[mask],
                "dynamic_data": dynamic_data,
                "feature_names": np.array(feature_names),
                "n_dynamic": len(indices),
                "total_pairs": len(metric_values)
            }

        mock_calculator.compute_dynamic_values = mock_compute_dynamic_values
        mock_feature_type.calculator = mock_calculator

        return mock_feature_type


class TestCrossTrajectoryReduction:
    """Test cross-trajectory reduction functionality with EXACT data comparisons."""

    def setup_method(self):
        """Create pipeline with mock trajectory setup."""
        self.pipeline = PipelineManager()

        # Create minimal topology for testing
        topology = md.Topology()
        chain = topology.add_chain()

        # Add 5 residues with CA atoms (smaller for cleaner tests)
        for i in range(5):
            residue = topology.add_residue(f"RES{i}", chain)
            topology.add_atom("CA", md.element.carbon, residue)

        # Create dummy trajectory
        coords = np.zeros((5, 5, 3))  # 5 frames, 5 atoms
        traj = md.Trajectory(coords, topology)

        # Set up pipeline data structure
        self.pipeline.data.trajectory_data.trajectories = [traj, traj]  # Two trajectories
        self.pipeline.data.trajectory_data.trajectory_names = ["traj1", "traj2"]

        # Create residue metadata for both trajectories
        for traj_idx in [0, 1]:
            self.pipeline.data.trajectory_data.res_label_data[traj_idx] = [
                {"seqid": i, "full_name": f"RES_{i}"} for i in range(5)
            ]

    def _create_test_data_with_known_metrics(self):
        """
        Create test data with documented max/min metrics for predictable results.

        Returns:
        --------
        tuple: (traj1_data, traj2_data) with known max values

        Max values:
        - Feature 0: max=10.1 (traj1), max=5.1 (traj2) → threshold > 7.0: Only traj1
        - Feature 1: max=8.0 (traj1), max=9.0 (traj2) → threshold > 7.0: Both pass
        - Feature 2: max=8.0 (traj1), max=2.1 (traj2) → threshold > 7.0: Only traj1
        - Feature 3: max=3.1 (traj1), max=14.0 (traj2) → threshold > 7.0: Only traj2
        - Feature 4: max=5.0 (traj1), max=4.0 (traj2) → threshold > 7.0: Neither pass
        """
        # Trajectory 1 data: columns [0,1,2] have max > 7.0
        traj1_data = np.array([
            [10.0, 2.0, 5.0, 3.0, 1.0],   # frame 0
            [10.1, 4.0, 8.0, 3.0, 5.0],   # frame 1
            [9.9, 6.0, 5.0, 3.1, 1.0],    # frame 2
            [10.0, 8.0, 8.0, 2.9, 5.0],   # frame 3
            [10.1, 2.0, 5.0, 3.0, 1.0],   # frame 4
        ])

        # Trajectory 2 data: columns [1,3] have max > 7.0
        traj2_data = np.array([
            [5.0, 3.0, 2.0, 7.0, 2.0],    # frame 0
            [5.1, 5.0, 2.0, 14.0, 4.0],   # frame 1
            [4.9, 7.0, 2.1, 7.0, 2.0],    # frame 2
            [5.0, 9.0, 1.9, 14.0, 4.0],   # frame 3
            [5.1, 3.0, 2.0, 7.0, 2.0],    # frame 4
        ])

        return traj1_data, traj2_data

    def test_independent_reduction_exact(self):
        """
        Test independent (non-cross-trajectory) reduction with EXACT expected arrays.

        Each trajectory is reduced independently, keeping different features.
        """
        traj1_data, traj2_data = self._create_test_data_with_known_metrics()

        # Mock the distance data in pipeline
        self.pipeline.data.feature_data = {
            "distances": {
                0: MockFeatureData(traj1_data),
                1: MockFeatureData(traj2_data)
            }
        }

        # Apply independent max reduction with threshold_min=7.0
        # Traj1: features [0,1,2] pass (max > 7.0)
        # Traj2: features [1,3] pass (max > 7.0)
        self.pipeline.feature.reduce.distances.max(
            threshold_min=7.0,
            cross_trajectory=False
        )

        # EXPECTED: Each trajectory keeps different columns based on max values
        expected_traj1_reduced = traj1_data[:, [0, 1, 2]]  # Columns 0,1,2 (max > 7.0)
        expected_traj2_reduced = traj2_data[:, [1, 3]]  # Columns 1,3 (max > 7.0)

        # Get actual reduced data
        actual_traj1_reduced = self.pipeline.data.feature_data["distances"][0].reduced_data
        actual_traj2_reduced = self.pipeline.data.feature_data["distances"][1].reduced_data

        # EXACT comparison - must match 100%
        np.testing.assert_array_almost_equal(
            actual_traj1_reduced,
            expected_traj1_reduced,
            decimal=10,
            err_msg="Trajectory 1 independent reduction does not match expected values exactly"
        )

        np.testing.assert_array_almost_equal(
            actual_traj2_reduced,
            expected_traj2_reduced,
            decimal=10,
            err_msg="Trajectory 2 independent reduction does not match expected values exactly"
        )

    def test_cross_trajectory_max_exact_intersection(self):
        """
        Test cross-trajectory max reduction with EXACT expected arrays.

        Only features that pass threshold in ALL trajectories are retained.
        """
        traj1_data, traj2_data = self._create_test_data_with_known_metrics()

        # Mock the distance data in pipeline
        self.pipeline.data.feature_data = {
            "distances": {
                0: MockFeatureData(traj1_data),
                1: MockFeatureData(traj2_data)
            }
        }

        # Apply cross-trajectory max reduction with threshold_min=7.0
        # Only feature [1] passes in BOTH trajectories (intersection)
        # Traj1: [0,1,2] pass, Traj2: [1,3] pass → Intersection: [1]
        self.pipeline.feature.reduce.distances.max(
            threshold_min=7.0,
            cross_trajectory=True
        )

        # EXPECTED: Only feature [1] remains (column 1 from original arrays)
        expected_traj1_reduced = traj1_data[:, [1]]  # Column 1 only
        expected_traj2_reduced = traj2_data[:, [1]]  # Column 1 only

        # Get actual reduced data
        actual_traj1_reduced = self.pipeline.data.feature_data["distances"][0].reduced_data
        actual_traj2_reduced = self.pipeline.data.feature_data["distances"][1].reduced_data

        # EXACT comparison - must match 100%
        np.testing.assert_array_almost_equal(
            actual_traj1_reduced,
            expected_traj1_reduced,
            decimal=10,
            err_msg="Trajectory 1 cross-trajectory reduced data does not match expected values exactly"
        )

        np.testing.assert_array_almost_equal(
            actual_traj2_reduced,
            expected_traj2_reduced,
            decimal=10,
            err_msg="Trajectory 2 cross-trajectory reduced data does not match expected values exactly"
        )

    def test_independent_vs_cross_trajectory_exact_difference(self):
        """
        Test exact difference between independent and cross-trajectory reduction.

        Uses data where trajectories have different low-min features to show the difference.
        """
        # Create data where trajectories have DIFFERENT low-min features
        # Traj1: Features [1,2] have min < 3.0
        # Traj2: Features [1,3] have min < 3.0
        # Cross-trajectory intersection: only feature 1

        traj1_data = np.array([
            [10.0, 1.0, 2.0, 5.0, 8.0],   # Features 1,2 have low min values
            [10.1, 2.5, 1.5, 5.1, 8.1],
            [9.9, 1.5, 2.5, 4.9, 7.9],
            [10.0, 2.0, 1.0, 5.0, 8.0],
            [10.1, 1.2, 2.2, 5.1, 8.1],
        ])

        traj2_data = np.array([
            [10.0, 1.5, 8.0, 2.0, 8.0],   # Features 1,3 have low min values
            [10.1, 2.5, 8.1, 1.0, 8.1],
            [9.9, 1.0, 7.9, 2.5, 7.9],
            [10.0, 2.0, 8.0, 1.5, 8.0],
            [10.1, 1.8, 8.1, 2.2, 8.1],
        ])

        # Mock data
        self.pipeline.data.feature_data = {
            "distances": {
                0: MockFeatureData(traj1_data),
                1: MockFeatureData(traj2_data)
            }
        }

        # Test independent reduction first
        self.pipeline.feature.reduce.distances.min(
            threshold_max=2.8,
            cross_trajectory=False
        )

        # EXPECTED for independent:
        # Traj1: features [1,2] (min < 2.8)
        # Traj2: features [1,3] (min < 2.8)
        expected_independent_traj1 = traj1_data[:, [1, 2]]
        expected_independent_traj2 = traj2_data[:, [1, 3]]

        # Get independent results
        independent_traj1 = self.pipeline.data.feature_data["distances"][0].reduced_data
        independent_traj2 = self.pipeline.data.feature_data["distances"][1].reduced_data

        # Verify independent results
        np.testing.assert_array_almost_equal(
            independent_traj1, expected_independent_traj1, decimal=10,
            err_msg="Independent reduction traj1 does not match expected"
        )
        np.testing.assert_array_almost_equal(
            independent_traj2, expected_independent_traj2, decimal=10,
            err_msg="Independent reduction traj2 does not match expected"
        )

        # Reset and test cross-trajectory
        self.pipeline.feature.reset_reduction("distances")

        # Re-mock data
        self.pipeline.data.feature_data = {
            "distances": {
                0: MockFeatureData(traj1_data),
                1: MockFeatureData(traj2_data)
            }
        }

        # Apply cross-trajectory reduction
        self.pipeline.feature.reduce.distances.min(
            threshold_max=2.8,
            cross_trajectory=True
        )

        # EXPECTED for cross-trajectory: only feature [1] (intersection)
        expected_cross_traj1 = traj1_data[:, [1]]
        expected_cross_traj2 = traj2_data[:, [1]]

        # Get cross-trajectory results
        cross_traj1 = self.pipeline.data.feature_data["distances"][0].reduced_data
        cross_traj2 = self.pipeline.data.feature_data["distances"][1].reduced_data

        # Verify cross-trajectory results
        np.testing.assert_array_almost_equal(
            cross_traj1, expected_cross_traj1, decimal=10,
            err_msg="Cross-trajectory reduction traj1 does not match expected"
        )
        np.testing.assert_array_almost_equal(
            cross_traj2, expected_cross_traj2, decimal=10,
            err_msg="Cross-trajectory reduction traj2 does not match expected"
        )

        # Verify the results are DIFFERENT
        assert independent_traj1.shape[1] == 2, "Independent should have 2 features for traj1"
        assert independent_traj2.shape[1] == 2, "Independent should have 2 features for traj2"
        assert cross_traj1.shape[1] == 1, "Cross-trajectory should have 1 feature for traj1"
        assert cross_traj2.shape[1] == 1, "Cross-trajectory should have 1 feature for traj2"

    def test_cross_trajectory_contacts_frequency_exact(self):
        """
        Test cross-trajectory contacts frequency reduction with EXACT binary arrays.

        Creates known binary contact data with exact frequency values.
        """
        # Create EXACT contact data with known frequencies
        # Feature 0: freq=0.6 (traj1), freq=0.4 (traj2) → NOT in intersection
        # Feature 1: freq=0.8 (traj1), freq=0.8 (traj2) → YES in intersection
        # Feature 2: freq=0.4 (traj1), freq=0.8 (traj2) → NOT in intersection
        # Feature 3: freq=0.2 (traj1), freq=0.6 (traj2) → NOT in intersection
        # Feature 4: freq=1.0 (traj1), freq=0.8 (traj2) → YES in intersection

        traj1_contacts = np.array([
            [1, 1, 0, 0, 1],  # frame 0
            [1, 1, 1, 0, 1],  # frame 1
            [1, 1, 0, 0, 1],  # frame 2
            [0, 1, 1, 1, 1],  # frame 3
            [1, 0, 0, 0, 1],  # frame 4
        ], dtype=np.int8)
        # Frequencies: [0.6, 0.8, 0.4, 0.2, 1.0]

        traj2_contacts = np.array([
            [0, 1, 1, 1, 1],  # frame 0
            [1, 1, 1, 1, 1],  # frame 1
            [0, 1, 1, 0, 1],  # frame 2
            [0, 1, 1, 1, 0],  # frame 3
            [1, 0, 1, 0, 1],  # frame 4
        ], dtype=np.int8)
        # Frequencies: [0.4, 0.8, 1.0, 0.6, 0.8]

        # Mock contact data
        self.pipeline.data.feature_data = {
            "contacts": {
                0: MockFeatureData(traj1_contacts),
                1: MockFeatureData(traj2_contacts)
            }
        }

        # Apply cross-trajectory frequency reduction with threshold_min=0.7
        # Features [1,4] pass: both have freq >= 0.7 in BOTH trajectories
        self.pipeline.feature.reduce.contacts.frequency(
            threshold_min=0.7,
            cross_trajectory=True
        )

        # EXPECTED: Features [1,4] remain (columns 1,4)
        expected_traj1_reduced = traj1_contacts[:, [1, 4]]
        expected_traj2_reduced = traj2_contacts[:, [1, 4]]

        # Get actual reduced data
        actual_traj1_reduced = self.pipeline.data.feature_data["contacts"][0].reduced_data
        actual_traj2_reduced = self.pipeline.data.feature_data["contacts"][1].reduced_data

        # EXACT comparison for binary data
        np.testing.assert_array_equal(
            actual_traj1_reduced,
            expected_traj1_reduced,
            err_msg="Trajectory 1 contact data does not match expected binary values exactly"
        )

        np.testing.assert_array_equal(
            actual_traj2_reduced,
            expected_traj2_reduced,
            err_msg="Trajectory 2 contact data does not match expected binary values exactly"
        )

    def test_cross_trajectory_empty_intersection_exact(self):
        """
        Test cross-trajectory reduction when NO features pass in all trajectories.

        Creates data where no feature meets the threshold in ALL trajectories.
        """
        # Create data where no overlap exists for high max threshold
        # Traj1: Feature 1 has high max, others low
        # Traj2: Feature 0 has high max, others low
        # No intersection for high threshold

        traj1_data = np.array([
            [1.0, 15.0, 2.0, 3.0, 1.0],  # Feature 1: max=15, rest: max<5
            [1.0, 10.0, 2.0, 3.0, 1.0],
            [1.1, 8.0, 2.1, 3.1, 1.1],
            [0.9, 12.0, 1.9, 2.9, 0.9],
            [1.0, 9.0, 2.0, 3.0, 1.0],
        ])

        traj2_data = np.array([
            [15.0, 2.0, 2.0, 3.0, 1.0],  # Feature 0: max=15, rest: max<5
            [10.0, 2.0, 2.0, 3.0, 1.0],
            [12.0, 2.1, 2.1, 3.1, 1.1],
            [8.0, 1.9, 1.9, 2.9, 0.9],
            [9.0, 2.0, 2.0, 3.0, 1.0],
        ])

        # Mock data
        self.pipeline.data.feature_data = {
            "distances": {
                0: MockFeatureData(traj1_data),
                1: MockFeatureData(traj2_data)
            }
        }

        # Apply high max threshold - no feature passes in both trajectories
        # Traj1 feature 1 max=15, Traj2 feature 0 max=15 → empty intersection
        self.pipeline.feature.reduce.distances.max(
            threshold_min=10.0,
            cross_trajectory=True
        )

        # Get actual results
        actual_traj1 = self.pipeline.data.feature_data["distances"][0].reduced_data
        actual_traj2 = self.pipeline.data.feature_data["distances"][1].reduced_data

        # EXPECTED: Empty result (no common features)
        assert actual_traj1 is None or actual_traj1.size == 0, "Expected empty result for trajectory 1"
        assert actual_traj2 is None or actual_traj2.size == 0, "Expected empty result for trajectory 2"


    def test_cross_trajectory_single_trajectory_exact(self):
        """
        Test cross-trajectory reduction with single trajectory produces exact results.

        When only one trajectory exists, cross-trajectory should behave like independent.
        """
        # Create minimal topology and trajectory
        topology = md.Topology()
        chain = topology.add_chain()
        for i in range(5):
            residue = topology.add_residue(f"RES{i}", chain)
            topology.add_atom("CA", md.element.carbon, residue)

        # Create dummy trajectory
        coords = np.zeros((5, 5, 3))  # 5 frames, 5 atoms
        traj = md.Trajectory(coords, topology)

        # Set up pipeline data structure with SINGLE trajectory
        self.pipeline.data.trajectory_data.trajectories = [traj]  # Only one trajectory
        self.pipeline.data.trajectory_data.trajectory_names = ["traj1"]

        # Create residue metadata for single trajectory
        self.pipeline.data.trajectory_data.res_label_data[0] = [
            {"name": "RES0", "resid": 1, "resname": "RES", "chain": "A"},
            {"name": "RES1", "resid": 2, "resname": "RES", "chain": "A"},
            {"name": "RES2", "resid": 3, "resname": "RES", "chain": "A"},
            {"name": "RES3", "resid": 4, "resname": "RES", "chain": "A"},
            {"name": "RES4", "resid": 5, "resname": "RES", "chain": "A"},
        ]

        # Create single trajectory data with known max values
        single_traj_data = np.array([
            [1.0, 5.0, 2.0, 3.0, 1.0],  # Features: max=1.1, max=15.0, max=2.1, max=3.1, max=1.1
            [1.0, 10.0, 2.0, 3.0, 1.0],
            [1.1, 15.0, 2.1, 3.1, 1.1],
            [0.9, 5.0, 1.9, 2.9, 0.9],
            [1.0, 10.0, 2.0, 3.0, 1.0],
        ])

        # Mock single trajectory
        self.pipeline.data.feature_data = {
            "distances": {
                0: MockFeatureData(single_traj_data)
            }
        }

        # Apply cross-trajectory reduction
        self.pipeline.feature.reduce.distances.max(
            threshold_min=10.0,
            cross_trajectory=True
        )

        # EXPECTED: Feature 1 should be retained (max=15.0 > 10.0), column 1
        expected_reduced = single_traj_data[:, [1]]

        # Get actual result
        actual_reduced = self.pipeline.data.feature_data["distances"][0].reduced_data

        # EXACT comparison
        np.testing.assert_array_almost_equal(
            actual_reduced,
            expected_reduced,
            decimal=10,
            err_msg="Single trajectory cross-trajectory reduction should match expected values exactly"
        )
