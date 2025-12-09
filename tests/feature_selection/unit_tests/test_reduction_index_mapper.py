# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0).
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
Unit tests for ReductionIndexMapper.

Tests the index mapping functionality that enables using original feature
data when the corresponding reduced feature provides the feature selection.
"""

import numpy as np

from mdxplain.feature_selection.helper.reduction_index_mapper import ReductionIndexMapper
from mdxplain.pipeline.entities.pipeline_data import PipelineData
from mdxplain.feature.entities.feature_data import FeatureData
from mdxplain.feature.feature_type.contacts.contacts import Contacts


class TestReductionIndexMapperMapReducedToOriginal:
    """Test map_reduced_to_original() method."""

    def test_simple_mapping(self):
        """Test basic index mapping."""
        # 5 features were kept during reduction
        kept_indices = np.array([5, 12, 45, 102, 203])

        # Want features 1 and 3 from reduced space
        reduced_indices = np.array([1, 3])

        # Map to original space
        original = ReductionIndexMapper.map_reduced_to_original(
            reduced_indices, kept_indices
        )

        # Should return original indices 12 and 102
        np.testing.assert_array_equal(original, np.array([12, 102]))

    def test_single_index(self):
        """Test mapping single index."""
        kept_indices = np.array([10, 20, 30, 40, 50])
        reduced_indices = np.array([2])

        original = ReductionIndexMapper.map_reduced_to_original(
            reduced_indices, kept_indices
        )

        np.testing.assert_array_equal(original, np.array([30]))

    def test_all_indices(self):
        """Test mapping all reduced indices."""
        kept_indices = np.array([100, 200, 300])
        reduced_indices = np.array([0, 1, 2])

        original = ReductionIndexMapper.map_reduced_to_original(
            reduced_indices, kept_indices
        )

        np.testing.assert_array_equal(original, np.array([100, 200, 300]))

    def test_order_preservation(self):
        """Test that index order is preserved."""
        kept_indices = np.array([3, 7, 15, 31, 63])
        reduced_indices = np.array([4, 2, 0, 3])

        original = ReductionIndexMapper.map_reduced_to_original(
            reduced_indices, kept_indices
        )

        # Order should match reduced_indices order
        np.testing.assert_array_equal(original, np.array([63, 15, 3, 31]))


class TestReductionIndexMapperGetKeptIndices:
    """Test get_kept_indices() method."""

    def test_get_kept_indices_exists(self):
        """Test getting kept indices when they exist."""
        # Create pipeline data with reduction info
        pipeline_data = PipelineData()
        feature_data = FeatureData(Contacts(cutoff=4.5))

        # Set up reduction info with kept_indices
        feature_data.reduction_info = {
            "reduction_method": "cv",
            "retention_rate": 0.2,
            "n_dynamic": 50,
            "total_pairs": 250,
            "kept_indices": np.array([5, 12, 45, 102, 203])
        }

        # Add to pipeline data
        pipeline_data.feature_data["contacts"] = {0: feature_data}

        # Get kept indices
        kept = ReductionIndexMapper.get_kept_indices(pipeline_data, "contacts", 0)

        assert kept is not None
        np.testing.assert_array_equal(kept, np.array([5, 12, 45, 102, 203]))

    def test_get_kept_indices_no_reduction_info(self):
        """Test getting kept indices when no reduction performed."""
        pipeline_data = PipelineData()
        feature_data = FeatureData(Contacts(cutoff=4.5))
        feature_data.reduction_info = None

        pipeline_data.feature_data["contacts"] = {0: feature_data}

        kept = ReductionIndexMapper.get_kept_indices(pipeline_data, "contacts", 0)

        assert kept is None

    def test_get_kept_indices_feature_not_exists(self):
        """Test getting kept indices when feature type doesn't exist."""
        pipeline_data = PipelineData()

        kept = ReductionIndexMapper.get_kept_indices(pipeline_data, "contacts", 0)

        assert kept is None

    def test_get_kept_indices_trajectory_not_exists(self):
        """Test getting kept indices when trajectory doesn't exist."""
        pipeline_data = PipelineData()
        feature_data = FeatureData(Contacts(cutoff=4.5))

        pipeline_data.feature_data["contacts"] = {0: feature_data}

        # Request trajectory 1 which doesn't exist
        kept = ReductionIndexMapper.get_kept_indices(pipeline_data, "contacts", 1)

        assert kept is None

    def test_get_kept_indices_multiple_trajectories(self):
        """Test getting kept indices from specific trajectory."""
        pipeline_data = PipelineData()

        # Trajectory 0
        feature_data_0 = FeatureData(Contacts(cutoff=4.5))
        feature_data_0.reduction_info = {
            "kept_indices": np.array([1, 2, 3])
        }

        # Trajectory 1
        feature_data_1 = FeatureData(Contacts(cutoff=4.5))
        feature_data_1.reduction_info = {
            "kept_indices": np.array([10, 20, 30])
        }

        pipeline_data.feature_data["contacts"] = {
            0: feature_data_0,
            1: feature_data_1
        }

        # Get from trajectory 0
        kept_0 = ReductionIndexMapper.get_kept_indices(pipeline_data, "contacts", 0)
        np.testing.assert_array_equal(kept_0, np.array([1, 2, 3]))

        # Get from trajectory 1
        kept_1 = ReductionIndexMapper.get_kept_indices(pipeline_data, "contacts", 1)
        np.testing.assert_array_equal(kept_1, np.array([10, 20, 30]))
