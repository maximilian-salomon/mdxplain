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
Helper for preparing time series data from feature importance.

Extracts top features and organizes trajectory data for time series visualization.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple, TYPE_CHECKING, Optional, Set
import numpy as np

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData

from .....feature_importance.managers.feature_importance_manager import FeatureImportanceManager
from .....utils.feature_metadata_utils import FeatureMetadataUtils
from ....helper.contact_to_distances_converter import ContactToDistancesConverter
from ....helper.feature_metadata_helper import FeatureMetadataHelper
from ....helper.validation_helper import ValidationHelper


class TimeSeriesDataPreparer:
    """
    Preparer for time series plot data.

    Extracts features from feature importance (union of top N) and
    organizes trajectory-level data for time series visualization.

    Examples
    --------
    >>> data, metadata, cutoff = TimeSeriesDataPreparer.prepare(
    ...     pipeline_data, "tree_analysis", n_top=5
    ... )
    """

    @staticmethod
    def prepare(
        pipeline_data: PipelineData,
        feature_importance_name: str,
        n_top: int,
        contact_transformation: bool = True
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str], Dict[str, Dict[str, Any]], Optional[float], str, bool]:
        """
        Prepare data for time series visualization.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_importance_name : str
            Feature importance analysis name
        n_top : int
            Top N features per sub-comparison (union across all)
        contact_transformation : bool, default=True
            Convert contacts to distances

        Returns
        -------
        feature_data : Dict[str, Dict[str, Any]]
            {feat_type: {feat_name: {metadata}}}
        feature_indices : Dict[int, str]
            Mapping feature_index -> feature_name
        metadata_map : Dict[str, Dict[str, Any]]
            Feature metadata: feat_type -> feat_name -> {type_metadata, features}
        contact_cutoff : Optional[float]
            Contact cutoff if transformed
        feature_selector_name : str
            Name of feature selector to use (may be transformed)
        is_temporary : bool
            True if temporary selector was created

        Examples
        --------
        >>> data, indices, metadata_map, cutoff, selector, is_temp = TimeSeriesDataPreparer.prepare(
        ...     pipeline_data, "analysis", 5
        ... )
        """
        fi_data = pipeline_data.feature_importance_data[feature_importance_name]
        fi_manager = FeatureImportanceManager(pipeline_data)

        original_selector = fi_data.feature_selector

        # Contact transformation using ContactToDistancesConverter
        if contact_transformation:
            continuous_selector, is_temporary, contact_cutoff = (
                ContactToDistancesConverter.convert_contacts_to_distances(
                    pipeline_data, original_selector
                )
            )
        else:
            continuous_selector = original_selector
            is_temporary = False
            contact_cutoff = None

        unique_features = TimeSeriesDataPreparer._get_union_features(
            pipeline_data, fi_manager, feature_importance_name, n_top
        )

        feature_metadata = pipeline_data.get_selected_metadata(continuous_selector)

        feature_data, feature_indices = TimeSeriesDataPreparer._organize_features(
            unique_features, feature_metadata
        )

        metadata_map = TimeSeriesDataPreparer._collect_feature_metadata(
            feature_data, feature_metadata
        )

        return feature_data, feature_indices, metadata_map, contact_cutoff, continuous_selector, is_temporary

    @staticmethod
    def prepare_from_manual_selection(
        pipeline_data: PipelineData,
        feature_selector: str,
        contact_transformation: bool = True
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str], Dict[str, Dict[str, Any]], Optional[float], str, bool]:
        """
        Prepare data for time series visualization from manual selection.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector : str
            Name of feature selector
        contact_transformation : bool, default=True
            Convert contacts to distances

        Returns
        -------
        feature_data : Dict[str, Dict[str, Any]]
            {feat_type: {feat_name: {metadata}}}
        feature_indices : Dict[int, str]
            Mapping feature_index -> feature_name
        metadata_map : Dict[str, Dict[str, Any]]
            Feature metadata: feat_type -> feat_name -> {type_metadata, features}
        contact_cutoff : Optional[float]
            Contact cutoff if transformed
        feature_selector_name : str
            Name of feature selector to use (may be transformed)
        is_temporary : bool
            True if temporary selector was created

        Examples
        --------
        >>> data, indices, metadata_map, cutoff, selector, is_temp = TimeSeriesDataPreparer.prepare_from_manual_selection(
        ...     pipeline_data, "my_selector"
        ... )
        """
        # Validate feature selector
        ValidationHelper.validate_feature_selector_exists(
            pipeline_data, feature_selector
        )

        # Contact transformation using ContactToDistancesConverter
        if contact_transformation:
            continuous_selector, is_temporary, contact_cutoff = (
                ContactToDistancesConverter.convert_contacts_to_distances(
                    pipeline_data, feature_selector
                )
            )
        else:
            continuous_selector = feature_selector
            is_temporary = False
            contact_cutoff = None

        # Get all features from selector
        feature_metadata = pipeline_data.get_selected_metadata(continuous_selector)
        unique_features = set(range(len(feature_metadata)))

        feature_data, feature_indices = TimeSeriesDataPreparer._organize_features(
            unique_features, feature_metadata
        )

        metadata_map = TimeSeriesDataPreparer._collect_feature_metadata(
            feature_data, feature_metadata
        )

        return feature_data, feature_indices, metadata_map, contact_cutoff, continuous_selector, is_temporary

    @staticmethod
    def _get_union_features(
        pipeline_data,
        fi_manager: FeatureImportanceManager,
        feature_importance_name: str,
        n_top: int
    ) -> Set[int]:
        """
        Get union of top N features from all sub-comparisons.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data
        fi_manager : FeatureImportanceManager
            Feature importance manager
        feature_importance_name : str
            Analysis name
        n_top : int
            Top N per comparison

        Returns
        -------
        Set[int]
            Unique feature indices

        Examples
        --------
        >>> indices = TimeSeriesDataPreparer._get_union_features(
        ...     pipeline_data, fi_manager, "analysis", 5
        ... )
        """
        fi_data = pipeline_data.feature_importance_data[feature_importance_name]
        all_indices = set()

        for comp_idx, comp_metadata in enumerate(fi_data.metadata):
            comp_name = comp_metadata.get("comparison", f"comparison_{comp_idx}")
            top_features = fi_manager.get_top_features(
                pipeline_data, feature_importance_name, comp_name, n_top
            )
            for feat_info in top_features:
                all_indices.add(feat_info["feature_index"])

        return all_indices

    @staticmethod
    def _organize_features(
        feature_indices: Set[int],
        feature_metadata: Optional[List[Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str]]:
        """
        Organize features by type and create index mapping.

        Parameters
        ----------
        feature_indices : Set[int]
            Feature indices
        feature_metadata : List or None
            Feature metadata

        Returns
        -------
        feature_data : Dict[str, Dict[str, Any]]
            Organized by type and name
        index_map : Dict[int, str]
            Index to name mapping

        Examples
        --------
        >>> data, index_map = TimeSeriesDataPreparer._organize_features(
        ...     {0, 5, 10}, metadata
        ... )
        """
        feature_data = {}
        index_map = {}

        for feat_idx in sorted(feature_indices):
            feat_name = FeatureMetadataUtils.get_feature_name(feature_metadata, feat_idx)
            feat_type = FeatureMetadataUtils.get_feature_type(feature_metadata, feat_idx)

            if feat_type not in feature_data:
                feature_data[feat_type] = {}

            feature_data[feat_type][feat_name] = {
                "feature_index": feat_idx,
                "feature_type": feat_type
            }
            index_map[feat_idx] = feat_name

        return feature_data, index_map

    @staticmethod
    def _collect_feature_metadata(
        feature_data: Dict[str, Dict[str, Any]],
        feature_metadata: Optional[List[Any]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Collect feature metadata for all features in feature data.

        Creates mapping from feature type and name to full metadata including
        top-level metadata (encoding, classes) and feature-specific info.

        Parameters
        ----------
        feature_data : Dict[str, Dict[str, Any]]
            Feature data structure (feat_type -> feat_name -> metadata)
        feature_metadata : List or None
            Feature metadata array from pipeline_data

        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]]
            Nested structure: feat_type -> feat_name -> {type_metadata, features}

        Examples
        --------
        >>> metadata_map = TimeSeriesDataPreparer._collect_feature_metadata(
        ...     feature_data, metadata_array
        ... )

        Notes
        -----
        Uses FeatureMetadataHelper for all metadata extraction operations.
        """
        if feature_metadata is None:
            return {}

        # Convert to numpy array if needed
        metadata_array = feature_metadata if isinstance(feature_metadata, np.ndarray) else np.array(feature_metadata)

        result = {}
        for feat_type in feature_data.keys():
            result[feat_type] = FeatureMetadataHelper.collect_metadata_for_type(
                feat_type, feature_data[feat_type], metadata_array
            )

        return result

    @staticmethod
    def get_x_values(
        pipeline_data: PipelineData,
        traj_idx: int,
        use_time: bool
    ) -> np.ndarray:
        """
        Get X-axis values for trajectory.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        traj_idx : int
            Trajectory index
        use_time : bool
            Use time (True) or frames (False)

        Returns
        -------
        np.ndarray
            X-axis values (time in ns or frame indices)

        Examples
        --------
        >>> # Get time values in nanoseconds
        >>> x_values = TimeSeriesDataPreparer.get_x_values(pipeline_data, 0, True)
        >>> print(x_values[0])  # First time point in ns

        >>> # Get frame indices
        >>> x_values = TimeSeriesDataPreparer.get_x_values(pipeline_data, 0, False)
        >>> print(x_values[0])  # 0

        Notes
        -----
        Central method for X-axis values used by all time series plotting helpers
        to ensure consistent axis ranges across feature and membership plots.
        """
        trajectory = pipeline_data.trajectory_data.trajectories[traj_idx]
        if use_time:
            return trajectory.time / 1000  # Convert ps to ns
        return np.arange(trajectory.n_frames)
