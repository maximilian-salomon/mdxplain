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
Base data preparer for feature importance plot types.

Provides shared data preparation logic for violin plots, density plots,
and other feature importance visualizations.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple, TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from .comparison_data_extractor import ComparisonDataExtractor
from .validation_helper import ValidationHelper
from .contact_to_distances_converter import ContactToDistancesConverter
from .color_mapping_helper import ColorMappingHelper
from .feature_metadata_helper import FeatureMetadataHelper
from ...feature_importance.manager.feature_importance_manager import FeatureImportanceManager
from ...utils.feature_metadata_utils import FeatureMetadataUtils


class BaseFeatureImportancePlotDataPreparer:
    """
    Base class for feature importance plot data preparation.

    Provides common data preparation logic for violin plots, density plots,
    and other visualizations based on feature importance analysis. Handles
    feature extraction, contact transformation, and metadata collection.

    Subclasses (ViolinDataPreparer, DensityDataPreparer) can inherit
    all functionality or override specific methods for customization.

    Examples
    --------
    >>> # Feature Importance mode
    >>> data, metadata, colors, cutoff = BaseFeatureImportancePlotDataPreparer.prepare_from_feature_importance(
    ...     pipeline_data, "tree_analysis", n_top=10
    ... )

    >>> # Manual Selection mode
    >>> data, metadata, colors, cutoff = BaseFeatureImportancePlotDataPreparer.prepare_from_manual_selection(
    ...     pipeline_data, "my_selector", ["cluster_0", "cluster_1"]
    ... )
    """

    @staticmethod
    def _collect_unique_features(
        all_top_features: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[int, str]:
        """
        Collect all unique features across comparisons.

        Parameters
        ----------
        all_top_features : Dict[str, List[Dict[str, Any]]]
            Top features per comparison

        Returns
        -------
        Dict[int, str]
            Mapping of feature_index -> feature_name

        Examples
        --------
        >>> features = {
        ...     "comp1": [{"feature_index": 5, "feature_name": "dist_CA_10_20"}],
        ...     "comp2": [{"feature_index": 5, "feature_name": "dist_CA_10_20"}]
        ... }
        >>> unique = BaseFeatureImportancePlotDataPreparer._collect_unique_features(features)
        >>> print(unique)  # {5: "dist_CA_10_20"}
        """
        feature_map = {}

        for _, features in all_top_features.items():
            for feat in features:
                feat_idx = feat["feature_index"]
                feat_name = feat["feature_name"]

                if feat_idx not in feature_map:
                    feature_map[feat_idx] = feat_name

        return feature_map

    @staticmethod
    def prepare_from_feature_importance(
        pipeline_data: PipelineData,
        feature_importance_name: str,
        n_top: int,
        contact_transformation: bool = True
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Dict[str, Dict[str, Any]]], Dict[str, str], Optional[float]]:
        """
        Complete preparation for Feature Importance mode.

        Coordinates all steps: validates feature importance, extracts top
        features, optionally converts contacts to distances, prepares plot data,
        collects feature metadata, and creates DataSelector color mapping.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_importance_name : str
            Name of feature importance analysis
        n_top : int
            Number of top features per comparison
        contact_transformation : bool, default=True
            If True, converts contact features to distances.
            If False, keeps contacts as binary (0/1) values.

        Returns
        -------
        plot_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure: feat_type -> feat_name -> data_selector_name -> values
        feature_metadata_map : Dict[str, Dict[str, Dict[str, Any]]]
            Feature metadata: feat_type -> feat_name -> {type_metadata, features}
        data_selector_colors : Dict[str, str]
            Mapping of data_selector_name -> color_hex
        contact_cutoff : Optional[float]
            Contact cutoff value if converted from contacts, None otherwise

        Raises
        ------
        ValueError
            If feature importance analysis not found

        Examples
        --------
        >>> # With contact transformation (default)
        >>> plot_data, metadata, colors, cutoff = BaseFeatureImportancePlotDataPreparer.prepare_from_feature_importance(
        ...     pipeline_data, "tree_analysis", n_top=10
        ... )
        >>> print(plot_data.keys())  # ["distances", "torsions"]
        >>> print(cutoff)  # 4.5

        >>> # Without contact transformation (binary contacts)
        >>> plot_data, metadata, colors, cutoff = BaseFeatureImportancePlotDataPreparer.prepare_from_feature_importance(
        ...     pipeline_data, "tree_analysis", n_top=10, contact_transformation=False
        ... )
        >>> print(plot_data.keys())  # ["contacts", "torsions"]
        >>> print(cutoff)  # None

        Notes
        -----
        Complete coordination method for Feature Importance mode.
        Returns same structure as prepare_from_manual_selection() for
        consistent downstream processing.

        When contact_transformation=False, contacts remain as binary features
        for visualization with Gaussian smoothing.
        """
        # Validate feature importance exists
        fi_data = ValidationHelper.validate_feature_importance_exists(
            pipeline_data, feature_importance_name
        )

        # Get top features from feature importance
        fi_manager = FeatureImportanceManager()
        all_top_features = fi_manager.get_all_top_features(
            pipeline_data, feature_importance_name, n_top
        )

        # Conditionally convert contacts to distances
        if contact_transformation:
            continuous_selector, is_temporary, contact_cutoff = (
                ContactToDistancesConverter.convert_contacts_to_distances(
                    pipeline_data, fi_data.feature_selector
                )
            )
        else:
            # Use original selector without conversion
            continuous_selector = fi_data.feature_selector
            is_temporary = False
            contact_cutoff = None

        # Prepare data for plots (grouped by feature type)
        comparison_name = fi_data.comparison_name

        # Collect unique features with their metadata
        feature_map = BaseFeatureImportancePlotDataPreparer._collect_unique_features(
            all_top_features
        )

        # Get all DataSelectors from comparison
        comparison_data_selector_names = (
            ComparisonDataExtractor.get_all_data_selectors_from_comparison(
                pipeline_data, comparison_name
            )
        )

        # Create DataSelector-to-color mapping (cluster-consistent)
        data_selector_colors = (
            ColorMappingHelper.create_data_selector_color_mapping(
                comparison_data_selector_names
            )
        )

        # Get metadata for type lookup
        metadata_array = pipeline_data.get_selected_metadata(continuous_selector)

        # Build plot data structure
        plot_data = BaseFeatureImportancePlotDataPreparer._build_plot_data(
            pipeline_data,
            continuous_selector,
            comparison_data_selector_names,
            feature_map,
            metadata_array
        )

        # Collect feature metadata for discrete feature support
        feature_metadata_map = BaseFeatureImportancePlotDataPreparer._collect_feature_metadata(
            plot_data, metadata_array
        )

        # Cleanup temporary selector if created
        if is_temporary:
            ContactToDistancesConverter.cleanup_temporary_selector(
                pipeline_data, continuous_selector
            )

        return plot_data, feature_metadata_map, data_selector_colors, contact_cutoff

    @staticmethod
    def prepare_from_manual_selection(
        pipeline_data: PipelineData,
        feature_selector_name: str,
        data_selectors: List[str],
        contact_transformation: bool = True
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Dict[str, Dict[str, Any]]], Dict[str, str], Optional[float]]:
        """
        Complete preparation for Manual selection mode.

        Coordinates all steps: optionally converts contacts to distances, gets all
        features from selector, builds plot data, collects feature metadata,
        and creates DataSelector color mapping.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector_name : str
            Name of feature selector
        data_selectors : List[str]
            DataSelector names to plot
        contact_transformation : bool, default=True
            If True, converts contact features to distances.
            If False, keeps contacts as binary (0/1) values.

        Returns
        -------
        plot_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure: feat_type -> feat_name -> data_selector_name -> values
        feature_metadata_map : Dict[str, Dict[str, Dict[str, Any]]]
            Feature metadata: feat_type -> feat_name -> {type_metadata, features}
        data_selector_colors : Dict[str, str]
            Mapping of data_selector_name -> color_hex
        contact_cutoff : Optional[float]
            Contact cutoff value if converted from contacts, None otherwise

        Raises
        ------
        ValueError
            If feature selector or data selectors not found

        Examples
        --------
        >>> # With contact transformation (default)
        >>> plot_data, metadata, colors, cutoff = BaseFeatureImportancePlotDataPreparer.prepare_from_manual_selection(
        ...     pipeline_data, "my_selector", ["cluster_0", "cluster_1"]
        ... )
        >>> print(plot_data.keys())  # ["distances", "torsions"]
        >>> print(cutoff)  # 4.5

        >>> # Without contact transformation (binary contacts)
        >>> plot_data, metadata, colors, cutoff = BaseFeatureImportancePlotDataPreparer.prepare_from_manual_selection(
        ...     pipeline_data, "my_selector", ["cluster_0", "cluster_1"],
        ...     contact_transformation=False
        ... )
        >>> print(plot_data.keys())  # ["contacts", "torsions"]
        >>> print(cutoff)  # None

        Notes
        -----
        Complete coordination method for Manual mode.
        Returns same structure as prepare_from_feature_importance() for
        consistent downstream processing.

        When contact_transformation=False, contacts remain as binary features
        for visualization with Gaussian smoothing.
        """
        # Validate feature selector exists
        ValidationHelper.validate_feature_selector_exists(
            pipeline_data, feature_selector_name
        )

        # Conditionally convert contacts to distances
        if contact_transformation:
            continuous_selector, is_temporary, contact_cutoff = (
                ContactToDistancesConverter.convert_contacts_to_distances(
                    pipeline_data, feature_selector_name
                )
            )
        else:
            # Use original selector without conversion
            continuous_selector = feature_selector_name
            is_temporary = False
            contact_cutoff = None

        # Get metadata ONCE for all operations
        metadata_array = pipeline_data.get_selected_metadata(continuous_selector)

        # Get all features from selector using central utility
        all_features = FeatureMetadataUtils.create_feature_map(metadata_array)

        # Build plot data
        plot_data = BaseFeatureImportancePlotDataPreparer._build_plot_data(
            pipeline_data, continuous_selector, data_selectors,
            all_features, metadata_array
        )

        # Collect feature metadata for discrete feature support
        feature_metadata_map = BaseFeatureImportancePlotDataPreparer._collect_feature_metadata(
            plot_data, metadata_array
        )

        # Create DataSelector-to-color mapping
        data_selector_colors = (
            ColorMappingHelper.create_data_selector_color_mapping(
                data_selectors
            )
        )

        # Cleanup temporary selector if created
        if is_temporary:
            ContactToDistancesConverter.cleanup_temporary_selector(
                pipeline_data, continuous_selector
            )

        return plot_data, feature_metadata_map, data_selector_colors, contact_cutoff

    @staticmethod
    def _build_plot_data(
        pipeline_data: PipelineData,
        feature_selector_name: str,
        data_selectors: List[str],
        features: Dict[int, str],
        metadata_array: np.ndarray
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Build plot data structure from features.

        Core method for building three-level nested dictionary organizing
        feature values by feature type, feature name, and DataSelector name.
        Shared by both feature_importance and manual selection modes.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector_name : str
            Name of feature selector containing features
        data_selectors : List[str]
            DataSelector names to extract values for
        features : Dict[int, str]
            Mapping of feature_index -> feature_name
        metadata_array : np.ndarray
            Feature metadata array for type lookup

        Returns
        -------
        Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure:
            feat_type -> feat_name -> data_selector_name -> values

        Examples
        --------
        >>> # Build plot data from features
        >>> features = {10: "dist_CA_5_10", 20: "phi_ALA_15"}
        >>> plot_data = BaseFeatureImportancePlotDataPreparer._build_plot_data(
        ...     pipeline_data, "my_selector", ["cluster_0", "cluster_1"],
        ...     features, metadata_array
        ... )
        >>> print(plot_data.keys())  # ["distances", "torsions"]

        Notes
        -----
        Implements proper memmap cleanup by building matrix, extracting
        all features to RAM, then closing memmap before next iteration.
        """
        result = {}

        for selector_name in data_selectors:
            # Build matrix (memmap)
            matrix = pipeline_data.get_selected_data(
                feature_selector_name, selector_name
            )

            # Extract ALL features from this matrix (copy to RAM)
            for feat_idx, feat_name in features.items():
                feat_type = metadata_array[feat_idx]["type"]

                # On-demand initialization
                if feat_type not in result:
                    result[feat_type] = {}
                if feat_name not in result[feat_type]:
                    result[feat_type][feat_name] = {}

                # Copy column to RAM to break memmap reference
                feature_values = np.array(matrix[:, feat_idx])
                result[feat_type][feat_name][selector_name] = feature_values

            # Close memmap explicitly to free file descriptor
            if hasattr(matrix, '_mmap') and matrix._mmap is not None:
                matrix._mmap.close()
            del matrix

        return result

    @staticmethod
    def _collect_feature_metadata(
        plot_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        metadata_array: np.ndarray
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Collect feature metadata for all features in plot data.

        Creates mapping from feature type and name to full metadata including
        top-level metadata (encoding, classes) and feature-specific info
        (dssp_class for one-hot, residue info, etc.).

        Parameters
        ----------
        plot_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Plot data structure (feat_type -> feat_name -> selector -> values)
        metadata_array : np.ndarray
            Selected metadata array from pipeline_data

        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]]
            Nested structure: feat_type -> feat_name -> {type_metadata, features}

        Examples
        --------
        >>> metadata_map = BaseFeatureImportancePlotDataPreparer._collect_feature_metadata(
        ...     plot_data, metadata_array
        ... )
        >>> print(metadata_map["dssp"]["LYS42_DSSP_H"].keys())  # ["type_metadata", "features"]
        >>> print(metadata_map["dssp"]["LYS42_DSSP_H"]["type_metadata"]["encoding"])  # "onehot"

        Notes
        -----
        Uses FeatureMetadataHelper for all metadata extraction operations.
        """
        result = {}

        for feat_type, features in plot_data.items():
            result[feat_type] = FeatureMetadataHelper.collect_metadata_for_type(
                feat_type, features, metadata_array
            )

        return result
