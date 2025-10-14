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
Helper class for preparing data for violin plots.

Processes feature importance results and extracts feature values
from trajectory matrices for visualization.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData
    from .....feature_importance.entities.feature_importance_data import FeatureImportanceData

from ....helper.comparison_data_extractor import ComparisonDataExtractor
from ....helper.validation_helper import ValidationHelper
from ....helper.contact_to_distances_converter import ContactToDistancesConverter
from ....helper.color_mapping_helper import ColorMappingHelper
from .....feature_importance.managers.feature_importance_manager import FeatureImportanceManager
from .....utils.feature_metadata_utils import FeatureMetadataUtils


class ViolinDataPreparer:
    """
    Helper class for preparing violin plot data from feature importance.

    Extracts feature values from trajectory matrices based on top features
    identified in feature importance analysis. Handles data extraction
    per comparison and organizes data for violin plot visualization.

    Examples
    --------
    >>> # Prepare data for violin plots
    >>> violin_data = ViolinDataPreparer.prepare_violin_data(
    ...     pipeline_data, fi_data, all_top_features, "key_features"
    ... )
    """

    @staticmethod
    def prepare_violin_data(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        all_top_features: Dict[str, List[Dict[str, Any]]],
        continuous_selector_name: str,
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Prepare violin plot data from feature importance top features.

        Collects all unique features across comparisons, groups them by
        feature type, and extracts their values for each comparison.
        Returns nested dictionary structure ready for violin plotting.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        fi_data : FeatureImportanceData
            Feature importance data object
        all_top_features : Dict[str, List[Dict[str, Any]]]
            Top features per comparison from get_all_top_features()
        continuous_selector_name : str
            Name of feature selector with continuous (distance) features

        Returns
        -------
        Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested dictionary structure:
            {
                "distances": {  # Feature-Type (outer)
                    "feature_name_1": {  # Feature-Name (middle)
                        "cluster_0_vs_rest": np.array([...]),  # Comparison (inner)
                        "cluster_1_vs_rest": np.array([...])
                    },
                    "feature_name_2": {...}
                },
                "torsions": {
                    "phi_ALA_15": {...}
                }
            }

        Examples
        --------
        >>> all_top = fi_manager.get_all_top_features("tree_analysis", n=10)
        >>> violin_data = ViolinDataPreparer.prepare_violin_data(
        ...     pipeline_data, fi_data, all_top, "features_distances"
        ... )
        >>> for feat_type, features in violin_data.items():
        ...     print(f"{feat_type}: {len(features)} features")

        Notes
        -----
        Algorithm:
        1. Collect all unique features across all comparisons
        2. Get all DataSelectors from comparison
        3. Build matrices ONCE per DataSelector (cache them)
        4. Extract all features from cached matrices
        5. Group results by feature type for organized plotting
        """
        comparison_name = fi_data.comparison_name
        result = {}  # Dict[feat_type, Dict[feat_name, Dict[comp_name, values]]]

        # Step 1: Collect unique features with their metadata
        feature_map = ViolinDataPreparer._collect_unique_features(
            all_top_features
        )

        # Step 2: Get all DataSelectors from comparison
        data_selector_names = (
            ComparisonDataExtractor.get_all_data_selectors_from_comparison(
                pipeline_data, comparison_name
            )
        )

        # Step 3: Get metadata for type lookup
        metadata_array = pipeline_data.get_selected_metadata(
            continuous_selector_name
        )

        # Step 4: Pre-initialize result structure for all features
        for feat_idx, feat_name in feature_map.items():
            feat_type = metadata_array[feat_idx]["type"]
            if feat_type not in result:
                result[feat_type] = {}
            if feat_name not in result[feat_type]:
                result[feat_type][feat_name] = {}

        # Step 5: Extract features from all DataSelectors with memmap cleanup
        ViolinDataPreparer._extract_features_from_selectors(
            pipeline_data,
            continuous_selector_name,
            data_selector_names,
            feature_map,
            metadata_array,
            result
        )

        return result

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
        >>> unique = ViolinDataPreparer._collect_unique_features(features)
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
    def _extract_features_from_selectors(
        pipeline_data: PipelineData,
        feature_selector_name: str,
        data_selectors: List[str],
        features: Dict[int, str],
        metadata_array: np.ndarray,
        result: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        Extract features from DataSelectors with memmap cleanup.

        Core extraction loop: build matrix → extract columns → close memmap.
        Modifies result dict in-place to avoid returning large data structures.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector_name : str
            Name of feature selector
        data_selectors : List[str]
            DataSelector names to process
        features : Dict[int, str]
            Mapping of feature_index -> feature_name
        metadata_array : np.ndarray
            Feature metadata array for type lookup
        result : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Result dictionary to populate (modified in-place)

        Returns
        -------
        None
            Modifies result dict in-place

        Examples
        --------
        >>> result = {}
        >>> _extract_features_from_selectors(
        ...     pipeline_data, "selector", ["cluster_0"],
        ...     {0: "feat1"}, metadata_array, result
        ... )
        >>> print(result.keys())  # ["distances"]

        Notes
        -----
        Implements the core bugfix: build matrix, extract all features,
        close memmap. Only keeps one matrix open at a time to prevent
        file descriptor exhaustion.
        """
        for selector_name in data_selectors:
            # Build matrix (memmap)
            matrix = pipeline_data.get_selected_data(
                feature_selector_name, selector_name
            )

            # Extract ALL features from this matrix (copy to RAM)
            for feat_idx, feat_name in features.items():
                feat_type = metadata_array[feat_idx]["type"]
                # Copy column to RAM to break memmap reference
                # This prevents file descriptor leaks
                feature_values = np.array(matrix[:, feat_idx])
                result[feat_type][feat_name][selector_name] = feature_values

            # Close memmap explicitly to free file descriptor
            if hasattr(matrix, '_mmap') and matrix._mmap is not None:
                matrix._mmap.close()
            del matrix

    @staticmethod
    def prepare_from_feature_importance(
        pipeline_data: PipelineData, feature_importance_name: str, n_top: int
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, str]]:
        """
        Complete preparation for Feature Importance mode.

        Coordinates all steps: validates feature importance, extracts top
        features, converts contacts to distances, prepares violin data,
        and creates DataSelector color mapping.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_importance_name : str
            Name of feature importance analysis
        n_top : int
            Number of top features per comparison

        Returns
        -------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure: feat_type -> feat_name -> data_selector_name -> values
        data_selector_colors : Dict[str, str]
            Mapping of data_selector_name -> color_hex

        Raises
        ------
        ValueError
            If feature importance analysis not found

        Examples
        --------
        >>> violin_data, colors = ViolinDataPreparer.prepare_from_feature_importance(
        ...     pipeline_data, "tree_analysis", n_top=10
        ... )
        >>> print(violin_data.keys())  # ["distances", "torsions"]

        Notes
        -----
        Complete coordination method for Feature Importance mode.
        Returns same structure as prepare_from_manual_selection() for
        consistent downstream processing.
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

        # Convert contacts to distances if needed (returns selector and temp flag)
        continuous_selector, is_temporary = (
            ContactToDistancesConverter.convert_contacts_to_distances(
                pipeline_data, fi_data.feature_selector
            )
        )

        # Prepare data for violins (grouped by feature type)
        violin_data = ViolinDataPreparer.prepare_violin_data(
            pipeline_data, fi_data, all_top_features, continuous_selector
        )

        # Get all DataSelectors from comparison
        data_selector_names = (
            ComparisonDataExtractor.get_all_data_selectors_from_comparison(
                pipeline_data, fi_data.comparison_name
            )
        )

        # Create DataSelector-to-color mapping (cluster-consistent)
        data_selector_colors = (
            ColorMappingHelper.create_data_selector_color_mapping(
                data_selector_names
            )
        )

        # Cleanup temporary selector if created
        if is_temporary:
            ContactToDistancesConverter.cleanup_temporary_selector(
                pipeline_data, continuous_selector
            )

        return violin_data, data_selector_colors

    @staticmethod
    def prepare_from_manual_selection(
        pipeline_data: PipelineData,
        feature_selector_name: str,
        data_selectors: List[str],
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, str]]:
        """
        Complete preparation for Manual selection mode.

        Coordinates all steps: converts contacts to distances, gets all
        features from selector, builds violin data, and creates DataSelector
        color mapping.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector_name : str
            Name of feature selector
        data_selectors : List[str]
            DataSelector names to plot

        Returns
        -------
        violin_data : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure: feat_type -> feat_name -> data_selector_name -> values
        data_selector_colors : Dict[str, str]
            Mapping of data_selector_name -> color_hex

        Raises
        ------
        ValueError
            If feature selector or data selectors not found

        Examples
        --------
        >>> violin_data, colors = ViolinDataPreparer.prepare_from_manual_selection(
        ...     pipeline_data, "my_selector", ["cluster_0", "cluster_1"]
        ... )
        >>> print(violin_data.keys())  # ["distances", "torsions"]

        Notes
        -----
        Complete coordination method for Manual mode.
        Returns same structure as prepare_from_feature_importance() for
        consistent downstream processing.
        """
        # Convert contacts to distances if needed (returns selector and temp flag)
        continuous_selector, is_temporary = (
            ContactToDistancesConverter.convert_contacts_to_distances(
                pipeline_data, feature_selector_name
            )
        )

        # Get all features from selector
        all_features = ViolinDataPreparer._get_all_features_from_selector(
            pipeline_data, continuous_selector
        )

        # Build violin data
        violin_data = ViolinDataPreparer._build_manual_violin_data(
            pipeline_data, continuous_selector, data_selectors, all_features
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

        return violin_data, data_selector_colors

    @staticmethod
    def _get_all_features_from_selector(
        pipeline_data: PipelineData, feature_selector_name: str
    ) -> Dict[int, str]:
        """
        Get all features from feature selector using metadata API.

        Directly accesses feature metadata to extract feature names
        without iterating through trajectory-level data. Handles both
        pair features (distances) and non-pair features (torsions).

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector_name : str
            Name of feature selector

        Returns
        -------
        Dict[int, str]
            Mapping of feature_index -> feature_name

        Raises
        ------
        ValueError
            If feature selector not found

        Examples
        --------
        >>> features = ViolinDataPreparer._get_all_features_from_selector(
        ...     pipeline_data, "my_selector"
        ... )
        >>> print(len(features))  # e.g., 150
        >>> print(features[42])  # "ALA_5_CA - GLU_10_CA"

        Notes
        -----
        Uses pipeline_data.get_selected_metadata() for direct access
        to feature information. Array index corresponds to feature index.
        """
        if feature_selector_name not in pipeline_data.selected_feature_data:
            available = list(pipeline_data.selected_feature_data.keys())
            raise ValueError(
                f"Feature selector '{feature_selector_name}' not found. "
                f"Available: {available}"
            )

        # Get metadata for all selected features
        metadata_array = pipeline_data.get_selected_metadata(
            feature_selector_name
        )

        # Use central utility for consistent feature name extraction
        feature_map = {}
        for idx in range(len(metadata_array)):
            feature_name = FeatureMetadataUtils.get_feature_name(
                metadata_array, idx
            )
            feature_map[idx] = feature_name

        return feature_map

    @staticmethod
    def _build_manual_violin_data(
        pipeline_data: PipelineData,
        feature_selector_name: str,
        data_selectors: List[str],
        resolved_features: Dict[int, str],
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Build violin data for manual selection mode.

        Creates three-level nested dictionary structure organizing feature
        values by feature type, feature name, and DataSelector name.
        Uses metadata API to get feature types directly.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_selector_name : str
            Name of feature selector containing features
        data_selectors : List[str]
            DataSelector names to extract values for
        resolved_features : Dict[int, str]
            Mapping of feature_index -> feature_name for selected features

        Returns
        -------
        Dict[str, Dict[str, Dict[str, np.ndarray]]]
            Three-level nested structure:
            feat_type -> feat_name -> data_selector_name -> values

        Examples
        --------
        >>> resolved = {10: "dist_CA_5_10", 20: "phi_ALA_15"}
        >>> violin_data = ViolinDataPreparer._build_manual_violin_data(
        ...     pipeline_data, "my_selector", ["cluster_0", "cluster_1"], resolved
        ... )
        >>> print(violin_data.keys())  # ["distances", "torsions"]
        """
        # Get metadata once for all features
        metadata_array = pipeline_data.get_selected_metadata(
            feature_selector_name
        )

        # Pre-initialize result structure for all features
        result = {}
        for feat_idx, feat_name in resolved_features.items():
            feat_type = metadata_array[feat_idx]["type"]
            if feat_type not in result:
                result[feat_type] = {}
            if feat_name not in result[feat_type]:
                result[feat_type][feat_name] = {}

        # Extract features from all DataSelectors with memmap cleanup
        ViolinDataPreparer._extract_features_from_selectors(
            pipeline_data,
            feature_selector_name,
            data_selectors,
            resolved_features,
            metadata_array,
            result
        )

        return result
