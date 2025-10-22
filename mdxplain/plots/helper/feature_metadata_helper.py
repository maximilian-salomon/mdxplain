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
Helper for feature metadata extraction and processing.

Centralized low-level metadata operations used across multiple
data preparers (TimeSeriesDataPreparer, ViolinDataPreparer, DensityDataPreparer).
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ...utils.feature_metadata_utils import FeatureMetadataUtils


class FeatureMetadataHelper:
    """
    Helper for feature metadata extraction.

    Provides static methods for extracting and organizing feature metadata
    from metadata arrays. Used by TimeSeriesDataPreparer and
    BaseFeatureImportancePlotDataPreparer to eliminate code duplication.

    Examples
    --------
    >>> # Collect metadata for all features of one type
    >>> meta = FeatureMetadataHelper.collect_metadata_for_type(
    ...     "dssp", features_dict, metadata_array
    ... )
    >>> print(meta.keys())  # Feature names

    >>> # Get top-level metadata for a feature type
    >>> top_level = FeatureMetadataHelper.get_top_level_metadata(
    ...     "dssp", metadata_array
    ... )
    >>> print(top_level.get("encoding"))  # "onehot"
    """

    @staticmethod
    def collect_metadata_for_type(
        feat_type: str,
        features: Dict[str, Any],
        metadata_array: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect metadata for all features of specific type.

        Parameters
        ----------
        feat_type : str
            Feature type name
        features : Dict[str, Any]
            Feature names for this type (keys used)
        metadata_array : np.ndarray
            Selected metadata array

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping feat_name -> {type_metadata, features}

        Examples
        --------
        >>> features = {"LYS42_DSSP_H": {...}, "ALA10_DSSP_C": {...}}
        >>> meta = FeatureMetadataHelper.collect_metadata_for_type(
        ...     "dssp", features, metadata_array
        ... )
        >>> print(meta["LYS42_DSSP_H"].keys())  # ["type_metadata", "features"]
        """
        result = {}
        top_level_metadata = FeatureMetadataHelper.get_top_level_metadata(
            feat_type, metadata_array
        )

        for feat_name in features.keys():
            feature_info = FeatureMetadataHelper.get_feature_info(
                feat_name, metadata_array
            )
            result[feat_name] = {
                "type_metadata": top_level_metadata,
                "features": feature_info
            }

        return result

    @staticmethod
    def get_top_level_metadata(
        feat_type: str,
        metadata_array: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get top-level metadata for feature type.

        Parameters
        ----------
        feat_type : str
            Feature type name
        metadata_array : np.ndarray
            Selected metadata array

        Returns
        -------
        Dict[str, Any]
            Top-level metadata (encoding, classes, visualization, etc.)

        Examples
        --------
        >>> top_level = FeatureMetadataHelper.get_top_level_metadata(
        ...     "dssp", metadata_array
        ... )
        >>> print(top_level.get("encoding"))  # "onehot"
        >>> print(top_level.get("classes"))  # ["H", "E", "C", ...]
        """
        for meta_entry in metadata_array:
            if meta_entry.get("type") == feat_type and "type_metadata" in meta_entry:
                return meta_entry["type_metadata"]
        return {}

    @staticmethod
    def get_feature_info(
        feat_name: str,
        metadata_array: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get feature-specific info from metadata array.

        Parameters
        ----------
        feat_name : str
            Feature name to find
        metadata_array : np.ndarray
            Selected metadata array

        Returns
        -------
        Dict[str, Any]
            Feature-specific metadata

        Examples
        --------
        >>> info = FeatureMetadataHelper.get_feature_info(
        ...     "LYS42_DSSP_H", metadata_array
        ... )
        >>> print(info.get("dssp_class"))  # "H"

        >>> # For pair features (distances)
        >>> info = FeatureMetadataHelper.get_feature_info(
        ...     "dist_CA_5_10", metadata_array
        ... )
        >>> print(len(info))  # 2 (both partners)
        """
        for idx in range(len(metadata_array)):
            if FeatureMetadataUtils.get_feature_name(metadata_array, idx) == feat_name:
                return FeatureMetadataHelper.extract_feature_info_from_entry(
                    metadata_array[idx]
                )
        return {}

    @staticmethod
    def extract_feature_info_from_entry(
        meta_entry: Dict[str, Any]
    ) -> list:
        """
        Extract feature info from metadata entry.

        Returns full features list for consistency. For pair features
        (distances), preserves both partners.

        Parameters
        ----------
        meta_entry : Dict[str, Any]
            Metadata entry containing features field

        Returns
        -------
        list
            Full features list from metadata entry

        Examples
        --------
        >>> # Single feature (DSSP, torsions)
        >>> entry = {"features": [{"dssp_class": "H"}], "type": "dssp"}
        >>> info = FeatureMetadataHelper.extract_feature_info_from_entry(entry)
        >>> print(info)  # [{"dssp_class": "H"}]

        >>> # Pair feature (distances)
        >>> entry = {
        ...     "features": [{"partner": 1}, {"partner": 2}],
        ...     "type": "distances"
        ... }
        >>> info = FeatureMetadataHelper.extract_feature_info_from_entry(entry)
        >>> print(info)  # [{"partner": 1}, {"partner": 2}]
        """
        if "features" not in meta_entry:
            return []

        return meta_entry["features"]
