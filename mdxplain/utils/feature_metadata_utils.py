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
Central utilities for feature metadata operations.

Provides generic methods for extracting feature names and types from
pipeline metadata, usable across all modules (plots, feature_importance, etc.).
"""

from typing import Optional, List, Any
import numpy as np


class FeatureMetadataUtils:
    """
    Central utility class for feature metadata operations.

    Provides static methods for extracting human-readable feature names
    and types from pipeline metadata. These utilities are shared across
    multiple modules to ensure consistency.

    Examples
    --------
    >>> # Get feature name from metadata
    >>> metadata = pipeline_data.get_selected_metadata("my_selector")
    >>> name = FeatureMetadataUtils.get_feature_name(metadata, 42)
    >>> print(name)  # "ALA_5_CA-GLU_10_CA" or "ALA_5_phi"

    >>> # Get feature type
    >>> ftype = FeatureMetadataUtils.get_feature_type(metadata, 42)
    >>> print(ftype)  # "distances" or "torsions"
    """

    @staticmethod
    def get_feature_name(
        feature_metadata: Optional[List[Any]],
        feature_idx: int
    ) -> str:
        """
        Get human-readable name for a feature index.

        Extracts the feature name from metadata for the given index,
        handling both pair features (distances) and non-pair features
        (torsions). Provides fallback when metadata is unavailable.

        Parameters
        ----------
        feature_metadata : list or None
            Feature metadata list from pipeline (e.g., from get_selected_metadata())
        feature_idx : int
            Index of the feature to get name for

        Returns
        -------
        str
            Human-readable feature name

        Examples
        --------
        >>> # Pair feature (distances)
        >>> name = FeatureMetadataUtils.get_feature_name(metadata, 10)
        >>> print(name)  # "ALA_5_CA-GLU_10_CA"

        >>> # Non-pair feature (torsions)
        >>> name = FeatureMetadataUtils.get_feature_name(metadata, 20)
        >>> print(name)  # "ALA_5_phi"

        >>> # Fallback when metadata unavailable
        >>> name = FeatureMetadataUtils.get_feature_name(None, 42)
        >>> print(name)  # "feature_42"

        Notes
        -----
        - Pair features (2 partners): Joined with "-" separator
        - Non-pair features (1 element): Single name
        - Uses numpy array iteration for partner extraction
        """
        # Fallback if no metadata available
        if feature_metadata is None or feature_idx >= len(feature_metadata):
            return f"feature_{feature_idx}"

        metadata_entry = feature_metadata[feature_idx]

        # Extract feature name from features metadata
        features_data = metadata_entry.get("features", {})

        # Extract feature name from features metadata (numpy array of partner residues)
        if isinstance(features_data, np.ndarray):
            # features_data contains the partner information for this specific feature
            partner_names = []
            for element in features_data:
                if isinstance(element, dict) and "full_name" in element:
                    partner_names.append(element["full_name"])

            if partner_names:
                # Join with "-" for multiple partners (contacts/distances) or return single name
                return "-".join(partner_names)

        return f"feature_{feature_idx}"

    @staticmethod
    def get_feature_type(
        feature_metadata: Optional[List[Any]],
        feature_idx: int
    ) -> str:
        """
        Get feature type for a feature index.

        Extracts the feature type from metadata for the given index,
        providing fallback when metadata is unavailable.

        Parameters
        ----------
        feature_metadata : list or None
            Feature metadata list from pipeline
        feature_idx : int
            Index of the feature to get type for

        Returns
        -------
        str
            Feature type name (e.g., "distances", "torsions", "sasa")

        Examples
        --------
        >>> # Get feature type
        >>> ftype = FeatureMetadataUtils.get_feature_type(metadata, 42)
        >>> print(ftype)  # "distances"

        >>> # Fallback when metadata unavailable
        >>> ftype = FeatureMetadataUtils.get_feature_type(None, 42)
        >>> print(ftype)  # "unknown"

        Notes
        -----
        Returns "unknown" when metadata is unavailable or feature index
        is out of bounds.
        """
        # Fallback if no metadata available
        if feature_metadata is None or feature_idx >= len(feature_metadata):
            return "unknown"

        metadata_entry = feature_metadata[feature_idx]
        return metadata_entry.get("type", "unknown")

    @staticmethod
    def create_feature_map(metadata_array: np.ndarray) -> dict:
        """
        Create feature index to name mapping from metadata array.

        Extracts all feature names from metadata array and creates
        a dictionary mapping feature indices to their names.

        Parameters
        ----------
        metadata_array : np.ndarray
            Feature metadata array from pipeline

        Returns
        -------
        Dict[int, str]
            Mapping of feature_index -> feature_name

        Examples
        --------
        >>> metadata = pipeline_data.get_selected_metadata("my_selector")
        >>> feature_map = FeatureMetadataUtils.create_feature_map(metadata)
        >>> print(feature_map[42])  # "ALA_5_CA-GLU_10_CA"

        Notes
        -----
        Uses get_feature_name() internally for consistent name extraction
        across all feature types (pairs and non-pairs).
        """
        feature_map = {}
        for idx in range(len(metadata_array)):
            feature_name = FeatureMetadataUtils.get_feature_name(
                metadata_array, idx
            )
            feature_map[idx] = feature_name
        return feature_map
