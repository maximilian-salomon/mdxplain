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
Central utilities for top features extraction and formatting.

This module provides utilities for extracting and formatting top important
features from feature importance analysis. Used by feature_importance and
structure_visualization modules.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .feature_metadata_utils import FeatureMetadataUtils

if TYPE_CHECKING:
    from ..pipeline.entities.pipeline_data import PipelineData
    from ..feature_importance.entities.feature_importance_data import (
        FeatureImportanceData,
    )


class TopFeaturesUtils:
    """
    Utility class for extracting and formatting top important features.

    Provides static methods for extracting top features from feature
    importance data and formatting them with human-readable names and
    metadata. Breaks down complex extraction into smaller, focused methods.

    Examples
    --------
    >>> # Get top features for specific comparison
    >>> features = TopFeaturesUtils.get_top_features_with_names(
    ...     pipeline_data, fi_data, "cluster_0_vs_rest", 5
    ... )

    >>> # Get top features averaged across all comparisons
    >>> features = TopFeaturesUtils.get_top_features_with_names(
    ...     pipeline_data, fi_data, n=10
    ... )
    """

    @staticmethod
    def get_top_features_with_names(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        comparison_identifier: Optional[str] = None,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top features with complete name mapping and formatting.

        Extracts top N most important features from feature importance data,
        retrieves metadata, and formats the results with human-readable names
        and types. This is the main entry point for top features processing.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing feature metadata
        fi_data : FeatureImportanceData
            Feature importance data object with importance scores
        comparison_identifier : str, optional
            Specific sub-comparison to get features from.
            If None, returns average importance across all sub-comparisons.
        n : int, default=10
            Number of top features to return

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with complete feature information.
            Each dictionary contains:
            - feature_index: int - Feature index in feature array
            - importance_score: float - Importance score
            - rank: int - Rank (1-indexed)
            - feature_name: str - Human-readable feature name
            - feature_type: str - Feature type (e.g., "distances", "torsions")
            - residue_seqids: List[int] - List of residue sequence IDs involved
            - residue_indices: List[int] - List of residue indices involved

        Examples
        --------
        >>> # Get top features for specific comparison
        >>> features = TopFeaturesUtils.get_top_features_with_names(
        ...     pipeline_data, fi_data, "cluster_0_vs_rest", 5
        ... )

        >>> # Get top features averaged across all comparisons
        >>> features = TopFeaturesUtils.get_top_features_with_names(
        ...     pipeline_data, fi_data, n=10
        ... )

        Notes
        -----
        - Requires feature_selector to be set in fi_data for name mapping
        - If comparison_identifier is None, uses average importance
        - Feature names extracted from metadata using FeatureMetadataUtils
        - Delegates to smaller methods for better code organization
        """
        indices_scores = TopFeaturesUtils._get_indices_and_scores(
            fi_data, comparison_identifier, n
        )
        metadata = TopFeaturesUtils._get_feature_metadata(pipeline_data, fi_data)
        return TopFeaturesUtils._format_features(indices_scores, metadata)

    @staticmethod
    def _get_indices_and_scores(
        fi_data: FeatureImportanceData,
        comparison_identifier: Optional[str],
        n: int
    ) -> List[Tuple[int, float]]:
        """
        Extract top feature indices and importance scores.

        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data object
        comparison_identifier : str, optional
            Specific comparison identifier or None for average
        n : int
            Number of top features to return

        Returns
        -------
        List[Tuple[int, float]]
            List of (feature_index, importance_score) tuples
        """
        if comparison_identifier is not None:
            return fi_data.get_top_features(comparison_identifier, n)

        # Get average importance across all comparisons
        avg_importance = fi_data.get_average_importance()
        if len(avg_importance) == 0:
            return []

        # Get top N indices sorted by importance
        sorted_indices = np.argsort(avg_importance)[::-1]
        top_n = min(n, len(avg_importance))

        return [
            (int(idx), float(avg_importance[idx]))
            for idx in sorted_indices[:top_n]
        ]

    @staticmethod
    def _get_feature_metadata(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData
    ) -> Optional[List[Any]]:
        """
        Get feature metadata from pipeline data.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        fi_data : FeatureImportanceData
            Feature importance data object

        Returns
        -------
        list or None
            Feature metadata list or None if not available
        """
        if fi_data.feature_selector:
            return pipeline_data.get_selected_metadata(fi_data.feature_selector)
        return None

    @staticmethod
    def _format_features(
        indices_scores: List[Tuple[int, float]],
        feature_metadata: Optional[List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format feature indices with names and metadata.

        Parameters
        ----------
        indices_scores : List[Tuple[int, float]]
            List of (feature_index, importance_score) tuples
        feature_metadata : list or None
            Feature metadata from pipeline

        Returns
        -------
        List[Dict[str, Any]]
            Formatted feature dictionaries with names, types, and residue seqids
        """
        result = []

        for feature_idx, importance_score in indices_scores:
            feature_info = {
                "feature_index": feature_idx,
                "importance_score": importance_score,
                "rank": len(result) + 1,
            }

            # Add feature name and type using central utility
            feature_info["feature_name"] = FeatureMetadataUtils.get_feature_name(
                feature_metadata, feature_idx
            )
            feature_info["feature_type"] = FeatureMetadataUtils.get_feature_type(
                feature_metadata, feature_idx
            )

            # Add residue info directly from metadata
            residues = FeatureMetadataUtils.get_feature_residues(
                feature_metadata, feature_idx
            )
            feature_info["residue_seqids"] = [
                res["seqid"] for res in residues
            ]
            feature_info["residue_indices"] = [
                res["index"] for res in residues
            ]

            result.append(feature_info)

        return result
