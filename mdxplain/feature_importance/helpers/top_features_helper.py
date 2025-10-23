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
Top features helper for feature importance operations.

This module provides helper methods for extracting and formatting
top important features from feature importance analysis results.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np

from ..entities.feature_importance_data import FeatureImportanceData
from ...utils.feature_metadata_utils import FeatureMetadataUtils

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class TopFeaturesHelper:
    """
    Helper class for processing top important features.
    
    Provides static methods for extracting top features from feature
    importance data and formatting them with human-readable names and
    metadata. These methods extract common logic from FeatureImportanceManager.
    
    Examples
    --------
    >>> # Get top features for specific comparison
    >>> top_features = TopFeaturesHelper.get_top_features_for_comparison(
    ...     fi_data, "folded_vs_rest", 5
    ... )
    
    >>> # Get top features averaged across comparisons
    >>> top_features = TopFeaturesHelper.get_top_features_averaged(
    ...     fi_data, 10
    ... )
    """
    
    @staticmethod
    def get_top_features_for_comparison(
        fi_data: FeatureImportanceData, 
        comparison_identifier: str, 
        n: int
    ) -> List[Tuple[int, float]]:
        """
        Get top N features for a specific comparison.
        
        Extracts the top N most important features for a specific
        sub-comparison from feature importance data.
        
        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data object
        comparison_identifier : str
            Name/identifier of the specific comparison
        n : int
            Number of top features to return
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (feature_index, importance_score) tuples
            
        Examples
        --------
        >>> indices_scores = TopFeaturesHelper.get_top_features_for_comparison(
        ...     fi_data, "folded_vs_rest", 5
        ... )
        >>> for idx, score in indices_scores:
        ...     print(f"Feature {idx}: {score:.3f}")
        """
        return fi_data.get_top_features(comparison_identifier, n)
    
    @staticmethod
    def get_top_features_averaged(
        fi_data: FeatureImportanceData,
        n: int
    ) -> List[Tuple[int, float]]:
        """
        Get top N features averaged across all comparisons.
        
        Computes average importance across all sub-comparisons and
        returns the top N features based on average importance.
        
        Parameters
        ----------
        fi_data : FeatureImportanceData
            Feature importance data object
        n : int
            Number of top features to return
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (feature_index, average_importance_score) tuples
            
        Examples
        --------
        >>> indices_scores = TopFeaturesHelper.get_top_features_averaged(
        ...     fi_data, 10
        ... )
        >>> for idx, score in indices_scores:
        ...     print(f"Feature {idx}: {score:.3f}")
        """
        # Get average importance across all comparisons
        avg_importance = fi_data.get_average_importance()
        if len(avg_importance) == 0:
            return []

        # Get top N indices
        sorted_indices = np.argsort(avg_importance)[::-1]
        top_n = min(n, len(avg_importance))
        
        return [
            (int(idx), float(avg_importance[idx]))
            for idx in sorted_indices[:top_n]
        ]
    
    @staticmethod
    def format_features_with_names(
        indices_scores: List[Tuple[int, float]],
        feature_metadata: Optional[List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format feature indices and scores with human-readable names.
        
        Converts a list of (index, score) tuples into dictionaries
        containing detailed feature information including names and types.
        
        Parameters
        ----------
        indices_scores : List[Tuple[int, float]]
            List of (feature_index, importance_score) tuples
        feature_metadata : list or None
            Feature metadata from pipeline for name mapping
            
        Returns
        -------
        List[Dict[str, Any]]
            List of feature info dictionaries with names and metadata
            
        Examples
        --------
        >>> formatted = TopFeaturesHelper.format_features_with_names(
        ...     [(42, 0.85), (15, 0.72)], metadata
        ... )
        >>> print(formatted[0]["feature_name"])  # "CA_distance_ALA_15_GLU_89"
        >>> print(formatted[0]["importance_score"])  # 0.85
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

            result.append(feature_info)
        
        return result
    
    @staticmethod
    def get_top_features_with_names(
        pipeline_data: PipelineData,
        fi_data: FeatureImportanceData,
        comparison_identifier: Optional[str] = None,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top features with complete name mapping and formatting.
        
        Complete method that extracts top features, retrieves metadata,
        and formats the results with human-readable names and types.
        This is the main entry point for top features processing.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing metadata
        fi_data : FeatureImportanceData
            Feature importance data object
        comparison_identifier : str, optional
            Specific sub-comparison to get features from.
            If None, returns average across all sub-comparisons.
        n : int, default=10
            Number of top features to return
            
        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with complete feature information
            
        Examples
        --------
        >>> # Get top features for specific comparison
        >>> top_features = TopFeaturesHelper.get_top_features_with_names(
        ...     pipeline_data, fi_data, "folded_vs_rest", 5
        ... )
        
        >>> # Get top features averaged across all comparisons
        >>> top_features = TopFeaturesHelper.get_top_features_with_names(
        ...     pipeline_data, fi_data, n=10
        ... )
        """
        # Get raw indices and scores
        if comparison_identifier is not None:
            indices_scores = TopFeaturesHelper.get_top_features_for_comparison(
                fi_data, comparison_identifier, n
            )
        else:
            indices_scores = TopFeaturesHelper.get_top_features_averaged(
                fi_data, n
            )

        # Get feature metadata for name mapping
        feature_metadata = None
        if fi_data.feature_selector:
            feature_metadata = pipeline_data.get_selected_metadata(
                fi_data.feature_selector
            )

        # Format with names and metadata
        return TopFeaturesHelper.format_features_with_names(
            indices_scores, feature_metadata
        )