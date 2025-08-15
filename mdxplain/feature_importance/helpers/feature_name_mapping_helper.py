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
Feature name mapping helper for feature importance operations.

This module provides helper methods for mapping feature indices to
human-readable feature names using pipeline metadata.
"""

from typing import Dict, Any, Optional


class FeatureNameMappingHelper:
    """
    Helper class for mapping feature indices to names.
    
    Provides static methods for converting feature indices to human-readable
    names using feature metadata from the pipeline. These methods extract
    common logic from FeatureImportanceManager to improve code organization.
    
    Examples:
    ---------
    >>> # Get feature name from metadata
    >>> name = FeatureNameMappingHelper.get_feature_name(
    ...     feature_metadata, feature_idx
    ... )
    
    >>> # Add names to feature info dictionary
    >>> FeatureNameMappingHelper.add_feature_names(
    ...     feature_info, feature_metadata, feature_idx
    ... )
    """
    
    @staticmethod
    def get_feature_metadata(pipeline_data, feature_selector: Optional[str]):
        """
        Get feature metadata for a feature selector.
        
        Safely retrieves feature metadata from pipeline_data for the given
        feature_selector, handling cases where metadata is unavailable.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing metadata
        feature_selector : str or None
            Name of the feature selector to get metadata for
            
        Returns:
        --------
        list or None
            Feature metadata list or None if unavailable
            
        Examples:
        ---------
        >>> metadata = FeatureNameMappingHelper.get_feature_metadata(
        ...     pipeline_data, "my_features"
        ... )
        >>> if metadata:
        ...     print(f"Found metadata for {len(metadata)} features")
        """
        if not feature_selector:
            return None
            
        return pipeline_data.get_selected_metadata(feature_selector)
    
    @staticmethod
    def get_feature_name(
        feature_metadata, 
        feature_idx: int
    ) -> str:
        """
        Get human-readable name for a feature index.
        
        Extracts the feature name from metadata for the given index,
        providing fallbacks when metadata is unavailable or incomplete.
        
        Parameters:
        -----------
        feature_metadata : list or None
            Feature metadata list from pipeline
        feature_idx : int
            Index of the feature to get name for
            
        Returns:
        --------
        str
            Human-readable feature name
            
        Examples:
        ---------
        >>> name = FeatureNameMappingHelper.get_feature_name(
        ...     metadata, 42
        ... )
        >>> print(name)  # "CA_distance_ALA_15_GLU_89" or "feature_42"
        """
        # Fallback if no metadata available
        if not feature_metadata or feature_idx >= len(feature_metadata):
            return f"feature_{feature_idx}"
        
        metadata_entry = feature_metadata[feature_idx]
        
        # Extract feature name from features metadata
        features_data = metadata_entry.get("features", {})
        if isinstance(features_data, dict):
            return features_data.get("name", f"feature_{feature_idx}")
        else:
            return f"feature_{feature_idx}"

    @staticmethod
    def get_feature_type(
        feature_metadata, 
        feature_idx: int
    ) -> str:
        """
        Get feature type for a feature index.
        
        Extracts the feature type from metadata for the given index,
        providing fallback when metadata is unavailable.
        
        Parameters:
        -----------
        feature_metadata : list or None
            Feature metadata list from pipeline
        feature_idx : int
            Index of the feature to get type for
            
        Returns:
        --------
        str
            Feature type name
            
        Examples:
        ---------
        >>> ftype = FeatureNameMappingHelper.get_feature_type(
        ...     metadata, 42
        ... )
        >>> print(ftype)  # "distances" or "unknown"
        """
        # Fallback if no metadata available
        if not feature_metadata or feature_idx >= len(feature_metadata):
            return "unknown"
        
        metadata_entry = feature_metadata[feature_idx]
        return metadata_entry.get("type", "unknown")
    
    @staticmethod
    def add_feature_names(
        feature_info: Dict[str, Any],
        feature_metadata,
        feature_idx: int
    ) -> None:
        """
        Add feature name and type to feature info dictionary.
        
        Enriches a feature info dictionary with human-readable name and
        type information extracted from metadata.
        
        Parameters:
        -----------
        feature_info : Dict[str, Any]
            Feature info dictionary to enrich (modified in-place)
        feature_metadata : list or None
            Feature metadata list from pipeline
        feature_idx : int
            Index of the feature to add names for
            
        Returns:
        --------
        None
            Modifies feature_info dictionary in-place
            
        Examples:
        ---------
        >>> feature_info = {"feature_index": 42, "importance_score": 0.85}
        >>> FeatureNameMappingHelper.add_feature_names(
        ...     feature_info, metadata, 42
        ... )
        >>> print(feature_info["feature_name"])  # "CA_distance_ALA_15_GLU_89"
        >>> print(feature_info["feature_type"])  # "distances"
        """
        feature_info["feature_name"] = FeatureNameMappingHelper.get_feature_name(
            feature_metadata, feature_idx
        )
        feature_info["feature_type"] = FeatureNameMappingHelper.get_feature_type(
            feature_metadata, feature_idx
        )