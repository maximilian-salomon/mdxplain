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
Metadata builder helper for feature importance operations.

This module provides helper methods for building metadata dictionaries
that describe feature importance analyses, extracting common logic from
FeatureImportanceManager methods.
"""

from typing import Dict, Any
import numpy as np

from ..analyzer_types.interfaces.analyzer_type_base import AnalyzerTypeBase


class MetadataBuilderHelper:
    """
    Helper class for building feature importance analysis metadata.
    
    Provides static methods for creating standardized metadata dictionaries
    that describe feature importance analyses. These dictionaries are stored
    alongside analysis results for documentation and debugging.
    
    Examples:
    ---------
    >>> # Build analysis metadata
    >>> metadata = MetadataBuilderHelper.build_analysis_metadata(
    ...     sub_comp, analyzer_type, X, y, result
    ... )
    
    >>> # Add group information
    >>> MetadataBuilderHelper.add_group_information(metadata, sub_comp)
    """
    
    @staticmethod
    def build_analysis_metadata(
        sub_comp: Dict[str, Any],
        analyzer_type: AnalyzerTypeBase,
        X: np.ndarray,
        y: np.ndarray,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build complete metadata dictionary for an analysis.
        
        Creates a comprehensive metadata dictionary that documents all
        aspects of a feature importance analysis including data dimensions,
        analyzer type, and model-specific metadata.
        
        Parameters:
        -----------
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary containing comparison metadata
        analyzer_type : AnalyzerTypeBase
            Analyzer instance used for the analysis
        X : np.ndarray
            Feature matrix used for analysis
        y : np.ndarray
            Label array used for analysis
        result : Dict[str, Any]
            Analysis result from the analyzer
            
        Returns:
        --------
        Dict[str, Any]
            Complete metadata dictionary for the analysis
            
        Examples:
        ---------
        >>> metadata = MetadataBuilderHelper.build_analysis_metadata(
        ...     sub_comp_dict, decision_tree, X, y, analysis_result
        ... )
        >>> print(metadata["comparison"])
        >>> print(metadata["n_samples"])
        >>> print(metadata["n_features"])
        """
        # Build base metadata
        metadata = {
            "comparison": sub_comp["name"],
            "analyzer_type": analyzer_type.get_type_name(),
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "model_metadata": result.get("metadata", {}),
        }
        
        # Add group information from sub-comparison
        MetadataBuilderHelper.add_group_information(metadata, sub_comp)
        
        return metadata
    
    @staticmethod
    def add_group_information(
        metadata: Dict[str, Any], sub_comp: Dict[str, Any]
    ) -> None:
        """
        Add group information to metadata dictionary.
        
        Adds information about the groups/selectors involved in the
        comparison to the metadata dictionary. Handles both binary
        and multiclass comparison modes.
        
        Parameters:
        -----------
        metadata : Dict[str, Any]
            Metadata dictionary to add group information to (modified in-place)
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary containing group information
            
        Returns:
        --------
        None
            Modifies metadata dictionary in-place
            
        Examples:
        ---------
        >>> metadata = {"comparison": "test"}
        >>> MetadataBuilderHelper.add_group_information(metadata, sub_comp)
        >>> print("group1" in metadata or "selectors" in metadata)  # True
        """
        # Add group information from sub-comparison
        if "group1_selectors" in sub_comp:
            # Binary or pairwise comparison
            metadata["group1"] = sub_comp["group1_selectors"]
            metadata["group2"] = sub_comp["group2_selectors"]
        elif "selectors" in sub_comp:
            # Multiclass comparison
            metadata["selectors"] = sub_comp["selectors"]
            metadata["mode"] = "multiclass"
    
    @staticmethod
    def build_base_metadata(
        comparison_name: str,
        analyzer_type: AnalyzerTypeBase,
        n_samples: int,
        n_features: int
    ) -> Dict[str, Any]:
        """
        Build basic metadata dictionary with core information.
        
        Creates a basic metadata dictionary with essential information
        about the analysis. This can be extended with additional information.
        
        Parameters:
        -----------
        comparison_name : str
            Name of the comparison being analyzed
        analyzer_type : AnalyzerTypeBase
            Analyzer instance used for the analysis
        n_samples : int
            Number of samples (frames) in the analysis
        n_features : int
            Number of features in the analysis
            
        Returns:
        --------
        Dict[str, Any]
            Basic metadata dictionary
            
        Examples:
        ---------
        >>> metadata = MetadataBuilderHelper.build_base_metadata(
        ...     "folded_vs_unfolded", decision_tree, 150, 50
        ... )
        >>> print(metadata["n_samples"])  # 150
        >>> print(metadata["analyzer_type"])  # "decision_tree"
        """
        return {
            "comparison": comparison_name,
            "analyzer_type": analyzer_type.get_type_name(),
            "n_samples": n_samples,
            "n_features": n_features,
        }
    
    @staticmethod
    def add_model_metadata(
        metadata: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """
        Add model-specific metadata to metadata dictionary.
        
        Extracts and adds model-specific metadata from the analysis result
        to the main metadata dictionary.
        
        Parameters:
        -----------
        metadata : Dict[str, Any]
            Metadata dictionary to add model metadata to (modified in-place)
        result : Dict[str, Any]
            Analysis result containing model metadata
            
        Returns:
        --------
        None
            Modifies metadata dictionary in-place
            
        Examples:
        ---------
        >>> metadata = {"comparison": "test"}
        >>> MetadataBuilderHelper.add_model_metadata(metadata, analysis_result)
        >>> print("model_metadata" in metadata)  # True
        """
        metadata["model_metadata"] = result.get("metadata", {})