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

from ..analyzer_type.interfaces.analyzer_type_base import AnalyzerTypeBase


class MetadataBuilderHelper:
    """
    Helper class for building feature importance analysis metadata.
    
    Provides static methods for creating standardized metadata dictionaries
    that describe feature importance analyses. These dictionaries are stored
    alongside analysis results for documentation and debugging.
    
    Examples
    --------
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
        _y: np.ndarray,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build complete metadata dictionary for an analysis.

        Creates a comprehensive metadata dictionary that documents all
        aspects of a feature importance analysis including data dimensions,
        analyzer type, and model-specific metadata.

        Parameters
        ----------
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary containing comparison metadata
        analyzer_type : AnalyzerTypeBase
            Analyzer instance used for the analysis
        X : np.ndarray
            Feature matrix used for analysis
        _y : np.ndarray
            Label array (unused, kept for API consistency)
        result : Dict[str, Any]
            Analysis result from the analyzer

        Returns
        -------
        Dict[str, Any]
            Complete metadata dictionary for the analysis

        Examples
        --------
        >>> metadata = MetadataBuilderHelper.build_analysis_metadata(
        ...     sub_comp_dict, decision_tree, X, y, analysis_result
        ... )
        >>> print(metadata["comparison"])
        >>> print(metadata["sub_comparison_name"])
        >>> print(metadata["class_names"])
        >>> print(metadata["n_samples"])
        >>> print(metadata["n_features"])
        """
        # Generate sub-comparison name and extract class names
        sub_comparison_name = MetadataBuilderHelper.generate_sub_comparison_name(sub_comp)
        class_names = MetadataBuilderHelper.extract_class_names(sub_comp)

        # Build base metadata
        metadata = {
            "comparison": sub_comp["name"],
            "sub_comparison_name": sub_comparison_name,
            "class_names": class_names,
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
        
        Parameters
        ----------
        metadata : Dict[str, Any]
            Metadata dictionary to add group information to (modified in-place)
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary containing group information
            
        Returns
        -------
        None
            Modifies metadata dictionary in-place
            
        Examples
        --------
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
        
        Parameters
        ----------
        comparison_name : str
            Name of the comparison being analyzed
        analyzer_type : AnalyzerTypeBase
            Analyzer instance used for the analysis
        n_samples : int
            Number of samples (frames) in the analysis
        n_features : int
            Number of features in the analysis
            
        Returns
        -------
        Dict[str, Any]
            Basic metadata dictionary
            
        Examples
        --------
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

        Parameters
        ----------
        metadata : Dict[str, Any]
            Metadata dictionary to add model metadata to (modified in-place)
        result : Dict[str, Any]
            Analysis result containing model metadata

        Returns
        -------
        None
            Modifies metadata dictionary in-place

        Examples
        --------
        >>> metadata = {"comparison": "test"}
        >>> MetadataBuilderHelper.add_model_metadata(metadata, analysis_result)
        >>> print("model_metadata" in metadata)  # True
        """
        metadata["model_metadata"] = result.get("metadata", {})

    @staticmethod
    def generate_sub_comparison_name(sub_comp: Dict[str, Any]) -> str:
        """
        Generate sub-comparison name from data selector names.

        Creates a descriptive name for the sub-comparison by joining
        data selector names with "_vs_" separator.

        Parameters
        ----------
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary containing selector information

        Returns
        -------
        str
            Generated sub-comparison name

        Examples
        --------
        >>> # Binary comparison
        >>> sub_comp = {
        ...     "group1_selectors": ["folded"],
        ...     "group2_selectors": ["unfolded"]
        ... }
        >>> name = MetadataBuilderHelper.generate_sub_comparison_name(sub_comp)
        >>> print(name)
        folded_vs_unfolded

        >>> # One-vs-rest comparison
        >>> sub_comp = {
        ...     "group1_selectors": ["cluster_2"],
        ...     "group2_selectors": ["cluster_0", "cluster_1", "cluster_3"]
        ... }
        >>> name = MetadataBuilderHelper.generate_sub_comparison_name(sub_comp)
        >>> print(name)
        cluster_2_vs_rest

        >>> # Multiclass comparison
        >>> sub_comp = {
        ...     "selectors": ["folded", "unfolded", "intermediate"]
        ... }
        >>> name = MetadataBuilderHelper.generate_sub_comparison_name(sub_comp)
        >>> print(name)
        folded_vs_unfolded_vs_intermediate
        """
        if "group1_selectors" in sub_comp and "group2_selectors" in sub_comp:
            # Binary, pairwise, or one-vs-rest comparison
            group1_name = sub_comp["group1_selectors"][0]
            group2_selectors = sub_comp["group2_selectors"]

            # Check if this is one-vs-rest (group2 has multiple selectors)
            if len(group2_selectors) > 1:
                return f"{group1_name}_vs_rest"
            else:
                group2_name = group2_selectors[0]
                return f"{group1_name}_vs_{group2_name}"
        elif "selectors" in sub_comp:
            # Multiclass comparison
            return "_vs_".join(sub_comp["selectors"])
        else:
            # Fallback to existing name if structure is unexpected
            return sub_comp.get("name", "unknown_comparison")

    @staticmethod
    def _extract_multiclass_names(sub_comp: Dict[str, Any]) -> list:
        """
        Extract class names for multiclass comparison.

        Parameters
        ----------
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary

        Returns
        -------
        list
            List of selector names
        """
        return sub_comp["selectors"]

    @staticmethod
    def _extract_binary_names(sub_comp: Dict[str, Any]) -> list:
        """
        Extract class names for binary comparison.

        Parameters
        ----------
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary

        Returns
        -------
        list
            List of two class names in correct label order
        """
        class1 = sub_comp["group1_selectors"][0]
        class2 = sub_comp["group2_selectors"][0]
        labels = sub_comp.get("labels", (0, 1))

        if labels == (1, 0):
            return [class2, class1]
        return [class1, class2]

    @staticmethod
    def _extract_one_vs_rest_names(sub_comp: Dict[str, Any]) -> list:
        """
        Extract class names for one-vs-rest comparison.

        Parameters
        ----------
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary

        Returns
        -------
        list
            List with one class and "all other classes" in label order [0, 1]
        """
        class1 = sub_comp["group1_selectors"][0]
        class2 = "all other classes"
        labels = sub_comp.get("labels", (0, 1))

        if labels == (1, 0):
            return [class2, class1]
        return [class1, class2]

    @staticmethod
    def extract_class_names(sub_comp: Dict[str, Any]) -> list:
        """
        Extract class names from sub-comparison data selectors.

        Extracts the data selector names that represent the classes
        in the comparison for use as human-readable class labels.
        Handles one-vs-rest comparisons by using "all other classes"
        label and ensures correct label-to-name mapping.

        Parameters
        ----------
        sub_comp : Dict[str, Any]
            Sub-comparison dictionary containing selector information

        Returns
        -------
        list
            List of class names (data selector names) in correct label order

        Examples
        --------
        >>> # Binary comparison
        >>> sub_comp = {
        ...     "group1_selectors": ["folded"],
        ...     "group2_selectors": ["unfolded"],
        ...     "labels": (0, 1)
        ... }
        >>> names = MetadataBuilderHelper.extract_class_names(sub_comp)
        >>> print(names)
        ['folded', 'unfolded']

        >>> # One-vs-rest comparison
        >>> sub_comp = {
        ...     "group1_selectors": ["cluster_0"],
        ...     "group2_selectors": ["cluster_1", "cluster_2", "cluster_3"],
        ...     "labels": (1, 0)
        ... }
        >>> names = MetadataBuilderHelper.extract_class_names(sub_comp)
        >>> print(names)
        ['all other classes', 'cluster_0']

        >>> # Multiclass comparison
        >>> sub_comp = {
        ...     "selectors": ["folded", "unfolded", "intermediate"]
        ... }
        >>> names = MetadataBuilderHelper.extract_class_names(sub_comp)
        >>> print(names)
        ['folded', 'unfolded', 'intermediate']
        """
        # Multiclass comparison
        if "selectors" in sub_comp:
            return MetadataBuilderHelper._extract_multiclass_names(sub_comp)

        # Binary or one-vs-rest comparison (Note: Pairwise same structure as binary)
        if "group1_selectors" in sub_comp and "group2_selectors" in sub_comp:
            if len(sub_comp["group2_selectors"]) == 1:
                return MetadataBuilderHelper._extract_binary_names(sub_comp)
            else:
                return MetadataBuilderHelper._extract_one_vs_rest_names(sub_comp)

        # Fallback
        return ["Class 0", "Class 1"]