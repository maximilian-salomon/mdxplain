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
Analysis runner helper for feature importance operations.

This module provides helper methods for running feature importance analyses
on individual sub-comparisons, extracting common logic from
FeatureImportanceManager methods.
"""

from typing import Dict, Any
import numpy as np

from ..analyzer_types.interfaces.analyzer_type_base import AnalyzerTypeBase
from ..entities.feature_importance_data import FeatureImportanceData
from .metadata_builder_helper import MetadataBuilderHelper
from ...pipeline.helper.comparison_data_helper import ComparisonDataHelper


class AnalysisRunnerHelper:
    """
    Helper class for running feature importance analyses.
    
    Provides static methods for executing feature importance analysis on
    individual sub-comparisons and processing the results. These methods
    extract common logic from FeatureImportanceManager to improve code
    organization and reusability.
    
    Examples:
    ---------
    >>> # Run analysis on single sub-comparison
    >>> result = AnalysisRunnerHelper.run_single_analysis(
    ...     analyzer_type, X, y, sub_comp
    ... )
    
    >>> # Process and store analysis result
    >>> AnalysisRunnerHelper.store_analysis_result(
    ...     fi_data, result, metadata
    ... )
    """
    
    @staticmethod
    def run_single_analysis(
        analyzer_type: AnalyzerTypeBase,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run feature importance analysis on a single sub-comparison.
        
        Executes the analyzer on the provided data and returns the complete
        result including importances, model, and metadata.
        
        Parameters:
        -----------
        analyzer_type : AnalyzerTypeBase
            Analyzer instance to run the analysis
        X : np.ndarray
            Feature matrix for the sub-comparison
        y : np.ndarray
            Label array for the sub-comparison
            
        Returns:
        --------
        Dict[str, Any]
            Analysis result containing importances, model, and metadata
            
        Raises:
        -------
        ValueError
            If analysis computation fails
            
        Examples:
        ---------
        >>> result = AnalysisRunnerHelper.run_single_analysis(
        ...     decision_tree_analyzer, X, y, sub_comp_dict
        ... )
        >>> print("importances" in result)  # True
        >>> print("model" in result)  # True
        """
        # Run feature importance analysis
        result = analyzer_type.compute(X, y)
        
        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError("Analyzer must return a dictionary")
        
        if "importances" not in result:
            raise ValueError("Analyzer result must contain 'importances' key")
        
        if "metadata" not in result:
            result["metadata"] = {}
        
        return result
    
    @staticmethod
    def store_analysis_result(
        fi_data: FeatureImportanceData,
        result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store analysis result in FeatureImportanceData object.
        
        Takes the analysis result and metadata and stores them properly
        in the FeatureImportanceData container.
        
        Parameters:
        -----------
        fi_data : FeatureImportanceData
            Feature importance data container to store result in
        result : Dict[str, Any]
            Analysis result from analyzer containing importances
        metadata : Dict[str, Any]
            Metadata dictionary describing the analysis
            
        Returns:
        --------
        None
            Stores result in the fi_data object
            
        Examples:
        ---------
        >>> AnalysisRunnerHelper.store_analysis_result(
        ...     fi_data, analysis_result, metadata_dict
        ... )
        """
        # Store the result using the data object's method
        fi_data.add_comparison_result(result["importances"], metadata)
    
    @staticmethod
    def run_comparison_analysis(
        pipeline_data,
        comp_data,
        analyzer_type: AnalyzerTypeBase,
        analysis_name: str
    ) -> FeatureImportanceData:
        """
        Run analysis on all sub-comparisons in a comparison.
        
        Processes all sub-comparisons within a comparison object, running
        the specified analyzer on each one and collecting results.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing data and comparisons
        comp_data : ComparisonData
            Comparison data object containing sub-comparisons
        analyzer_type : AnalyzerTypeBase
            Analyzer instance to use for all sub-comparisons
        analysis_name : str
            Name for the analysis (for metadata)
            
        Returns:
        --------
        FeatureImportanceData
            Complete feature importance data with all sub-comparison results
            
        Examples:
        ---------
        >>> fi_data = AnalysisRunnerHelper.run_comparison_analysis(
        ...     pipeline_data, comp_data, analyzer, "my_analysis"
        ... )
        >>> print(len(fi_data.data))  # Number of sub-comparisons analyzed
        """
        # Create feature importance data container
        fi_data = FeatureImportanceData(analysis_name)
        fi_data.analyzer_type = analyzer_type.get_type_name()
        fi_data.comparison_name = comp_data.name
        fi_data.feature_selector = comp_data.feature_selector

        # Run analysis on each sub-comparison
        for sub_comp in comp_data.sub_comparisons:
            # Get data for this sub-comparison using ComparisonDataHelper
            X, y = ComparisonDataHelper.get_sub_comparison_data(
                pipeline_data, comp_data, sub_comp["name"]
            )
            
            # Run single analysis
            result = AnalysisRunnerHelper.run_single_analysis(
                analyzer_type, X, y
            )
            
            # Build metadata
            metadata = MetadataBuilderHelper.build_analysis_metadata(
                sub_comp, analyzer_type, X, y, result
            )
            
            # Store result
            AnalysisRunnerHelper.store_analysis_result(fi_data, result, metadata)
        
        return fi_data
