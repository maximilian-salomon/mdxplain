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
Validation helper for feature importance management.

This module provides the FeatureImportanceValidationHelper class with static methods
for validating feature importance inputs, analyses, and dependencies. Extracted from
FeatureImportanceManager to improve code organization and testability.
"""

from ..analyzer_types.interfaces.analyzer_type_base import AnalyzerTypeBase


class FeatureImportanceValidationHelper:
    """
    Static helper class for feature importance validation operations.
    
    Provides validation methods for analysis names, analyzer types, comparisons,
    and dependencies. All methods are static and stateless.
    """

    @staticmethod
    def validate_analysis_name(pipeline_data, analysis_name: str, force: bool) -> None:
        """
        Validate analysis name doesn't already exist unless force is used.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing existing analyses
        analysis_name : str
            Name of the analysis to validate
        force : bool
            Whether to allow overwriting existing analysis
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if name exists and force=False
            
        Raises:
        -------
        ValueError
            If analysis name already exists and force=False
        """
        if analysis_name in pipeline_data.feature_importance_data and not force:
            raise ValueError(
                f"Feature importance analysis '{analysis_name}' already exists. "
                "Use force=True to overwrite."
            )

    @staticmethod
    def validate_analysis_exists(pipeline_data, analysis_name: str) -> None:
        """
        Validate that feature importance analysis with given name exists.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing analysis data
        analysis_name : str
            Name of the analysis to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if analysis not found
            
        Raises:
        -------
        ValueError
            If analysis not found in pipeline_data
        """
        if analysis_name not in pipeline_data.feature_importance_data:
            available = list(pipeline_data.feature_importance_data.keys())
            raise ValueError(
                f"Feature importance analysis '{analysis_name}' not found. "
                f"Available analyses: {available}"
            )

    @staticmethod
    def validate_comparison_exists(pipeline_data, comparison_name: str) -> None:
        """
        Validate that comparison with given name exists.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing comparison data
        comparison_name : str
            Name of the comparison to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if comparison not found
            
        Raises:
        -------
        ValueError
            If comparison not found in pipeline_data
        """
        if comparison_name not in pipeline_data.comparison_data:
            available = list(pipeline_data.comparison_data.keys())
            raise ValueError(
                f"Comparison '{comparison_name}' not found. "
                f"Available comparisons: {available}"
            )

    @staticmethod
    def validate_analyzer_type(analyzer_type) -> None:
        """
        Validate analyzer type instance and its required interface.
        
        Checks that the analyzer_type is properly instantiated and implements
        the required methods for feature importance analysis. Validates both
        type inheritance and method availability.
        
        Parameters:
        -----------
        analyzer_type : AnalyzerTypeBase
            Analyzer instance to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises errors for invalid analyzer types
            
        Raises:
        -------
        TypeError
            If analyzer_type is not an instance of AnalyzerTypeBase
        ValueError
            If analyzer_type doesn't implement required methods
        """        
        if not isinstance(analyzer_type, AnalyzerTypeBase):
            raise TypeError(
                f"analyzer_type must be an instance of AnalyzerTypeBase, "
                f"got {type(analyzer_type).__name__}"
            )

        if not hasattr(analyzer_type, "compute") or not callable(
            getattr(analyzer_type, "compute")
        ):
            raise ValueError("analyzer_type must implement compute() method")

        if not hasattr(analyzer_type, "get_type_name") or not callable(
            getattr(analyzer_type, "get_type_name")
        ):
            raise ValueError("analyzer_type must implement get_type_name() method")
    