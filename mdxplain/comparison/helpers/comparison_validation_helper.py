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
Validation helper for comparison management.

This module provides the ComparisonValidationHelper class with static methods
for validating comparison inputs, modes, and dependencies. Extracted from
ComparisonManager to improve code organization and testability.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class ComparisonValidationHelper:
    """
    Static helper class for comparison validation operations.
    
    Provides validation methods for comparison names, modes, selectors,
    and dependencies. All methods are static and stateless.
    """

    @staticmethod
    def validate_comparison_name(pipeline_data: PipelineData, name: str) -> None:
        """
        Validate that comparison name doesn't already exist.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing existing comparisons
        name : str
            Name of the comparison to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if name already exists
        """
        if name in pipeline_data.comparison_data:
            raise ValueError(f"Comparison '{name}' already exists.")

    @staticmethod
    def validate_comparison_exists(pipeline_data: PipelineData, name: str) -> None:
        """
        Validate that a comparison with given name exists.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing comparison data
        name : str
            Name of the comparison to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if comparison not found
        """
        if name not in pipeline_data.comparison_data:
            available = list(pipeline_data.comparison_data.keys())
            raise ValueError(
                f"Comparison '{name}' not found. Available comparisons: {available}"
            )

    @staticmethod
    def validate_mode(mode: str) -> None:
        """
        Validate that comparison mode is supported.
        
        Parameters:
        -----------
        mode : str
            Comparison mode to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if mode invalid
        """
        valid_modes = ["binary", "pairwise", "one_vs_rest", "multiclass"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")

    @staticmethod
    def validate_feature_selector(pipeline_data: PipelineData, feature_selector: str) -> None:
        """
        Validate that a feature selector with given name exists.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature selector data
        feature_selector : str
            Name of the feature selector to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if selector not found
        """
        if feature_selector not in pipeline_data.selected_feature_data:
            available = list(pipeline_data.selected_feature_data.keys())
            raise ValueError(
                f"Feature selector '{feature_selector}' not found. "
                f"Available selectors: {available}"
            )

    @staticmethod
    def validate_data_selectors(pipeline_data: PipelineData, data_selectors: List[str]) -> None:
        """
        Validate that all specified data selectors exist.
        
        Checks that the list is non-empty and that all data selector names
        are found in the pipeline data. Reports any missing selectors.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing data selector data
        data_selectors : List[str]
            List of data selector names to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if selectors missing or empty
        """
        if not data_selectors:
            raise ValueError("At least one data selector is required")

        available_selectors = set(pipeline_data.data_selector_data.keys())
        missing_selectors = []

        for selector in data_selectors:
            if selector not in available_selectors:
                missing_selectors.append(selector)

        if missing_selectors:
            raise ValueError(
                f"Data selectors not found: {missing_selectors}. "
                f"Available selectors: {list(available_selectors)}"
            )
        