# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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
Feature binding helper for analysis method binding.

This module provides utilities for binding analysis methods from calculators
to feature data objects with automatic data selection based on reduction state.
"""

from typing import Set, Callable, Any

from ..entities.feature_data import FeatureData
from .bound_method import BoundMethod
from .bound_stats import BoundStats


class FeatureBindingHelper:
    """
    Helper class for binding analysis methods to feature data.

    Provides static methods for creating bound methods that automatically
    use the appropriate data (original or reduced) based on feature state.
    """

    @staticmethod
    def bind_stats_methods(feature_data: FeatureData) -> None:
        """
        Bind analysis methods from calculator to feature_data.analysis.

        Creates bound methods that automatically pass the current data
        (reduced if available, otherwise original) to calculator analysis methods.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object to bind methods to

        Returns:
        --------
        None
            Creates feature_data.analysis object with bound methods

        Examples:
        ---------
        >>> FeatureBindingHelper.bind_stats_methods(feature_data)
        >>> # Now feature_data.analysis.mean() works automatically
        >>> mean_values = feature_data.analysis.mean()
        """
        if not hasattr(feature_data.feature_type.calculator, "analysis"):
            return

        # Create analysis object to hold bound methods
        feature_data.analysis = BoundStats()

        # Get methods that require full data
        requires_full_data = getattr(
            feature_data.feature_type.calculator.analysis, 
            "REQUIRES_FULL_DATA", 
            set()
        )

        # Bind each analysis method
        FeatureBindingHelper._bind_analysis_methods(
            feature_data, requires_full_data
        )

    @staticmethod
    def _bind_analysis_methods(
        feature_data: FeatureData, 
        requires_full_data: Set[str]
    ) -> None:
        """
        Bind individual analysis methods to feature data.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object to bind to
        requires_full_data : Set[str]
            Set of method names that require full data

        Returns:
        --------
        None
            Binds methods to feature_data.analysis
        """
        calculator_analysis = feature_data.feature_type.calculator.analysis

        for method_name in dir(calculator_analysis):
            if FeatureBindingHelper._should_bind_method(
                method_name, calculator_analysis
            ):
                original_method = getattr(calculator_analysis, method_name)
                bound_method = FeatureBindingHelper._create_bound_method(
                    feature_data, original_method, method_name, requires_full_data
                )
                setattr(feature_data.analysis, method_name, bound_method)

    @staticmethod
    def _should_bind_method(method_name: str, calculator_analysis: Any) -> bool:
        """
        Check if method should be bound (public and callable).

        Parameters:
        -----------
        method_name : str
            Name of method to check
        calculator_analysis : Any
            Calculator analysis object

        Returns:
        --------
        bool
            True if method should be bound, False otherwise
        """
        return (
            not method_name.startswith("_") 
            and callable(getattr(calculator_analysis, method_name))
        )

    @staticmethod
    def _create_bound_method(
        feature_data: FeatureData,
        original_method: Callable,
        method_name: str,
        requires_full_data: Set[str],
    ) -> BoundMethod:
        """
        Create bound method with automatic data selection.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object with data
        original_method : Callable
            Original method to wrap
        method_name : str
            Name of method being wrapped
        requires_full_data : Set[str]
            Set of methods that require full data

        Returns:
        --------
        BoundMethod
            Bound method that automatically uses appropriate data
        """
        return BoundMethod(feature_data, original_method, method_name, requires_full_data)

    @staticmethod
    def repair_bound_methods(feature_data: FeatureData) -> None:
        """
        Repair BoundMethod objects after unpickling.
        
        This method restores the feature_data and original_method references
        that were removed during pickling.
        
        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object with analysis attribute to repair
        
        Returns:
        --------
        None
            Repairs BoundMethod objects in-place
        """
        if not hasattr(feature_data, 'analysis'):
            return
            
        if not hasattr(feature_data.feature_type.calculator, 'analysis'):
            return
            
        calculator_analysis = feature_data.feature_type.calculator.analysis
        
        for method_name in dir(feature_data.analysis):
            if method_name.startswith('_'):
                continue
                
            bound_method = getattr(feature_data.analysis, method_name)
            if isinstance(bound_method, BoundMethod):
                # Restore missing references
                bound_method.feature_data = feature_data
                
                # Find the original method
                # Remove "compute_" prefix if present for method lookup
                original_method_name = method_name
                if not hasattr(calculator_analysis, original_method_name):
                    original_method_name = f"compute_{method_name}"
                
                if hasattr(calculator_analysis, original_method_name):
                    bound_method.original_method = getattr(calculator_analysis, original_method_name)
