# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
Feature reset helper for managing feature data cleanup.

This module provides utilities for resetting and clearing feature data
in the FeatureManager, handling both single and multiple feature resets
with different error handling modes.
"""

from typing import Callable, List, Union, Any

from .feature_computation_helper import FeatureComputationHelper
from ...utils.data_utils import DataUtils


class FeatureResetHelper:
    """
    Helper class for feature reset operations.

    Provides static methods for handling feature reset logic, including
    selective resets, error handling modes, and status reporting.
    """

    @staticmethod
    def reset_all_features(pipeline_data) -> None:
        """
        Reset all features in the pipeline data.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to reset

        Returns:
        --------
        None
            Clears all feature data and prints summary

        Examples:
        ---------
        >>> FeatureResetHelper.reset_all_features(pipeline_data)
        Reset 3 feature(s): distances, contacts, angles
        All feature data has been cleared. Features must be recalculated.
        """
        feature_list = list(pipeline_data.feature_data.keys())
        pipeline_data.feature_data.clear()
        print(f"Reset {len(feature_list)} feature(s): {', '.join(feature_list)}")
        print("All feature data has been cleared. Features must be recalculated.")

    @staticmethod
    def reset_specific_features(
        pipeline_data,
        feature_types: List[Any],
        strict: bool = False,
    ) -> None:
        """
        Reset specific feature types with optional strict error handling.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to reset
        feature_types : List[Any]
            List of feature types to reset (strings, instances, or classes)
        strict : bool, default=False
            Whether to raise errors for non-existent features

        Returns:
        --------
        None
            Resets specified features and prints status

        Raises:
        -------
        ValueError
            If strict=True and feature types are not found

        Examples:
        ---------
        >>> FeatureResetHelper.reset_specific_features(
        ...     pipeline_data, ["distances", "contacts"], get_key_fn
        ... )
        Reset 2 feature(s): distances, contacts
        """
        reset_keys = []
        missing_keys = []

        for ft in feature_types:
            key = DataUtils.get_type_key(ft)
            if key in pipeline_data.feature_data:
                del pipeline_data.feature_data[key]
                reset_keys.append(key)
            else:
                missing_keys.append(key)

        # Handle missing keys based on strict mode
        FeatureResetHelper._handle_missing_keys(missing_keys, strict, pipeline_data)

        # Report results
        FeatureResetHelper._report_reset_results(reset_keys, missing_keys, strict)

    @staticmethod
    def _handle_missing_keys(
        missing_keys: List[str], 
        strict: bool, 
        pipeline_data
    ) -> None:
        """
        Handle missing feature keys based on strict mode.

        Parameters:
        -----------
        missing_keys : List[str]
            List of missing feature keys
        strict : bool
            Whether to raise errors for missing keys
        pipeline_data : PipelineData
            Pipeline data for available features info

        Returns:
        --------
        None
            Handles missing keys or raises ValueError

        Raises:
        -------
        ValueError
            If strict=True and keys are missing
        """
        if missing_keys and strict:
            available_features = list(pipeline_data.feature_data.keys())
            raise ValueError(
                f"Feature type(s) not found: {', '.join(missing_keys)}. "
                f"Available features: {available_features}"
            )

    @staticmethod
    def _report_reset_results(
        reset_keys: List[str], 
        missing_keys: List[str], 
        strict: bool
    ) -> None:
        """
        Report results of reset operation.

        Parameters:
        -----------
        reset_keys : List[str]
            List of successfully reset feature keys
        missing_keys : List[str]
            List of missing feature keys
        strict : bool
            Whether strict mode was used

        Returns:
        --------
        None
            Prints appropriate status messages
        """
        if reset_keys:
            print(f"Reset {len(reset_keys)} feature(s): {', '.join(reset_keys)}")

        if missing_keys and not strict:
            print(
                f"WARNING: {len(missing_keys)} feature(s) not found: {', '.join(missing_keys)}"
            )

        if not reset_keys and not missing_keys:
            print("No features to reset.")

    @staticmethod
    def normalize_feature_types(feature_type: Union[Any, List[Any]]) -> List[Any]:
        """
        Normalize feature_type parameter to list format.

        Parameters:
        -----------
        feature_type : Any or List[Any]
            Single feature type or list of feature types

        Returns:
        --------
        List[Any]
            Normalized list of feature types

        Examples:
        ---------
        >>> types = FeatureResetHelper.normalize_feature_types("distances")
        >>> print(types)
        ["distances"]

        >>> types = FeatureResetHelper.normalize_feature_types(["distances", "contacts"])
        >>> print(types)
        ["distances", "contacts"]
        """
        if not isinstance(feature_type, list):
            return [feature_type]
        return feature_type

    @staticmethod
    def check_has_features(pipeline_data) -> bool:
        """
        Check if pipeline data has any features to reset.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to check

        Returns:
        --------
        bool
            True if features exist, False otherwise

        Examples:
        ---------
        >>> has_features = FeatureResetHelper.check_has_features(pipeline_data)
        >>> if not has_features:
        ...     print("No features to reset.")
        """
        return bool(pipeline_data.feature_data)
