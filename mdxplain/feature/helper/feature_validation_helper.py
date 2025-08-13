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
Feature validation helper for parameter and state validation.

This module provides utilities for validating feature-related parameters
and states in the FeatureManager, separating validation logic from core
business logic.
"""

import numbers
from typing import Optional, Union


class FeatureValidationHelper:
    """
    Helper class for feature parameter and state validation.

    Provides static methods for validating various types of parameters
    used in feature management, including threshold parameters, feature
    existence checks, and dependency validation.
    """

    @staticmethod
    def validate_threshold_parameters(
        threshold_min: Optional[Union[int, float]],
        threshold_max: Optional[Union[int, float]],
        metric: str,
    ) -> None:
        """
        Validate threshold parameters for data reduction.

        Parameters:
        -----------
        threshold_min : float or None
            Minimum threshold value to validate
        threshold_max : float or None
            Maximum threshold value to validate
        metric : str
            Metric name for context in error messages

        Returns:
        --------
        None
            Validates parameters or raises ValueError

        Raises:
        -------
        ValueError
            If threshold parameters are invalid (wrong type or invalid range)

        Examples:
        ---------
        >>> FeatureValidationHelper.validate_threshold_parameters(0.1, 0.9, "cv")
        >>> # No error - valid parameters

        >>> FeatureValidationHelper.validate_threshold_parameters(0.9, 0.1, "cv")
        ValueError: Invalid threshold range...
        """
        # Type validation - supports all numeric types including numpy scalars
        if threshold_min is not None and not isinstance(threshold_min, numbers.Number):
            raise ValueError(
                f"threshold_min must be a number, got {type(threshold_min).__name__}"
            )

        if threshold_max is not None and not isinstance(threshold_max, numbers.Number):
            raise ValueError(
                f"threshold_max must be a number, got {type(threshold_max).__name__}"
            )

        # Range validation
        if threshold_min is not None and threshold_max is not None:
            if threshold_min > threshold_max:
                raise ValueError(
                    f"Invalid threshold range: threshold_min ({threshold_min}) > "
                    f"threshold_max ({threshold_max}) for metric '{metric}'. "
                    f"threshold_min must be <= threshold_max."
                )
            if threshold_min == threshold_max:
                print(
                    f"WARNING: threshold_min == threshold_max ({threshold_min}) "
                    f"for metric '{metric}'. This may result in very low data retention."
                )

    @staticmethod
    def validate_feature_exists(pipeline_data, feature_key: str) -> bool:
        """
        Check if feature exists and has computed data.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to check
        feature_key : str
            Feature key to validate

        Returns:
        --------
        bool
            True if feature exists and has data, False otherwise

        Examples:
        ---------
        >>> exists = FeatureValidationHelper.validate_feature_exists(
        ...     pipeline_data, "distances"
        ... )
        """
        return (
            feature_key in pipeline_data.feature_data
            and pipeline_data.feature_data[feature_key].data is not None
        )

    @staticmethod
    def validate_dependencies(pipeline_data, feature_type, feature_key: str) -> None:
        """
        Validate that all feature dependencies are available.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to check
        feature_type : FeatureTypeBase
            Feature type object to check dependencies for
        feature_key : str
            Feature key for error messages

        Returns:
        --------
        None
            Validates dependencies or raises ValueError

        Raises:
        -------
        ValueError
            If required dependency is missing

        Examples:
        ---------
        >>> FeatureValidationHelper.validate_dependencies(
        ...     pipeline_data, contacts_feature, "contacts"
        ... )
        """
        dependencies = feature_type.get_dependencies()
        for dep in dependencies:
            if (
                dep not in pipeline_data.feature_data
                or pipeline_data.feature_data[dep].get_data() is None
            ):
                raise ValueError(
                    f"Dependency '{dep}' must be computed before '{feature_key}'."
                )

    @staticmethod
    def validate_computation_requirements(pipeline_data, feature_type) -> None:
        """
        Validate that all requirements for feature computation are met.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to validate
        feature_type : FeatureTypeBase
            Feature type object to validate

        Returns:
        --------
        None
            Validates requirements or raises ValueError

        Raises:
        -------
        ValueError
            If trajectory labels are not set or trajectories are not loaded

        Examples:
        ---------
        >>> FeatureValidationHelper.validate_computation_requirements(
        ...     pipeline_data, distances_feature
        ... )
        """
        if pipeline_data.trajectory_data.res_label_data is None:
            raise ValueError("Trajectory labels must be set before computing features.")

        if (
            feature_type.get_input() is None
            and pipeline_data.trajectory_data.trajectories is None
        ):
            raise ValueError("Trajectories must be loaded before computing features.")

    @staticmethod
    def validate_reduction_state(pipeline_data, feature_key: str) -> None:
        """
        Validate that feature is ready for reduction.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to validate
        feature_key : str
            Feature key to validate

        Returns:
        --------
        None
            Validates state or raises ValueError

        Raises:
        -------
        ValueError
            If feature has no data or reduction already performed

        Examples:
        ---------
        >>> FeatureValidationHelper.validate_reduction_state(
        ...     pipeline_data, "distances"
        ... )
        """
        if pipeline_data.feature_data[feature_key].data is None:
            raise ValueError(
                "No data available. Add the feature to the trajectory data first."
            )

        if pipeline_data.feature_data[feature_key].reduced_data is not None:
            raise ValueError(
                "Reduction already performed. Reset the reduction first using reset_reduction()."
            )
