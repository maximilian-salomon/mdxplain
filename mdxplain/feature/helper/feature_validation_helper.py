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
Feature validation helper for parameter and state validation.

This module provides utilities for validating feature-related parameters
and states in the FeatureManager, separating validation logic from core
business logic.
"""
from __future__ import annotations

import numbers
from typing import Optional, Union, List, TYPE_CHECKING

from ..feature_type.interfaces.feature_type_base import FeatureTypeBase

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


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

        Parameters
        ----------
        threshold_min : float or None
            Minimum threshold value to validate
        threshold_max : float or None
            Maximum threshold value to validate
        metric : str
            Metric name for context in error messages

        Returns
        -------
        None
            Validates parameters or raises ValueError

        Raises
        ------
        ValueError
            If threshold parameters are invalid (wrong type or invalid range)

        Examples
        --------
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
    def validate_feature_exists(pipeline_data: PipelineData, feature_key: str) -> bool:
        """
        Check if feature exists and has computed data.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to check
        feature_key : str
            Feature key to validate

        Returns
        -------
        bool
            True if feature exists and has data, False otherwise

        Examples
        --------
        >>> exists = FeatureValidationHelper.validate_feature_exists(
        ...     pipeline_data, "distances"
        ... )
        """
        return (
            feature_key in pipeline_data.feature_data
            and pipeline_data.feature_data[feature_key].data is not None
        )

    @staticmethod
    def validate_dependencies(pipeline_data: PipelineData, feature_type: FeatureTypeBase, feature_key: str, traj_indices: List[int]) -> None:
        """
        Validate that all feature dependencies are available.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to check
        feature_type : FeatureTypeBase
            Feature type object to check dependencies for
        feature_key : str
            Feature key for error messages

        Returns
        -------
        None
            Validates dependencies or raises ValueError

        Raises
        ------
        ValueError
            If required dependency is missing

        Examples
        --------
        >>> FeatureValidationHelper.validate_dependencies(
        ...     pipeline_data, contacts_feature, "contacts"
        ... )
        """
        dependencies = feature_type.get_dependencies()
        for dep in dependencies:
            if dep not in pipeline_data.feature_data:
                raise ValueError(f"Dependency '{dep}' must be computed before '{feature_key}'.")
            
            # Check if dependency exists for ALL specified trajectories
            dep_dict = pipeline_data.feature_data[dep]
            missing_trajs = [
                traj_idx for traj_idx in traj_indices
                if traj_idx not in dep_dict or dep_dict[traj_idx].get_data() is None
            ]
            
            if missing_trajs:
                raise ValueError(
                    f"Dependency '{dep}' must be computed for trajectories {missing_trajs} "
                    f"before '{feature_key}'."
                )

    @staticmethod
    def validate_computation_requirements(pipeline_data: PipelineData, feature_type: FeatureTypeBase) -> None:
        """
        Validate that all requirements for feature computation are met.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to validate
        feature_type : FeatureTypeBase
            Feature type object to validate

        Returns
        -------
        None
            Validates requirements or raises ValueError

        Raises
        ------
        ValueError
            If trajectory labels are not set or trajectories are not loaded

        Examples
        --------
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
    def validate_reduction_state(pipeline_data: PipelineData, feature_key: str, traj_indices: List[int]) -> None:
        """
        Validate that feature is ready for reduction for specified trajectories.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to validate
        feature_key : str
            Feature key to validate
        traj_indices : List[int]
            List of trajectory indices to validate

        Returns
        -------
        None
            Validates state or raises ValueError

        Raises
        ------
        ValueError
            If feature has no data or reduction already performed for specified trajectories

        Examples
        --------
        >>> FeatureValidationHelper.validate_reduction_state(
        ...     pipeline_data, "distances", [0, 1, 2]
        ... )
        """
        # Check if feature exists and has trajectory data
        if feature_key not in pipeline_data.feature_data:
            raise ValueError(
                f"Feature '{feature_key}' not found. Add the feature to the trajectory data first."
            )
        
        feature_traj_dict = pipeline_data.feature_data[feature_key]
        if not isinstance(feature_traj_dict, dict) or not feature_traj_dict:
            raise ValueError(
                "No trajectory data available. Add the feature to the trajectory data first."
            )

        # Validate using helper methods
        FeatureValidationHelper._check_trajectories_exist(feature_key, feature_traj_dict, traj_indices)
        FeatureValidationHelper._check_trajectories_have_data(feature_key, feature_traj_dict, traj_indices)
        FeatureValidationHelper._check_no_existing_reduction(feature_key, feature_traj_dict, traj_indices)

    @staticmethod
    def _check_trajectories_exist(feature_key: str, feature_traj_dict: dict, traj_indices: List[int]) -> None:
        """
        Check if feature exists for all specified trajectories.

        Parameters
        ----------
        feature_key : str
            Feature key for error messages
        feature_traj_dict : dict
            Feature trajectory dictionary
        traj_indices : List[int]
            List of trajectory indices to validate

        Returns
        -------
        None
            Validates existence or raises ValueError

        Raises
        ------
        ValueError
            If feature not computed for any of the specified trajectories
        """
        missing_trajectories = [
            traj_idx for traj_idx in traj_indices 
            if traj_idx not in feature_traj_dict
        ]
        
        if missing_trajectories:
            raise ValueError(
                f"Feature '{feature_key}' not computed for trajectories {missing_trajectories}. "
                f"Add the feature to these trajectories first."
            )

    @staticmethod
    def _check_trajectories_have_data(feature_key: str, feature_traj_dict: dict, traj_indices: List[int]) -> None:
        """
        Check if trajectories have actual feature data.

        Parameters
        ----------
        feature_key : str
            Feature key for error messages
        feature_traj_dict : dict
            Feature trajectory dictionary
        traj_indices : List[int]
            List of trajectory indices to validate

        Returns
        -------
        None
            Validates data existence or raises ValueError

        Raises
        ------
        ValueError
            If any trajectory has no computed feature data
        """
        trajectories_without_data = [
            traj_idx for traj_idx in traj_indices
            if traj_idx in feature_traj_dict and feature_traj_dict[traj_idx].data is None
        ]
        
        if trajectories_without_data:
            raise ValueError(
                f"No data available for feature '{feature_key}' in trajectories {trajectories_without_data}. "
                f"Compute the feature for these trajectories first."
            )

    @staticmethod
    def _check_no_existing_reduction(feature_key: str, feature_traj_dict: dict, traj_indices: List[int]) -> None:
        """
        Check if reduction has already been performed.

        Parameters
        ----------
        feature_key : str
            Feature key for error messages
        feature_traj_dict : dict
            Feature trajectory dictionary
        traj_indices : List[int]
            List of trajectory indices to validate

        Returns
        -------
        None
            Validates no existing reduction or raises ValueError

        Raises
        ------
        ValueError
            If reduction already performed for any of the specified trajectories
        """
        trajectories_with_reduced_data = [
            traj_idx for traj_idx in traj_indices
            if traj_idx in feature_traj_dict and feature_traj_dict[traj_idx].reduced_data is not None
        ]
        
        if trajectories_with_reduced_data:
            raise ValueError(
                f"Reduction already performed for feature '{feature_key}' in trajectories {trajectories_with_reduced_data}. "
                f"Reset the reduction first using reset_reduction()."
            )
