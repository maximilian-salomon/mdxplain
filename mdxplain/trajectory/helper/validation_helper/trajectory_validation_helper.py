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
Trajectory validation helper for pipeline operations.

Provides validation logic for trajectory-changing operations in the pipeline,
ensuring that existing features are properly handled before modifications.
"""

from typing import List


class TrajectoryValidationHelper:
    """Helper class for validating trajectory operations in pipeline context."""

    @staticmethod
    def check_features_before_trajectory_changes(
        pipeline_data, force: bool, operation_name: str, traj_indices: List[int] = None
    ):
        """
        Check if features exist before trajectory modification.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        force : bool
            Whether to force the operation despite existing features
        operation_name : str
            Name of the operation for error messages
        traj_indices : List[int], optional
            Trajectory indices to check for features.
            If None, checks if ANY features exist at all.

        Raises:
        -------
        ValueError
            If features exist and force=False
        """
        if not pipeline_data.feature_data:
            return

        if traj_indices is None:
            TrajectoryValidationHelper._check_any_features_exist(
                pipeline_data, force, operation_name
            )
        else:
            TrajectoryValidationHelper._check_specific_trajectory_features(
                pipeline_data, force, operation_name, traj_indices
            )

    @staticmethod
    def _check_any_features_exist(pipeline_data, force: bool, operation_name: str):
        """
        Check if any features exist in pipeline.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        force : bool
            Whether to force the operation despite existing features
        operation_name : str
            Name of the operation for error messages

        Raises:
        -------
        ValueError
            If features exist and force=False
        """
        any_features_exist = any(
            bool(feature_dict) 
            for feature_dict in pipeline_data.feature_data.values()
        )
        
        if any_features_exist and not force:
            raise ValueError(
                f"Features exist in pipeline. "
                f"Operation '{operation_name}' will invalidate them. "
                f"So you need to recalculate features after this operation. "
                f"Use force=True to proceed anyway."
            )
        elif any_features_exist and force:
            affected_features = [
                feature_type for feature_type, feature_dict 
                in pipeline_data.feature_data.items() 
                if feature_dict
            ]
            print(
                f"WARNING: Operation '{operation_name}' will invalidate "
                f"existing features. Affected feature types: {affected_features}. "
                f"These features must be recalculated."
            )

    @staticmethod
    def _check_specific_trajectory_features(
        pipeline_data, force: bool, operation_name: str, traj_indices: List[int]
    ):
        """
        Check if specific trajectories have features.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        force : bool
            Whether to force the operation despite existing features
        operation_name : str
            Name of the operation for error messages
        traj_indices : List[int]
            Trajectory indices to check for features

        Raises:
        -------
        ValueError
            If features exist for trajectories and force=False
        """
        trajectories_with_features = []
        
        for traj_idx in traj_indices:
            for feature_type, feature_dict in pipeline_data.feature_data.items():
                if traj_idx in feature_dict:
                    trajectories_with_features.append(traj_idx)
                    break
        
        unique_trajectories = list(set(trajectories_with_features))
        
        if not unique_trajectories:
            return
        
        if not force:
            raise ValueError(
                f"Features exist for trajectories {unique_trajectories}. "
                f"Operation '{operation_name}' will invalidate them. "
                f"So you need to recalculate features after this operation. "
                f"Use force=True to proceed anyway."
            )
        
        affected_features = []
        for feature_type in pipeline_data.feature_data.keys():
            for traj_idx in unique_trajectories:
                if traj_idx in pipeline_data.feature_data[feature_type]:
                    affected_features.append(feature_type)
                    break
        
        print(
            f"WARNING: Operation '{operation_name}' will invalidate features for "
            f"trajectories {unique_trajectories}. "
            f"Affected feature types: {affected_features}. "
            f"These features must be recalculated."
        )
