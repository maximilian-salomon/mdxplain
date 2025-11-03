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

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ....pipeline.entities.pipeline_data import PipelineData
    from ...entities.trajectory_data import TrajectoryData


class TrajectoryValidationHelper:
    """Helper class for validating trajectory operations in pipeline context."""

    @staticmethod
    def check_features_before_trajectory_changes(
        pipeline_data: PipelineData, force: bool, operation_name: str, traj_indices: Optional[List[int]] = None
    ) -> None:
        """
        Check if features exist before trajectory modification.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        force : bool
            Whether to force the operation despite existing features
        operation_name : str
            Name of the operation for error messages
        traj_indices : List[int], optional
            Trajectory indices to check for features.
            If None, checks if ANY features exist at all.

        Raises
        ------
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
    def _check_any_features_exist(pipeline_data: PipelineData, force: bool, operation_name: str) -> None:
        """
        Check if any features exist in pipeline.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        force : bool
            Whether to force the operation despite existing features
        operation_name : str
            Name of the operation for error messages

        Raises
        ------
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
    def validate_slice_parameters(frames: Optional[Union[int, slice, List[int]]]) -> None:
        """
        Validate parameters for trajectory slicing.
        
        Parameters
        ----------
        frames : int, slice, list, or None
            Frame specification for slicing
        
        Returns
        -------
        None
            Performs validation, does not return value
        
        Raises
        ------
        ValueError
            If frames parameter is invalid
        
        Examples
        --------
        >>> TrajectoryValidationHelper.validate_slice_parameters(1000)  # OK
        >>> TrajectoryValidationHelper.validate_slice_parameters(slice(100, 500))  # OK
        >>> TrajectoryValidationHelper.validate_slice_parameters([0, 10, 20])  # OK
        >>> TrajectoryValidationHelper.validate_slice_parameters(-100)  # ValueError
        """
        if frames is not None:
            if isinstance(frames, int) and frames < 0:
                raise ValueError("Frame count must be non-negative.")
            elif isinstance(frames, slice):
                if frames.start is not None and frames.start < 0:
                    raise ValueError("Slice start must be non-negative.")
                if frames.stop is not None and frames.stop < 0:
                    raise ValueError("Slice stop must be non-negative.")
            elif isinstance(frames, (list, np.ndarray)):
                if len(frames) == 0:
                    raise ValueError("Frame index list cannot be empty.")

    @staticmethod
    def validate_data_selector(pipeline_data: PipelineData, data_selector: str) -> Any:
        """
        Validate DataSelector exists and has data.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        data_selector : str
            Name of DataSelector to validate
        
        Returns
        -------
        DataSelectorData
            The validated selector data object
        
        Raises
        ------
        ValueError
            If DataSelector does not exist or has no data
            
        Examples
        --------
        >>> selector_data = TrajectoryValidationHelper.validate_data_selector(
        ...     pipeline_data, "folded_frames"
        ... )
        >>> print(f"Selector has {len(selector_data.trajectory_frames)} trajectories")
        """
        if data_selector not in pipeline_data.data_selector_data:
            available_selectors = list(pipeline_data.data_selector_data.keys())
            raise ValueError(
                f"DataSelector '{data_selector}' not found. "
                f"Available selectors: {available_selectors}"
            )
        
        selector_data = pipeline_data.data_selector_data[data_selector]
        
        if selector_data.is_empty():
            raise ValueError(f"DataSelector '{data_selector}' has no selected frames.")
            
        return selector_data

    @staticmethod
    def resolve_and_validate_traj_pair(
        trajectory_data: TrajectoryData,
        target_traj: Union[int, str],
        source_traj: Union[int, str],
    ) -> Tuple[int, int]:
        """
        Resolve and validate trajectory pair for join/stack operations.

        Parameters
        ----------
        trajectory_data : TrajectoryData
            Trajectory data container
        target_traj : int or str
            Target trajectory selector
        source_traj : int or str
            Source trajectory selector

        Returns
        -------
        tuple
            (target_idx, source_idx) as integers

        Raises
        ------
        ValueError
            If selectors don't resolve to exactly one trajectory each
        ValueError
            If target and source refer to same trajectory
        """
        target_indices = trajectory_data.get_trajectory_indices(target_traj)
        source_indices = trajectory_data.get_trajectory_indices(source_traj)

        if len(target_indices) != 1:
            raise ValueError(f"target_traj must resolve to exactly 1 trajectory, got {len(target_indices)}")
        if len(source_indices) != 1:
            raise ValueError(f"source_traj must resolve to exactly 1 trajectory, got {len(source_indices)}")

        target_idx = target_indices[0]
        source_idx = source_indices[0]

        if target_idx == source_idx:
            raise ValueError("target_traj and source_traj must be different trajectories")

        return target_idx, source_idx

    @staticmethod
    def _check_specific_trajectory_features(
        pipeline_data: PipelineData, force: bool, operation_name: str, traj_indices: List[int]
    ) -> None:
        """
        Check if specific trajectories have features.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        force : bool
            Whether to force the operation despite existing features
        operation_name : str
            Name of the operation for error messages
        traj_indices : List[int]
            Trajectory indices to check for features

        Raises
        ------
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

    @staticmethod
    def validate_superpose_parameters(
        pipeline_data: PipelineData,
        reference_traj: int,
        reference_frame: int,
        atom_selection: str,
        traj_selection: Union[int, str, List[Union[int, str]], str]
    ) -> tuple:
        """
        Validate all parameters for superpose operation.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        reference_traj : int
            Index of trajectory containing the reference frame
        reference_frame : int
            Frame index within reference trajectory
        atom_selection : str
            MDTraj selection string for alignment atoms
        traj_selection : various
            Selection of trajectories to align

        Returns
        -------
        tuple
            (reference_trajectory, traj_indices, ref_frame, ref_atom_indices)

        Raises
        ------
        ValueError
            If any parameter is invalid

        Examples
        --------
        >>> ref_traj, indices, ref_frame, atom_indices = \\
        ...     TrajectoryValidationHelper.validate_superpose_parameters(
        ...         pipeline_data, 0, 0, "backbone", "all"
        ...     )
        """
        # Check trajectories are loaded
        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("No trajectories loaded for superposition.")

        # Validate reference trajectory index
        if reference_traj < 0 or reference_traj >= len(pipeline_data.trajectory_data.trajectories):
            raise ValueError(
                f"Reference trajectory index {reference_traj} is invalid. "
                f"Valid range: 0 to {len(pipeline_data.trajectory_data.trajectories) - 1}"
            )

        reference_trajectory = pipeline_data.trajectory_data.trajectories[reference_traj]

        # Validate reference frame index
        if reference_frame < 0 or reference_frame >= reference_trajectory.n_frames:
            raise ValueError(
                f"Reference frame index {reference_frame} is invalid for trajectory {reference_traj}. "
                f"Valid range: 0 to {reference_trajectory.n_frames - 1}"
            )

        # Get and validate trajectory selection
        traj_indices = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)
        if not traj_indices:
            raise ValueError("No trajectories found matching the selection criteria.")

        # Get reference frame and validate atom selection
        ref_frame = reference_trajectory[reference_frame]
        try:
            ref_atom_indices = ref_frame.topology.select(atom_selection)
            if len(ref_atom_indices) == 0:
                raise ValueError(f"Atom selection '{atom_selection}' produced no atoms in reference frame.")
        except Exception as e:
            raise ValueError(f"Invalid atom selection '{atom_selection}': {e}")

        return reference_trajectory, traj_indices, ref_frame, ref_atom_indices
