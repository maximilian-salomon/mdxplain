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

"""Trajectory processing utilities."""

from __future__ import annotations

from typing import Any, List, Optional, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...entities.trajectory_data import TrajectoryData
    from ....pipeline.entities.pipeline_data import PipelineData

from ..validation_helper.trajectory_validation_helper import TrajectoryValidationHelper


class TrajectoryProcessHelper:
    """Utility class for processing individual trajectory objects."""

    @staticmethod
    def apply_slicing(
        pipeline_data: PipelineData, 
        indices: List[int], 
        frames: Optional[Union[int, slice, List[int]]], 
        data_selector: Optional[str],
        stride: Optional[int],
        cut: Optional[int]
    ) -> None:
        """
        Apply slicing to trajectories using frames OR DataSelector.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing trajectory data
        indices : List[int]
            List of trajectory indices to process
        frames : int, slice, list, or None
            Frame specification for slicing (ignored if data_selector is used)
        data_selector : str or None
            Name of DataSelector to use for frame selection
        stride : int or None
            Stride value for frame subsampling
        cut : int or None
            Frame number after which to cut trajectories
        
        Returns:
        --------
        None
            Modifies trajectories in-place
            
        Examples:
        ---------
        >>> # Using frames parameter
        >>> TrajectoryProcessHelper.apply_slicing(
        ...     pipeline_data, [0, 1, 2], frames=1000, data_selector=None, stride=2, cut=500
        ... )
        >>> # Using DataSelector
        >>> TrajectoryProcessHelper.apply_slicing(
        ...     pipeline_data, [0, 1], frames=None, data_selector="folded_frames", stride=2, cut=None
        ... )
        """
        # Get frames from DataSelector or use frames parameter
        if data_selector is not None:
            frames_dict = TrajectoryProcessHelper._get_frames_from_selector(pipeline_data, data_selector)
        else:
            TrajectoryValidationHelper.validate_slice_parameters(frames)
            frames_dict = {idx: frames for idx in indices}
        
        # Default stride
        if stride is None or stride <= 0:
            stride = 1
        
        # Apply slicing to each trajectory
        for idx in indices:
            traj = pipeline_data.trajectory_data.trajectories[idx]
            frame_spec = frames_dict.get(idx)
            
            if frame_spec is not None and isinstance(frame_spec, list) and not frame_spec:
                print(f"No frames for trajectory {idx} - keeping unchanged")
                continue
            
            # Build indices and apply ONE slice
            indices_to_use = TrajectoryProcessHelper._combine_slice_params(frame_spec, stride, cut, traj.n_frames)
            if indices_to_use is not None:
                sliced_traj = traj.slice(indices_to_use)
                pipeline_data.trajectory_data.trajectories[idx] = sliced_traj
                
                if data_selector:
                    print(f"Sliced trajectory {idx} to {sliced_traj.n_frames} frames using DataSelector '{data_selector}'")
                else:
                    print(f"Sliced trajectory {idx} to {sliced_traj.n_frames} frames")

    @staticmethod
    def _get_frames_from_selector(pipeline_data: PipelineData, data_selector: str) -> Dict[int, List[int]]:
        """
        Get frames dictionary from DataSelector.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        data_selector : str
            Name of DataSelector
            
        Returns:
        --------
        Dict[int, List[int]]
            Dictionary mapping trajectory index to frame list
        """
        selector_data = TrajectoryValidationHelper.validate_data_selector(pipeline_data, data_selector)
        return selector_data.trajectory_frames

    @staticmethod
    def _combine_slice_params(
        frames: Optional[Union[int, slice, List[int]]], 
        stride: int, 
        cut: Optional[int], 
        n_frames: int
    ) -> Optional[List[int]]:
        """
        Combine frames, stride, and cut into one index list.
        
        Parameters:
        -----------
        frames : int, slice, list, or None
            Frame specification
        stride : int
            Stride value
        cut : int or None
            Cut value
        n_frames : int
            Total frames in trajectory
            
        Returns:
        --------
        list or None
            Combined indices for slicing, or None if no slicing needed
        """
        # Start with base indices from frames parameter
        if frames is None:
            if stride <= 1 and cut is None:
                return None  # No slicing needed
            indices = list(range(n_frames))
        elif isinstance(frames, int):
            indices = list(range(min(frames, n_frames)))
        elif isinstance(frames, slice):
            indices = list(range(*frames.indices(n_frames)))
        else:  # list or array
            indices = list(frames)
        
        # Apply stride
        if stride > 1:
            indices = indices[::stride]
        
        # Apply cut
        if cut is not None and len(indices) > cut:
            indices = indices[:cut]
        
        return indices

    @staticmethod
    def apply_selection_to_new_trajectories(
        new_trajectories: List[Any], selection: Optional[str]
    ) -> List[Any]:
        """
        Apply atom selection to new trajectories if provided.

        Parameters:
        -----------
        new_trajectories : list
            List of trajectory objects to process
        selection : str or None
            MDTraj selection string to apply

        Returns:
        --------
        list
            List of trajectories with selection applied

        Examples:
        ---------
        >>> processed_trajs = TrajectoryProcessHelper.apply_selection_to_new_trajectories(
        ...     trajectories, "protein"
        ... )
        """
        if selection is not None:
            for i, traj in enumerate(new_trajectories):
                atom_indices = traj.topology.select(selection)
                new_trajectories[i] = traj.atom_slice(atom_indices)
        return new_trajectories

    @staticmethod
    def execute_removal(
        traj_data: TrajectoryData, indices_to_remove: List[int]
    ) -> None:
        """
        Remove trajectories and names from trajectory data.

        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        indices_to_remove : list
            List of trajectory indices to remove (must be sorted in reverse order)

        Returns:
        --------
        None
            Removes trajectories from traj_data

        Examples:
        ---------
        >>> # Remove trajectories at indices [3, 1, 0] (reverse sorted)
        >>> TrajectoryProcessHelper.execute_removal(traj_data, [3, 1, 0])
        """
        removed_trajectories = []
        for idx in indices_to_remove:
            removed_name = traj_data.trajectory_names[idx]
            removed_trajectories.append(f"{removed_name} (was at index {idx})")
            del traj_data.trajectories[idx]
            del traj_data.trajectory_names[idx]

        # Also remove tags for removed trajectories
        for idx in indices_to_remove:
            traj_data.trajectory_tags.pop(idx, None)

        # Print removal summary
        for removed_info in removed_trajectories:
            print(f"Removed trajectory: {removed_info}")

        print(
            f"Removed {len(indices_to_remove)} trajectory(ies). "
            f"{len(traj_data.trajectories)} trajectory(ies) remaining."
        )

    @staticmethod
    def validate_name_mapping(
        traj_data: TrajectoryData, name_mapping: Union[Dict, List]
    ) -> None:
        """
        Validate name mapping input before processing.

        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        name_mapping : dict or list
            Name mapping to validate

        Returns:
        --------
        None
            Validation successful
        
        Examples:
        ---------
        >>> traj_data = "TrajectoryData"()
        >>> # Load some trajectories first
        >>> TrajectoryProcessHelper.validate_name_mapping(traj_data, ['new1', 'new2'])
        >>> TrajectoryProcessHelper.validate_name_mapping(traj_data, {0: 'renamed_first'})

        Raises:
        -------
        ValueError
            If trajectories are not loaded or mapping is invalid
        """
        if not traj_data.trajectories:
            raise ValueError("No trajectories loaded to rename.")

        if not isinstance(name_mapping, (dict, list)):
            raise ValueError("name_mapping must be dict or list")

    @staticmethod
    def rename_with_list(traj_data: "TrajectoryData", name_list: List[str]) -> List[str]:
        """
        Process list-based positional renaming with validation.

        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        name_list : list
            List of new names for positional assignment

        Returns:
        --------
        list
            Validated list of new trajectory names
        
        Examples:
        ---------
        >>> traj_data = "TrajectoryData"()
        >>> # Assuming 3 trajectories are loaded
        >>> new_names = TrajectoryProcessHelper.rename_with_list(
        ...     traj_data, ['system1', 'system2', 'system3']
        ... )
        >>> print(new_names)
        ['system1', 'system2', 'system3']

        Raises:
        -------
        ValueError
            If list length doesn't match trajectory count
        """
        if len(name_list) != len(traj_data.trajectory_names):
            raise ValueError(
                f"List length ({len(name_list)}) doesn't match "
                f"trajectory count ({len(traj_data.trajectory_names)})"
            )
        return list(name_list)

    @staticmethod
    def rename_with_dict(
        traj_data: TrajectoryData, name_dict: Dict[Union[int, str], str]
    ) -> List[str]:
        """
        Process dict-based selective renaming with validation.

        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        name_dict : dict
            Dictionary mapping old identifiers to new names

        Returns:
        --------
        list
            New list of trajectory names with applied changes
        
        Examples:
        ---------
        >>> traj_data = "TrajectoryData"()
        >>> # Rename specific trajectories by index
        >>> new_names = TrajectoryProcessHelper.rename_with_dict(
        ...     traj_data, {0: 'first_system', 2: 'third_system'}
        ... )
        >>> # Rename by existing name
        >>> new_names = TrajectoryProcessHelper.rename_with_dict(
        ...     traj_data, {'old_name': 'new_name'}
        ... )

        Raises:
        -------
        ValueError
            If dictionary keys reference invalid trajectories

        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        name_dict : dict
            Dictionary mapping old identifiers to new names

        Returns:
        --------
        list
            New list of trajectory names with applied changes

        Raises:
        -------
        ValueError
            If dictionary keys reference invalid trajectories
        """
        new_names = list(traj_data.trajectory_names)
        TrajectoryProcessHelper._apply_rename_dict(traj_data, name_dict, new_names)
        return new_names
    
    @staticmethod
    def _apply_rename_dict(
        traj_data: TrajectoryData, 
        name_dict: Dict[Union[int, str], str], 
        new_names: List[str]
    ) -> None:
        """
        Apply renaming dictionary to names list.
        
        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        name_dict : Dict[Union[int, str], str]
            Dictionary mapping old identifiers to new names
        new_names : List[str]
            List of trajectory names to modify
        
        Returns:
        --------
        None
            Modifies new_names list in-place
        """
        for old_identifier, new_name in name_dict.items():
            TrajectoryProcessHelper._process_single_rename(
                traj_data, old_identifier, new_name, new_names
            )
    
    @staticmethod
    def _process_single_rename(
        traj_data: TrajectoryData, 
        old_identifier: Union[int, str], 
        new_name: str, 
        new_names: List[str]
    ) -> None:
        """
        Process a single rename operation.
        
        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        old_identifier : Union[int, str]
            Old trajectory identifier (index or name)
        new_name : str
            New trajectory name
        new_names : List[str]
            List of trajectory names to modify
        
        Returns:
        --------
        None
            Modifies new_names list in-place
        
        Raises:
        -------
        ValueError
            If identifier type is not int or str
        """
        if isinstance(old_identifier, int):
            TrajectoryProcessHelper._rename_by_index(old_identifier, new_name, new_names)
        elif isinstance(old_identifier, str):
            TrajectoryProcessHelper._rename_by_name(traj_data, old_identifier, new_name, new_names)
        else:
            raise ValueError(f"Invalid identifier type: {type(old_identifier)}")
    
    @staticmethod
    def _rename_by_index(old_index: int, new_name: str, new_names: List[str]) -> None:
        """
        Rename trajectory by index.
        
        Parameters:
        -----------
        old_index : int
            Trajectory index to rename
        new_name : str
            New trajectory name
        new_names : List[str]
            List of trajectory names to modify
        
        Returns:
        --------
        None
            Modifies new_names list in-place
        
        Raises:
        -------
        ValueError
            If index is out of range
        """
        if 0 <= old_index < len(new_names):
            new_names[old_index] = new_name
        else:
            raise ValueError(
                f"Trajectory index {old_index} out of range. We have {len(new_names)} trajectories."
            )
    
    @staticmethod
    def _rename_by_name(
        traj_data: TrajectoryData, old_name: str, new_name: str, new_names: List[str]
    ) -> None:
        """
        Rename trajectory by name.
        
        Parameters:
        -----------
        traj_data : "TrajectoryData"
            Trajectory data object
        old_name : str
            Old trajectory name to find
        new_name : str
            New trajectory name
        new_names : List[str]
            List of trajectory names to modify
        
        Returns:
        --------
        None
            Modifies new_names list in-place
        
        Raises:
        -------
        ValueError
            If old_name is not found in trajectory names
        """
        try:
            idx = traj_data.trajectory_names.index(old_name)
            new_names[idx] = new_name
        except ValueError:
            raise ValueError(f"Trajectory name '{old_name}' not found")
