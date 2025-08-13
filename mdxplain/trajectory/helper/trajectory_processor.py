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

"""Trajectory processing utilities."""

from typing import Any, List, Optional, Union, Dict

from ..entities.trajectory_data import TrajectoryData


class TrajectoryProcessor:
    """Utility class for processing individual trajectory objects."""

    @staticmethod
    def process_trajectory_cuts(traj: Any, cut: Optional[int], stride: int) -> Any:
        """
        Apply cut and stride operations to a single trajectory.

        Parameters:
        -----------
        traj : Trajectory
            Trajectory object
        cut : int, optional
            Frame number after which to cut the trajectory.
        stride : int
            Take every stride-th frame.

        Returns:
        --------
        Trajectory
            Processed trajectory object

        Examples:
        ---------
        >>> processed_traj = TrajectoryProcessor.process_trajectory_cuts(
        ...     traj, cut=1000, stride=2
        ... )
        """
        # Apply cut if specified
        if cut is not None and cut < traj.n_frames:
            traj = traj[:cut]

        # Apply stride if not 1
        if stride > 1:
            traj = traj[::stride]

        return traj

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
        >>> processed_trajs = TrajectoryProcessor.apply_selection_to_new_trajectories(
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
        traj_data : TrajectoryData
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
        >>> TrajectoryProcessor.execute_removal(traj_data, [3, 1, 0])
        """
        removed_trajectories = []
        for idx in indices_to_remove:
            removed_name = traj_data.trajectory_names[idx]
            removed_trajectories.append(f"{removed_name} (was at index {idx})")
            del traj_data.trajectories[idx]
            del traj_data.trajectory_names[idx]

        # Also remove keywords for removed trajectories
        for idx in indices_to_remove:
            traj_data.trajectory_keywords.pop(idx, None)

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
        traj_data : TrajectoryData
            Trajectory data object
        name_mapping : dict or list
            Name mapping to validate

        Returns:
        --------
        None
            Validation successful

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
    def rename_with_list(traj_data: TrajectoryData, name_list: List[str]) -> List[str]:
        """
        Process list-based positional renaming with validation.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        name_list : list
            List of new names for positional assignment

        Returns:
        --------
        list
            Validated list of new trajectory names

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
        traj_data : TrajectoryData
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
        # Create a copy of existing names
        new_names = list(traj_data.trajectory_names)

        for old_identifier, new_name in name_dict.items():
            if isinstance(old_identifier, int):
                # Index-based renaming
                if 0 <= old_identifier < len(new_names):
                    new_names[old_identifier] = new_name
                else:
                    raise ValueError(
                        f"Trajectory index {old_identifier} out of range. We have {len(new_names)} trajectories."
                    )

            elif isinstance(old_identifier, str):
                # Name-based renaming
                try:
                    idx = traj_data.trajectory_names.index(old_identifier)
                    new_names[idx] = new_name
                except ValueError:
                    raise ValueError(f"Trajectory name '{old_identifier}' not found")

            else:
                raise ValueError(f"Invalid identifier type: {type(old_identifier)}")

        return new_names
