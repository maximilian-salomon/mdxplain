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

"""Trajectory selection resolution utilities."""

from typing import List, Optional, Union

from ..entities.trajectory_data import TrajectoryData


class TrajectorySelectionResolver:
    """Utility class for resolving trajectory selections to indices."""

    @staticmethod
    def resolve_selection(
        traj_data: TrajectoryData, selection: List[Union[int, str]]
    ) -> List[int]:
        """
        Resolve selection list to trajectory indices.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selection : list
            List of trajectory indices (int) or names (str)

        Returns:
        --------
        list
            List of trajectory indices

        Raises:
        -------
        ValueError
            If selection contains invalid indices or names

        Examples:
        ---------
        >>> indices = TrajectorySelectionResolver.resolve_selection(
        ...     traj_data, [0, "traj1", 2]
        ... )
        """
        indices = []
        for item in selection:
            idx = TrajectorySelectionResolver._resolve_single_selection_item(
                item, traj_data
            )
            indices.append(idx)

        return indices

    @staticmethod
    def get_indices_to_process(
        traj_data: TrajectoryData, selection: Optional[List[Union[int, str]]]
    ) -> List[int]:
        """
        Get list of trajectory indices to process.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selection : list, optional
            List of trajectory indices (int) or names (str) to process.
            If None, all trajectories will be processed.

        Returns:
        --------
        list
            List of trajectory indices to process

        Examples:
        ---------
        >>> # Process all trajectories
        >>> indices = TrajectorySelectionResolver.get_indices_to_process(traj_data, None)

        >>> # Process specific trajectories
        >>> indices = TrajectorySelectionResolver.get_indices_to_process(
        ...     traj_data, [0, "traj1"]
        ... )
        """
        if selection is None:
            return list(range(len(traj_data.trajectories)))
        else:
            return TrajectorySelectionResolver.resolve_selection(traj_data, selection)

    @staticmethod
    def prepare_removal_indices(
        traj_data: TrajectoryData, trajs: Union[int, str, List[Union[int, str]]]
    ) -> List[int]:
        """
        Convert trajs input to sorted indices for removal.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        trajs : int, str, or list
            Trajectory index (int), name (str), or list of indices/names to remove.

        Returns:
        --------
        list
            List of trajectory indices to remove, sorted in reverse order

        Examples:
        ---------
        >>> indices = TrajectorySelectionResolver.prepare_removal_indices(
        ...     traj_data, [0, "traj1", 2]
        ... )
        """
        if not isinstance(trajs, list):
            trajs = [trajs]
        indices_to_remove = TrajectorySelectionResolver.resolve_selection(
            traj_data, trajs
        )
        indices_to_remove.sort(reverse=True)  # Avoid index shifting issues
        return indices_to_remove

    @staticmethod
    def _resolve_single_selection_item(
        item: Union[int, str], traj_data: TrajectoryData
    ) -> int:
        """
        Resolve a single selection item to trajectory index.

        Parameters:
        -----------
        item : int or str
            Selection item to resolve
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        int
            Index of the resolved selection item

        Raises:
        -------
        ValueError
            If selection item is not an integer or string
        """
        if isinstance(item, int):
            return TrajectorySelectionResolver._resolve_index_selection(item, traj_data)
        elif isinstance(item, str):
            return TrajectorySelectionResolver._resolve_name_selection(item, traj_data)
        else:
            raise ValueError(
                f"Selection item must be int (index) or str (name), got {type(item)}"
            )

    @staticmethod
    def _resolve_index_selection(item: int, traj_data: TrajectoryData) -> int:
        """
        Resolve integer index selection.

        Parameters:
        -----------
        item : int
            Index to resolve
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        int
            Index of the resolved selection item

        Raises:
        -------
        ValueError
            If index is out of range
        """
        if 0 <= item < len(traj_data.trajectories):
            return item
        else:
            raise ValueError(
                f"Trajectory index {item} out of range (0-{len(traj_data.trajectories)-1})"
            )

    @staticmethod
    def _resolve_name_selection(item: str, traj_data: TrajectoryData) -> int:
        """
        Resolve string name selection.

        Parameters:
        -----------
        item : str
            Name to resolve
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        int
            Index of the resolved selection item

        Raises:
        -------
        ValueError
            If name is not found in trajectory names
        """
        if item in traj_data.trajectory_names:
            return traj_data.trajectory_names.index(item)
        else:
            raise ValueError(
                f"Trajectory name '{item}' not found. Available names: {traj_data.trajectory_names}"
            )
