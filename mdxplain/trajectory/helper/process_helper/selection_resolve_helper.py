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

"""Trajectory selection resolution utilities."""

from typing import List, Union

from ...entities.trajectory_data import TrajectoryData


class SelectionResolveHelper:
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
        >>> indices = SelectionResolveHelper.resolve_selection(
        ...     traj_data, [0, "traj1", 2]
        ... )
        """
        indices = []
        for item in selection:
            idx = SelectionResolveHelper._resolve_single_selection_item(
                item, traj_data
            )
            indices.append(idx)

        return indices

    @staticmethod
    def get_indices_to_process(
        traj_data: TrajectoryData, traj_selection: Union[int, str, List[Union[int, str]], "all"]
    ) -> List[int]:
        """
        Get list of trajectory indices to process.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        traj_selection : int, str, list, or "all"
            Selection of trajectories to process:
            - int: trajectory index
            - str: trajectory name or "all" for all trajectories  
            - list: list of indices/names

        Returns:
        --------
        list
            List of trajectory indices to process

        Examples:
        ---------
        >>> # Process all trajectories
        >>> indices = SelectionResolveHelper.get_indices_to_process(traj_data, "all")

        >>> # Process single trajectory
        >>> indices = SelectionResolveHelper.get_indices_to_process(traj_data, 0)

        >>> # Process specific trajectories
        >>> indices = SelectionResolveHelper.get_indices_to_process(
        ...     traj_data, [0, "traj1"]
        ... )
        """
        # Handle "all" string
        if traj_selection == "all":
            return list(range(len(traj_data.trajectories)))
        else:
            if not isinstance(traj_selection, list):
                traj_selection = [traj_selection]
            return SelectionResolveHelper.resolve_selection(
                traj_data, traj_selection
            )

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
            return SelectionResolveHelper._resolve_index_selection(item, traj_data)
        elif isinstance(item, str):
            return SelectionResolveHelper._resolve_name_selection(item, traj_data)
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
