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

"""Tag management utilities for trajectory data."""
from __future__ import annotations

from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ...entities.trajectory_data import TrajectoryData


class TagHelper:
    """Helper class for tag management and bulk trajectory tag operations."""

    @staticmethod
    def resolve_trajectory_selectors(
        traj_data: TrajectoryData,
        trajectory_selector: Union[int, str, list, range, dict],
        tags: List[str] = None,
    ) -> List[tuple]:
        """
        Resolve trajectory selectors to list of (indices, tags) assignments.

        This method handles all selector types and returns a list of assignments
        for the TrajectoryManager to apply. It does NOT modify the trajectory data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        trajectory_selector : int, str, list, range, dict
            Flexible selector for trajectories
        tags : list, optional
            List of tag strings. Required when trajectory_selector is not dict.

        Returns:
        --------
        List[tuple]
            List of (indices, tags) tuples for TrajectoryManager to apply

        Examples:
        ---------
        >>> # Returns: [(indices, tags), ...]
        >>> assignments = TagHelper.resolve_trajectory_selectors(
        ...     traj_data, 0, ["system_A"]
        ... )

        Raises:
        -------
        ValueError
            If tags is None when trajectory_selector is not dict
        """
        assignments = []

        # Dict mode - bulk assignment
        if isinstance(trajectory_selector, dict):
            for selector, tag_list in trajectory_selector.items():
                resolved_indices = traj_data.get_trajectory_indices(selector)
                assignments.append((resolved_indices, tag_list))
        else:
            # Single selector mode
            if tags is None:
                raise ValueError(
                    "tags parameter is required when trajectory_selector is not dict"
                )

            resolved_indices = traj_data.get_trajectory_indices(trajectory_selector)
            assignments.append((resolved_indices, tags))

        return assignments

