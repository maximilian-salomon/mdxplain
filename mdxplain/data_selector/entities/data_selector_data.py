# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
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
Data selector data entity for storing frame selection configurations.

This module contains the DataSelectorData class that stores frame
selection configurations including selected frame indices and
selection criteria. This is the row-selection counterpart to
FeatureSelectorData (which selects columns).
"""

from typing import Dict, List, Any


class DataSelectorData:
    """
    Data entity for storing trajectory-specific data selector configurations.

    Stores frame indices per trajectory and selection criteria for trajectory
    frame selection. This entity serves as the counterpart to FeatureSelectorData,
    focusing on row selection instead of column selection.

    The DataSelectorData works with the 2-level dictionary structure where
    each trajectory has its own frame indices, supporting the new trajectory-
    specific architecture.

    Attributes:
    -----------
    name : str
        Name identifier for this data selector configuration
    trajectory_frames : Dict[int, List[int]]
        Dictionary mapping trajectory indices to their selected frame indices
    selection_criteria : Dict[str, Any]
        Dictionary containing the criteria used for selection
    n_selected_frames : int
        Total number of selected frames across all trajectories (convenience property)

    Examples:
    ---------
    >>> selector_data = DataSelectorData("folded_frames")
    >>> selector_data.set_trajectory_frames({0: [0, 5, 12], 1: [18, 25]})
    >>> print(f"Selected {selector_data.n_selected_frames} frames")
    5

    >>> # With selection criteria
    >>> criteria = {"type": "cluster", "clustering": "conformations", "cluster_ids": [0]}
    >>> selector_data.append_selection_criteria(criteria)
    >>> print(selector_data.get_selection_info())
    """

    def __init__(self, name: str):
        """
        Initialize data selector data with given name.

        Parameters:
        -----------
        name : str
            Name identifier for this data selector configuration

        Returns:
        --------
        None
            Initializes empty DataSelectorData with given name

        Examples:
        ---------
        >>> selector_data = DataSelectorData("active_site_frames")
        >>> print(selector_data.name)
        'active_site_frames'
        >>> print(len(selector_data.trajectory_frames))
        0
        """
        self.name = name
        self.trajectory_frames: Dict[int, List[int]] = {}
        self.selection_criteria: Dict[str, Any] = {}

    @property
    def n_selected_frames(self) -> int:
        """
        Get total number of selected frames across all trajectories.

        Returns:
        --------
        int
            Total number of frames in the selection

        Examples:
        ---------
        >>> selector_data.set_trajectory_frames({0: [1, 5, 10], 1: [2, 7]})
        >>> print(selector_data.n_selected_frames)
        5
        """
        return sum(len(frames) for frames in self.trajectory_frames.values())

    def set_trajectory_frames(self, trajectory_frames: Dict[int, List[int]]) -> None:
        """
        Set the selected frame indices per trajectory.

        Parameters:
        -----------
        trajectory_frames : Dict[int, List[int]]
            Dictionary mapping trajectory indices to frame indices

        Returns:
        --------
        None
            Updates the trajectory_frames attribute

        Examples:
        ---------
        >>> selector_data = DataSelectorData("test")
        >>> selector_data.set_trajectory_frames({0: [0, 10, 20], 1: [5, 15]})
        >>> print(selector_data.trajectory_frames)
        {0: [0, 10, 20], 1: [5, 15]}
        """
        self.trajectory_frames = {}
        for traj_idx, frames in trajectory_frames.items():
            self.trajectory_frames[traj_idx] = frames.copy() if frames else []

    def get_trajectory_frames(self) -> Dict[int, List[int]]:
        """
        Get the selected frame indices per trajectory.

        Returns:
        --------
        Dict[int, List[int]]
            Dictionary mapping trajectory indices to frame indices

        Examples:
        ---------
        >>> traj_frames = selector_data.get_trajectory_frames()
        >>> print(f"Trajectory 0 frames: {traj_frames.get(0, [])}")
        """
        return {traj_idx: frames.copy() for traj_idx, frames in self.trajectory_frames.items()}

    def append_selection_criteria(self, criteria: Dict[str, Any]) -> None:
        """
        Add selection criteria to chronological operations list.

        Creates operations list on first call, appends to existing list on
        subsequent calls. Maintains chronological order of operations for
        full reproducibility of frame selection process.

        Parameters:
        -----------
        criteria : Dict[str, Any]
            Dictionary containing selection criteria for this operation

        Returns:
        --------
        None
            Updates the selection_criteria attribute with operations list

        Examples:
        ---------
        >>> # First operation - creates operations list
        >>> selector_data.append_selection_criteria({
        ...     "type": "cluster", "clustering": "conformations", 
        ...     "cluster_ids": [0], "mode": "add"
        ... })
        >>> # Second operation - appends to list
        >>> selector_data.append_selection_criteria({
        ...     "type": "tags", "tags": ["system_A"], "mode": "intersect"
        ... })
        >>> # Result: {"operations": [operation1, operation2]}
        """
        if not criteria:
            return

        if not self.selection_criteria:
            # First operation - create operations list
            self.selection_criteria = {"operations": [criteria.copy()]}
        else:
            # Add operation to existing list
            self.selection_criteria["operations"].append(criteria.copy())

    def get_selection_criteria(self) -> Dict[str, Any]:
        """
        Get the selection criteria.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing selection criteria

        Examples:
        ---------
        >>> criteria = selector_data.get_selection_criteria()
        >>> print(f"Selection type: {criteria.get('type', 'unknown')}")
        """
        return self.selection_criteria.copy()

    def add_trajectory_frames(self, trajectory_frames: Dict[int, List[int]]) -> None:
        """
        Add additional frame indices to existing trajectories.

        Parameters:
        -----------
        trajectory_frames : Dict[int, List[int]]
            Dictionary mapping trajectory indices to additional frame indices

        Returns:
        --------
        None
            Extends existing trajectory frame lists with new indices

        Examples:
        ---------
        >>> selector_data.set_trajectory_frames({0: [0, 5, 10]})
        >>> selector_data.add_trajectory_frames({0: [15, 20], 1: [2, 7]})
        >>> print(selector_data.trajectory_frames)
        {0: [0, 5, 10, 15, 20], 1: [2, 7]}
        """
        for traj_idx, frames in trajectory_frames.items():
            if frames:
                if traj_idx not in self.trajectory_frames:
                    self.trajectory_frames[traj_idx] = []
                self.trajectory_frames[traj_idx].extend(frames)

    def remove_duplicates(self) -> None:
        """
        Remove duplicate frame indices and sort the lists for each trajectory.

        Returns:
        --------
        None
            Removes duplicates from trajectory frame lists and sorts them

        Examples:
        ---------
        >>> selector_data.trajectory_frames = {0: [5, 1, 5, 3], 1: [8, 1, 8]}
        >>> selector_data.remove_duplicates()
        >>> print(selector_data.trajectory_frames)
        {0: [1, 3, 5], 1: [1, 8]}
        """
        for traj_idx in self.trajectory_frames:
            self.trajectory_frames[traj_idx] = sorted(list(set(self.trajectory_frames[traj_idx])))

    def clear_selection(self) -> None:
        """
        Clear all selected frame indices and criteria.

        Returns:
        --------
        None
            Resets trajectory_frames and selection_criteria to empty

        Examples:
        ---------
        >>> selector_data.clear_selection()
        >>> print(selector_data.n_selected_frames)
        0
        >>> print(selector_data.selection_criteria)
        {}
        """
        self.trajectory_frames = {}
        self.selection_criteria = {}

    def get_selection_info(self) -> Dict[str, Any]:
        """
        Get summary information about this selection.

        Returns:
        --------
        Dict[str, Any]
            Dictionary with selection summary information

        Examples:
        ---------
        >>> info = selector_data.get_selection_info()
        >>> print(f"Name: {info['name']}")
        >>> print(f"Frames: {info['n_frames']}")
        >>> print(f"Type: {info['selection_type']}")
        """
        # Determine selection type from operations
        operations = self.selection_criteria.get("operations", [])
        if not operations:
            selection_type = "empty"
        elif len(operations) == 1:
            selection_type = operations[0].get("type", "unknown")
        else:
            # Multiple operations - show summary
            types = [op.get("type", "unknown") for op in operations]
            selection_type = f"multi_operation ({len(operations)} ops: {', '.join(set(types))})"
            
        return {
            "name": self.name,
            "n_frames": self.n_selected_frames,
            "selection_type": selection_type,
            "n_operations": len(operations),
            "operations": operations,
            "frame_range": self._get_frame_range_summary(),
            "trajectories": list(self.trajectory_frames.keys()),
        }
    
    def _get_frame_range_summary(self) -> str:
        """
        Get a summary of frame ranges across all trajectories.
        
        Returns:
        --------
        str
            Summary string of frame ranges
        """
        if not self.trajectory_frames:
            return "none"
        
        ranges = []
        for traj_idx, frames in self.trajectory_frames.items():
            if frames:
                ranges.append(f"traj{traj_idx}:{min(frames)}-{max(frames)}")
        
        return ", ".join(ranges) if ranges else "none"

    def is_empty(self) -> bool:
        """
        Check if the selection is empty.

        Returns:
        --------
        bool
            True if no frames are selected, False otherwise

        Examples:
        ---------
        >>> empty_selector = DataSelectorData("empty")
        >>> print(empty_selector.is_empty())
        True
        """
        return len(self.trajectory_frames) == 0 or self.n_selected_frames == 0

    def __len__(self) -> int:
        """
        Get number of selected frames.

        Returns:
        --------
        int
            Total number of selected frames across all trajectories

        Examples:
        ---------
        >>> selector_data.set_trajectory_frames({0: [1, 2, 3], 1: [4, 5]})
        >>> print(len(selector_data))
        5
        """
        return self.n_selected_frames

    def __contains__(self, traj_frame_tuple) -> bool:
        """
        Check if a (trajectory_index, frame_index) tuple is in the selection.

        Parameters:
        -----------
        traj_frame_tuple : tuple
            Tuple of (trajectory_index, frame_index) to check

        Returns:
        --------
        bool
            True if the trajectory-frame combination is in the selection

        Examples:
        ---------
        >>> selector_data.set_trajectory_frames({0: [1, 5, 10], 1: [2, 7]})
        >>> print((0, 5) in selector_data)
        True
        >>> print((1, 10) in selector_data)
        False
        """
        if not isinstance(traj_frame_tuple, tuple) or len(traj_frame_tuple) != 2:
            return False
        traj_idx, frame_idx = traj_frame_tuple
        return traj_idx in self.trajectory_frames and frame_idx in self.trajectory_frames[traj_idx]

    def __repr__(self) -> str:
        """
        String representation of the DataSelectorData.

        Returns:
        --------
        str
            String representation

        Examples:
        ---------
        >>> print(repr(selector_data))
        DataSelectorData(name='folded_frames', n_frames=150, n_trajectories=3)
        """
        return f"DataSelectorData(name='{self.name}', n_frames={self.n_selected_frames}, n_trajectories={len(self.trajectory_frames)})"
    