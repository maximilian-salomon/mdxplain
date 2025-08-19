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
    Data entity for storing data selector configurations.

    Stores frame indices and selection criteria for trajectory frame selection.
    This entity serves as the counterpart to FeatureSelectorData, focusing on
    row selection instead of column selection.

    The DataSelectorData follows the same pattern as FeatureSelectorData,
    providing a container for frame selection configurations that can be
    stored and retrieved by name.

    Attributes:
    -----------
    name : str
        Name identifier for this data selector configuration
    frame_indices : List[int]
        List of selected trajectory frame indices
    selection_criteria : Dict[str, Any]
        Dictionary containing the criteria used for selection
    n_selected_frames : int
        Number of selected frames (convenience property)

    Examples:
    ---------
    >>> selector_data = DataSelectorData("folded_frames")
    >>> selector_data.set_frame_indices([0, 5, 12, 18, 25])
    >>> print(f"Selected {selector_data.n_selected_frames} frames")
    5

    >>> # With selection criteria
    >>> criteria = {"type": "cluster", "clustering": "conformations", "cluster_ids": [0]}
    >>> selector_data.set_selection_criteria(criteria)
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
        >>> print(len(selector_data.frame_indices))
        0
        """
        self.name = name
        self.frame_indices: List[int] = []
        self.selection_criteria: Dict[str, Any] = {}

    @property
    def n_selected_frames(self) -> int:
        """
        Get number of selected frames.

        Returns:
        --------
        int
            Number of frames in the selection

        Examples:
        ---------
        >>> selector_data.set_frame_indices([1, 5, 10])
        >>> print(selector_data.n_selected_frames)
        3
        """
        return len(self.frame_indices)

    def set_frame_indices(self, indices: List[int]) -> None:
        """
        Set the selected frame indices.

        Parameters:
        -----------
        indices : List[int]
            List of frame indices to select

        Returns:
        --------
        None
            Updates the frame_indices attribute

        Examples:
        ---------
        >>> selector_data = DataSelectorData("test")
        >>> selector_data.set_frame_indices([0, 10, 20, 30])
        >>> print(selector_data.frame_indices)
        [0, 10, 20, 30]
        """
        self.frame_indices = indices.copy() if indices else []

    def get_frame_indices(self) -> List[int]:
        """
        Get the selected frame indices.

        Returns:
        --------
        List[int]
            List of selected frame indices

        Examples:
        ---------
        >>> indices = selector_data.get_frame_indices()
        >>> print(f"Selected frames: {indices[:5]}...")  # First 5
        """
        return self.frame_indices.copy()

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

    def add_frame_indices(self, indices: List[int]) -> None:
        """
        Add additional frame indices to the existing selection.

        Parameters:
        -----------
        indices : List[int]
            Frame indices to add to the current selection

        Returns:
        --------
        None
            Extends the frame_indices list with new indices

        Examples:
        ---------
        >>> selector_data.set_frame_indices([0, 5, 10])
        >>> selector_data.add_frame_indices([15, 20])
        >>> print(selector_data.frame_indices)
        [0, 5, 10, 15, 20]
        """
        if indices:
            self.frame_indices.extend(indices)

    def remove_duplicates(self) -> None:
        """
        Remove duplicate frame indices and sort the list.

        Returns:
        --------
        None
            Removes duplicates from frame_indices and sorts the list

        Examples:
        ---------
        >>> selector_data.frame_indices = [5, 1, 5, 3, 1, 8]
        >>> selector_data.remove_duplicates()
        >>> print(selector_data.frame_indices)
        [1, 3, 5, 8]
        """
        self.frame_indices = sorted(list(set(self.frame_indices)))

    def clear_selection(self) -> None:
        """
        Clear all selected frame indices and criteria.

        Returns:
        --------
        None
            Resets frame_indices and selection_criteria to empty

        Examples:
        ---------
        >>> selector_data.clear_selection()
        >>> print(selector_data.n_selected_frames)
        0
        >>> print(selector_data.selection_criteria)
        {}
        """
        self.frame_indices = []
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
            "frame_range": (
                f"{min(self.frame_indices)}-{max(self.frame_indices)}"
                if self.frame_indices
                else "none"
            ),
        }

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
        return len(self.frame_indices) == 0

    def __len__(self) -> int:
        """
        Get number of selected frames.

        Returns:
        --------
        int
            Number of selected frames

        Examples:
        ---------
        >>> selector_data.set_frame_indices([1, 2, 3])
        >>> print(len(selector_data))
        3
        """
        return len(self.frame_indices)

    def __contains__(self, frame_index: int) -> bool:
        """
        Check if a frame index is in the selection.

        Parameters:
        -----------
        frame_index : int
            Frame index to check

        Returns:
        --------
        bool
            True if frame_index is in the selection, False otherwise

        Examples:
        ---------
        >>> selector_data.set_frame_indices([1, 5, 10])
        >>> print(5 in selector_data)
        True
        >>> print(7 in selector_data)
        False
        """
        return frame_index in self.frame_indices

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
        DataSelectorData(name='folded_frames', n_frames=150)
        """
        return f"DataSelectorData(name='{self.name}', n_frames={self.n_selected_frames})"
    