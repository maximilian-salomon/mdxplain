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

from __future__ import annotations

from typing import List, Union, TYPE_CHECKING
import fnmatch
import re

if TYPE_CHECKING:
    from ...entities.trajectory_data import TrajectoryData

class SelectionResolveHelper:
    """Utility class for resolving trajectory selections to indices."""

    @staticmethod
    def _resolve_selection(
        traj_data: TrajectoryData, selection: List[Union[int, str]]
    ) -> List[int]:
        """
        Resolve selection list to trajectory indices.

        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        selection : int, str, list, or "all"
            Selection criteria:
            - int: trajectory index
            - str: trajectory name, tag (prefixed with "tag:") or advanced formats:
                * Range: "0-3", "id 0-3" → [0, 1, 2, 3]
                * Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
                * Single number: "7", "id 7" → [7]
                * Pattern: "system_*" → fnmatch pattern matching
            - list: mix of indices/names/tags/patterns

        Returns
        -------
        list
            List of trajectory indices

        Raises
        ------
        ValueError
            If selection contains invalid indices or names
        """
        indices = []
        for item in selection:
            result = SelectionResolveHelper._resolve_single_selection_item(
                item, traj_data
            )
            # Handle both single int and List[int] results
            if isinstance(result, list):
                indices.extend(result)
            else:
                indices.append(result)

        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)

        return unique_indices

    @staticmethod
    def get_indices_to_process(
        traj_data: TrajectoryData, traj_selection: Union[int, str, List[Union[int, str]], "all"]
    ) -> List[int]:
        """
        Get list of trajectory indices to process.

        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        traj_selection : int, str, list, or "all"
            Selection criteria:
            
            - int: trajectory index
            - str: trajectory name, tag (prefixed with "tag:") or advanced formats:
                
                * Range: "0-3", "id 0-3" → [0, 1, 2, 3]
                * Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
                * Single number: "7", "id 7" → [7]
                * Pattern: "system_*" → fnmatch pattern matching
            - list: mix of indices/names/tags/patterns
            - "all": all trajectories

        Returns
        -------
        list
            List of trajectory indices to process

        Examples
        --------
        >>> # Process all trajectories
        >>> indices = SelectionResolveHelper.get_indices_to_process(traj_data, "all")

        >>> # Process single trajectory
        >>> indices = SelectionResolveHelper.get_indices_to_process(traj_data, 0)

        >>> # Process by tag
        >>> indices = SelectionResolveHelper.get_indices_to_process(traj_data, "tag:system_A")

        >>> # Process specific trajectories
        >>> indices = SelectionResolveHelper.get_indices_to_process(
        ...     traj_data, [0, "traj1", "tag:biased"]
        ... )
        """
        # Handle "all" string
        if traj_selection == "all":
            return list(range(len(traj_data.trajectories)))
        else:
            if not isinstance(traj_selection, list):
                traj_selection = [traj_selection]
            return SelectionResolveHelper._resolve_selection(
                traj_data, traj_selection
            )

    @staticmethod
    def _resolve_single_selection_item(
        item: Union[int, str], traj_data: TrajectoryData
    ) -> Union[int, List[int]]:
        """
        Resolve a single selection item to trajectory index(es).

        Now supports advanced string formats from TagHelper:
        - Range: "0-3", "id 0-3" → [0, 1, 2, 3]
        - Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
        - Single number: "7", "id 7" → 7
        - Pattern: "system_*" → [matching indices]
        - Tags: "tag:system_A" → [matching indices]

        Parameters
        ----------
        item : int or str
            Selection item to resolve
        traj_data : TrajectoryData
            Trajectory data object

        Returns
        -------
        int or List[int]
            Index(es) of the resolved selection item

        Raises
        ------
        ValueError
            If selection item is not an integer or string
        """
        if isinstance(item, int):
            return SelectionResolveHelper._resolve_index_selection(item, traj_data)
        elif isinstance(item, str):
            return SelectionResolveHelper._resolve_string_selection(item, traj_data)
    
    @staticmethod
    def _resolve_string_selection(item: str, traj_data: TrajectoryData) -> Union[int, List[int]]:
        """
        Resolve string selection with all supported formats.
        
        Supports:
        - Range: "0-3", "id 0-3" → [0, 1, 2, 3]
        - Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5] 
        - Single number: "7", "id 7" → 7
        - Pattern: "system_*" → [matching indices]
        - Tags: "tag:system_A" → [matching indices]
        - Names: "traj1" → single index
        - "all" → all trajectories
        
        Parameters
        ----------
        item : str
            String to resolve
        traj_data : TrajectoryData
            Trajectory data object
            
        Returns
        -------
        int or List[int]
            Index(es) of the resolved selection
        """
        clean_selector = SelectionResolveHelper._clean_selector_string(item)
        return SelectionResolveHelper._route_selector_by_format(traj_data, clean_selector)
        
    @staticmethod
    def _clean_selector_string(selector: str) -> str:
        """
        Clean selector string by removing optional id prefix.
        
        Parameters
        ----------
        selector : str
            Raw selector string potentially with "id " prefix
        
        Returns
        -------
        str
            Cleaned selector string
        """
        return selector.removeprefix("id ").strip()
    
    @staticmethod
    def _route_selector_by_format(traj_data: TrajectoryData, clean_selector: str) -> List[int]:
        """
        Route selector to appropriate resolver based on format.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        clean_selector : str
            Cleaned selector string
        
        Returns
        -------
        List[int]
            List of resolved trajectory indices
        """
        # Handle "all" special case
        if clean_selector == "all":
            return list(range(len(traj_data.trajectories)))
        elif SelectionResolveHelper._is_range_format(clean_selector):
            return SelectionResolveHelper._resolve_range_string(traj_data, clean_selector)
        elif SelectionResolveHelper._is_comma_list_format(clean_selector):
            return SelectionResolveHelper._resolve_comma_list(traj_data, clean_selector)
        elif SelectionResolveHelper._is_single_number_format(clean_selector):
            return SelectionResolveHelper._resolve_int_selector(traj_data, int(clean_selector))
        else:
            return SelectionResolveHelper._resolve_pattern_format(traj_data, clean_selector)
        
    @staticmethod
    def _is_range_format(clean_selector: str) -> bool:
        """
        Check if selector is range format.
        
        Parameters
        ----------
        clean_selector : str
            Cleaned selector string to check
        
        Returns
        -------
        bool
            True if selector matches range format (e.g., "0-3")
        """
        return "-" in clean_selector and "," not in clean_selector
    
    @staticmethod
    def _is_comma_list_format(clean_selector: str) -> bool:
        """
        Check if selector is comma list format.
        
        Parameters
        ----------
        clean_selector : str
            Cleaned selector string to check
        
        Returns
        -------
        bool
            True if selector matches comma list format (e.g., "1,2,4")
        """
        return "," in clean_selector
    
    @staticmethod
    def _is_single_number_format(clean_selector: str) -> bool:
        """
        Check if selector is single number format.
        
        Parameters
        ----------
        clean_selector : str
            Cleaned selector string to check
        
        Returns
        -------
        bool
            True if selector is a single digit string (e.g., "7")
        """
        return clean_selector.isdigit()
    
    @staticmethod
    def _resolve_pattern_format(traj_data: TrajectoryData, clean_selector: str) -> List[int]:
        """
        Resolve pattern format (regex or fnmatch) or fallback to name/tag.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        clean_selector : str
            Pattern selector string
        
        Returns
        -------
        List[int]
            List of matching trajectory indices
        """
        if clean_selector.startswith("regex "):
            regex_pattern = clean_selector.removeprefix("regex ").strip()
            return SelectionResolveHelper._resolve_regex_pattern(traj_data, regex_pattern)
        elif clean_selector.startswith("tag:"):
            tag_name = clean_selector.removeprefix("tag:").strip()
            return SelectionResolveHelper._find_trajectories_with_tag(tag_name, traj_data)
        return SelectionResolveHelper._resolve_fnmatch_pattern(traj_data, clean_selector)
    
    @staticmethod
    def _resolve_index_selection(item: int, traj_data: TrajectoryData) -> int:
        """
        Resolve integer index selection.

        Parameters
        ----------
        item : int
            Index to resolve
        traj_data : TrajectoryData
            Trajectory data object

        Returns
        -------
        int
            Index of the resolved selection item

        Raises
        ------
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
    def _resolve_tag_selection(tag: str, traj_data: TrajectoryData) -> int:
        """
        Resolve tag-based selection to first matching trajectory index.

        Parameters
        ----------
        tag : str
            Tag to search for
        traj_data : TrajectoryData
            Trajectory data object

        Returns
        -------
        int
            Index of first trajectory with the specified tag

        Raises
        ------
        ValueError
            If tag is not found in any trajectory
        """
        matching_indices = SelectionResolveHelper._find_trajectories_with_tag(
            tag, traj_data
        )
        if not matching_indices:
            available_tags = SelectionResolveHelper._get_all_tags(traj_data)
            raise ValueError(
                f"Tag '{tag}' not found in any trajectory. Available tags: {available_tags}"
            )
        return matching_indices[0]

    @staticmethod
    def _find_trajectories_with_tag(tag: str, traj_data: TrajectoryData) -> List[int]:
        """
        Find all trajectories that have the specified tag.

        Parameters
        ----------
        tag : str
            Tag to search for
        traj_data : TrajectoryData
            Trajectory data object

        Returns
        -------
        List[int]
            List of trajectory indices that have the tag
        """
        matching_indices = []
        if hasattr(traj_data, 'trajectory_tags') and traj_data.trajectory_tags:
            for traj_idx, tags in traj_data.trajectory_tags.items():
                if tag in tags:
                    matching_indices.append(traj_idx)
        return matching_indices
    
    @staticmethod  
    def _resolve_range_string(traj_data: TrajectoryData, selector: str) -> List[int]:
        """
        Parse range string and resolve to trajectory indices.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : str
            Range string like "0-3", "5-10"
            
        Returns
        -------
        List[int]
            List of trajectory indices from parsed range
        """
        try:
            start, end = selector.split("-")
            start_idx = int(start.strip())
            end_idx = int(end.strip())
            indices = list(range(start_idx, end_idx + 1))
            
            # Validate indices
            max_index = len(traj_data.trajectories) - 1
            for idx in indices:
                if idx < 0 or idx > max_index:
                    raise ValueError(f"Range contains invalid index {idx}")
            return indices
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid range format '{selector}'. Expected format: 'start-end'")
            raise
    
    @staticmethod
    def _resolve_comma_list(traj_data: TrajectoryData, selector: str) -> List[int]:
        """
        Parse comma-separated list and resolve to trajectory indices.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : str
            Comma list like "1,2,4,5"
            
        Returns
        -------
        List[int]
            List of trajectory indices from parsed list
        """
        try:
            indices = [int(idx.strip()) for idx in selector.split(",")]
            
            # Validate indices
            max_index = len(traj_data.trajectories) - 1
            for idx in indices:
                if idx < 0 or idx > max_index:
                    raise ValueError(f"List contains invalid index {idx}")
            return indices
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid comma list format '{selector}'. Expected format: 'num1,num2,num3'")
            raise
    
    @staticmethod
    def _resolve_int_selector(traj_data: TrajectoryData, selector: int) -> List[int]:
        """
        Resolve integer selector to single-element list.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : int
            Trajectory index
            
        Returns
        -------
        List[int]
            Single-element list with trajectory index
        """
        if 0 <= selector < len(traj_data.trajectories):
            return [selector]
        else:
            raise ValueError(f"Trajectory index {selector} out of range")
    
    @staticmethod
    def _resolve_fnmatch_pattern(traj_data: TrajectoryData, pattern: str) -> List[int]:
        """
        Resolve fnmatch pattern against trajectory names.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        pattern : str
            Fnmatch pattern (e.g., "system_*")
            
        Returns
        -------
        List[int]
            List of matching trajectory indices
        """
        matching_indices = []
        
        for idx, name in enumerate(traj_data.trajectory_names):
            if fnmatch.fnmatch(name, pattern):
                matching_indices.append(idx)
                
        if not matching_indices:
            raise ValueError(f"No trajectories match pattern '{pattern}'")
        return matching_indices
    
    @staticmethod
    def _resolve_regex_pattern(traj_data: TrajectoryData, pattern: str) -> List[int]:
        """
        Resolve regex pattern against trajectory names.
        
        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
        pattern : str
            Regex pattern
            
        Returns
        -------
        List[int]
            List of matching trajectory indices
        """
        matching_indices = []
        
        try:
            regex = re.compile(pattern)
            for idx, name in enumerate(traj_data.trajectory_names):
                if regex.search(name):
                    matching_indices.append(idx)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
            
        if not matching_indices:
            raise ValueError(f"No trajectories match regex pattern '{pattern}'")
        return matching_indices

    @staticmethod
    def _get_all_tags(traj_data: TrajectoryData) -> List[str]:
        """
        Get all unique tags from all trajectories.

        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns
        -------
        List[str]
            List of all unique tags
        """
        all_tags = set()
        if hasattr(traj_data, 'trajectory_tags') and traj_data.trajectory_tags:
            for tags in traj_data.trajectory_tags.values():
                all_tags.update(tags)
        return sorted(list(all_tags))

    @staticmethod
    def get_all_trajectories_with_tag(
        tag: str, traj_data: TrajectoryData
    ) -> List[int]:
        """
        Get all trajectory indices that have the specified tag.

        Parameters
        ----------
        tag : str
            Tag to search for
        traj_data : TrajectoryData
            Trajectory data object

        Returns
        -------
        List[int]
            List of trajectory indices that have the tag

        Examples
        --------
        >>> # Get all trajectories with "system_A" tag
        >>> indices = SelectionResolveHelper.get_all_trajectories_with_tag(
        ...     "system_A", traj_data
        ... )
        """
        return SelectionResolveHelper._find_trajectories_with_tag(tag, traj_data)
