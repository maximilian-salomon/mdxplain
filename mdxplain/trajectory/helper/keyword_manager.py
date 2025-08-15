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

"""Keyword management utilities for trajectory data."""

import fnmatch
from typing import List, Union, Dict
import re

from ..entities.trajectory_data import TrajectoryData


class KeywordManager:
    """Utility class for managing trajectory keywords."""
    #TODO: Use Tag instead of keywords.
    @staticmethod
    def resolve_trajectory_selectors(
        traj_data: TrajectoryData,
        trajectory_selector: Union[int, str, list, range, dict],
        keywords: List[str] = None,
    ) -> List[tuple]:
        """
        Resolve trajectory selectors to list of (indices, keywords) assignments.

        This method handles all selector types and returns a list of assignments
        for the TrajectoryManager to apply. It does NOT modify the trajectory data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        trajectory_selector : int, str, list, range, dict
            Flexible selector for trajectories
        keywords : list, optional
            List of keyword strings. Required when trajectory_selector is not dict.

        Returns:
        --------
        List[tuple]
            List of (indices, keywords) tuples for TrajectoryManager to apply

        Examples:
        ---------
        >>> # Returns: [(indices, keywords), ...]
        >>> assignments = KeywordManager.resolve_trajectory_selectors(
        ...     traj_data, 0, ["system_A"]
        ... )

        Raises:
        -------
        ValueError
            If keywords is None when trajectory_selector is not dict
        """
        assignments = []

        # Dict mode - bulk assignment
        if isinstance(trajectory_selector, dict):
            for selector, kw_list in trajectory_selector.items():
                resolved_indices = KeywordManager._resolve_trajectory_selectors(
                    traj_data, selector
                )
                assignments.append((resolved_indices, kw_list))
        else:
            # Single selector mode
            if keywords is None:
                raise ValueError(
                    "keywords parameter is required when trajectory_selector is not dict"
                )

            resolved_indices = KeywordManager._resolve_trajectory_selectors(
                traj_data, trajectory_selector
            )
            assignments.append((resolved_indices, keywords))

        return assignments

    @staticmethod
    def _resolve_trajectory_selectors(traj_data: TrajectoryData, selector) -> List[int]:
        """
        Resolve trajectory selectors to list of trajectory indices.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : int, str, list, range
            Trajectory selector to resolve

        Returns:
        --------
        list
            List of trajectory indices

        Raises:
        -------
        ValueError
            If selector contains invalid indices or names
        """
        if isinstance(selector, int):
            return KeywordManager._resolve_int_selector(traj_data, selector)
        elif isinstance(selector, str):
            return KeywordManager._resolve_string_selector(traj_data, selector)
        elif isinstance(selector, range):
            return KeywordManager._resolve_range_selector(traj_data, selector)
        elif isinstance(selector, list):
            return KeywordManager._resolve_list_selector(traj_data, selector)
        else:
            raise ValueError(f"Unsupported selector type: {type(selector)}")

    @staticmethod
    def _resolve_int_selector(traj_data: TrajectoryData, selector: int) -> List[int]:
        """
        Resolve integer selector to trajectory index.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : int
            Trajectory index

        Returns:
        --------
        list
            Single-element list with trajectory index

        Raises:
        -------
        ValueError
            If index is out of range
        """
        if 0 <= selector < len(traj_data.trajectories):
            return [selector]
        else:
            raise ValueError(f"Trajectory index {selector} out of range")

    @staticmethod
    def _resolve_string_selector(traj_data: TrajectoryData, selector: str) -> List[int]:
        """
        Resolve string selector with flexible syntax to trajectory indices.

        Supports multiple string formats:
        - Range: "0-3", "id 0-3" → [0, 1, 2, 3]
        - Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
        - Single number: "7", "id 7" → [7]
        - Fnmatch pattern: "system_*" → fnmatch pattern matching
        - Regex pattern: "regex system_\d+" → regex pattern matching

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : str
            Trajectory selector string

        Returns:
        --------
        list
            List of matching trajectory indices

        Raises:
        -------
        ValueError
            If selector format is invalid or no trajectories match
        """
        # Remove optional "id " prefix for parsing
        clean_selector = selector.removeprefix("id ").strip()

        # 1. Range: "0-3" → use range string parser
        if "-" in clean_selector and "," not in clean_selector:
            return KeywordManager._resolve_range_string(traj_data, clean_selector)

        # 2. Comma list: "1,2,4,5" → use comma list parser
        elif "," in clean_selector:
            return KeywordManager._resolve_comma_list(traj_data, clean_selector)

        # 3. Single number: "7" → use existing int resolver
        elif clean_selector.isdigit():
            return KeywordManager._resolve_int_selector(traj_data, int(clean_selector))

        # 4. Pattern: regex vs fnmatch
        else:
            if clean_selector.startswith("regex "):
                # Regex pattern
                regex_pattern = clean_selector.removeprefix(
                    "regex "
                ).strip()  # Remove "regex " prefix
                return KeywordManager._resolve_regex_pattern(traj_data, regex_pattern)
            else:
                # Fnmatch pattern
                return KeywordManager._resolve_fnmatch_pattern(
                    traj_data, clean_selector
                )

    @staticmethod
    def _resolve_range_selector(
        traj_data: TrajectoryData, selector: range
    ) -> List[int]:
        """
        Resolve range selector to trajectory indices.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : range
            Range of trajectory indices

        Returns:
        --------
        list
            List of trajectory indices from range

        Raises:
        -------
        ValueError
            If range contains invalid indices
        """
        indices = list(selector)
        max_index = len(traj_data.trajectories) - 1
        for idx in indices:
            if idx < 0 or idx > max_index:
                raise ValueError(f"Range contains invalid index {idx}")
        return indices

    @staticmethod
    def _resolve_list_selector(traj_data: TrajectoryData, selector: list) -> List[int]:
        """
        Resolve list selector recursively to trajectory indices.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : list
            List of nested selectors

        Returns:
        --------
        list
            List of unique trajectory indices (duplicates removed, order preserved)
        """
        all_indices = []
        for sub_selector in selector:
            resolved = KeywordManager._resolve_trajectory_selectors(
                traj_data, sub_selector
            )
            all_indices.extend(resolved)

        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in all_indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)

        return unique_indices

    @staticmethod
    def _resolve_range_string(traj_data: TrajectoryData, selector: str) -> List[int]:
        """
        Parse range string and resolve using existing range resolver.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : str
            Range string like "0-3", "5-10"

        Returns:
        --------
        list
            List of trajectory indices from parsed range

        Raises:
        -------
        ValueError
            If range format is invalid or indices are out of range
        """
        try:
            start, end = selector.split("-")
            start, end = int(start.strip()), int(end.strip())
        except ValueError:
            raise ValueError(f"Invalid range format: '{selector}'. Use 'X-Y'")

        # Create range and use existing resolver (reuses all validation)
        range_obj = range(start, end + 1)  # inclusive
        return KeywordManager._resolve_range_selector(traj_data, range_obj)

    @staticmethod
    def _resolve_comma_list(traj_data: TrajectoryData, selector: str) -> List[int]:
        """
        Parse comma-separated list and resolve using existing list resolver.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selector : str
            Comma-separated string like "1,2,4,5"

        Returns:
        --------
        list
            List of trajectory indices from parsed comma list

        Raises:
        -------
        ValueError
            If comma list format is invalid or indices are out of range
        """
        try:
            indices = [int(x.strip()) for x in selector.split(",")]
        except ValueError:
            raise ValueError(f"Invalid comma list: '{selector}'. Use 'X,Y,Z'")

        # Create list and use existing resolver (reuses all validation)
        return KeywordManager._resolve_list_selector(traj_data, indices)

    @staticmethod
    def _resolve_regex_pattern(traj_data: TrajectoryData, pattern: str) -> List[int]:
        """
        Resolve regex pattern against trajectory names.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        pattern : str
            Regex pattern to match against trajectory names

        Returns:
        --------
        list
            List of trajectory indices matching the regex pattern

        Raises:
        -------
        ValueError
            If regex pattern is invalid or no trajectories match
        """
        try:
            compiled_regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

        matched_indices = []
        for idx, name in enumerate(traj_data.trajectory_names):
            if compiled_regex.search(name):
                matched_indices.append(idx)

        if not matched_indices:
            raise ValueError(f"No trajectories match regex pattern '{pattern}'")
        return matched_indices

    @staticmethod
    def _resolve_fnmatch_pattern(traj_data: TrajectoryData, pattern: str) -> List[int]:
        """
        Resolve fnmatch pattern against trajectory names.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        pattern : str
            Fnmatch pattern to match against trajectory names (e.g., "system_*")

        Returns:
        --------
        list
            List of trajectory indices matching the fnmatch pattern

        Raises:
        -------
        ValueError
            If no trajectories match the pattern
        """
        matched_indices = []
        for idx, name in enumerate(traj_data.trajectory_names):
            if fnmatch.fnmatch(name, pattern):
                matched_indices.append(idx)

        if not matched_indices:
            raise ValueError(f"No trajectories match fnmatch pattern '{pattern}'")
        return matched_indices

    @staticmethod
    def build_frame_keyword_mapping(traj_data: TrajectoryData) -> Dict[int, List[str]]:
        """
        Build mapping from global frame indices to trajectory keywords.

        This method creates a mapping that allows quick lookup of keywords
        for any frame across all trajectories. Returns the mapping without
        modifying the trajectory data.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        Dict[int, List[str]]
            Dictionary mapping global frame indices to keyword lists

        Examples:
        ---------
        >>> frame_mapping = KeywordManager.build_frame_keyword_mapping(traj_data)
        >>> traj_data.frame_keyword_mapping = frame_mapping
        """
        frame_keyword_mapping = {}
        frame_offset = 0

        for traj_idx, traj in enumerate(traj_data.trajectories):
            # Get keywords for this trajectory
            keywords = traj_data.get_trajectory_keywords(traj_idx)

            if keywords:
                # Map all frames from this trajectory to these keywords
                for frame_idx in range(traj.n_frames):
                    global_frame_idx = frame_offset + frame_idx
                    frame_keyword_mapping[global_frame_idx] = keywords

            frame_offset += traj.n_frames

        return frame_keyword_mapping
