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
Criteria builder helper for data selector operations.

This module provides helper methods for building selection criteria
dictionaries that describe how frame selections were created.
"""

from typing import List, Dict, Any, Union


class CriteriaBuilderHelper:
    """
    Helper class for building selection criteria dictionaries.
    
    Provides static methods for creating standardized criteria dictionaries
    that describe how frame selections were created. These dictionaries are
    stored in DataSelectorData objects for documentation and debugging.
    
    Examples:
    ---------
    >>> # Build tag criteria
    >>> criteria = CriteriaBuilderHelper.build_tag_criteria(
    ...     ["system_A", "biased"], True, 150
    ... )
    
    >>> # Build cluster criteria
    >>> criteria = CriteriaBuilderHelper.build_cluster_criteria(
    ...     "conformations", [0, 1], [0, 1], 300
    ... )
    """
    
    @staticmethod
    def build_tag_criteria(
        tags: List[str], match_all: bool, n_matches: int
    ) -> Dict[str, Any]:
        """
        Build criteria dictionary for tag-based selection.
        
        Creates a standardized criteria dictionary that documents how
        a tag-based frame selection was performed.
        
        Parameters:
        -----------
        tags : List[str]
            List of tags used for selection
        match_all : bool
            Whether all tags were required (True) or any tag (False)
        n_matches : int
            Number of frames that matched the criteria
            
        Returns:
        --------
        Dict[str, Any]
            Criteria dictionary with selection documentation
            
        Examples:
        ---------
        >>> criteria = CriteriaBuilderHelper.build_tag_criteria(
        ...     ["system_A", "biased"], True, 150
        ... )
        >>> print(criteria["type"])  # "tags"
        >>> print(criteria["n_matches"])  # 150
        """
        return {
            "type": "tags",
            "tags": tags,
            "match_all": match_all,
            "n_matches": n_matches,
        }
    
    @staticmethod
    def build_cluster_criteria(
        clustering_name: str,
        cluster_ids: List[Union[int, str]],
        resolved_cluster_ids: List[int],
        n_matches: int
    ) -> Dict[str, Any]:
        """
        Build criteria dictionary for cluster-based selection.
        
        Creates a standardized criteria dictionary that documents how
        a cluster-based frame selection was performed, including both
        original and resolved cluster IDs.
        
        Parameters:
        -----------
        clustering_name : str
            Name of the clustering used for selection
        cluster_ids : List[Union[int, str]]
            Original cluster identifiers (may include names)
        resolved_cluster_ids : List[int]
            Numeric cluster IDs after name resolution
        n_matches : int
            Number of frames that matched the criteria
            
        Returns:
        --------
        Dict[str, Any]
            Criteria dictionary with selection documentation
            
        Examples:
        ---------
        >>> criteria = CriteriaBuilderHelper.build_cluster_criteria(
        ...     "conformations", ["folded", 1], [0, 1], 300
        ... )
        >>> print(criteria["type"])  # "cluster"
        >>> print(criteria["clustering_name"])  # "conformations"
        """
        return {
            "type": "cluster",
            "clustering_name": clustering_name,
            "cluster_ids": cluster_ids,
            "resolved_cluster_ids": resolved_cluster_ids,
            "n_matches": n_matches,
        }
    
    @staticmethod
    def build_indices_criteria(trajectory_frames: Dict[int, List[int]], n_matches: int) -> Dict[str, Any]:
        """
        Build criteria dictionary for trajectory-specific explicit indices selection.
        
        Creates a standardized criteria dictionary that documents how
        a trajectory-specific explicit indices-based frame selection was performed.
        
        Parameters:
        -----------
        trajectory_frames : Dict[int, List[int]]
            Dictionary mapping trajectory indices to their selected frame indices
        n_matches : int
            Number of frames selected
            
        Returns:
        --------
        Dict[str, Any]
            Criteria dictionary with selection documentation
            
        Examples:
        ---------
        >>> trajectory_frames = {0: [0, 10, 20], 1: [100, 200]}
        >>> criteria = CriteriaBuilderHelper.build_indices_criteria(trajectory_frames, 5)
        >>> print(criteria["type"])  # "indices"
        >>> print(criteria["n_matches"])  # 5
        >>> print(criteria["n_trajectories"])  # 2
        """
        return {
            "type": "indices",
            "trajectory_frames": trajectory_frames,
            "n_matches": n_matches,
            "n_trajectories": len(trajectory_frames),
        }
    
    @staticmethod
    def build_combination_criteria(
        source_selectors: List[str], combination_mode: str, n_matches: int
    ) -> Dict[str, Any]:
        """
        Build criteria dictionary for combination-based selection.
        
        Creates a standardized criteria dictionary that documents how
        a combination of multiple selectors was performed.
        
        Parameters:
        -----------
        source_selectors : List[str]
            Names of the source selectors that were combined
        combination_mode : str
            Mode used for combination ("union", "intersection", "difference")
        n_matches : int
            Number of frames in the final combined selection
            
        Returns:
        --------
        Dict[str, Any]
            Criteria dictionary with selection documentation
            
        Examples:
        ---------
        >>> criteria = CriteriaBuilderHelper.build_combination_criteria(
        ...     ["folded_frames", "biased_frames"], "intersection", 75
        ... )
        >>> print(criteria["type"])  # "combination"
        >>> print(criteria["combination_mode"])  # "intersection"
        """
        return {
            "type": "combination",
            "source_selectors": source_selectors,
            "combination_mode": combination_mode,
            "n_matches": n_matches,
        }