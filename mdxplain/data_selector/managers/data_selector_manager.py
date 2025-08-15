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
Data selector manager for trajectory frame selection.

This module provides the DataSelectorManager class that manages frame
selection (row selection) as the counterpart to FeatureSelector's
column selection. It supports selection based on keywords, clusters,
and combinations thereof.
"""

from typing import List, Union, Optional, Dict, Any, Set
import numpy as np

from ..entities.data_selector_data import DataSelectorData
from ..helpers.frame_selection_helper import FrameSelectionHelper
from ..helpers.criteria_builder_helper import CriteriaBuilderHelper


class DataSelectorManager:
    """
    Manager for creating and managing trajectory frame selections.

    This class provides methods to select trajectory frames (rows) based on
    various criteria such as keywords, cluster assignments, or combinations.
    It serves as the counterpart to FeatureSelectorManager, focusing on
    row selection instead of column selection.

    The manager supports:
    - Keyword-based frame selection
    - Cluster-based frame selection
    - Combination of multiple selections
    - Frame index range selection

    Examples:
    ---------
    Pipeline mode (automatic injection):

    >>> pipeline = PipelineManager()
    >>> pipeline.data_selector.create("folded_frames")
    >>> pipeline.data_selector.select_by_cluster("folded_frames", "conformations", [0])

    Standalone mode:

    >>> pipeline_data = PipelineData()
    >>> manager = DataSelectorManager()
    >>> manager.create(pipeline_data, "folded_frames")
    >>> manager.select_by_cluster(pipeline_data, "folded_frames", "conformations", [0])
    """

    def __init__(self):
        """
        Initialize the data selector manager.

        Returns:
        --------
        None
            Initializes DataSelectorManager instance
        """
        pass

    def create(self, pipeline_data, name: str) -> None:
        """
        Create a new data selector with given name.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.create("folded_frames")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.create(pipeline_data, "folded_frames")  # WITH pipeline_data parameter

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to store the selector
        name : str
            Name for the new data selector

        Returns:
        --------
        None
            Creates empty DataSelectorData in pipeline_data

        Raises:
        -------
        ValueError
            If a selector with the given name already exists

        Examples:
        ---------
        >>> manager = DataSelectorManager()
        >>> manager.create(pipeline_data, "folded_frames")
        >>> manager.create(pipeline_data, "system_A_frames")
        """
        if name in pipeline_data.data_selector_data:
            raise ValueError(f"Data selector '{name}' already exists.")

        pipeline_data.data_selector_data[name] = DataSelectorData(name)

    def select_by_keywords(
        self,
        pipeline_data,
        name: str,
        keywords: List[str],
        match_all: bool = True,
        mode: str = "add",
    ) -> None:
        """
        Select frames based on trajectory keywords.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.select_by_keywords("biased_system_A", ["system_A", "biased"])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.select_by_keywords(pipeline_data, "biased_system_A", ["system_A", "biased"])  # WITH pipeline_data parameter

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing trajectory data
        name : str
            Name of the data selector to populate
        keywords : List[str]
            List of keywords to search for
        match_all : bool, default=True
            If True, frame must have ALL keywords. If False, ANY keyword matches.
        mode : str, default="add"
            Selection mode: "add" (union), "subtract" (difference), "intersect" (intersection)
        # TODO: We could use an Enum for modes

        Returns:
        --------
        None
            Updates DataSelectorData with selected frame indices

        Raises:
        -------
        ValueError
            If selector name doesn't exist or no trajectories loaded

        Examples:
        ---------
        >>> # Add frames with all specified keywords
        >>> manager.select_by_keywords(
        ...     pipeline_data, "biased_system_A", ["system_A", "biased"], match_all=True, mode="add"
        ... )

        >>> # Remove frames with any of the keywords
        >>> manager.select_by_keywords(
        ...     pipeline_data, "my_frames", ["system_A", "system_B"], match_all=False, mode="subtract"
        ... )

        >>> # Keep only frames that have these keywords
        >>> manager.select_by_keywords(
        ...     pipeline_data, "my_frames", ["production"], mode="intersect"
        ... )
        """
        # Validation using helper
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)
        FrameSelectionHelper.validate_trajectories_loaded(pipeline_data)

        selector_data = pipeline_data.data_selector_data[name]
        
        # Frame selection using helper
        frame_indices = FrameSelectionHelper.select_frames_by_keywords(
            pipeline_data.trajectory_data, keywords, match_all
        )
        
        # Update selector indices
        self._update_selector_indices(selector_data, frame_indices, mode)
        
        # Build and store criteria using helper
        criteria = CriteriaBuilderHelper.build_keyword_criteria(
            keywords, match_all, len(frame_indices)
        )
        criteria["mode"] = mode  # Add mode to criteria
        
        # Use appropriate criteria method based on mode and existing data
        selector_data.append_selection_criteria(criteria)

    def select_by_cluster(
        self,
        pipeline_data,
        name: str,
        clustering_name: str,
        cluster_ids: List[Union[int, str]],
        mode: str = "add",
    ) -> None:
        """
        Select frames based on cluster assignments.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.select_by_cluster("structured", "conformations", [0, 1])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.select_by_cluster(pipeline_data, "structured", "conformations", [0, 1])  # WITH pipeline_data parameter

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing cluster data
        name : str
            Name of the data selector to populate
        clustering_name : str
            Name of the clustering result to use
        cluster_ids : List[Union[int, str]]
            List of cluster IDs to select (can be numeric IDs or cluster names)
        mode : str, default="add"
            Selection mode: "add" (union), "subtract" (difference), "intersect" (intersection)

        Returns:
        --------
        None
            Updates DataSelectorData with selected frame indices

        Raises:
        -------
        ValueError
            If selector name doesn't exist or clustering not found

        Examples:
        ---------
        >>> # Add frames from specific clusters
        >>> manager.select_by_cluster(
        ...     pipeline_data, "structured", "conformations", [0, 1], mode="add"
        ... )

        >>> # Remove frames from specific clusters
        >>> manager.select_by_cluster(
        ...     pipeline_data, "my_frames", "conformations", ["noise"], mode="subtract"
        ... )

        >>> # Keep only frames from these clusters
        >>> manager.select_by_cluster(
        ...     pipeline_data, "my_frames", "conformations", ["folded", "intermediate"], mode="intersect"
        ... )
        """
        # Validation using helpers
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)
        FrameSelectionHelper.validate_clustering_exists(pipeline_data, clustering_name)

        selector_data = pipeline_data.data_selector_data[name]

        # Get cluster data and validate
        cluster_data = pipeline_data.cluster_data[clustering_name]
        labels = cluster_data.get_labels()
        if labels is None:
            raise ValueError(f"No cluster labels found for clustering '{clustering_name}'")

        # Resolve cluster IDs using helper
        resolved_cluster_ids = FrameSelectionHelper.resolve_cluster_ids(
            cluster_data, cluster_ids, clustering_name
        )

        # Select frames using helper
        frame_indices = FrameSelectionHelper.select_frames_by_cluster(
            labels, resolved_cluster_ids
        )

        # Update selector indices
        self._update_selector_indices(selector_data, frame_indices, mode)

        # Build and store criteria using helper
        criteria = CriteriaBuilderHelper.build_cluster_criteria(
            clustering_name, cluster_ids, resolved_cluster_ids, len(frame_indices)
        )
        criteria["mode"] = mode  # Add mode to criteria

        # Use appropriate criteria method based on mode and existing data
        selector_data.append_selection_criteria(criteria)

    def select_by_indices(
        self,
        pipeline_data,
        name: str,
        indices: List[int],
        mode: str = "add",
    ) -> None:
        """
        Select frames by explicit frame indices.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the data selector to populate
        indices : List[int]
            List of frame indices to select
        mode : str, default="add"
            Selection mode: "add" (union), "subtract" (difference), "intersect" (intersection)

        Returns:
        --------
        None
            Updates DataSelectorData with specified frame indices

        Examples:
        ---------
        >>> # Add specific frame indices
        >>> manager.select_by_indices(
        ...     pipeline_data, "custom_frames", [0, 10, 20, 100, 200], mode="add"
        ... )

        >>> # Remove specific frame indices
        >>> manager.select_by_indices(
        ...     pipeline_data, "my_frames", [5, 15, 25], mode="subtract"
        ... )
        """
        # Validation using helper
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)

        selector_data = pipeline_data.data_selector_data[name]
        
        # Update selector indices
        self._update_selector_indices(selector_data, indices, mode)

        # Build and store criteria using helper
        criteria = CriteriaBuilderHelper.build_indices_criteria(indices, len(indices))
        criteria["mode"] = mode  # Add mode to criteria

        # Use appropriate criteria method based on mode and existing data
        selector_data.append_selection_criteria(criteria)


    def get_selection_info(self, pipeline_data, name: str) -> Dict[str, Any]:
        """
        Get information about a data selection.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the data selector

        Returns:
        --------
        Dict[str, Any]
            Dictionary with selection information

        Examples:
        ---------
        >>> info = manager.get_selection_info(pipeline_data, "folded_frames")
        >>> print(f"Selected {info['n_frames']} frames")
        >>> print(f"Selection type: {info['selection_type']}")
        """
        self._validate_selector_exists(pipeline_data, name)
        selector_data = pipeline_data.data_selector_data[name]
        return selector_data.get_selection_info()

    def list_selectors(self, pipeline_data) -> List[str]:
        """
        List all available data selectors.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns:
        --------
        List[str]
            List of selector names

        Examples:
        ---------
        >>> selectors = manager.list_selectors(pipeline_data)
        >>> print(f"Available selectors: {selectors}")
        """
        return list(pipeline_data.data_selector_data.keys())

    def clear_selector(self, pipeline_data, name: str) -> None:
        """
        Clear all frames and criteria from a data selector.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.clear_selector("my_frames")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.clear_selector(pipeline_data, "my_frames")  # WITH pipeline_data parameter

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the selector to clear

        Returns:
        --------
        None
            Clears all frames and criteria from the selector

        Examples:
        ---------
        >>> manager.clear_selector(pipeline_data, "my_selection")
        """
        self._validate_selector_exists(pipeline_data, name)
        pipeline_data.data_selector_data[name].clear_selection()

    def remove_selector(self, pipeline_data, name: str) -> None:
        """
        Remove a data selector.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the selector to remove

        Returns:
        --------
        None
            Removes the selector from pipeline_data

        Examples:
        ---------
        >>> manager.remove_selector(pipeline_data, "old_selection")
        """
        self._validate_selector_exists(pipeline_data, name)
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)
        del pipeline_data.data_selector_data[name]

    # Private helper methods
    
    def _update_selector_indices(
        self, selector_data, frame_indices: List[int], mode: str
    ) -> None:
        """
        Update selector indices based on mode.
        
        Helper method that handles frame index updates for different modes:
        add (union), subtract (difference), and intersect (intersection).
        
        Parameters:
        -----------
        selector_data : DataSelectorData
            Selector data object to update
        frame_indices : List[int]
            Frame indices for the operation
        mode : str
            Operation mode: "add", "subtract", or "intersect"
            
        Returns:
        --------
        None
            Updates selector_data in-place
        """
        current_indices = set(selector_data.get_frame_indices())
        new_indices = set(frame_indices)
        
        if mode == "add":
            # Union: add new indices to existing
            if not current_indices:
                # First operation - just set the indices
                result_indices = new_indices
            else:
                result_indices = current_indices | new_indices
        elif mode == "subtract":
            # Difference: remove new indices from existing
            result_indices = current_indices - new_indices
        elif mode == "intersect":
            # Intersection: keep only common indices
            if not current_indices:
                # If no existing indices, intersection is empty
                result_indices = set()
            else:
                result_indices = current_indices & new_indices
        else:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: 'add', 'subtract', 'intersect'")
        
        # Update the selector with sorted indices
        selector_data.set_frame_indices(sorted(list(result_indices)))

    def _validate_selector_exists(self, pipeline_data, name: str) -> None:
        """
        Validate that a data selector with given name exists.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing selector data
        name : str
            Name of the selector to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if selector not found
        """
        if name not in pipeline_data.data_selector_data:
            available = list(pipeline_data.data_selector_data.keys())
            raise ValueError(
                f"Data selector '{name}' not found. Available selectors: {available}"
            )