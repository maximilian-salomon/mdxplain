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
column selection. It supports selection based on tags, clusters,
and combinations thereof.
"""

from __future__ import annotations

from typing import List, Union, Dict, Any, TYPE_CHECKING

from ..entities.data_selector_data import DataSelectorData
from ..helpers.frame_selection_helper import FrameSelectionHelper
from ..helpers.criteria_builder_helper import CriteriaBuilderHelper
from ...utils.data_utils import DataUtils

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class DataSelectorManager:
    """
    Manager for creating and managing trajectory frame selections.

    This class provides methods to select trajectory frames (rows) based on
    various criteria such as tags, cluster assignments, or combinations.
    It serves as the counterpart to FeatureSelectorManager, focusing on
    row selection instead of column selection.

    The manager supports:

    - Tag-based frame selection
    - Cluster-based frame selection
    - Combination of multiple selections
    - Frame index range selection

    Examples
    --------
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

    def __init__(self) -> None:
        """
        Initialize the data selector manager.

        Returns
        -------
        None
            Initializes DataSelectorManager instance
        """
        pass

    def create(self, pipeline_data: PipelineData, name: str) -> None:
        """
        Create a new data selector with given name.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.create("folded_frames")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.create(pipeline_data, "folded_frames")  # WITH pipeline_data parameter

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to store the selector
        name : str
            Name for the new data selector

        Returns
        -------
        None
            Creates empty DataSelectorData in pipeline_data

        Raises
        ------
        ValueError
            If a selector with the given name already exists

        Examples
        --------
        >>> manager = DataSelectorManager()
        >>> manager.create(pipeline_data, "folded_frames")
        >>> manager.create(pipeline_data, "system_A_frames")
        """
        if name in pipeline_data.data_selector_data:
            raise ValueError(f"Data selector '{name}' already exists.")

        pipeline_data.data_selector_data[name] = DataSelectorData(name)

    def select_by_tags(
        self,
        pipeline_data: PipelineData,
        name: str,
        tags: List[str],
        match_all: bool = True,
        mode: str = "add",
        stride: int = 1,
    ) -> None:
        """
        Select frames based on trajectory tags.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.select_by_tags("biased_system_A", ["system_A", "biased"])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.select_by_tags(pipeline_data, "biased_system_A", ["system_A", "biased"])  # WITH pipeline_data parameter

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing trajectory data
        name : str
            Name of the data selector to populate
        tags : List[str]
            List of tags to search for
        match_all : bool, default=True
            If True, frame must have ALL tags. If False, ANY tag matches.
        mode : str, default="add"
            Selection mode: "add" (union), "subtract" (difference), "intersect" (intersection)
        stride : int, default=1
            Minimum distance between consecutive frames (per trajectory).
            stride=1 returns all frames, stride=10 returns every 10th frame.
        
        # TODO: We could use an Enum for modes

        Returns
        -------
        None
            Updates DataSelectorData with selected frame indices

        Raises
        ------
        ValueError
            If selector name doesn't exist or no trajectories loaded

        Examples
        --------
        >>> # Add frames with all specified tags
        >>> manager.select_by_tags(
        ...     pipeline_data, "biased_system_A", ["system_A", "biased"], match_all=True, mode="add"
        ... )

        >>> # Select every 5th frame from tagged trajectories
        >>> manager.select_by_tags(
        ...     pipeline_data, "biased_sparse", ["biased"], stride=5
        ... )

        >>> # Keep only frames that have these tags
        >>> manager.select_by_tags(
        ...     pipeline_data, "my_frames", ["production"], mode="intersect"
        ... )
        """
        # Validation using helper
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)
        FrameSelectionHelper.validate_trajectories_loaded(pipeline_data)

        selector_data = pipeline_data.data_selector_data[name]
        
        # Trajectory-specific frame selection using helper
        trajectory_frames = FrameSelectionHelper.select_frames_by_tags(
            pipeline_data.trajectory_data, tags, match_all, stride
        )
        
        # Update selector indices with trajectory-specific data
        self._update_selector_indices(selector_data, trajectory_frames, mode)
        
        # Build and store criteria using helper
        n_frames = sum(len(frames) for frames in trajectory_frames.values())
        criteria = CriteriaBuilderHelper.build_tag_criteria(
            tags, match_all, n_frames
        )
        criteria["mode"] = mode  # Add mode to criteria
        criteria["stride"] = stride  # Add stride to criteria
        
        # Use appropriate criteria method based on mode and existing data
        selector_data.append_selection_criteria(criteria)

    def select_by_cluster(
        self,
        pipeline_data: PipelineData,
        name: str,
        clustering_name: str,
        cluster_ids: List[Union[int, str]],
        mode: str = "add",
        stride: int = 1,
    ) -> None:
        """
        Select frames based on cluster assignments.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.select_by_cluster("structured", "conformations", [0, 1])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.select_by_cluster(pipeline_data, "structured", "conformations", [0, 1])  # WITH pipeline_data parameter

        Parameters
        ----------
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
        stride : int, default=1
            Minimum distance between consecutive frames (per trajectory).
            Applied after cluster selection to maintain cluster representation.

        Returns
        -------
        None
            Updates DataSelectorData with selected frame indices

        Raises
        ------
        ValueError
            If selector name doesn't exist or clustering not found

        Examples
        --------
        >>> # Add frames from specific clusters
        >>> manager.select_by_cluster(
        ...     pipeline_data, "structured", "conformations", [0, 1], mode="add"
        ... )

        >>> # Select every 10th frame from clusters (sparse sampling)
        >>> manager.select_by_cluster(
        ...     pipeline_data, "structured_sparse", "conformations", [0, 1], stride=10
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
        
        # Require frame_mapping for trajectory-specific operation
        frame_mapping = cluster_data.get_frame_mapping()
        if not frame_mapping:
            raise ValueError(
                f"Clustering '{clustering_name}' has no frame_mapping. "
                f"Re-run clustering to generate trajectory-specific mapping."
            )

        # Resolve cluster IDs using helper
        resolved_cluster_ids = FrameSelectionHelper.resolve_cluster_ids(
            cluster_data, cluster_ids, clustering_name
        )

        # Trajectory-specific frame selection using new helper
        trajectory_frames = FrameSelectionHelper.select_frames_by_cluster(
            labels, resolved_cluster_ids, frame_mapping, stride
        )

        # Update selector indices with trajectory-specific data
        self._update_selector_indices(selector_data, trajectory_frames, mode)

        # Build and store criteria using helper
        n_frames = sum(len(frames) for frames in trajectory_frames.values())
        criteria = CriteriaBuilderHelper.build_cluster_criteria(
            clustering_name, cluster_ids, resolved_cluster_ids, n_frames
        )
        criteria["mode"] = mode  # Add mode to criteria
        criteria["stride"] = stride  # Add stride to criteria

        # Use appropriate criteria method based on mode and existing data
        selector_data.append_selection_criteria(criteria)

    def select_by_indices(
        self,
        pipeline_data: PipelineData,
        name: str,
        trajectory_indices: Union[Dict[int, List[int]], Dict[str, List[int]], Dict[int, str], Dict[str, str]],
        mode: str = "add",
    ) -> None:
        """
        Select frames by explicit trajectory-specific frame indices.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.select_by_indices("custom_frames", {0: [10, 20], 1: [5, 15]})  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.select_by_indices(pipeline_data, "custom_frames", {0: [10, 20], 1: [5, 15]})  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the data selector to populate
        trajectory_indices : Dict[traj_selection, frame_selection]
            Dictionary mapping trajectory selectors to frame specifications.
            
            traj_selection:

                - int: trajectory index (0, 1, 2...)
                - str: trajectory name ("system_A"), tag ("tag:biased"), pattern ("system_*")
                - Can resolve to multiple trajectories (e.g., tags apply frames to all matching)
            
            frame_selection:

                - int: single frame (42)
                - List[int]: explicit frames ([10, 20, 30])
                - str: various formats:

                    * Single: "42"
                    * Range: "10-20" → [10, 11, ..., 20]
                    * Comma list: "10,20,30" → [10, 20, 30]
                    * Combined: "10-20,30-40,50" → [10...20, 30...40, 50]
                    * All: "all" → all frames in trajectory

                - dict: with stride support:
                
                    * {"frames": frame_selection, "stride": N}
                    * stride = minimum distance between consecutive frames
                    * Example: {"frames": "0-100", "stride": 10} → [0, 10, 20, ..., 100]
        mode : str, default="add"
            Selection mode: "add" (union), "subtract" (difference), "intersect" (intersection)

        Returns
        -------
        None
            Updates DataSelectorData with specified trajectory frame indices

        Examples
        --------
        >>> # Direct trajectory-specific selection
        >>> manager.select_by_indices(
        ...     pipeline_data, "custom_frames", 
        ...     {0: [10, 20, 30], 1: [5, 15, 25]}, mode="add"
        ... )

        >>> # Combined ranges
        >>> manager.select_by_indices(
        ...     pipeline_data, "complex_frames",
        ...     {0: "10-20,30-40,50", "system_A": "100-200"}, mode="add"
        ... )

        >>> # All frames from tagged trajectories
        >>> manager.select_by_indices(
        ...     pipeline_data, "all_biased",
        ...     {"tag:biased": "all"}, mode="add"
        ... )

        >>> # With stride for sparse sampling
        >>> manager.select_by_indices(
        ...     pipeline_data, "sparse_frames",
        ...     {0: {"frames": "0-1000", "stride": 50}}, mode="add"
        ... )

        >>> # Complex mixed example
        >>> manager.select_by_indices(
        ...     pipeline_data, "mixed_selection",
        ...     {
        ...         "system_A": {"frames": "10-20,100-200", "stride": 5},
        ...         "tag:biased": "all",
        ...         1: [42, 84, 126]
        ...     }, mode="add"
        ... )
        """
        # Validation using helper
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)

        selector_data = pipeline_data.data_selector_data[name]
        
        # Parse input to trajectory-specific format using helper
        trajectory_frames = FrameSelectionHelper.select_frames_by_indices(
            trajectory_indices, pipeline_data.trajectory_data
        )
        
        # Update selector indices with trajectory-specific data
        self._update_selector_indices(selector_data, trajectory_frames, mode)

        # Build and store criteria using helper
        n_frames = sum(len(frames) for frames in trajectory_frames.values())
        criteria = CriteriaBuilderHelper.build_indices_criteria(trajectory_frames, n_frames)
        criteria["mode"] = mode  # Add mode to criteria

        # Use appropriate criteria method based on mode and existing data
        selector_data.append_selection_criteria(criteria)


    def get_selection_info(self, pipeline_data: PipelineData, name: str) -> Dict[str, Any]:
        """
        Get information about a data selection.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the data selector

        Returns
        -------
        Dict[str, Any]
            Dictionary with selection information

        Examples
        --------
        >>> info = manager.get_selection_info(pipeline_data, "folded_frames")
        >>> print(f"Selected {info['n_frames']} frames")
        >>> print(f"Selection type: {info['selection_type']}")
        """
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)
        selector_data = pipeline_data.data_selector_data[name]
        return selector_data.get_selection_info()

    def list_selectors(self, pipeline_data: PipelineData) -> List[str]:
        """
        List all available data selectors.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns
        -------
        List[str]
            List of selector names

        Examples
        --------
        >>> selectors = manager.list_selectors(pipeline_data)
        >>> print(f"Available selectors: {selectors}")
        """
        return list(pipeline_data.data_selector_data.keys())

    def clear_selector(self, pipeline_data: PipelineData, name: str) -> None:
        """
        Clear all frames and criteria from a data selector.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.clear_selector("my_frames")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.clear_selector(pipeline_data, "my_frames")  # WITH pipeline_data parameter

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the selector to clear

        Returns
        -------
        None
            Clears all frames and criteria from the selector

        Examples
        --------
        >>> manager.clear_selector(pipeline_data, "my_selection")
        """
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)
        pipeline_data.data_selector_data[name].clear_selection()

    def remove_selector(self, pipeline_data: PipelineData, name: str) -> None:
        """
        Remove a data selector.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Name of the selector to remove

        Returns
        -------
        None
            Removes the selector from pipeline_data

        Examples
        --------
        >>> manager.remove_selector(pipeline_data, "old_selection")
        """
        FrameSelectionHelper.validate_selector_exists(pipeline_data, name)
        del pipeline_data.data_selector_data[name]

    # Private helper methods
    
    def _update_selector_indices(
        self, selector_data: DataSelectorData, trajectory_frames: Dict[int, List[int]], mode: str
    ) -> None:
        """
        Update selector with trajectory-specific frame indices based on mode.
        
        IMPORTANT: Automatically removes duplicates when storing frames.
        
        Parameters
        ----------
        selector_data : DataSelectorData
            Selector data object to update
        trajectory_frames : Dict[int, List[int]]
            Dictionary mapping trajectory indices to frame indices
        mode : str
            Operation mode: "add", "subtract", or "intersect"
            
        Returns
        -------
        None
            Updates selector_data in-place
        """
        current = selector_data.get_trajectory_frames()
        
        if mode == "add":
            result = self._union_frames(current, trajectory_frames)
        elif mode == "subtract":
            result = self._subtract_frames(current, trajectory_frames)
        elif mode == "intersect":
            result = self._intersect_frames(current, trajectory_frames)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Valid: 'add', 'subtract', 'intersect'")
        
        # Store with automatic duplicate removal
        selector_data.set_trajectory_frames(result)
    
    def _union_frames(self, current: Dict[int, List[int]], new_frames: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Union of frame sets per trajectory.
        Automatically removes duplicates and sorts.

        Parameters
        ----------
        current : Dict[int, List[int]]
            Current trajectory to frame mapping
        new_frames : Dict[int, List[int]]
            New trajectory to frame mapping to add

        Returns
        -------
        Dict[int, List[int]]
            Union of frame sets per trajectory
        """
        result = current.copy()
        
        for traj_idx, frames in new_frames.items():
            if traj_idx in result:
                # Merge with existing, remove duplicates, sort
                combined = set(result[traj_idx]) | set(frames)
                result[traj_idx] = sorted(list(combined))
            else:
                # New trajectory, ensure no duplicates and sorted
                result[traj_idx] = sorted(list(set(frames)))
        
        return result
    
    def _subtract_frames(self, current: Dict[int, List[int]], new_frames: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Subtract frame sets per trajectory.
        Result is always sorted with no duplicates.

        Parameters
        ----------
        current : Dict[int, List[int]]
            Current trajectory to frame mapping
        new_frames : Dict[int, List[int]]
            New trajectory to frame mapping to subtract

        Returns
        -------
        Dict[int, List[int]]
            Subtraction of frame sets per trajectory
        """
        result = current.copy()
        
        for traj_idx, frames in new_frames.items():
            if traj_idx in result:
                remaining = set(result[traj_idx]) - set(frames)
                if remaining:
                    result[traj_idx] = sorted(list(remaining))
                else:
                    # No frames left, remove trajectory
                    del result[traj_idx]
        
        return result
    
    def _intersect_frames(self, current: Dict[int, List[int]], new_frames: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Intersect frame sets per trajectory.
        Result is always sorted with no duplicates.

        Parameters
        ----------
        current : Dict[int, List[int]]
            Current trajectory to frame mapping
        new_frames : Dict[int, List[int]]
            New trajectory to frame mapping to intersect

        Returns
        -------
        Dict[int, List[int]]
            Intersection of frame sets per trajectory
        """
        result = {}
        
        for traj_idx in current:
            if traj_idx in new_frames:
                common = set(current[traj_idx]) & set(new_frames[traj_idx])
                if common:
                    result[traj_idx] = sorted(list(common))
        
        return result
    
    def save(self, pipeline_data: PipelineData, save_path: str) -> None:
        """
        Save all data selector data to single file.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.save('data_selector.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.save(pipeline_data, 'data_selector.npy')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with data selector data
        save_path : str
            Path where to save all data selector data in one file

        Returns
        -------
        None
            Saves all data selector data to the specified file
            
        Examples
        --------
        >>> manager.save(pipeline_data, 'data_selector.npy')
        """
        DataUtils.save_object(pipeline_data.data_selector_data, save_path)

    def load(self, pipeline_data: PipelineData, load_path: str) -> None:
        """
        Load all data selector data from single file.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.load('data_selector.npy')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.load(pipeline_data, 'data_selector.npy')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container to load data selector data into
        load_path : str
            Path to saved data selector data file

        Returns
        -------
        None
            Loads all data selector data from the specified file
            
        Examples
        --------
        >>> manager.load(pipeline_data, 'data_selector.npy')
        """
        temp_dict = {}
        DataUtils.load_object(temp_dict, load_path)
        pipeline_data.data_selector_data = temp_dict

    def print_info(self, pipeline_data: PipelineData) -> None:
        """
        Print data selector information.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.data_selector.print_info()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = DataSelectorManager()
        >>> manager.print_info(pipeline_data)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with data selector data

        Returns
        -------
        None
            Prints data selector information to console

        Examples
        --------
        >>> manager.print_info(pipeline_data)
        """
        if len(pipeline_data.data_selector_data) == 0:
            print("No dataselectordata data available.")
            return

        print("=== DataSelectorData Information ===")
        data_names = list(pipeline_data.data_selector_data.keys())
        print(f"DataSelectorData Names: {len(data_names)} ({', '.join(data_names)})")
        
        for name, data in pipeline_data.data_selector_data.items():
            print(f"\n--- {name} ---")
            data.print_info()
