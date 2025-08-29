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
Frame selection helper for data selector operations.

This module provides helper methods for selecting frames based on
tags and cluster assignments, extracting common logic from
DataSelectorManager methods.
"""

from __future__ import annotations

from typing import List, Union, Dict, Optional, TYPE_CHECKING
import numpy as np

from ...trajectory.entities.trajectory_data import TrajectoryData
from ...clustering.entities.cluster_data import ClusterData

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class FrameSelectionHelper:
    """
    Helper class for frame selection operations.
    
    Provides static methods for selecting trajectory frames based on
    various criteria such as tags and cluster assignments. These
    methods extract common logic from DataSelectorManager to improve
    code organization and reusability.
    
    Examples:
    ---------
    >>> # Select frames by tags
    >>> indices = FrameSelectionHelper.select_frames_by_tags(
    ...     trajectory_data, ["system_A", "biased"], match_all=True
    ... )
    
    >>> # Select frames by cluster
    >>> indices = FrameSelectionHelper.select_frames_by_cluster(
    ...     labels, [0, 1, 2]
    ... )
    """
    
    @staticmethod
    def select_frames_by_tags(
        trajectory_data: TrajectoryData, tags: List[str], match_all: bool, stride: int = 1
    ) -> Dict[int, List[int]]:
        """
        Select frames from trajectories that match tag criteria.
        
        Returns all frames from trajectories whose tags match the criteria,
        optionally applying stride for sparse sampling.
        
        Parameters:
        -----------
        trajectory_data : TrajectoryData
            Trajectory data object containing trajectory tags
        tags : List[str]
            List of tags to search for in trajectory tags
        match_all : bool
            If True, trajectory must have ALL tags. If False, ANY tag matches.
        stride : int, default=1
            Minimum distance between consecutive frames (per trajectory).
            stride=1 returns all frames, stride=10 returns every 10th frame.
            
        Returns:
        --------
        Dict[int, List[int]]
            Dictionary mapping trajectory indices to their frame indices
            
        Examples:
        ---------
        >>> # Select every 10th frame from matching trajectories
        >>> frames = select_frames_by_tags(
        ...     trajectory_data, ["system_A"], match_all=False, stride=10
        ... )
        """
        trajectory_frames = {}
        
        for traj_idx, traj in enumerate(trajectory_data.trajectories):
            if FrameSelectionHelper._trajectory_matches_tags(
                trajectory_data, traj_idx, tags, match_all
            ):
                frames = list(range(traj.n_frames))
                # Apply stride if specified
                if stride > 1:
                    frames = FrameSelectionHelper._apply_stride(frames, stride)
                trajectory_frames[traj_idx] = frames
        
        return trajectory_frames
    
    @staticmethod
    def _trajectory_matches_tags(trajectory_data: TrajectoryData, traj_idx: int, tags: List[str], match_all: bool) -> bool:
        """Check if trajectory tags match criteria."""
        if traj_idx not in trajectory_data.trajectory_tags:
            return False
        
        traj_tags = set(trajectory_data.trajectory_tags[traj_idx])
        
        if match_all:
            return all(tag in traj_tags for tag in tags)
        else:
            return any(tag in traj_tags for tag in tags)
    
    @staticmethod
    def select_frames_by_cluster(
        labels: List[int], cluster_ids: List[int], frame_mapping: Optional[Dict[int, int]] = None, stride: int = 1
    ) -> Dict[int, List[int]]:
        """
        Select frames based on cluster assignments.
        
        Requires frame_mapping for trajectory-specific selection.
        Optionally applies stride for sparse sampling per trajectory.
        
        Parameters:
        -----------
        labels : List[int]
            List of cluster labels for each frame
        cluster_ids : List[int]
            List of cluster IDs to select frames from
        frame_mapping : Dict[int, tuple], optional
            Mapping from global frame index to (traj_idx, local_frame_idx)
        stride : int, default=1
            Minimum distance between consecutive frames (per trajectory).
            Applied after cluster selection to maintain cluster representation.
            
        Returns:
        --------
        Dict[int, List[int]]
            Dictionary mapping trajectory indices to their selected frame indices
            
        Examples:
        ---------
        >>> # Select every 5th frame from clusters (per trajectory)
        >>> frames = select_frames_by_cluster(
        ...     labels, [0, 1], frame_mapping, stride=5
        ... )
        """
        if frame_mapping is None:
            raise ValueError("frame_mapping is required for trajectory-specific selection")
        
        # Collect frames per trajectory
        trajectory_frames = FrameSelectionHelper._collect_cluster_frames(
            labels, cluster_ids, frame_mapping
        )
        
        # Apply stride per trajectory
        return FrameSelectionHelper._apply_stride_per_trajectory(trajectory_frames, stride)
    
    @staticmethod
    def _collect_cluster_frames(labels: np.ndarray, cluster_ids: List[int], frame_mapping: Dict[int, int]) -> Dict[int, List[int]]:
        """Collect frames belonging to specified clusters."""
        trajectory_frames = {}
        
        for global_idx, label in enumerate(labels):
            if label in cluster_ids:
                traj_idx, local_idx = frame_mapping[global_idx]
                if traj_idx not in trajectory_frames:
                    trajectory_frames[traj_idx] = []
                trajectory_frames[traj_idx].append(local_idx)
        
        # Sort frames per trajectory
        for traj_idx in trajectory_frames:
            trajectory_frames[traj_idx] = sorted(trajectory_frames[traj_idx])
        
        return trajectory_frames
    
    @staticmethod
    def _apply_stride_per_trajectory(trajectory_frames: Dict[int, List[int]], stride: int) -> Dict[int, List[int]]:
        """Sort and apply stride to frames per trajectory."""
        result = {}
        for traj_idx, frames in trajectory_frames.items():
            sorted_frames = sorted(frames)
            if stride > 1:
                sorted_frames = FrameSelectionHelper._apply_stride(sorted_frames, stride)
            result[traj_idx] = sorted_frames
        return result
    
    @staticmethod
    def resolve_cluster_ids(
        cluster_data: ClusterData, cluster_ids: List[Union[int, str]], clustering_name: str
    ) -> List[int]:
        """
        Convert cluster names to numeric IDs if necessary.
        
        Processes a list of cluster identifiers, converting string cluster names
        to their corresponding numeric IDs using the cluster's name mappings.
        Numeric IDs are passed through unchanged.
        
        Parameters:
        -----------
        cluster_data : ClusterData
            Cluster data object containing labels and optional cluster names
        cluster_ids : List[Union[int, str]]
            List of cluster identifiers (numeric IDs or string names)
        clustering_name : str
            Name of the clustering (used for error messages)
            
        Returns:
        --------
        List[int]
            List of numeric cluster IDs corresponding to input identifiers
            
        Examples:
        ---------
        >>> # Convert mixed IDs and names
        >>> resolved = FrameSelectionHelper.resolve_cluster_ids(
        ...     cluster_data, [0, "folded", 2], "conformations"
        ... )
        """
        resolved_ids = []

        for cluster_id in cluster_ids:
            if isinstance(cluster_id, int):
                resolved_ids.append(cluster_id)
            elif isinstance(cluster_id, str):
                # Check if cluster has named clusters
                if cluster_data.cluster_names:
                    # Find numeric ID for this name
                    found = False
                    for numeric_id, name in cluster_data.cluster_names.items():
                        if name == cluster_id:
                            resolved_ids.append(numeric_id)
                            found = True
                            break
                    if not found:
                        available_names = list(cluster_data.cluster_names.values())
                        raise ValueError(
                            f"Cluster name '{cluster_id}' not found in clustering "
                            f"'{clustering_name}'. Available names: {available_names}"
                        )
                else:
                    raise ValueError(
                        f"String cluster ID '{cluster_id}' provided but clustering "
                        f"'{clustering_name}' has no named clusters"
                    )
            else:
                raise TypeError(f"Cluster ID must be int or str, got {type(cluster_id)}")

        return resolved_ids
    
    @staticmethod
    def select_frames_by_indices(
        input_data: Union[str, dict, List[int]], trajectory_data: TrajectoryData
    ) -> Dict[int, List[int]]:
        """
        Parse trajectory and frame selections from various input formats.
        
        Structure:
        ----------
        input_data : Dict[traj_selection, frame_selection]
        
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
        
        Parameters:
        -----------
        input_data : dict
            Dictionary with trajectory keys and frame specifications
        trajectory_data : TrajectoryData
            Trajectory data object for validation and resolution
            
        Returns:
        --------
        Dict[int, List[int]]
            Dictionary mapping trajectory indices to frame indices
            
        Examples:
        ---------
        >>> # Combined ranges
        >>> frames = select_frames_by_indices({0: "10-20,30-40,50"}, trajectory_data)
        
        >>> # All frames
        >>> frames = select_frames_by_indices({"tag:biased": "all"}, trajectory_data)
        
        >>> # With stride
        >>> frames = select_frames_by_indices({
        ...     0: {"frames": "0-1000", "stride": 50}
        ... }, trajectory_data)
        
        >>> # Complex example
        >>> frames = select_frames_by_indices({
        ...     "system_A": {"frames": "10-20,100-200", "stride": 5},
        ...     "tag:biased": "all",
        ...     1: [42, 84, 126]
        ... }, trajectory_data)
        """
        if not isinstance(input_data, dict):
            raise TypeError(f"Expected dict, got {type(input_data)}")
        
        trajectory_frames = {}
        
        for traj_key, frame_value in input_data.items():
            # Resolve trajectory key - can return MULTIPLE trajectories
            traj_indices = FrameSelectionHelper._resolve_trajectory_key(traj_key, trajectory_data)
            # Apply frames to ALL resolved trajectories
            for traj_idx in traj_indices:
                frames = FrameSelectionHelper._parse_frame_value(
                    frame_value, trajectory_data, traj_idx
                )

                # Validate and store frames
                FrameSelectionHelper._validate_and_store_frames(
                    trajectory_frames, traj_idx, frames, trajectory_data
                )
        
        # Remove duplicates per trajectory
        return FrameSelectionHelper._finalize_trajectory_frames(trajectory_frames)
    
    @staticmethod
    def _validate_and_store_frames(trajectory_frames: Dict[int, List[int]], traj_idx: int, frames: List[int], trajectory_data: TrajectoryData) -> None:
        """
        Validate and store frames for a trajectory.
        
        Validates that frame indices are within bounds for the trajectory
        and adds them to the trajectory_frames dictionary.
        
        Parameters:
        -----------
        trajectory_frames : Dict[int, List[int]]
            Dictionary mapping trajectory indices to frame lists to update
        traj_idx : int
            Index of the trajectory to store frames for
        frames : List[int]
            Frame indices to validate and store
        trajectory_data : TrajectoryData
            Trajectory data object for validation
            
        Returns:
        --------
        None
            Updates trajectory_frames dictionary in-place
        """
        FrameSelectionHelper._validate_frame_indices(frames, traj_idx, trajectory_data)
        
        if traj_idx not in trajectory_frames:
            trajectory_frames[traj_idx] = []
        trajectory_frames[traj_idx].extend(frames)
    
    @staticmethod  
    def _finalize_trajectory_frames(trajectory_frames: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Remove duplicates and sort frames per trajectory.
        
        Takes a dictionary of trajectory frames and returns a cleaned version
        with duplicates removed and frames sorted in ascending order.
        
        Parameters:
        -----------
        trajectory_frames : Dict[int, List[int]]
            Dictionary mapping trajectory indices to lists of frame indices
            
        Returns:
        --------
        Dict[int, List[int]]
            Dictionary with deduplicated and sorted frame indices per trajectory
        """
        result = {}
        for traj_idx, frames in trajectory_frames.items():
            result[traj_idx] = sorted(list(set(frames)))
        return result
    
    @staticmethod
    def _resolve_trajectory_key(traj_key: Union[int, str, List], trajectory_data: TrajectoryData) -> List[int]:
        """
        Resolve trajectory key to list of indices using get_trajectory_indices().
        
        Converts various trajectory selection formats to a list of trajectory indices.
        Handles single integers, strings (names, tags, patterns), and lists.
        
        Parameters:
        -----------
        traj_key : Union[int, str, List]
            Trajectory selection key to resolve:
            - int: single trajectory index
            - str: trajectory name, tag, or pattern
            - List: list of indices/names/tags
        trajectory_data : TrajectoryData
            Trajectory data object for index resolution
            
        Returns:
        --------
        List[int]
            List of resolved trajectory indices
        """
        if isinstance(traj_key, int):
            return [traj_key]
        
        # Use get_trajectory_indices for string resolution
        return trajectory_data.get_trajectory_indices(traj_key)
    
    @staticmethod
    def _parse_frame_value(frame_value: Union[str, dict, List[int]], trajectory_data: Optional[TrajectoryData] = None, traj_idx: Optional[int] = None) -> List[int]:
        """Parse frame specification from various formats including stride support."""
        # Handle dict with stride
        if isinstance(frame_value, dict):
            return FrameSelectionHelper._parse_dict_frame_value(
                frame_value, trajectory_data, traj_idx
            )
        
        # Handle simple types
        if isinstance(frame_value, list):
            return frame_value
        
        if isinstance(frame_value, int):
            return [frame_value]
        
        if isinstance(frame_value, str):
            return FrameSelectionHelper._parse_string_frame_value(
                frame_value, trajectory_data, traj_idx
            )
        
        raise TypeError(f"Frame value must be dict, list, int or str, got {type(frame_value)}")
    
    @staticmethod
    def _parse_dict_frame_value(frame_dict: dict, trajectory_data: TrajectoryData, traj_idx: int) -> List[int]:
        """Parse dict format with optional stride."""
        if "frames" not in frame_dict:
            raise ValueError("Dict format requires 'frames' key")
        
        frames = frame_dict["frames"]
        stride = frame_dict.get("stride", 1)
        
        # Parse frames based on type
        if isinstance(frames, str):
            parsed_frames = FrameSelectionHelper._parse_frame_string(frames)
        elif isinstance(frames, list):
            parsed_frames = frames
        elif isinstance(frames, int):
            parsed_frames = [frames]
        else:
            raise TypeError(f"frames must be str, list or int, got {type(frames)}")
        
        # Handle "all" keyword
        if parsed_frames == "all":
            parsed_frames = FrameSelectionHelper._get_all_frames(trajectory_data, traj_idx)
        
        # Apply stride
        return FrameSelectionHelper._apply_stride(parsed_frames, stride) if stride > 1 else parsed_frames
    
    @staticmethod
    def _parse_string_frame_value(frame_str: str, trajectory_data: TrajectoryData, traj_idx: int) -> List[int]:
        """Parse string frame value, handling 'all' keyword."""
        parsed = FrameSelectionHelper._parse_frame_string(frame_str)
        if parsed == "all":
            return FrameSelectionHelper._get_all_frames(trajectory_data, traj_idx)
        return parsed
    
    @staticmethod
    def _get_all_frames(trajectory_data: TrajectoryData, traj_idx: int) -> List[int]:
        """Get all frame indices for a trajectory."""
        if trajectory_data is None or traj_idx is None:
            raise ValueError("'all' keyword requires trajectory_data and traj_idx")
        n_frames = trajectory_data.trajectories[traj_idx].n_frames
        return list(range(n_frames))
    
    @staticmethod
    def _parse_frame_string(frame_str: str) -> List[int]:
        """Parse frame string: ranges, comma lists, combined formats, or 'all'."""
        frame_str = frame_str.strip()
        
        # Check for "all" keyword
        if frame_str.lower() == "all":
            return "all"
        
        # Handle combined ranges and comma lists
        if "," in frame_str:
            return FrameSelectionHelper._parse_combined_frames(frame_str)
        
        # Check for single range
        if "-" in frame_str:
            return FrameSelectionHelper._parse_range(frame_str)
        
        # Single number
        return [int(frame_str)]
    
    @staticmethod
    def _parse_combined_frames(frame_str: str) -> List[int]:
        """Parse comma-separated frame specifications."""
        frames = []
        for part in frame_str.split(","):
            part = part.strip()
            if "-" in part:
                frames.extend(FrameSelectionHelper._parse_range(part))
            else:
                frames.append(int(part))
        return sorted(list(set(frames)))  # Remove duplicates and sort
    
    @staticmethod
    def _parse_range(range_str: str) -> List[int]:
        """Parse a single range like '10-20'."""
        parts = range_str.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {range_str}")
        start = int(parts[0].strip())
        end = int(parts[1].strip())
        if start > end:
            raise ValueError(f"Invalid range: start ({start}) > end ({end})")
        return list(range(start, end + 1))
    
    @staticmethod
    def _apply_stride(frames: List[int], stride: int) -> List[int]:
        """
        Apply stride to frame list (minimum distance between consecutive frames).
        
        Parameters:
        -----------
        frames : List[int]
            Sorted list of frame indices
        stride : int
            Minimum distance between consecutive frames
            
        Returns:
        --------
        List[int]
            Filtered frame indices with stride applied
        """
        if stride <= 1 or not frames:
            return frames
        
        sorted_frames = sorted(frames)
        result = [sorted_frames[0]]
        
        for frame in sorted_frames[1:]:
            if frame - result[-1] >= stride:
                result.append(frame)
        
        return result
    
    @staticmethod
    def _validate_frame_indices(frames: List[int], traj_idx: int, trajectory_data: TrajectoryData) -> None:
        """Validate that frame indices are within trajectory bounds."""
        n_frames = len(trajectory_data.trajectories[traj_idx])
        invalid = [f for f in frames if f < 0 or f >= n_frames]
        if invalid:
            raise ValueError(
                f"Frame indices {invalid} out of range for trajectory {traj_idx} (0-{n_frames-1})"
            )
    
    @staticmethod
    def validate_selector_exists(pipeline_data: PipelineData, name: str) -> None:
        """
        Validate that a data selector with given name exists.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing data selectors
        name : str
            Name of the data selector to validate
            
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
    
    @staticmethod
    def validate_trajectories_loaded(pipeline_data: PipelineData) -> None:
        """
        Validate that trajectory data is available for frame selection.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object to check for trajectory data
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if no trajectories loaded
        """
        if not pipeline_data.has_trajectories():
            raise ValueError("No trajectories loaded. Load trajectories first.")
    
    @staticmethod
    def validate_clustering_exists(pipeline_data: PipelineData, clustering_name: str) -> None:
        """
        Validate that a clustering result with given name exists.
        
        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing cluster data
        clustering_name : str
            Name of the clustering result to validate
            
        Returns:
        --------
        None
            Method returns nothing, raises ValueError if clustering not found
        """
        if clustering_name not in pipeline_data.cluster_data:
            available = list(pipeline_data.cluster_data.keys())
            raise ValueError(
                f"Clustering '{clustering_name}' not found. "
                f"Available clusterings: {available}"
            )