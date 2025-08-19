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

from typing import List, Union


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
        trajectory_data, tags: List[str], match_all: bool
    ) -> List[int]:
        """
        Select frame indices based on tag criteria.
        
        Searches through trajectory frame tag mappings to find frames
        that contain the specified tags according to the matching mode.
        
        Parameters:
        -----------
        trajectory_data : TrajectoryData
            Trajectory data object containing frame tag mappings
        tags : List[str]
            List of tags to search for in frame mappings
        match_all : bool
            If True, frame must contain ALL tags. If False, ANY tag matches.
            
        Returns:
        --------
        List[int]
            List of global frame indices that match the tag criteria
            
        Examples:
        ---------
        >>> # Select frames with all tags
        >>> indices = FrameSelectionHelper.select_frames_by_tags(
        ...     trajectory_data, ["system_A", "biased"], match_all=True
        ... )
        
        >>> # Select frames with any tag
        >>> indices = FrameSelectionHelper.select_frames_by_tags(
        ...     trajectory_data, ["system_A", "system_B"], match_all=False
        ... )
        """
        frame_indices = []

        for frame_idx, frame_tags in trajectory_data.frame_tag_mapping.items():
            if frame_tags is None:
                continue

            frame_tag_set = set(frame_tags)

            if match_all:
                # Frame must have ALL specified tags
                if all(tag in frame_tag_set for tag in tags):
                    frame_indices.append(frame_idx)
            else:
                # Frame must have ANY of the specified tags
                if any(tag in frame_tag_set for tag in tags):
                    frame_indices.append(frame_idx)

        return frame_indices
    
    @staticmethod
    def select_frames_by_cluster(
        labels: List[int], cluster_ids: List[int]
    ) -> List[int]:
        """
        Select frame indices based on cluster assignments.
        
        Finds frames that belong to the specified cluster IDs by examining
        the cluster label array.
        
        Parameters:
        -----------
        labels : List[int]
            List of cluster labels for each frame
        cluster_ids : List[int]
            List of cluster IDs to select frames from
            
        Returns:
        --------
        List[int]
            List of frame indices belonging to the specified clusters
            
        Examples:
        ---------
        >>> # Select frames from clusters 0 and 1
        >>> indices = FrameSelectionHelper.select_frames_by_cluster(
        ...     labels, [0, 1]
        ... )
        """
        return [i for i, label in enumerate(labels) if label in cluster_ids]
    
    @staticmethod
    def resolve_cluster_ids(
        cluster_data, cluster_ids: List[Union[int, str]], clustering_name: str
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
    def validate_selector_exists(pipeline_data, name: str) -> None:
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
    def validate_trajectories_loaded(pipeline_data) -> None:
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
    def validate_clustering_exists(pipeline_data, clustering_name: str) -> None:
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