# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
Cluster data container for computed clustering results.

Container for clustering results (DBSCAN, HDBSCAN, DPA) with associated metadata
and hyperparameters. Stores cluster labels with clustering information.
"""

import numpy as np
from typing import Dict, Optional, Any


class ClusterData:
    """
    Container for clustering results with metadata and hyperparameters.

    Stores results from clustering methods (DBSCAN, HDBSCAN, DPA) along
    with clustering metadata and hyperparameters used for computation.
    """

    def __init__(self, cluster_type: str, cache_path: str = "./cache"):
        """
        Initialize cluster data container.

        Parameters:
        -----------
        cluster_type : str
            Type of clustering algorithm used (e.g., "dbscan", "hdbscan", "dpa")
        cache_path : str, optional
            Path for cached results, default="./cache"

        Returns:
        --------
        None
            Initializes cluster data container

        Examples:
        ---------
        >>> # Basic initialization
        >>> cluster_data = ClusterData("dbscan")

        >>> # With custom cache path
        >>> cluster_data = ClusterData(
        ...     "hdbscan",
        ...     cache_path="./cache/clustering"
        ... )
        """
        self.cluster_type = cluster_type
        self.cache_path = cache_path
        
        # Initialize data attributes
        self.labels: Optional[np.ndarray] = None
        self.metadata: Optional[Dict[str, Any]] = None

    def get_labels(self) -> Optional[np.ndarray]:
        """
        Get cluster labels for each trajectory frame.

        Returns:
        --------
        numpy.ndarray or None
            Array of cluster labels corresponding to trajectory frame indices,
            or None if clustering has not been computed yet

        Examples:
        ---------
        >>> cluster_data = ClusterData("dbscan")
        >>> labels = cluster_data.get_labels()
        >>> if labels is not None:
        ...     print(f"Number of frames: {len(labels)}")
        """
        return self.labels

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get clustering metadata including parameters and metrics.

        Returns:
        --------
        dict or None
            Dictionary containing clustering parameters, metrics, and other
            metadata, or None if clustering has not been computed yet

        Examples:
        ---------
        >>> cluster_data = ClusterData("dbscan")
        >>> metadata = cluster_data.get_metadata()
        >>> if metadata is not None:
        ...     print(f"Number of clusters: {metadata.get('n_clusters', 'Unknown')}")
        ...     print(f"Algorithm: {metadata.get('algorithm', 'Unknown')}")
        """
        return self.metadata

    def get_cluster_type(self) -> str:
        """
        Get the clustering algorithm type.

        Returns:
        --------
        str
            The type of clustering algorithm used

        Examples:
        ---------
        >>> cluster_data = ClusterData("dbscan")
        >>> print(cluster_data.get_cluster_type())
        'dbscan'
        """
        return self.cluster_type

    def get_cache_path(self) -> str:
        """
        Get the cache path for clustering results.

        Returns:
        --------
        str
            Path where clustering results are cached

        Examples:
        ---------
        >>> cluster_data = ClusterData("dbscan", cache_path="./my_cache")
        >>> print(cluster_data.get_cache_path())
        './my_cache'
        """
        return self.cache_path

    def has_data(self) -> bool:
        """
        Check if clustering data has been computed and stored.

        Returns:
        --------
        bool
            True if both labels and metadata are available, False otherwise

        Examples:
        ---------
        >>> cluster_data = ClusterData("dbscan")
        >>> print(cluster_data.has_data())
        False
        >>> # After clustering computation...
        >>> # cluster_data.labels = computed_labels
        >>> # cluster_data.metadata = computed_metadata
        >>> # print(cluster_data.has_data())
        >>> # True
        """
        return self.labels is not None and self.metadata is not None

    def get_n_clusters(self) -> Optional[int]:
        """
        Get the number of clusters found.

        Returns:
        --------
        int or None
            Number of clusters found, or None if clustering has not been computed
            or if the information is not available in metadata

        Examples:
        ---------
        >>> cluster_data = ClusterData("dbscan")
        >>> n_clusters = cluster_data.get_n_clusters()
        >>> if n_clusters is not None:
        ...     print(f"Found {n_clusters} clusters")
        """
        if self.metadata is not None:
            return self.metadata.get('n_clusters')
        return None

    def get_n_frames(self) -> Optional[int]:
        """
        Get the number of trajectory frames that were clustered.

        Returns:
        --------
        int or None
            Number of trajectory frames, or None if clustering has not been computed

        Examples:
        ---------
        >>> cluster_data = ClusterData("dbscan")
        >>> n_frames = cluster_data.get_n_frames()
        >>> if n_frames is not None:
        ...     print(f"Clustered {n_frames} frames")
        """
        if self.labels is not None:
            return len(self.labels)
        return None