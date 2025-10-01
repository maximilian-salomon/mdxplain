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
Cluster data container for computed clustering results.

Container for clustering results (DBSCAN, HDBSCAN, DPA) with associated metadata
and hyperparameters. Stores cluster labels with clustering information.
"""

import numpy as np
from typing import Dict, Optional, Any

from ...utils.data_utils import DataUtils

class ClusterData:
    """
    Container for clustering results with metadata and hyperparameters.

    Stores results from clustering methods (DBSCAN, HDBSCAN, DPA) along
    with clustering metadata and hyperparameters used for computation.

    Attributes
    ----------
    cluster_type : str
        Type of clustering algorithm used (e.g., "dbscan", "hdbscan", "dpa")
    cache_path : str
        Path for cached results
    labels : numpy.ndarray or None
        Array of cluster labels for each trajectory frame, or None if not computed
    metadata : dict or None
        Dictionary containing clustering parameters, metrics, and other metadata,
        or None if not computed
    frame_mapping : dict or None
        Mapping from global_frame_index to (trajectory_index, local_frame_index),
        or None if not computed
    """

    def __init__(self, cluster_type: str, cache_path: str = "./cache"):
        """
        Initialize cluster data container.

        Parameters
        ----------
        cluster_type : str
            Type of clustering algorithm used (e.g., "dbscan", "hdbscan", "dpa")
        cache_path : str, optional
            Path for cached results, default="./cache"

        Returns
        -------
        None
            Initializes cluster data container

        Examples
        --------
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
        self.frame_mapping: Optional[Dict[int, tuple]] = None

    def get_labels(self) -> Optional[np.ndarray]:
        """
        Get cluster labels for each trajectory frame.

        Returns
        -------
        numpy.ndarray or None
            Array of cluster labels corresponding to trajectory frame indices,
            or None if clustering has not been computed yet

        Examples
        --------
        >>> cluster_data = ClusterData("dbscan")
        >>> labels = cluster_data.get_labels()
        >>> if labels is not None:
        ...     print(f"Number of frames: {len(labels)}")
        """
        return self.labels

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get clustering metadata including parameters and metrics.

        Returns
        -------
        dict or None
            Dictionary containing clustering parameters, metrics, and other
            metadata, or None if clustering has not been computed yet

        Examples
        --------
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

        Returns
        -------
        str
            The type of clustering algorithm used

        Examples
        --------
        >>> cluster_data = ClusterData("dbscan")
        >>> print(cluster_data.get_cluster_type())
        'dbscan'
        """
        return self.cluster_type

    def get_cache_path(self) -> str:
        """
        Get the cache path for clustering results.

        Returns
        -------
        str
            Path where clustering results are cached

        Examples
        --------
        >>> cluster_data = ClusterData("dbscan", cache_path="./my_cache")
        >>> print(cluster_data.get_cache_path())
        './my_cache'
        """
        return self.cache_path

    def has_data(self) -> bool:
        """
        Check if clustering data has been computed and stored.

        Returns
        -------
        bool
            True if both labels and metadata are available, False otherwise

        Examples
        --------
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

        Returns
        -------
        int or None
            Number of clusters found, or None if clustering has not been computed
            or if the information is not available in metadata

        Examples
        --------
        >>> cluster_data = ClusterData("dbscan")
        >>> n_clusters = cluster_data.get_n_clusters()
        >>> if n_clusters is not None:
        ...     print(f"Found {n_clusters} clusters")
        """
        if self.metadata is not None:
            return self.metadata.get("n_clusters")
        return None

    def get_n_frames(self) -> Optional[int]:
        """
        Get the number of trajectory frames that were clustered.

        Returns
        -------
        int or None
            Number of trajectory frames, or None if clustering has not been computed

        Examples
        --------
        >>> cluster_data = ClusterData("dbscan")
        >>> n_frames = cluster_data.get_n_frames()
        >>> if n_frames is not None:
        ...     print(f"Clustered {n_frames} frames")
        """
        if self.labels is not None:
            return len(self.labels)
        return None

    def get_frame_mapping(self) -> Optional[Dict[int, tuple]]:
        """
        Get frame mapping from global frame indices to trajectory origins.

        Returns
        -------
        dict or None
            Mapping from global_frame_index to (trajectory_index, local_frame_index),
            or None if clustering has not been computed or mapping is not available

        Examples
        --------
        >>> cluster_data = ClusterData("dbscan")
        >>> frame_mapping = cluster_data.get_frame_mapping()
        >>> if frame_mapping is not None:
        ...     print(f"Frame 0 comes from: {frame_mapping[0]}")  # (traj_idx, local_frame_idx)
        """
        return self.frame_mapping
        
    def set_frame_mapping(self, frame_mapping: Dict[int, tuple]) -> None:
        """
        Set frame mapping from global frame indices to trajectory origins.

        Parameters
        ----------
        frame_mapping : dict
            Mapping from global_frame_index to (trajectory_index, local_frame_index)

        Returns
        -------
        None
            Sets the frame mapping for trajectory tracking

        Examples
        --------
        >>> cluster_data = ClusterData("dbscan")
        >>> mapping = {0: (0, 10), 1: (0, 11), 2: (1, 5)}  # global -> (traj, local)
        >>> cluster_data.set_frame_mapping(mapping)
        """
        self.frame_mapping = frame_mapping

    def save(self, save_path: str) -> None:
        """
        Save ClusterData object to disk.

        Parameters
        ----------
        save_path : str
            Path where to save the ClusterData object

        Returns
        -------
        None
            Saves the ClusterData object to the specified path

        Examples
        --------
        >>> cluster_data.save('analysis_results/dbscan_clustering.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load ClusterData object from disk.

        Parameters
        ----------
        load_path : str
            Path to the saved ClusterData file

        Returns
        -------
        None
            Loads the ClusterData object from the specified path

        Examples
        --------
        >>> cluster_data.load('analysis_results/dbscan_clustering.pkl')
        """
        DataUtils.load_object(self, load_path)

    def print_info(self) -> None:
        """
        Print comprehensive cluster information.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints cluster information to console

        Examples
        --------
        >>> cluster_data.print_info()
        === ClusterData ===
        Cluster Type: DBSCAN
        Number of Clusters: 5
        Number of Frames: 1000
        Noise Points: 127 (12.7%)
        """
        if not self.has_data():
            print("No cluster data available.")
            return

        self._print_cluster_header()
        self._print_cluster_details()
        if self.frame_mapping is not None:
            self._print_frame_mapping_info()

    def _print_cluster_header(self) -> None:
        """
        Print header with cluster type.

        Returns
        -------
        None
        """
        print("=== ClusterData ===")
        print(f"Cluster Type: {self.cluster_type.upper()}")

    def _print_cluster_details(self) -> None:
        """
        Print detailed cluster information.

        Returns
        -------
        None
        """
        n_clusters = self.get_n_clusters()
        n_frames = self.get_n_frames()
        
        if n_clusters is not None:
            print(f"Number of Clusters: {n_clusters}")
        
        if n_frames is not None:
            print(f"Number of Frames: {n_frames}")
            
            # Calculate noise points if labels contain -1
            if self.labels is not None:
                unique_labels = set(self.labels)
                if -1 in unique_labels:
                    noise_count = np.sum(self.labels == -1)
                    noise_percent = (noise_count / len(self.labels)) * 100
                    print(f"Noise Points: {noise_count} ({noise_percent:.1f}%)")

        # Show hyperparameters if available
        if self.metadata is not None:
            hyperparams = self.metadata.get("hyperparameters", {})
            if hyperparams:
                param_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
                print(f"Hyperparameters: {param_str}")

    def _print_frame_mapping_info(self) -> None:
        """
        Print information about frame mapping.

        Returns
        -------
        None
        """
        if self.frame_mapping:
            n_mapped_frames = len(self.frame_mapping)
            # Count unique trajectories
            trajectory_indices = set()
            for traj_idx, _ in self.frame_mapping.values():
                trajectory_indices.add(traj_idx)
            n_trajectories = len(trajectory_indices)
            print(f"Frame Mapping: {n_mapped_frames} frames from {n_trajectories} trajectories")
