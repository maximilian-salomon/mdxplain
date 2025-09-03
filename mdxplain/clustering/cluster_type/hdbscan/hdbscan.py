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
HDBSCAN cluster type implementation.

This module provides the HDBSCAN cluster type that implements hierarchical
density-based clustering for molecular dynamics trajectory analysis.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np

from ..interfaces.cluster_type_base import ClusterTypeBase
from .hdbscan_calculator import HDBSCANCalculator


class HDBSCAN(ClusterTypeBase):
    """
    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) cluster type.

    HDBSCAN extends DBSCAN by converting it into a hierarchical clustering algorithm,
    and then using a technique to extract a flat clustering based on the stability
    of clusters. It's particularly useful for identifying conformational states
    in molecular dynamics trajectories with varying densities.

    Parameters:
    -----------
    min_cluster_size : int, optional
        The minimum size of clusters. Default is 5.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point. If not specified, defaults to min_cluster_size.
    cluster_selection_epsilon : float, optional
        A distance threshold for cluster selection. Default is 0.0.
    cluster_selection_method : str, optional
        The method used to select clusters from the condensed tree.
        Options are 'eom' (Excess of Mass) or 'leaf'. Default is 'eom'.

    Examples:
    ---------
    >>> # Create HDBSCAN with default parameters
    >>> hdbscan = HDBSCAN()

    >>> # Create HDBSCAN with custom parameters
    >>> hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5)

    >>> # Initialize and compute clustering
    >>> hdbscan.init_calculator()
    >>> labels, metadata = hdbscan.compute(data)
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        method: str = "standard",
        sample_fraction: float = 0.1,
        knn_neighbors: int = 5,
        force: bool = False,
    ) -> None:
        """
        Initialize HDBSCAN cluster type.

        Parameters:
        -----------
        min_cluster_size : int, optional
            Minimum size of clusters. Default is 5.
        min_samples : int, optional
            Minimum samples in neighborhood for core point.
            If None, defaults to min_cluster_size.
        cluster_selection_epsilon : float, optional
            Distance threshold for cluster selection. Default is 0.0.
        cluster_selection_method : str, optional
            Method for cluster selection ('eom' or 'leaf'). Default is 'eom'.
        method : str, default="standard"
            Clustering method:
            - "standard": Load all data into memory (default)
            - "sampling_approximate": Sample data + approximate_predict for large datasets
            - "sampling_knn": Sample data + k-NN classifier fallback
        sample_fraction : float, default=0.1
            Fraction of data to sample for sampling-based methods (10%)
            Final sample size: max(50000, min(100000, sample_fraction * n_samples))
        knn_neighbors : int, default=5
            Number of neighbors for k-NN classifier in knn_sampling method
        force : bool, default=False
            Override memory and dimensionality checks (converts errors to warnings)

        Returned Metadata:
        ------------------
        algorithm : str
            Always "hdbscan"
        hyperparameters : dict
            Dictionary containing all HDBSCAN parameters used
        original_shape : tuple
            Shape of the input data (n_samples, n_features)
        n_clusters : int
            Number of clusters found (excluding noise points)
        n_noise : int
            Number of noise points identified (label -1)
        silhouette_score : float or None
            Silhouette score for clustering quality assessment
        computation_time : float
            Time taken for clustering computation in seconds
        cluster_probabilities : list or None
            Cluster membership probabilities for each point
        outlier_scores : list or None
            Outlier scores for each point
        cache_path : str
            Path used for caching results
        """
        super().__init__()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.method = method
        self.sample_fraction = sample_fraction
        self.knn_neighbors = knn_neighbors
        self.force = force
        self._validate_parameters()

    @classmethod
    def get_type_name(cls) -> str:
        """
        Return unique string identifier for HDBSCAN cluster type.

        Returns:
        --------
        str
            The string 'hdbscan'
        """
        return "hdbscan"

    def init_calculator(
        self, 
        cache_path: str = "./cache", 
        max_memory_gb: float = 2.0,
        chunk_size: int = 1000,
        use_memmap: bool = False
    ) -> None:
        """
        Initialize the HDBSCAN calculator.

        Parameters:
        -----------
        cache_path : str, optional
            Directory path for cache files. Default is './cache'.
        max_memory_gb : float, optional
            Maximum memory threshold in GB. Default is 2.0.
        chunk_size : int, optional
            Chunk size for processing large datasets. Default is 1000.
        use_memmap : bool, optional
            Whether to use memory mapping for large datasets. Default is False.
        """
        self.calculator = HDBSCANCalculator(
            cache_path=cache_path, 
            max_memory_gb=max_memory_gb,
            chunk_size=chunk_size,
            use_memmap=use_memmap
        )

    def compute(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute HDBSCAN clustering.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - cluster_labels: Cluster labels for each sample (-1 for noise)
            - metadata: Dictionary with clustering information

        Raises:
        -------
        ValueError
            If calculator is not initialized
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        return self.calculator.compute(
            data,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            method=self.method,
            sample_fraction=self.sample_fraction,
            knn_neighbors=self.knn_neighbors,
            force=self.force,
        )

    def _validate_parameters(self):
        """
        Validate HDBSCAN parameters.

        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        if not isinstance(self.min_cluster_size, int) or self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be an integer >= 2")

        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")

        if (
            not isinstance(self.cluster_selection_epsilon, (int, float))
            or self.cluster_selection_epsilon < 0
        ):
            raise ValueError("cluster_selection_epsilon must be a non-negative number")

        if self.cluster_selection_method not in ["eom", "leaf"]:
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")

        if self.method not in ["standard", "approximate_predict", "knn_sampling"]:
            raise ValueError("method must be 'standard', 'approximate_predict', or 'knn_sampling'")

        if not isinstance(self.sample_fraction, (int, float)) or not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be a number between 0 and 1")

        if not isinstance(self.knn_neighbors, int) or self.knn_neighbors < 1:
            raise ValueError("knn_neighbors must be a positive integer")

        if not isinstance(self.force, bool):
            raise ValueError("force must be a boolean")
