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
DBSCAN cluster type implementation.

This module provides the DBSCAN cluster type that implements density-based
clustering for molecular dynamics trajectory analysis.
"""

from typing import Dict, Tuple, Any
import numpy as np

from ..interfaces.cluster_type_base import ClusterTypeBase
from .dbscan_calculator import DBSCANCalculator


class DBSCAN(ClusterTypeBase):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) cluster type.

    DBSCAN groups together points that are closely packed while marking points
    in low-density regions as outliers. It's particularly useful for identifying
    conformational states in molecular dynamics trajectories.

    Parameters:
    -----------
    eps : float, optional
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. Default is 0.5.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point. Default is 5.

    Examples:
    ---------
    >>> # Create DBSCAN with default parameters
    >>> dbscan = DBSCAN()

    >>> # Create DBSCAN with custom parameters
    >>> dbscan = DBSCAN(eps=0.3, min_samples=10)

    >>> # Initialize and compute clustering
    >>> dbscan.init_calculator()
    >>> labels, metadata = dbscan.compute(data)
    """

    def __init__(
        self, 
        eps: float = 0.5, 
        min_samples: int = 5,
        method: str = "standard",
        sample_fraction: float = 0.1,
        force: bool = False,
    ) -> None:
        """
        Initialize DBSCAN cluster type.

        Parameters:
        -----------
        eps : float, optional
            Maximum distance between samples in neighborhood. Default is 0.5.
        min_samples : int, optional
            Minimum samples in neighborhood for core point. Default is 5.
        method : str, default="standard"
            Clustering method:
            - "standard": Load all data into memory (default)
            - "sampling_approximate": Sample data + approximate_predict for large datasets
            - "sampling_knn": Sample data + k-NN classifier fallback
        sample_fraction : float, default=0.1
            Fraction of data to sample for sampling-based methods (10%)
            Final sample size: max(50000, min(100000, sample_fraction * n_samples))
        force : bool, default=False
            Override memory and dimensionality checks (converts errors to warnings)

        Returned Metadata:
        ------------------
        algorithm : str
            Always "dbscan"
        hyperparameters : dict
            Dictionary containing all DBSCAN parameters used
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
        core_sample_indices : list
            List of core sample indices
        cache_path : str
            Path used for caching results
        """
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.method = method
        self.sample_fraction = sample_fraction
        self.force = force
        self._validate_parameters()

    @classmethod
    def get_type_name(cls) -> str:
        """
        Return unique string identifier for DBSCAN cluster type.

        Returns:
        --------
        str
            The string 'dbscan'
        """
        return "dbscan"

    def init_calculator(
        self, 
        cache_path: str = "./cache", 
        max_memory_gb: float = 2.0,
        chunk_size: int = 1000,
        use_memmap: bool = False
    ) -> None:
        """
        Initialize the DBSCAN calculator.

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
        self.calculator = DBSCANCalculator(
            cache_path=cache_path, 
            max_memory_gb=max_memory_gb,
            chunk_size=chunk_size,
            use_memmap=use_memmap
        )

    def compute(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute DBSCAN clustering.

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
            raise ValueError("Calculator not initialized. Call init_calculator() first.")
        
        return self.calculator.compute(
            data, 
            eps=self.eps, 
            min_samples=self.min_samples,
            method=self.method,
            sample_fraction=self.sample_fraction,
            force=self.force,
        )
    
    def _validate_parameters(self):
        """
        Validate DBSCAN parameters.

        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        if not isinstance(self.eps, (int, float)) or self.eps <= 0:
            raise ValueError("eps must be a positive number")

        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")

        if self.method not in ["standard", "precomputed", "knn_sampling"]:
            raise ValueError("method must be 'standard', 'precomputed', or 'knn_sampling'")

        if not isinstance(self.sample_fraction, (int, float)) or not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be a number between 0 and 1")

        if not isinstance(self.force, bool):
            raise ValueError("force must be a boolean")
