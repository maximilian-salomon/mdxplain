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
HDBSCAN cluster type implementation.

This module provides the HDBSCAN cluster type that implements hierarchical
density-based clustering for molecular dynamics trajectory analysis.
"""

from typing import Dict, Tuple

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
    
    def __init__(self, min_cluster_size=5, min_samples=None, 
                 cluster_selection_epsilon=0.0, cluster_selection_method="eom"):
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
    
    def init_calculator(self, cache_path="./cache"):
        """
        Initialize the HDBSCAN calculator.
        
        Parameters:
        -----------
        cache_path : str, optional
            Directory path for cache files. Default is './cache'.
        """
        self.calculator = HDBSCANCalculator(cache_path=cache_path)
    
    def compute(self, data) -> Tuple[np.ndarray, Dict]:
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
            raise ValueError("Calculator not initialized. Call init_calculator() first.")
        
        return self.calculator.compute(
            data, 
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method
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
        
        if not isinstance(self.cluster_selection_epsilon, (int, float)) or self.cluster_selection_epsilon < 0:
            raise ValueError("cluster_selection_epsilon must be a non-negative number")
        
        if self.cluster_selection_method not in ["eom", "leaf"]:
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")