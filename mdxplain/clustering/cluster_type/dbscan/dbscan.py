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
DBSCAN cluster type implementation.

This module provides the DBSCAN cluster type that implements density-based
clustering for molecular dynamics trajectory analysis.
"""

from typing import Dict, Tuple

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
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN cluster type.
        
        Parameters:
        -----------
        eps : float, optional
            Maximum distance between samples in neighborhood. Default is 0.5.
        min_samples : int, optional
            Minimum samples in neighborhood for core point. Default is 5.
            
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
    
    def init_calculator(self, cache_path="./cache"):
        """
        Initialize the DBSCAN calculator.
        
        Parameters:
        -----------
        cache_path : str, optional
            Directory path for cache files. Default is './cache'.
        """
        self.calculator = DBSCANCalculator(cache_path=cache_path)
    
    def compute(self, data) -> Tuple[np.ndarray, Dict]:
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
            min_samples=self.min_samples
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
