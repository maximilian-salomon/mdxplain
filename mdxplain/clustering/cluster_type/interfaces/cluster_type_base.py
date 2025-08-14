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
Abstract base class defining the interface for all cluster types.

Defines the interface that all cluster types (DBSCAN, HDBSCAN, DPA, etc.)
must implement for consistency across different clustering methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from .cluster_type_meta import ClusterTypeMeta


class ClusterTypeBase(ABC, metaclass=ClusterTypeMeta):
    """
    Abstract base class for all cluster types.

    Defines the interface that all cluster types (DBSCAN, HDBSCAN, DPA, etc.)
    must implement. Each cluster type encapsulates computation logic
    for a specific type of clustering analysis.

    Examples:
    ---------
    >>> class MyCluster(ClusterTypeBase):
    ...     @classmethod
    ...     def get_type_name(cls):
    ...         return 'my_cluster'
    ...     def init_calculator(self, **kwargs):
    ...         self.calculator = MyCalculator(**kwargs)
    ...     def compute(self, data, **kwargs):
    ...         return self.calculator.compute(data, **kwargs)
    """

    def __init__(self):
        """
        Initialize the cluster type.

        Sets up the cluster type instance with an empty calculator that
        will be initialized later through init_calculator().

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Examples:
        ---------
        >>> # Create cluster type instance
        >>> cluster = MyCluster()
        >>> print(f"Type: {cluster.get_type_name()}")
        """
        self.calculator = None

    @classmethod
    @abstractmethod
    def get_type_name(cls) -> str:
        """
        Return unique string identifier for this cluster type.

        Used as the key for storing clustering results in TrajectoryData
        dictionaries and for type identification.

        Parameters:
        -----------
        cls : type
            The cluster type class

        Returns:
        --------
        str
            Unique string identifier (e.g., 'dbscan', 'hdbscan', 'dpa')

        Examples:
        ---------
        >>> print(DBSCAN.get_type_name())
        'dbscan'
        >>> print(HDBSCAN.get_type_name())
        'hdbscan'
        """
        pass

    @abstractmethod
    def init_calculator(self, cache_path="./cache"):
        """
        Initialize the calculator instance for this cluster type.

        Parameters:
        -----------
        cache_path : str, optional
            Directory path for cache files

        Returns:
        --------
        None
            Sets self.calculator to initialized calculator instance

        Examples:
        ---------
        >>> # Basic initialization
        >>> dbscan = DBSCAN()
        >>> dbscan.init_calculator()

        >>> # With custom cache path
        >>> dbscan.init_calculator(cache_path='./my_cache/')
        """
        pass

    @abstractmethod
    def compute(self, data) -> Tuple[np.ndarray, Dict]:
        """
        Compute clustering using the initialized calculator.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - cluster_labels: Cluster labels for each sample (n_samples,)
            - metadata: Dictionary with clustering information including
              hyperparameters, number of clusters, silhouette score, etc.

        Examples:
        ---------
        >>> # Compute DBSCAN clustering
        >>> dbscan = DBSCAN(eps=0.5, min_samples=5)
        >>> dbscan.init_calculator()
        >>> data = np.random.rand(100, 50)
        >>> labels, metadata = dbscan.compute(data)
        >>> print(f"Number of clusters: {metadata['n_clusters']}")

        Raises:
        -------
        ValueError
            If calculator is not initialized or input data is invalid
        """
        pass
