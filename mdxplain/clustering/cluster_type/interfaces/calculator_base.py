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
Abstract base class for clustering calculators.

Defines the interface that all clustering calculators must implement
for consistency across different clustering methods.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import silhouette_score

from ....utils.data_utils import DataUtils


class CalculatorBase(ABC):
    """
    Abstract base class for clustering calculators.

    Defines the interface that all clustering calculators (DBSCAN, HDBSCAN,
    DPA) must implement for consistency across different clustering methods.

    Examples:
    ---------
    >>> class MyCalculator(CalculatorBase):
    ...     def __init__(self, cache_path="./cache"):
    ...         super().__init__(cache_path)
    ...     def compute(self, data, **kwargs):
    ...         # Implement computation logic
    ...         return cluster_labels, metadata
    """

    def __init__(self, cache_path="./cache"):
        """
        Initialize the clustering calculator.

        Parameters:
        -----------
        cache_path : str, optional
            Path for cache files

        Returns:
        --------
        None
            Initializes calculator with specified configuration

        Examples:
        ---------
        >>> # Basic initialization
        >>> calc = MyCalculator()

        >>> # With custom cache path
        >>> calc = MyCalculator(cache_path="./my_cache/")
        """
        self.cache_path = cache_path

    @abstractmethod
    def compute(self, data, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute clustering of input data.

        This method performs the actual clustering computation
        and returns the cluster labels along with metadata about the
        clustering process.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        **kwargs : dict
            Additional parameters specific to the clustering method

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - cluster_labels: Cluster labels for each sample (n_samples,)
            - metadata: Dictionary with clustering information including
              hyperparameters, number of clusters, silhouette score, etc.

        Examples:
        ---------
        >>> # Compute clustering
        >>> calc = MyCalculator()
        >>> data = np.random.rand(100, 50)
        >>> labels, metadata = calc.compute(data, eps=0.5, min_samples=5)
        >>> print(f"Number of clusters: {metadata['n_clusters']}")
        >>> print(f"Silhouette score: {metadata['silhouette_score']}")
        """
        pass

    def _validate_input_data(self, data):
        """
        Validate input data for clustering.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data to validate

        Returns:
        --------
        None
            Validates input data format and shape

        Raises:
        -------
        ValueError
            If data format is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array")
        if data.shape[0] < 2:
            raise ValueError("Input data must have at least 2 samples")
        if data.shape[1] < 1:
            raise ValueError("Input data must have at least 1 feature")

    def _prepare_metadata(self, hyperparameters, original_shape, n_clusters, n_noise=0):
        """
        Prepare base metadata dictionary.

        Parameters:
        -----------
        hyperparameters : dict
            Hyperparameters used for clustering
        original_shape : tuple
            Shape of original input data
        n_clusters : int
            Number of clusters found
        n_noise : int, optional
            Number of noise points (for algorithms that identify noise)

        Returns:
        --------
        dict
            Base metadata dictionary with common information
        """
        return {
            "hyperparameters": hyperparameters,
            "original_shape": original_shape,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cache_path": self.cache_path,
        }

    def _count_clusters(self, labels):
        """
        Count number of clusters (excluding noise).

        Parameters:
        -----------
        labels : numpy.ndarray
            Cluster labels (-1 indicates noise)

        Returns:
        --------
        int
            Number of clusters found
        """
        unique_labels = np.unique(labels)
        # Exclude noise label (-1) from count
        return len(unique_labels[unique_labels != -1])

    def _count_noise_points(self, labels, noise_cluster=-1):
        """
        Count number of noise points.

        Parameters:
        -----------
        labels : numpy.ndarray
            Cluster labels (-1 indicates noise)
        noise_cluster : int
            The value or label which is assigned to noise data points
        Returns:
        --------
        int
            Number of noise points
        """
        return np.sum(labels == noise_cluster)

    def _compute_silhouette_score(self, data, labels):
        """
        Compute silhouette score for clustering quality assessment.

        Parameters:
        -----------
        data : numpy.ndarray
            Original data used for clustering
        labels : numpy.ndarray
            Cluster labels

        Returns:
        --------
        float or None
            Silhouette score, or None if cannot be computed
        """
        # Remove noise points for silhouette calculation
        non_noise_mask = labels != -1

        if np.sum(non_noise_mask) < 2:
            return None

        # Check if we have at least 2 different clusters
        unique_labels = np.unique(labels[non_noise_mask])
        if len(unique_labels) < 2:
            return None

        # Compute silhouette score on non-noise points
        return silhouette_score(data[non_noise_mask], labels[non_noise_mask])

    def _load_cache_files(self, algorithm_name):
        """
        Load cached clustering results if both label and metadata files exist.

        Parameters:
        -----------
        algorithm_name : str
            Name of the clustering algorithm (e.g., 'dbscan', 'hdbscan', 'dpa')

        Returns:
        --------
        Tuple[numpy.ndarray, Dict] or None
            Cached results if valid, None otherwise
        """
        labels_path = DataUtils.get_cache_file_path(
            f"{algorithm_name}_labels.npy", self.cache_path
        )
        metadata_path = DataUtils.get_cache_file_path(
            f"{algorithm_name}_metadata.npy", self.cache_path
        )

        if not (os.path.exists(labels_path) and os.path.exists(metadata_path)):
            return None

        labels = np.load(labels_path, allow_pickle=True)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        return labels, metadata

    def _save_cache_files(self, algorithm_name, labels, metadata):
        """
        Save clustering results to separate label and metadata cache files.

        Parameters:
        -----------
        algorithm_name : str
            Name of the clustering algorithm
        labels : numpy.ndarray
            Cluster labels to save
        metadata : dict
            Clustering metadata to save
        """
        labels_path = DataUtils.get_cache_file_path(
            f"{algorithm_name}_labels.npy", self.cache_path
        )
        metadata_path = DataUtils.get_cache_file_path(
            f"{algorithm_name}_metadata.npy", self.cache_path
        )

        np.save(labels_path, labels, allow_pickle=True)
        np.save(metadata_path, metadata, allow_pickle=True)

    def _compute_with_cache(self, data, algorithm_name, compute_func, **kwargs):
        """
        Compute clustering with caching support.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data for clustering
        algorithm_name : str
            Name of the clustering algorithm
        compute_func : callable
            Function to perform actual computation
        **kwargs : dict
            Algorithm parameters

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Cluster labels and metadata
        """
        # Try to load from cache
        cached_result = self._load_cache_files(algorithm_name)
        if cached_result is not None:
            return cached_result

        # Compute if not cached
        labels, metadata = compute_func(data, **kwargs)

        # Save to cache
        self._save_cache_files(algorithm_name, labels, metadata)

        return labels, metadata
