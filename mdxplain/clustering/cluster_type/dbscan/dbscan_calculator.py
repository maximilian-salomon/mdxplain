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
DBSCAN calculator implementation.

This module provides the DBSCANCalculator class that performs the actual
DBSCAN clustering computation using scikit-learn.
"""

import time
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from ..interfaces.calculator_base import CalculatorBase


class DBSCANCalculator(CalculatorBase):
    """
    Calculator for DBSCAN clustering.

    This class implements the actual DBSCAN clustering computation using
    scikit-learn's DBSCAN implementation and computes clustering quality metrics.

    Examples:
    ---------
    >>> # Create calculator and compute clustering
    >>> calc = DBSCANCalculator()
    >>> data = np.random.rand(100, 10)
    >>> labels, metadata = calc.compute(data, eps=0.5, min_samples=5)
    >>> print(f"Found {metadata['n_clusters']} clusters")
    """

    def __init__(self, cache_path="./cache"):
        """
        Initialize DBSCAN calculator.

        Parameters:
        -----------
        cache_path : str, optional
            Path for cache files. Default is './cache'.
        """
        super().__init__(cache_path)

    def compute(self, data, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute DBSCAN clustering.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        **kwargs : dict
            DBSCAN parameters including:
            - eps : float, maximum distance between samples
            - min_samples : int, minimum samples in neighborhood

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - cluster_labels: Cluster labels for each sample (-1 for noise)
            - metadata: Dictionary with clustering information

        Raises:
        -------
        ValueError
            If input data is invalid or required parameters are missing
        """
        self._validate_input_data(data)

        # Use caching functionality
        return self._compute_with_cache(
            data, "dbscan", self._compute_without_cache, **kwargs
        )

    def _compute_without_cache(self, data, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Perform DBSCAN clustering without caching.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster
        **kwargs : dict
            DBSCAN parameters

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Cluster labels and metadata
        """
        eps, min_samples = self._extract_parameters(kwargs)

        cluster_labels, dbscan_model, computation_time = self._perform_clustering(
            data, eps, min_samples
        )
        metadata = self._build_metadata(
            data, cluster_labels, dbscan_model, eps, min_samples, computation_time
        )

        return cluster_labels, metadata

    def _extract_parameters(self, kwargs):
        """
        Extract and validate DBSCAN parameters.

        Parameters:
        -----------
        kwargs : dict
            Keyword arguments containing eps and min_samples

        Returns:
        --------
        Tuple[float, int]
            Validated eps and min_samples parameters

        Raises:
        -------
        ValueError
            If required parameters are missing
        """
        eps = kwargs.get("eps")
        min_samples = kwargs.get("min_samples")

        if eps is None or min_samples is None:
            raise ValueError("Both 'eps' and 'min_samples' parameters are required")

        return eps, min_samples

    def _perform_clustering(self, data, eps, min_samples):
        """
        Perform DBSCAN clustering computation.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data to cluster
        eps : float
            Maximum distance between samples
        min_samples : int
            Minimum samples in neighborhood

        Returns:
        --------
        Tuple[numpy.ndarray, SklearnDBSCAN, float]
            Cluster labels, DBSCAN model, and computation time
        """
        start_time = time.time()

        dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data)

        computation_time = time.time() - start_time

        return cluster_labels, dbscan, computation_time

    def _build_metadata(
        self, data, cluster_labels, dbscan_model, eps, min_samples, computation_time
    ):
        """
        Build comprehensive metadata dictionary.

        Parameters:
        -----------
        data : numpy.ndarray
            Original input data
        cluster_labels : numpy.ndarray
            Computed cluster labels
        dbscan_model : SklearnDBSCAN
            Fitted DBSCAN model
        eps : float
            Epsilon parameter used
        min_samples : int
            Min samples parameter used
        computation_time : float
            Time taken for computation

        Returns:
        --------
        dict
            Complete metadata dictionary
        """
        n_clusters = self._count_clusters(cluster_labels)
        n_noise = self._count_noise_points(cluster_labels)
        silhouette = self._compute_silhouette_score(data, cluster_labels)

        hyperparameters = {"eps": eps, "min_samples": min_samples}
        metadata = self._prepare_metadata(
            hyperparameters, data.shape, n_clusters, n_noise
        )

        metadata.update(
            {
                "algorithm": "dbscan",
                "silhouette_score": silhouette,
                "computation_time": computation_time,
                "core_sample_indices": self._get_core_sample_indices(dbscan_model),
            }
        )

        return metadata

    def _get_core_sample_indices(self, dbscan_model):
        """
        Extract core sample indices from DBSCAN model.

        Parameters:
        -----------
        dbscan_model : SklearnDBSCAN
            Fitted DBSCAN model

        Returns:
        --------
        list
            List of core sample indices
        """
        if len(dbscan_model.core_sample_indices_) > 0:
            return dbscan_model.core_sample_indices_.tolist()
        return []
