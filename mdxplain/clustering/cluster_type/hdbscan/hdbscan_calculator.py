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
HDBSCAN calculator implementation.

This module provides the HDBSCANCalculator class that performs the actual
HDBSCAN clustering computation using scikit-learn.
"""

import time
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

from ..interfaces.calculator_base import CalculatorBase


class HDBSCANCalculator(CalculatorBase):
    """
    Calculator for HDBSCAN clustering.

    This class implements the actual HDBSCAN clustering computation using
    scikit-learn's HDBSCAN implementation and computes clustering quality metrics.

    Examples:
    ---------
    >>> # Create calculator and compute clustering
    >>> calc = HDBSCANCalculator()
    >>> data = np.random.rand(100, 10)
    >>> labels, metadata = calc.compute(data, min_cluster_size=5, min_samples=5)
    >>> print(f"Found {metadata['n_clusters']} clusters")
    """

    def __init__(self, cache_path: str = "./cache") -> None:
        """
        Initialize HDBSCAN calculator.

        Parameters:
        -----------
        cache_path : str, optional
            Path for cache files. Default is './cache'.
        """
        super().__init__(cache_path)

    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute HDBSCAN clustering.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        **kwargs : dict
            HDBSCAN parameters including:
            - min_cluster_size : int, minimum size of clusters
            - min_samples : int, minimum samples in neighborhood
            - cluster_selection_epsilon : float, distance threshold
            - cluster_selection_method : str, cluster selection method

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

        # Extract parameters and perform clustering
        parameters = self._extract_parameters(kwargs)

        cluster_labels, hdbscan_model, computation_time = self._perform_clustering(
            data, parameters
        )
        metadata = self._build_metadata(
            data, cluster_labels, hdbscan_model, parameters, computation_time
        )

        return cluster_labels, metadata


    def _extract_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate HDBSCAN parameters.

        Parameters:
        -----------
        kwargs : dict
            Keyword arguments containing HDBSCAN parameters

        Returns:
        --------
        dict
            Validated HDBSCAN parameters
        """
        return {
            "min_cluster_size": kwargs.get("min_cluster_size", 5),
            "min_samples": kwargs.get("min_samples", None),
            "cluster_selection_epsilon": kwargs.get("cluster_selection_epsilon", 0.0),
            "cluster_selection_method": kwargs.get("cluster_selection_method", "eom"),
        }

    def _perform_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Any, float]:
        """
        Perform HDBSCAN clustering computation.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data to cluster
        parameters : dict
            HDBSCAN parameters

        Returns:
        --------
        Tuple[numpy.ndarray, SklearnHDBSCAN, float]
            Cluster labels, HDBSCAN model, and computation time
        """
        start_time = time.time()

        hdbscan = SklearnHDBSCAN(
            min_cluster_size=parameters["min_cluster_size"],
            min_samples=parameters["min_samples"],
            cluster_selection_epsilon=parameters["cluster_selection_epsilon"],
            cluster_selection_method=parameters["cluster_selection_method"],
        )
        cluster_labels = hdbscan.fit_predict(data)

        computation_time = time.time() - start_time

        return cluster_labels, hdbscan, computation_time

    def _build_metadata(
        self, data: np.ndarray, cluster_labels: np.ndarray, hdbscan_model: SklearnHDBSCAN, parameters: Dict[str, Any], computation_time: float
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata dictionary.

        Parameters:
        -----------
        data : numpy.ndarray
            Original input data
        cluster_labels : numpy.ndarray
            Computed cluster labels
        hdbscan_model : SklearnHDBSCAN
            Fitted HDBSCAN model
        parameters : dict
            HDBSCAN parameters used
        computation_time : float
            Time taken for computation

        Returns:
        --------
        dict
            Complete metadata dictionary
        """
        n_clusters = self._count_clusters(cluster_labels)
        n_noise = self._count_noise_points(cluster_labels)
        metadata = self._prepare_metadata(parameters, data.shape, n_clusters, n_noise)

        metadata.update(
            {
                "algorithm": "hdbscan",
                "silhouette_score": self._compute_silhouette_score(
                    data, cluster_labels
                ),
                "computation_time": computation_time,
                "cluster_probabilities": self._get_cluster_probabilities(hdbscan_model),
                "outlier_scores": self._get_outlier_scores(hdbscan_model),
            }
        )

        return metadata

    def _get_cluster_probabilities(self, hdbscan_model: SklearnHDBSCAN) -> Optional[List[float]]:
        """
        Extract cluster membership probabilities from HDBSCAN model.

        Parameters:
        -----------
        hdbscan_model : SklearnHDBSCAN
            Fitted HDBSCAN model

        Returns:
        --------
        list or None
            List of cluster probabilities or None if not available
        """
        if (
            hasattr(hdbscan_model, "probabilities_")
            and hdbscan_model.probabilities_ is not None
        ):
            return hdbscan_model.probabilities_.tolist()
        return None

    def _get_outlier_scores(self, hdbscan_model: SklearnHDBSCAN) -> Optional[List[float]]:
        """
        Extract outlier scores from HDBSCAN model.

        Parameters:
        -----------
        hdbscan_model : SklearnHDBSCAN
            Fitted HDBSCAN model

        Returns:
        --------
        list or None
            List of outlier scores or None if not available
        """
        if (
            hasattr(hdbscan_model, "outlier_scores_")
            and hdbscan_model.outlier_scores_ is not None
        ):
            return hdbscan_model.outlier_scores_.tolist()
        return None
