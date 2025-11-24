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
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import numpy as np

from mdxplain.utils.progress_utils import ProgressUtils

from ..interfaces.calculator_base import CalculatorBase


class HDBSCANCalculator(CalculatorBase):
    """
    Calculator for HDBSCAN clustering.

    This class implements the actual HDBSCAN clustering computation using
    scikit-learn's HDBSCAN implementation and computes clustering quality metrics.

    Examples
    --------
    >>> # Create calculator and compute clustering
    >>> calc = HDBSCANCalculator()
    >>> data = np.random.rand(100, 10)
    >>> labels, metadata = calc.compute(data, min_cluster_size=5, min_samples=5)
    >>> print(f"Found {metadata['n_clusters']} clusters")
    """

    def __init__(
        self, 
        cache_path: str = "./cache", 
        max_memory_gb: float = 2.0,
        chunk_size: int = 1000,
        use_memmap: bool = False
    ) -> None:
        """
        Initialize HDBSCAN calculator.

        Parameters
        ----------
        cache_path : str, optional
            Path for cache files. Default is './cache'.
        max_memory_gb : float, optional
            Maximum memory threshold in GB. Default is 2.0.
        chunk_size : int, optional
            Chunk size for processing large datasets in sampling methods.
            Used for chunked k-NN prediction and approximate_predict. Default is 1000.
        use_memmap : bool, optional
            Whether to use memory mapping for large datasets. Default is False.

        Returns
        -------
        None
        """
        super().__init__(cache_path, max_memory_gb, chunk_size, use_memmap)

    def compute(self, data: np.ndarray, center_method: str = "centroid", **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute HDBSCAN clustering.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        center_method : str, optional
            Method for calculating cluster centers, default="centroid":

            - "centroid": Representative point (medoid - closest to mean)
            - "mean": Average of cluster members
            - "median": Coordinate-wise median (robust to outliers)
            - "density_peak": Point with highest local density
            - "median_centroid": Medoid from median (more robust to outliers)
            - "rmsd_centroid": Centroid using RMSD metric (better for structural comparisons)
        kwargs : dict
            HDBSCAN parameters including:

            - min_cluster_size : int, minimum size of clusters
            - min_samples : int, minimum samples in neighborhood
            - cluster_selection_epsilon : float, distance threshold
            - cluster_selection_method : str, cluster selection method
            - method : str, clustering method ('standard', 'sampling_approximate', 'sampling_knn')
            - sample_fraction : float, fraction of data to sample
            - force : bool, override memory and dimensionality checks

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            
            - cluster_labels: Cluster labels for each sample (-1 for noise)
            - metadata: Dictionary with clustering information

        Raises
        ------
        ValueError
            If input data is invalid or required parameters are missing
        """
        self._validate_input_data(data)
        parameters = self._extract_parameters(kwargs, data)

        self._validate_memory_and_dimensionality(data, parameters)

        cluster_labels, hdbscan_model, computation_time = self._perform_clustering(
            data, parameters
        )
        metadata = self._build_metadata(
            data, cluster_labels, hdbscan_model, parameters, computation_time, center_method
        )

        return cluster_labels, metadata


    def _extract_parameters(self, kwargs: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """
        Extract and validate HDBSCAN parameters.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments containing HDBSCAN parameters
        data : numpy.ndarray
            Input data to calculate sample size

        Returns
        -------
        dict
            Validated HDBSCAN parameters
        """
        # Calculate sample size directly
        sample_fraction = kwargs.get("sample_fraction", 0.1)
        sample_size = self._calculate_sample_size(
            data.shape[0], sample_fraction,
            min_samples=50000, max_samples=100000
        )

        return {
            "min_cluster_size": kwargs.get("min_cluster_size", 5),
            "min_samples": kwargs.get("min_samples", None),
            "cluster_selection_epsilon": kwargs.get("cluster_selection_epsilon", 0.0),
            "cluster_selection_method": kwargs.get("cluster_selection_method", "eom"),
            "method": kwargs.get("method", "standard"),
            "sample_size": sample_size,
            "knn_neighbors": kwargs.get("knn_neighbors", 5),
            "force": kwargs.get("force", False),
        }

    def _perform_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Any, float]:
        """
        Perform HDBSCAN clustering computation.

        Parameters
        ----------
        data : numpy.ndarray
            Input data to cluster
        parameters : dict
            HDBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, hdbscan.HDBSCAN, float]
            Cluster labels, HDBSCAN model, and computation time
        """
        start_time = time.time()
        
        method = parameters.get("method", "standard")
        
        if method == "standard":
            cluster_labels, hdbscan_model = self._perform_standard_clustering(data, parameters)
        elif method == "approximate_predict":
            cluster_labels, hdbscan_model = self._perform_approximate_predict(data, parameters)
        elif method == "knn_sampling":
            cluster_labels, hdbscan_model = self._perform_knn_sampling(data, parameters)
        else:
            raise ValueError(f"Unknown method: {method}")

        computation_time = time.time() - start_time
        return cluster_labels, hdbscan_model, computation_time

    def _perform_standard_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """
        Perform standard HDBSCAN clustering by loading all data.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            HDBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, hdbscan.HDBSCAN]
            Cluster labels and fitted model
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=parameters["min_cluster_size"],
            min_samples=parameters["min_samples"],
            cluster_selection_epsilon=parameters["cluster_selection_epsilon"],
            cluster_selection_method=parameters["cluster_selection_method"],
        )
        cluster_labels = clusterer.fit_predict(data)
        
        # Apply memmap finalization
        cluster_labels = self._finalize_labels(cluster_labels, "hdbscan", "standard")
        
        return cluster_labels, clusterer

    def _perform_approximate_predict(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """
        HDBSCAN Sampling + approximate_predict (Gold Standard).

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            HDBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, hdbscan.HDBSCAN]
            Cluster labels and fitted model
        """
        n_samples = data.shape[0]
        sample_size = parameters["sample_size"]

        # Sample data randomly (50k-100k points)
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        sample_data = data[sample_indices]
        
        print(f"Using approximate_predict with {sample_size:,} samples "
              f"({sample_size/n_samples*100:.1f}% of {n_samples:,} total)")
        
        # Fit HDBSCAN on sample with prediction_data=True
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=parameters["min_cluster_size"],
            min_samples=parameters["min_samples"],
            cluster_selection_epsilon=parameters["cluster_selection_epsilon"],
            cluster_selection_method=parameters["cluster_selection_method"],
            prediction_data=True  # Enable approximate_predict functionality
        )
        clusterer.fit(sample_data)
        
        # Use approximate_predict for all data in chunks (direct memmap/array writing)
        full_labels = self._prepare_labels_storage(n_samples, "hdbscan", "approximate_predict")
        chunk_size = self.chunk_size
        
        for start in ProgressUtils.iterate(
            range(0, n_samples, chunk_size),
            desc="HDBSCAN approximate_predict",
            unit="chunks",
        ):
            end = min(start + chunk_size, n_samples)
            chunk_labels, _ = hdbscan.approximate_predict(clusterer, data[start:end])
            full_labels[start:end] = chunk_labels
        
        return full_labels, clusterer

    def _perform_knn_sampling(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """
        Perform HDBSCAN on sample + k-NN classifier for rest.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            HDBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, hdbscan.HDBSCAN]
            Cluster labels and fitted model
        """
        n_samples = data.shape[0]
        sample_size = parameters["sample_size"]
        
        # Sample data randomly
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        sample_data = data[sample_indices]
        
        # Fit HDBSCAN on sample
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=parameters["min_cluster_size"],
            min_samples=parameters["min_samples"],
            cluster_selection_epsilon=parameters["cluster_selection_epsilon"],
            cluster_selection_method=parameters["cluster_selection_method"],
        )
        sample_labels = clusterer.fit_predict(sample_data)
        
        # Use base class generic k-NN sampling implementation
        full_labels = super()._perform_knn_sampling(
            data, parameters, sample_indices, sample_labels, "hdbscan", -1
        )
        
        return full_labels, clusterer

    def _build_metadata(
        self, data: np.ndarray, cluster_labels: np.ndarray, hdbscan_model: hdbscan.HDBSCAN,
        parameters: Dict[str, Any], computation_time: float, center_method: str
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata dictionary.

        Parameters
        ----------
        data : numpy.ndarray
            Original input data
        cluster_labels : numpy.ndarray
            Computed cluster labels
        hdbscan_model : hdbscan.HDBSCAN
            Fitted HDBSCAN model
        parameters : dict
            HDBSCAN parameters used
        computation_time : float
            Time taken for computation
        center_method : str
            Method for calculating cluster centers

        Returns
        -------
        dict
            Complete metadata dictionary including centers
        """
        n_clusters = self._count_clusters(cluster_labels)
        n_noise = self._count_noise_points(cluster_labels)
        metadata = self._prepare_metadata(parameters, data.shape, n_clusters, n_noise)

        metadata.update(
            {
                "algorithm": "hdbscan",
                "silhouette_score": self._compute_silhouette_score(
                    data, cluster_labels, sample_size=parameters["sample_size"]
                ),
                "computation_time": computation_time,
                "cluster_probabilities": self._get_cluster_probabilities(hdbscan_model),
                "outlier_scores": self._get_outlier_scores(hdbscan_model),
            }
        )

        # Calculate cluster centers using base class method
        centers, method_used = self._calculate_centers(
            data, cluster_labels, hdbscan_model, center_method
        )
        metadata["centers"] = centers
        metadata["center_method"] = method_used

        return metadata

    def _get_cluster_probabilities(self, hdbscan_model: hdbscan.HDBSCAN) -> Optional[List[float]]:
        """
        Extract cluster membership probabilities from HDBSCAN model.

        Parameters
        ----------
        hdbscan_model : hdbscan.HDBSCAN
            Fitted HDBSCAN model

        Returns
        -------
        list or None
            List of cluster probabilities or None if not available
        """
        if (
            hasattr(hdbscan_model, "probabilities_")
            and hdbscan_model.probabilities_ is not None
        ):
            return hdbscan_model.probabilities_.tolist()
        return None

    def _get_outlier_scores(self, hdbscan_model: hdbscan.HDBSCAN) -> Optional[List[float]]:
        """
        Extract outlier scores from HDBSCAN model.

        Parameters
        ----------
        hdbscan_model : hdbscan.HDBSCAN
            Fitted HDBSCAN model

        Returns
        -------
        list or None
            List of outlier scores or None if not available
        """
        if (
            hasattr(hdbscan_model, "outlier_scores_")
            and hdbscan_model.outlier_scores_ is not None
        ):
            return hdbscan_model.outlier_scores_.tolist()
        return None
