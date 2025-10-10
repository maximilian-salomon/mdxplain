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
DBSCAN calculator implementation.

This module provides the DBSCANCalculator class that performs the actual
DBSCAN clustering computation using scikit-learn.
"""

import time
from typing import Dict, Tuple, Any
from scipy.sparse import vstack

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ..interfaces.calculator_base import CalculatorBase


class DBSCANCalculator(CalculatorBase):
    """
    Calculator for DBSCAN clustering.

    This class implements the actual DBSCAN clustering computation using
    scikit-learn's DBSCAN implementation and computes clustering quality metrics.

    Examples
    --------
    >>> # Create calculator and compute clustering
    >>> calc = DBSCANCalculator()
    >>> data = np.random.rand(100, 10)
    >>> labels, metadata = calc.compute(data, eps=0.5, min_samples=5)
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
        Initialize DBSCAN calculator.

        Parameters
        ----------
        cache_path : str, optional
            Path for cache files. Default is './cache'.
        max_memory_gb : float, optional
            Maximum memory threshold in GB. Default is 2.0.
        chunk_size : int, optional
            Chunk size for processing large datasets in sampling methods.
            Used for chunked k-NN prediction and precomputed distance matrix. Default is 1000.
        use_memmap : bool, optional
            Whether to use memory mapping for large datasets. Default is False.
        """
        super().__init__(cache_path, max_memory_gb, chunk_size, use_memmap)

    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute DBSCAN clustering.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        \*\*kwargs : dict
            DBSCAN parameters including:

            - eps : float, maximum distance between samples
            - min_samples : int, minimum samples in neighborhood
            - method : str, clustering method ('standard', 'precomputed', 'knn_sampling')
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

        cluster_labels, dbscan_model, computation_time = self._perform_clustering(
            data, parameters
        )

        metadata = self._build_metadata(
            data, cluster_labels, dbscan_model, parameters, computation_time
        )

        return cluster_labels, metadata


    def _extract_parameters(self, kwargs: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """
        Extract and validate DBSCAN parameters.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments containing DBSCAN parameters
        data : numpy.ndarray
            Input data to calculate sample size

        Returns
        -------
        dict
            Validated DBSCAN parameters
        """
        # Calculate sample size directly
        sample_fraction = kwargs.get("sample_fraction", 0.1)
        sample_size = self._calculate_sample_size(
            data.shape[0], sample_fraction, 
            min_samples=50000, max_samples=100000
        )
        
        return {
            "eps": kwargs.get("eps", 0.5),
            "min_samples": kwargs.get("min_samples", 5),
            "method": kwargs.get("method", "standard"),
            "sample_size": sample_size,
            "force": kwargs.get("force", False),
            "knn_neighbors": kwargs.get("knn_neighbors", 5),
        }

    def _perform_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Any, float]:
        """
        Perform DBSCAN clustering computation.

        Parameters
        ----------
        data : numpy.ndarray
            Input data to cluster
        parameters : dict
            DBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, SklearnDBSCAN, float]
            Cluster labels, DBSCAN model, and computation time
        """
        start_time = time.time()
        
        method = parameters.get("method", "standard")
        
        if method == "standard":
            cluster_labels, dbscan_model = self._perform_standard_clustering(data, parameters)
        elif method == "precomputed":
            cluster_labels, dbscan_model = self._perform_precomputed_clustering(data, parameters)
        elif method == "knn_sampling":
            cluster_labels, dbscan_model = self._perform_knn_sampling(data, parameters)
        else:
            raise ValueError(f"Unknown method: {method}")

        computation_time = time.time() - start_time
        return cluster_labels, dbscan_model, computation_time

    def _perform_standard_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, SklearnDBSCAN]:
        """
        Perform standard DBSCAN clustering by loading all data.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            DBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, SklearnDBSCAN]
            Cluster labels and fitted model
        """
        dbscan = SklearnDBSCAN(
            eps=parameters["eps"],
            min_samples=parameters["min_samples"],
        )
        cluster_labels = dbscan.fit_predict(data)
        
        # Apply memmap finalization
        cluster_labels = self._finalize_labels(cluster_labels, "dbscan", "standard")
        
        return cluster_labels, dbscan

    def _perform_precomputed_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, SklearnDBSCAN]:
        """
        DBSCAN with precomputed distance matrix (exact but slow).

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            DBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, SklearnDBSCAN]
            Cluster labels and fitted model
        """
        n_samples = data.shape[0]
        eps = parameters["eps"]
        
        print(f"Using precomputed method for {n_samples:,} samples (exact but slow)")
        
        # Build index ONCE before loop
        print("Building neighbors index on full dataset (this may take a while)...")
        nbrs = NearestNeighbors(radius=eps, n_jobs=-1)
        nbrs.fit(data)
        
        # Compute radius neighbors graph chunk-wise
        chunk_size = self.chunk_size
        sparse_matrices = []
        
        for chunk_start in tqdm(range(0, n_samples, chunk_size), 
                                desc="Building sparse matrix", unit="chunks"):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_data = data[chunk_start:chunk_end]
            
            # Only query (no rebuilding index!)
            sparse_matrix = nbrs.radius_neighbors_graph(chunk_data, mode='distance')
            sparse_matrices.append(sparse_matrix)
        
        # Combine sparse matrices
        full_sparse_matrix = vstack(sparse_matrices)
        
        # DBSCAN with precomputed metric
        dbscan = SklearnDBSCAN(
            eps=eps,
            min_samples=parameters["min_samples"],
            metric='precomputed',
            n_jobs=-1
        )
        cluster_labels = dbscan.fit_predict(full_sparse_matrix)
        
        # Apply memmap finalization
        cluster_labels = self._finalize_labels(cluster_labels, "dbscan", "precomputed")
        
        print("Precomputed clustering finished.")
        return cluster_labels, dbscan

    def _perform_knn_sampling(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, SklearnDBSCAN]:
        """
        DBSCAN Sampling + k-NN (fast approximation).

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            DBSCAN parameters

        Returns
        -------
        Tuple[numpy.ndarray, SklearnDBSCAN]
            Cluster labels and fitted model
        """
        n_samples = data.shape[0]
        sample_size = parameters["sample_size"]

        # Sample data randomly
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        sample_data = data[sample_indices]
        
        # DBSCAN on sample
        dbscan = SklearnDBSCAN(
            eps=parameters["eps"],
            min_samples=parameters["min_samples"],
        )
        sample_labels = dbscan.fit_predict(sample_data)
        
        # Use base class generic k-NN sampling implementation
        full_labels = super()._perform_knn_sampling(
            data, parameters, sample_indices, sample_labels, "dbscan", -1
        )
        
        return full_labels, dbscan

    def _build_metadata(
        self, data: np.ndarray, cluster_labels: np.ndarray, dbscan_model: SklearnDBSCAN, parameters: Dict[str, Any], computation_time: float
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata dictionary.

        Parameters
        ----------
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

        Returns
        -------
        dict
            Complete metadata dictionary
        """
        n_clusters = self._count_clusters(cluster_labels)
        n_noise = self._count_noise_points(cluster_labels)
        silhouette = self._compute_silhouette_score(
            data, cluster_labels, sample_size=parameters["sample_size"]
        )

        metadata = self._prepare_metadata(
            parameters, data.shape, n_clusters, n_noise
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

    def _get_core_sample_indices(self, dbscan_model: SklearnDBSCAN) -> np.ndarray:
        """
        Extract core sample indices from DBSCAN model.

        Parameters
        ----------
        dbscan_model : SklearnDBSCAN
            Fitted DBSCAN model

        Returns
        -------
        list
            List of core sample indices
        """
        if len(dbscan_model.core_sample_indices_) > 0:
            return dbscan_model.core_sample_indices_.tolist()
        return []
