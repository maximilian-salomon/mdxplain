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
DPA calculator implementation.

This module provides the DPACalculator class that performs the actual
DPA clustering computation using the DPA package from conda environment.
"""

import time
import warnings
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
from Pipeline.DPA import DensityPeakAdvanced

from ..interfaces.calculator_base import CalculatorBase


class DPACalculator(CalculatorBase):
    """
    Calculator for DPA clustering.

    This class implements the actual DPA clustering computation using
    the DPA package and computes clustering quality metrics.

    Examples
    --------
    >>> # Create calculator and compute clustering
    >>> calc = DPACalculator()
    >>> data = np.random.rand(100, 10)
    >>> labels, metadata = calc.compute(data, Z=2.0, affinity='euclidean',
    ...                                nn_distances=10, density_algo='knn',
    ...                                k_max=20, block_ratio=0.1, blockAn=False,
    ...                                frac=1.0, halos=False)
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
        Initialize DPA calculator.

        Parameters
        ----------
        cache_path : str, optional
            Path for cache files. Default is './cache'.
        max_memory_gb : float, optional
            Maximum memory threshold in GB. Default is 2.0.
        chunk_size : int, optional
            Chunk size for processing large datasets in sampling methods.
            Used for chunked k-NN prediction. Default is 1000.
        use_memmap : bool, optional
            Whether to use memory mapping for large datasets. Default is False.

        Returns
        -------
        None
        """
        super().__init__(cache_path, max_memory_gb, chunk_size, use_memmap)

    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute DPA clustering.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        '**'kwargs : dict
            DPA parameters including:
            See DPA init docstring for more information or
            https://github.com/mariaderrico/DPA and https://github.com/mariaderrico/DPA/blob/master/DPA_analysis.ipynb

            - Z : float, density threshold parameter
            - affinity : str, affinity metric for distance calculation
            - nn_distances : int, number of nearest neighbors
            - density_algo : str, algorithm for density computation
            - k_max : int, maximum number of clusters
            - block_ratio : float, block ratio parameter
            - blockAn : bool, whether to use block analysis
            - frac : float, fraction parameter for sampling
            - halos : bool, whether to return halo points assigned to cluster 0
            - method : str, clustering method ('standard', 'knn_sampling')
            - sample_fraction : float, fraction of data to sample
            - force : bool, override memory and dimensionality checks

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            
            - cluster_labels: Cluster labels for each sample
            - metadata: Dictionary with clustering information

        Raises
        ------
        ValueError
            If input data is invalid or required parameters are missing
        ImportError
            If DPA package is not available
        """
        self._validate_input_data(data)
        parameters = self._extract_parameters(kwargs, data)
        
        self._validate_memory_and_dimensionality(data, parameters)

        cluster_labels, dpa_model, computation_time = self._perform_clustering(
            data, parameters
        )
        metadata = self._build_metadata(
            data, cluster_labels, dpa_model, parameters, computation_time
        )

        return cluster_labels, metadata


    def _extract_parameters(self, kwargs: Dict[str, Any], data: np.ndarray) -> Dict[str, Any]:
        """
        Extract and validate DPA parameters.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments containing DPA parameters
        data : numpy.ndarray
            Input data to calculate sample size

        Returns
        -------
        dict
            Validated DPA parameters
        """
        required_params = [
            "Z",
            "metric",
            "affinity",
            "density_algo",
            "k_max",
            "D_thr",
            "dim_algo",
            "blockAn",
            "block_ratio",
            "frac",
        ]

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Required parameter '{param}' is missing")

        # Calculate sample size directly for DPA (smaller limits due to complexity)
        sample_fraction = kwargs.get("sample_fraction", 0.1)
        sample_size = self._calculate_sample_size(
            data.shape[0], sample_fraction, 
            min_samples=10000, max_samples=50000
        )

        return {
            "Z": kwargs["Z"],
            "metric": kwargs["metric"],
            "affinity": kwargs["affinity"],
            "density_algo": kwargs["density_algo"],
            "k_max": kwargs["k_max"],
            "D_thr": kwargs["D_thr"],
            "dim_algo": kwargs["dim_algo"],
            "blockAn": kwargs["blockAn"],
            "block_ratio": kwargs["block_ratio"],
            "frac": kwargs["frac"],
            "halos": kwargs.get("halos", False),
            "method": kwargs.get("method", "standard"),
            "sample_size": sample_size,
            "knn_neighbors": kwargs.get("knn_neighbors", 5),
            "force": kwargs.get("force", False),
        }

    def _perform_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Any, float]:
        """
        Perform DPA clustering computation.

        Parameters
        ----------
        data : numpy.ndarray
            Input data to cluster
        parameters : dict
            DPA parameters

        Returns
        -------
        Tuple[numpy.ndarray, object, float]
            Cluster labels, DPA model, and computation time
        """
        start_time = time.time()
        
        method = parameters.get("method", "standard")
        
        if method == "standard":
            cluster_labels, dpa_model = self._perform_standard_clustering(data, parameters)
        elif method == "knn_sampling":
            cluster_labels, dpa_model = self._perform_knn_sampling(data, parameters)
        else:
            raise ValueError(f"Unknown method: {method}")

        computation_time = time.time() - start_time
        return cluster_labels, dpa_model, computation_time

    def _perform_standard_clustering(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
        """
        Perform standard DPA clustering by loading all data.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            DPA parameters

        Returns
        -------
        Tuple[numpy.ndarray, DensityPeakAdvanced]
            Cluster labels and fitted model
        """
        dpa_model = self._create_dpa_model(data, parameters)
        cluster_labels = self._extract_labels(dpa_model, parameters["halos"])
        
        # Apply memmap finalization
        cluster_labels = self._finalize_labels(cluster_labels, "dpa", "standard")
        
        return cluster_labels, dpa_model

    def _perform_knn_sampling(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
        """
        DPA Sampling + k-NN (only practical option for large data).

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            DPA parameters

        Returns
        -------
        Tuple[numpy.ndarray, DensityPeakAdvanced]
            Cluster labels and fitted model
        """
        n_samples = data.shape[0]
        sample_size = parameters["sample_size"]

        print(f"Using knn_sampling with {sample_size:,} samples for DPA "
              f"({sample_size/n_samples*100:.1f}% of {n_samples:,} total)")
        
        # Sample data randomly
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        sample_data = data[sample_indices]
        
        # DPA on sample
        dpa_model = self._create_dpa_model(sample_data, parameters)
        sample_labels = self._extract_labels(dpa_model, parameters["halos"])
        
        # Use base class generic k-NN sampling with DPA-specific noise label
        noise_label = -1
        full_labels = super()._perform_knn_sampling(
            data, parameters, sample_indices, sample_labels, "dpa", noise_label
        )
        
        return full_labels, dpa_model

    def _create_dpa_model(self, data: np.ndarray, parameters: Dict[str, Any]) -> Any:
        """
        Create and fit DPA model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data to cluster
        parameters : dict
            DPA parameters

        Returns
        -------
        object
            Fitted DPA model
        """
        dpa = DensityPeakAdvanced(
            Z=parameters["Z"],
            metric=parameters["metric"],
            affinity=parameters["affinity"],
            density_algo=parameters["density_algo"],
            k_max=parameters["k_max"],
            D_thr=parameters["D_thr"],
            dim_algo=parameters["dim_algo"],
            blockAn=parameters["blockAn"],
            block_ratio=parameters["block_ratio"],
            frac=parameters["frac"],
            n_jobs=-1,  # Always use all available cores
        )

        dpa.fit(data)
        return dpa

    def _extract_labels(self, dpa_model: DensityPeakAdvanced, use_halos: bool) -> np.ndarray:
        """
        Extract cluster labels from DPA model.

        Parameters
        ----------
        dpa_model : object
            Fitted DPA model
        use_halos : bool
            Whether to return halo points or regular labels

        Returns
        -------
        numpy.ndarray
            Cluster labels
        """
        if use_halos and hasattr(dpa_model, "halos_"):
            return dpa_model.halos_
        elif hasattr(dpa_model, "labels_"):
            return dpa_model.labels_
        else:
            raise ValueError("DPA model does not have expected label attributes")

    def _build_metadata(
        self, 
        data: np.ndarray, 
        cluster_labels: np.ndarray, 
        dpa_model: DensityPeakAdvanced, 
        parameters: Dict[str, Any], 
        computation_time: float
    ) -> Dict[str, Any]:
        """
        Build comprehensive metadata dictionary.

        Parameters
        ----------
        data : numpy.ndarray
            Original input data
        cluster_labels : numpy.ndarray
            Computed cluster labels
        dpa_model : object
            Fitted DPA model
        parameters : dict
            DPA parameters used
        computation_time : float
            Time taken for computation

        Returns
        -------
        dict
            Complete metadata dictionary
        """
        n_clusters = self._count_clusters(cluster_labels)
        if parameters["halos"]:
            n_noise = self._count_noise_points(cluster_labels, noise_cluster=0)
        else:
            n_noise = 0
        metadata = self._prepare_metadata(parameters, data.shape, n_clusters, n_noise)

        metadata.update(
            {
                "algorithm": "dpa",
                "silhouette_score": self._compute_silhouette_score(
                    data, cluster_labels, sample_size=parameters["sample_size"]
                ),
                "computation_time": computation_time,
                "cluster_centers": self._get_cluster_centers(dpa_model),
                "densities": self._get_densities(dpa_model),
                "nn_distances": self._get_nn_distances(dpa_model),
                "nn_indices": self._get_nn_indices(dpa_model),
                "topography": self._get_topography(dpa_model),
                "error_densities": self._get_error_densities(dpa_model),
            }
        )

        return metadata

    def _get_cluster_centers(
            self, 
            dpa_model: DensityPeakAdvanced
    ) -> Optional[List[float]]:
        """
        Extract cluster centers from DPA model.

        Parameters
        ----------
        dpa_model : object
            Fitted DPA model

        Returns
        -------
        list or None
            List of cluster center indices or None if not available
        """
        if hasattr(dpa_model, "centers_") and dpa_model.centers_ is not None:
            if hasattr(dpa_model.centers_, "tolist"):
                return dpa_model.centers_.tolist()
            else:
                return list(dpa_model.centers_)
        return None

    def _get_densities(self, dpa_model: DensityPeakAdvanced) -> Optional[List[float]]:
        """
        Extract density values from DPA model.

        Parameters
        ----------
        dpa_model : object
            Fitted DPA model

        Returns
        -------
        list or None
            List of density values or None if not available
        """
        if hasattr(dpa_model, "densities_") and dpa_model.densities_ is not None:
            if hasattr(dpa_model.densities_, "tolist"):
                return dpa_model.densities_.tolist()
            else:
                return list(dpa_model.densities_)
        return None

    def _get_nn_indices(self, dpa_model: DensityPeakAdvanced) -> Optional[List[int]]:
        """
        Extract indices of the k_max neighbors of each points.

        Parameters
        ----------
        dpa_model : object
            Fitted DPA model

        Returns
        -------
        list or None
            List of nn_indice values or None if not available
        """
        if hasattr(dpa_model, "nn_indices_") and dpa_model.nn_indices_ is not None:
            if hasattr(dpa_model.nn_indices_, "tolist"):
                return dpa_model.nn_indices_.tolist()
            else:
                return list(dpa_model.nn_indices_)
        return None

    def _get_topography(self, dpa_model: DensityPeakAdvanced) -> Optional[List[Any]]:
        """
        Extract the topography values, which consists in a Nclus x Nclus symmetric matrix,
        in which the diagonal entries are the heights of the peaks and the off-diagonal entries are the
        heights of the saddle points.

        Parameters
        ----------
        dpa_model : object
            Fitted DPA model

        Returns
        -------
        list or None
            List of topography values or None if not available

        References
        ----------
        Parameter descriptions adapted from the DPA package documentation.
        See: https://github.com/mariaderrico/DPA
        """
        if hasattr(dpa_model, "topography_") and dpa_model.topography_ is not None:
            if hasattr(dpa_model.topography_, "tolist"):
                return dpa_model.topography_.tolist()
            else:
                return list(dpa_model.topography_)
        return None

    def _get_error_densities(self, dpa_model: DensityPeakAdvanced) -> Optional[List[float]]:
        """
        Extract uncertainty values of the density estimation from DPA model.

        Parameters
        ----------
        dpa_model : object
            Fitted DPA model

        Returns
        -------
        list or None
            List of error density values or None if not available
        """
        if (
            hasattr(dpa_model, "err_densities_")
            and dpa_model.err_densities_ is not None
        ):
            if hasattr(dpa_model.err_densities_, "tolist"):
                return dpa_model.err_densities_.tolist()
            else:
                return list(dpa_model.err_densities_)
        return None

    def _get_nn_distances(self, dpa_model: DensityPeakAdvanced) -> Optional[List[float]]:
        """
        Extract distances to k_max neighbors from DPA model.

        Parameters
        ----------
        dpa_model : object
            Fitted DPA model

        Returns
        -------
        list or None
            List of nn_distance values or None if not available
        """
        if hasattr(dpa_model, "nn_distances_") and dpa_model.nn_distances_ is not None:
            if hasattr(dpa_model.nn_distances_, "tolist"):
                return dpa_model.nn_distances_.tolist()
            else:
                return list(dpa_model.nn_distances_)
        return None
