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
DPA calculator implementation.

This module provides the DPACalculator class that performs the actual
DPA clustering computation using the DPA package from conda environment.
"""

import time
from typing import Dict, Tuple

import numpy as np

from ..interfaces.calculator_base import CalculatorBase

try:
    from Pipeline.DPA import DensityPeakAdvanced
except ImportError:
    raise ImportError(
        "DPA package not found. Please install DPA package from conda environment. "
        "See: https://github.com/mariaderrico/DPA"
    )


class DPACalculator(CalculatorBase):
    """
    Calculator for DPA clustering.

    This class implements the actual DPA clustering computation using
    the DPA package and computes clustering quality metrics.

    Examples:
    ---------
    >>> # Create calculator and compute clustering
    >>> calc = DPACalculator()
    >>> data = np.random.rand(100, 10)
    >>> labels, metadata = calc.compute(data, Z=2.0, affinity='euclidean',
    ...                                nn_distances=10, density_algo='knn',
    ...                                k_max=20, block_ratio=0.1, blockAn=False,
    ...                                frac=1.0, halos=False)
    >>> print(f"Found {metadata['n_clusters']} clusters")
    """

    def __init__(self, cache_path="./cache"):
        """
        Initialize DPA calculator.

        Parameters:
        -----------
        cache_path : str, optional
            Path for cache files. Default is './cache'.
        """
        super().__init__(cache_path)

    def compute(self, data, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute DPA clustering.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        **kwargs : dict
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

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - cluster_labels: Cluster labels for each sample
            - metadata: Dictionary with clustering information

        Raises:
        -------
        ValueError
            If input data is invalid or required parameters are missing
        ImportError
            If DPA package is not available
        """
        self._validate_input_data(data)

        # Use caching functionality
        return self._compute_with_cache(
            data, 
            "dpa", 
            self._compute_without_cache, 
            **kwargs
        )

    def _compute_without_cache(self, data, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Perform DPA clustering without caching.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster
        **kwargs : dict
            DPA parameters

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Cluster labels and metadata
        """
        parameters = self._extract_parameters(kwargs)

        cluster_labels, dpa_model, computation_time = self._perform_clustering(
            data, parameters
        )
        metadata = self._build_metadata(
            data, cluster_labels, dpa_model, parameters, computation_time
        )

        return cluster_labels, metadata

    def _extract_parameters(self, kwargs):
        """
        Extract and validate DPA parameters.

        Parameters:
        -----------
        kwargs : dict
            Keyword arguments containing DPA parameters

        Returns:
        --------
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
        }

    def _perform_clustering(self, data, parameters):
        """
        Perform DPA clustering computation.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data to cluster
        parameters : dict
            DPA parameters

        Returns:
        --------
        Tuple[numpy.ndarray, object, float]
            Cluster labels, DPA model, and computation time
        """
        start_time = time.time()

        dpa_model = self._create_dpa_model(data, parameters)
        cluster_labels = self._extract_labels(dpa_model, parameters["halos"])

        computation_time = time.time() - start_time

        return cluster_labels, dpa_model, computation_time

    def _create_dpa_model(self, data, parameters):
        """
        Create and fit DPA model.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data to cluster
        parameters : dict
            DPA parameters

        Returns:
        --------
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

    def _extract_labels(self, dpa_model, use_halos):
        """
        Extract cluster labels from DPA model.

        Parameters:
        -----------
        dpa_model : object
            Fitted DPA model
        use_halos : bool
            Whether to return halo points or regular labels

        Returns:
        --------
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
        self, data, cluster_labels, dpa_model, parameters, computation_time
    ):
        """
        Build comprehensive metadata dictionary.

        Parameters:
        -----------
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

        Returns:
        --------
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
                    data, cluster_labels
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

    def _get_cluster_centers(self, dpa_model):
        """
        Extract cluster centers from DPA model.

        Parameters:
        -----------
        dpa_model : object
            Fitted DPA model

        Returns:
        --------
        list or None
            List of cluster center indices or None if not available
        """
        if hasattr(dpa_model, "centers_") and dpa_model.centers_ is not None:
            if hasattr(dpa_model.centers_, "tolist"):
                return dpa_model.centers_.tolist()
            else:
                return list(dpa_model.centers_)
        return None

    def _get_densities(self, dpa_model):
        """
        Extract density values from DPA model.

        Parameters:
        -----------
        dpa_model : object
            Fitted DPA model

        Returns:
        --------
        list or None
            List of density values or None if not available
        """
        if hasattr(dpa_model, "densities_") and dpa_model.densities_ is not None:
            if hasattr(dpa_model.densities_, "tolist"):
                return dpa_model.densities_.tolist()
            else:
                return list(dpa_model.densities_)
        return None

    def _get_nn_indices(self, dpa_model):
        """
        Extract indices of the k_max neighbors of each points.

        Parameters:
        -----------
        dpa_model : object
            Fitted DPA model

        Returns:
        --------
        list or None
            List of nn_indice values or None if not available
        """
        if hasattr(dpa_model, "nn_indices_") and dpa_model.nn_indices_ is not None:
            if hasattr(dpa_model.nn_indices_, "tolist"):
                return dpa_model.nn_indices_.tolist()
            else:
                return list(dpa_model.nn_indices_)
        return None

    def _get_topography(self, dpa_model):
        """
        Extract the topography values, which consists in a Nclus x Nclus symmetric matrix,
        in which the diagonal entries are the heights of the peaks and the off-diagonal entries are the
        heights of the saddle points.

        Parameters:
        -----------
        dpa_model : object
            Fitted DPA model

        Returns:
        --------
        list or None
            List of topography values or None if not available

        References:
        -----------
        Parameter descriptions adapted from the DPA package documentation.
        See: https://github.com/mariaderrico/DPA
        """
        if hasattr(dpa_model, "topography_") and dpa_model.topography_ is not None:
            if hasattr(dpa_model.topography_, "tolist"):
                return dpa_model.topography_.tolist()
            else:
                return list(dpa_model.topography_)
        return None

    def _get_error_densities(self, dpa_model):
        """
        Extract uncertainty values of the density estimation from DPA model.

        Parameters:
        -----------
        dpa_model : object
            Fitted DPA model

        Returns:
        --------
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

    def _get_nn_distances(self, dpa_model):
        """
        Extract distances to k_max neighbors from DPA model.

        Parameters:
        -----------
        dpa_model : object
            Fitted DPA model

        Returns:
        --------
        list or None
            List of nn_distance values or None if not available
        """
        if hasattr(dpa_model, "nn_distances_") and dpa_model.nn_distances_ is not None:
            if hasattr(dpa_model.nn_distances_, "tolist"):
                return dpa_model.nn_distances_.tolist()
            else:
                return list(dpa_model.nn_distances_)
        return None
