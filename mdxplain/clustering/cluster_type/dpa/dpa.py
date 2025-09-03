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
DPA cluster type implementation.

This module provides the DPA (Density Peak Advanced) cluster type that implements
density-based clustering for molecular dynamics trajectory analysis using the
DPA package from conda environment.
"""

from typing import Dict, Tuple, Any
import numpy as np

from ..interfaces.cluster_type_base import ClusterTypeBase
from .dpa_calculator import DPACalculator


class DPA(ClusterTypeBase):
    """
    DPA (Density Peak Advanced) cluster type.

    DPA is a density-based clustering algorithm that identifies cluster centers
    as points with high density that are far from other high-density points.
    It's particularly useful for identifying conformational states in molecular
    dynamics trajectories with complex cluster shapes and varying densities.


    Examples:
    ---------
    >>> # Create DPA with default parameters
    >>> dpa = DPA()

    >>> # Create DPA with custom parameters
    >>> dpa = DPA(Z=1.5, metric='euclidean', density_algo='PAk',
    ...           k_max=500, blockAn=True, block_ratio=10)

    >>> # Initialize and compute clustering
    >>> dpa.init_calculator()
    >>> labels, metadata = dpa.compute(data)

    References:
    -----------
    M. d'Errico, E. Facco, A. Laio, A. Rodriguez, Information Sciences, Volume 560,
    June 2021, 476-492. See: https://github.com/mariaderrico/DPA
    """

    def __init__(
        self,
        Z: float = 1.0,
        metric: str = "euclidean",
        affinity: str = "precomputed",
        density_algo: str = "PAk",
        k_max: int = 1000,
        D_thr: float = 23.92812698,
        dim_algo: str = "twoNN",
        blockAn: bool = True,
        block_ratio: int = 20,
        frac: float = 1.0,
        halos: bool = False,
        method: str = "standard",
        sample_fraction: float = 0.1,
        force: bool = False,
    ) -> None:
        """
        Initialize DPA cluster type.

        Parameters:
        -----------
        Z : float, default=1
            The number of standard deviations, which fixes the level of statistical
            confidence at which one decides to consider a cluster meaningful.

        metric : string or callable, default="euclidean"
            The distance metric to use. If metric is a string, it must be one of the
            options allowed by scipy.spatial.distance.pdist for its metric parameter,
            or a metric listed in VALID_METRIC = [precomputed, euclidean, cosine].
            If metric is "precomputed", X is assumed to be a distance matrix.
            Alternatively, if metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays from X as input and return a value indicating
            the distance between them.

        affinity : string or callable, default='precomputed'
            How to construct the affinity matrix.
            - "nearest_neighbors": construct the affinity matrix by computing a
            graph of nearest neighbors.
            - "rbf": construct the affinity matrix using a radial basis function
            (RBF) kernel.
            - "precomputed": interpret X as a precomputed affinity matrix.
            - "precomputed_nearest_neighbors": interpret X as a sparse graph
            of precomputed nearest neighbors, and constructs the affinity matrix
            by selecting the n_neighbors nearest neighbors.
            - one of the kernels supported by sklearn.metrics.pairwise_kernels.

        density_algo : string, default="PAk"
            Define the algorithm to use as density estimator. It must be one of the
            options allowed by VALID_DENSITY = [PAk, kNN].

        k_max : int, default=1000
            This parameter is considered if density_algo is "PAk" or "kNN", it is
            ignored otherwise. k_max set the maximum number of nearest-neighbors
            considered by the density estimator. If density_algo="PAk", k_max is used
            by the algorithm in the search for the largest number of neighbors k_hat
            for which the condition of constant density holds, within a given level
            of confidence. If density_algo="kNN", k_max set the number of neighbors
            to be used by the standard k-Nearest Neighbor algorithm. If the number
            of points in the sample N is less than the default value, k_max will be
            set automatically to the value N/2.

        D_thr : float, default=23.92812698
            This parameter is considered if density_algo is "PAk", it is ignored
            otherwise. Set the level of confidence in the PAk density estimator.
            The default value corresponds to a p-value of 10^-6 for a χ² distribution
            with one degree of freedom.

        dim_algo : string, default="twoNN"
            Method for intrinsic dimensionality calculation. If dim_algo is "auto",
            dim is assumed to be equal to n_samples. If dim_algo is a string, it must
            be one of the options allowed by VALID_DIM = [auto, twoNN].

        blockAn : bool, default=True
            This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
            If blockAn is True the algorithm perform a block analysis that allows
            discriminating the relevant dimensions as a function of the block size.
            This allows to study the stability of the estimation with respect to
            changes in the neighborhood size, which is crucial for ID estimations
            when the data lie on a manifold perturbed by a high-dimensional noise.

        block_ratio : int, default=20
            This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
            Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False,
            block_ratio is ignored.

        frac : float, default=1.0
            This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
            Define the fraction of points in the data set used for ID calculation.
            By default the full data set is used.

        halos : bool, default=False
            Whether to return halo points. If True, returns dpa.halos_,
            otherwise returns dpa.labels_.
            If true frames which are on a low density are set to 0.
            So kind of a -1 in sklearn clustering algorithms.
            If false, each frame is assigned to its most probable cluster.

        method : str, default="standard"
            Clustering method:
            - "standard": Load all data into memory (default)
            - "sampling_approximate": Sample data + approximate_predict for large datasets
            - "sampling_knn": Sample data + k-NN classifier fallback

        sample_fraction : float, default=0.1
            Fraction of data to sample for sampling-based methods (10%)
            Final sample size: max(50000, min(100000, sample_fraction * n_samples))

        force : bool, default=False
            Override memory and dimensionality checks (converts errors to warnings)

        Returned Metadata:
        ------------------
        algorithm : str
            Always "dpa"
        hyperparameters : dict
            Dictionary containing all DPA parameters used
        original_shape : tuple
            Shape of the input data (n_samples, n_features)
        n_clusters : int
            Number of clusters found (excluding noise/halo points)
        n_noise : int
            Number of noise/halo points identified
        silhouette_score : float or None
            Silhouette score for clustering quality assessment
        computation_time : float
            Time taken for clustering computation in seconds
        cluster_centers : list or None
            Indices of cluster center points
        densities : list or None
            Density values for each point
        nn_distances : list or None
            Distances to k_max neighbors for each point
        nn_indices : list or None
            Indices of k_max neighbors for each point
        topography : list or None
            Topography matrix with peak heights and saddle points
        error_densities : list or None
            Uncertainty values of density estimation
        cache_path : str
            Path used for caching results

        Note:
        -----
        The n_jobs parameter is automatically set to -1 (use all processors) for
        optimal performance in molecular dynamics analysis.

        References:
        -----------
        Parameter descriptions adapted from the DPA package documentation.
        See: https://github.com/mariaderrico/DPA
        """
        super().__init__()
        self.Z = Z
        self.metric = metric
        self.affinity = affinity
        self.density_algo = density_algo
        self.k_max = k_max
        self.D_thr = D_thr
        self.dim_algo = dim_algo
        self.blockAn = blockAn
        self.block_ratio = block_ratio
        self.frac = frac
        self.halos = halos
        self.method = method
        self.sample_fraction = sample_fraction
        self.force = force
        self._validate_parameters()

    @classmethod
    def get_type_name(cls) -> str:
        """
        Return unique string identifier for DPA cluster type.

        Returns:
        --------
        str
            The string 'dpa'
        """
        return "dpa"

    def init_calculator(
        self, 
        cache_path: str = "./cache", 
        max_memory_gb: float = 2.0,
        chunk_size: int = 1000,
        use_memmap: bool = False
    ) -> None:
        """
        Initialize the DPA calculator.

        Parameters:
        -----------
        cache_path : str, optional
            Directory path for cache files. Default is './cache'.
        max_memory_gb : float, optional
            Maximum memory threshold in GB. Default is 2.0.
        chunk_size : int, optional
            Chunk size for processing large datasets. Default is 1000.
        use_memmap : bool, optional
            Whether to use memory mapping for large datasets. Default is False.
        """
        self.calculator = DPACalculator(
            cache_path=cache_path, 
            max_memory_gb=max_memory_gb,
            chunk_size=chunk_size,
            use_memmap=use_memmap
        )

    def compute(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute DPA clustering.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - cluster_labels: Cluster labels for each sample
            - metadata: Dictionary with clustering information

        Raises:
        -------
        ValueError
            If calculator is not initialized
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        return self.calculator.compute(
            data,
            Z=self.Z,
            metric=self.metric,
            affinity=self.affinity,
            density_algo=self.density_algo,
            k_max=self.k_max,
            D_thr=self.D_thr,
            dim_algo=self.dim_algo,
            blockAn=self.blockAn,
            block_ratio=self.block_ratio,
            frac=self.frac,
            halos=self.halos,
            method=self.method,
            sample_fraction=self.sample_fraction,
            force=self.force,
        )

    def _validate_parameters(self):
        """
        Validate DPA parameters.

        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        # Validate Z
        if not isinstance(self.Z, (int, float)) or self.Z <= 0:
            raise ValueError("Z must be a positive number")

        # Validate metric (can be string or callable)
        if not (isinstance(self.metric, str) or callable(self.metric)):
            raise ValueError("metric must be a string or callable")

        if isinstance(self.metric, str):
            valid_metrics = ["precomputed", "euclidean", "cosine"]
            if self.metric not in valid_metrics:
                raise ValueError(f"metric must be one of {valid_metrics} or a callable")

        # Validate affinity (can be string or callable)
        if not (isinstance(self.affinity, str) or callable(self.affinity)):
            raise ValueError("affinity must be a string or callable")

        if isinstance(self.affinity, str):
            valid_affinities = [
                "precomputed",
                "rbf",
                "nearest_neighbors",
                "precomputed_nearest_neighbors",
            ]
            if self.affinity not in valid_affinities:
                raise ValueError(
                    f"affinity must be one of {valid_affinities} or a callable"
                )

        # Validate density_algo
        if self.density_algo not in ["PAk", "kNN"]:
            raise ValueError("density_algo must be 'PAk' or 'kNN'")

        # Validate k_max
        if not isinstance(self.k_max, int) or self.k_max < 1:
            raise ValueError("k_max must be a positive integer")

        # Validate D_thr
        if not isinstance(self.D_thr, (int, float)) or self.D_thr <= 0:
            raise ValueError("D_thr must be a positive number")

        # Validate dim_algo
        if self.dim_algo not in ["auto", "twoNN"]:
            raise ValueError("dim_algo must be 'auto' or 'twoNN'")

        # Validate blockAn
        if not isinstance(self.blockAn, bool):
            raise ValueError("blockAn must be a boolean")

        # Validate block_ratio
        if not isinstance(self.block_ratio, int) or self.block_ratio < 1:
            raise ValueError("block_ratio must be a positive integer")

        # Validate frac
        if not isinstance(self.frac, (int, float)) or not (0 < self.frac <= 1):
            raise ValueError("frac must be a number between 0 and 1")

        # Validate halos
        if not isinstance(self.halos, bool):
            raise ValueError("halos must be a boolean")

        # Validate method
        if self.method not in ["standard", "knn_sampling"]:
            raise ValueError("method must be 'standard' or 'knn_sampling'")

        # Validate sample_fraction
        if not isinstance(self.sample_fraction, (int, float)) or not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be a number between 0 and 1")

        # Validate force
        if not isinstance(self.force, bool):
            raise ValueError("force must be a boolean")
