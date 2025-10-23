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

"""Service for adding DPA clustering with flexible center method selection."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.cluster_manager import ClusterManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..cluster_type import DPA


class DPAAddService:
    """
    Service for adding DPA clustering.

    Uses centroid (medoid) center calculation by default. The centroid is the
    actual data point closest to the cluster mean, ensuring the cluster center
    is a real conformational state from the trajectory.

    For alternative center methods, use:
    - .with_mean_centers() - Arithmetic mean (may not be real state)
    - .with_median_centers() - Feature-wise median (robust to outliers)
    - .with_density_peak_centers() - Highest density point
    - .with_median_centroid_centers() - Medoid from median
    - .with_rmsd_centroid_centers() - RMSD-based centroid

    Examples
    --------
    >>> # Standard call with default centroid centers
    >>> pipeline.clustering.add.dpa("features", Z=2.0)

    >>> # Explicit center method selection
    >>> pipeline.clustering.add.dpa.with_median_centers("features", Z=2.0)
    >>> pipeline.clustering.add.dpa.with_density_peak_centers("features", Z=2.0)
    """

    def __init__(self, manager: ClusterManager, pipeline_data: PipelineData) -> None:
        """
        Initialize DPA service.

        Parameters
        ----------
        manager : ClusterManager
            Cluster manager instance
        pipeline_data : PipelineData
            Pipeline data container

        Returns
        -------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data

    def __call__(
        self,
        selection_name: str,
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
        knn_neighbors: int = 5,
        use_decomposed: bool = True,
        cluster_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
        override_cache: bool = False,
    ) -> None:
        """
        Add DPA (Density Peak Analysis) clustering algorithm.

        DPA identifies cluster centers as points with high local density
        and high distance to points with higher density. It's particularly
        effective for finding clusters with irregular shapes.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        Z : float, default=1.0
            Number of standard deviations for statistical confidence
        metric : str, default="euclidean"
            Distance metric ('euclidean', 'cosine', 'precomputed')
        affinity : str, default="precomputed"
            How to construct affinity matrix ('precomputed', 'rbf', 'nearest_neighbors')
        density_algo : str, default="PAk"
            Density estimator algorithm ('PAk' or 'kNN')
        k_max : int, default=1000
            Maximum number of nearest neighbors for density estimation
        D_thr : float, default=23.92812698
            Confidence threshold for PAk density estimator
        dim_algo : str, default="twoNN"
            Intrinsic dimensionality calculation method ('auto' or 'twoNN')
        blockAn : bool, default=True
            Whether to perform block analysis for twoNN dimensionality
        block_ratio : int, default=20
            Minimum block size as n_samples/block_ratio for twoNN
        frac : float, default=1.0
            Fraction of points used for dimensionality calculation
        halos : bool, default=False
            Whether to return halo points (low-density outliers)
        method : str, default="standard"
            Clustering method ('standard', 'sampling_approximate', 'sampling_knn')
        sample_fraction : float, default=0.1
            Fraction of data to sample for sampling-based methods
        knn_neighbors : int, default=5
            Number of neighbors for k-NN classifier in sampling methods
        use_decomposed : bool, default=True
            Whether to use decomposed data if available
        cluster_name : str, optional
            Name for the clustering result
        data_selector_name : str, optional
            Name of data selector to apply before clustering
        force : bool, default=False
            Force recalculation or override automatic method selection
        override_cache : bool, default=False
            Override cache settings for this clustering

        Returns
        -------
        None
            Adds DPA clustering results to pipeline data

        Examples
        --------
        >>> # Basic DPA clustering
        >>> pipeline.clustering.add.dpa("my_features", Z=2.0)

        >>> # DPA with custom density algorithm
        >>> pipeline.clustering.add.dpa(
        ...     "distance_features",
        ...     Z=1.5,
        ...     density_algo="kNN",
        ...     k_max=500,
        ...     cluster_name="density_peaks"
        ... )

        >>> # DPA with sampling for large datasets
        >>> pipeline.clustering.add.dpa(
        ...     "conformational_features",
        ...     Z=1.8,
        ...     method="sampling_knn",
        ...     sample_fraction=0.2,
        ...     data_selector_name="equilibrated_frames"
        ... )

        Notes
        -----
        **Center Method:** Uses centroid (medoid) by default. The centroid is the
        actual data point closest to the cluster mean, ensuring the cluster center
        represents a real conformational state. Use .with_xxx_centers() for alternatives.

        DPA is in practice easy to use for molecular dynamics data and
        does not require extensive parameter tuning. It can find clusters
        of varying shapes and densities without prior knowledge of the
        data distribution. Just vary the Z parameter to adjust the number
        of clusters.
        """
        return self._execute(
            selection_name, Z, metric, affinity, density_algo, k_max, D_thr,
            dim_algo, blockAn, block_ratio, frac, halos, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="centroid"
        )

    def with_centroid_centers(
        self,
        selection_name: str,
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
        knn_neighbors: int = 5,
        use_decomposed: bool = True,
        cluster_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
        override_cache: bool = False,
    ) -> None:
        """
        Add DPA with centroid (medoid) centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        Z : float, default=1.0
            Z-score threshold
        metric : str, default="euclidean"
            Distance metric
        affinity : str, default="precomputed"
            Affinity type
        density_algo : str, default="PAk"
            Density algorithm
        k_max : int, default=1000
            Maximum k for density
        D_thr : float, default=23.92812698
            Density threshold
        dim_algo : str, default="twoNN"
            Dimensionality algorithm
        blockAn : bool, default=True
            Block analysis
        block_ratio : int, default=20
            Block ratio
        frac : float, default=1.0
            Fraction of data
        halos : bool, default=False
            Include halos
        method : str, default="standard"
            Clustering method
        sample_fraction : float, default=0.1
            Sampling fraction
        knn_neighbors : int, default=5
            K-NN neighbors
        use_decomposed : bool, default=True
            Use decomposed data if available
        cluster_name : str, optional
            Name for clustering result
        data_selector_name : str, optional
            Data selector to apply
        force : bool, default=False
            Force recalculation
        override_cache : bool, default=False
            Override cache settings

        Returns
        -------
        None
        """
        return self._execute(
            selection_name, Z, metric, affinity, density_algo, k_max, D_thr,
            dim_algo, blockAn, block_ratio, frac, halos, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="centroid"
        )

    def with_mean_centers(
        self,
        selection_name: str,
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
        knn_neighbors: int = 5,
        use_decomposed: bool = True,
        cluster_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
        override_cache: bool = False,
    ) -> None:
        """
        Add DPA with mean centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        Z : float, default=1.0
            Z-score threshold
        metric : str, default="euclidean"
            Distance metric
        affinity : str, default="precomputed"
            Affinity type
        density_algo : str, default="PAk"
            Density algorithm
        k_max : int, default=1000
            Maximum k for density
        D_thr : float, default=23.92812698
            Density threshold
        dim_algo : str, default="twoNN"
            Dimensionality algorithm
        blockAn : bool, default=True
            Block analysis
        block_ratio : int, default=20
            Block ratio
        frac : float, default=1.0
            Fraction of data
        halos : bool, default=False
            Include halos
        method : str, default="standard"
            Clustering method
        sample_fraction : float, default=0.1
            Sampling fraction
        knn_neighbors : int, default=5
            K-NN neighbors
        use_decomposed : bool, default=True
            Use decomposed data if available
        cluster_name : str, optional
            Name for clustering result
        data_selector_name : str, optional
            Data selector to apply
        force : bool, default=False
            Force recalculation
        override_cache : bool, default=False
            Override cache settings

        Returns
        -------
        None
        """
        return self._execute(
            selection_name, Z, metric, affinity, density_algo, k_max, D_thr,
            dim_algo, blockAn, block_ratio, frac, halos, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="mean"
        )

    def with_median_centers(
        self,
        selection_name: str,
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
        knn_neighbors: int = 5,
        use_decomposed: bool = True,
        cluster_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
        override_cache: bool = False,
    ) -> None:
        """
        Add DPA with median centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        Z : float, default=1.0
            Z-score threshold
        metric : str, default="euclidean"
            Distance metric
        affinity : str, default="precomputed"
            Affinity type
        density_algo : str, default="PAk"
            Density algorithm
        k_max : int, default=1000
            Maximum k for density
        D_thr : float, default=23.92812698
            Density threshold
        dim_algo : str, default="twoNN"
            Dimensionality algorithm
        blockAn : bool, default=True
            Block analysis
        block_ratio : int, default=20
            Block ratio
        frac : float, default=1.0
            Fraction of data
        halos : bool, default=False
            Include halos
        method : str, default="standard"
            Clustering method
        sample_fraction : float, default=0.1
            Sampling fraction
        knn_neighbors : int, default=5
            K-NN neighbors
        use_decomposed : bool, default=True
            Use decomposed data if available
        cluster_name : str, optional
            Name for clustering result
        data_selector_name : str, optional
            Data selector to apply
        force : bool, default=False
            Force recalculation
        override_cache : bool, default=False
            Override cache settings

        Returns
        -------
        None
        """
        return self._execute(
            selection_name, Z, metric, affinity, density_algo, k_max, D_thr,
            dim_algo, blockAn, block_ratio, frac, halos, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="median"
        )

    def with_density_peak_centers(
        self,
        selection_name: str,
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
        knn_neighbors: int = 5,
        use_decomposed: bool = True,
        cluster_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
        override_cache: bool = False,
    ) -> None:
        """
        Add DPA with density peak centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        Z : float, default=1.0
            Z-score threshold
        metric : str, default="euclidean"
            Distance metric
        affinity : str, default="precomputed"
            Affinity type
        density_algo : str, default="PAk"
            Density algorithm
        k_max : int, default=1000
            Maximum k for density
        D_thr : float, default=23.92812698
            Density threshold
        dim_algo : str, default="twoNN"
            Dimensionality algorithm
        blockAn : bool, default=True
            Block analysis
        block_ratio : int, default=20
            Block ratio
        frac : float, default=1.0
            Fraction of data
        halos : bool, default=False
            Include halos
        method : str, default="standard"
            Clustering method
        sample_fraction : float, default=0.1
            Sampling fraction
        knn_neighbors : int, default=5
            K-NN neighbors
        use_decomposed : bool, default=True
            Use decomposed data if available
        cluster_name : str, optional
            Name for clustering result
        data_selector_name : str, optional
            Data selector to apply
        force : bool, default=False
            Force recalculation
        override_cache : bool, default=False
            Override cache settings

        Returns
        -------
        None
        """
        return self._execute(
            selection_name, Z, metric, affinity, density_algo, k_max, D_thr,
            dim_algo, blockAn, block_ratio, frac, halos, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="density_peak"
        )

    def with_median_centroid_centers(
        self,
        selection_name: str,
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
        knn_neighbors: int = 5,
        use_decomposed: bool = True,
        cluster_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
        override_cache: bool = False,
    ) -> None:
        """
        Add DPA with median centroid centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        Z : float, default=1.0
            Z-score threshold
        metric : str, default="euclidean"
            Distance metric
        affinity : str, default="precomputed"
            Affinity type
        density_algo : str, default="PAk"
            Density algorithm
        k_max : int, default=1000
            Maximum k for density
        D_thr : float, default=23.92812698
            Density threshold
        dim_algo : str, default="twoNN"
            Dimensionality algorithm
        blockAn : bool, default=True
            Block analysis
        block_ratio : int, default=20
            Block ratio
        frac : float, default=1.0
            Fraction of data
        halos : bool, default=False
            Include halos
        method : str, default="standard"
            Clustering method
        sample_fraction : float, default=0.1
            Sampling fraction
        knn_neighbors : int, default=5
            K-NN neighbors
        use_decomposed : bool, default=True
            Use decomposed data if available
        cluster_name : str, optional
            Name for clustering result
        data_selector_name : str, optional
            Data selector to apply
        force : bool, default=False
            Force recalculation
        override_cache : bool, default=False
            Override cache settings

        Returns
        -------
        None
        """
        return self._execute(
            selection_name, Z, metric, affinity, density_algo, k_max, D_thr,
            dim_algo, blockAn, block_ratio, frac, halos, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="median_centroid"
        )

    def with_rmsd_centroid_centers(
        self,
        selection_name: str,
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
        knn_neighbors: int = 5,
        use_decomposed: bool = True,
        cluster_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
        override_cache: bool = False,
    ) -> None:
        """
        Add DPA with RMSD centroid centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        Z : float, default=1.0
            Z-score threshold
        metric : str, default="euclidean"
            Distance metric
        affinity : str, default="precomputed"
            Affinity type
        density_algo : str, default="PAk"
            Density algorithm
        k_max : int, default=1000
            Maximum k for density
        D_thr : float, default=23.92812698
            Density threshold
        dim_algo : str, default="twoNN"
            Dimensionality algorithm
        blockAn : bool, default=True
            Block analysis
        block_ratio : int, default=20
            Block ratio
        frac : float, default=1.0
            Fraction of data
        halos : bool, default=False
            Include halos
        method : str, default="standard"
            Clustering method
        sample_fraction : float, default=0.1
            Sampling fraction
        knn_neighbors : int, default=5
            K-NN neighbors
        use_decomposed : bool, default=True
            Use decomposed data if available
        cluster_name : str, optional
            Name for clustering result
        data_selector_name : str, optional
            Data selector to apply
        force : bool, default=False
            Force recalculation
        override_cache : bool, default=False
            Override cache settings

        Returns
        -------
        None
        """
        return self._execute(
            selection_name, Z, metric, affinity, density_algo, k_max, D_thr,
            dim_algo, blockAn, block_ratio, frac, halos, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="rmsd_centroid"
        )

    def _execute(
        self,
        selection_name: str,
        Z: float,
        metric: str,
        affinity: str,
        density_algo: str,
        k_max: int,
        D_thr: float,
        dim_algo: str,
        blockAn: bool,
        block_ratio: int,
        frac: float,
        halos: bool,
        method: str,
        sample_fraction: float,
        knn_neighbors: int,
        use_decomposed: bool,
        cluster_name: Optional[str],
        data_selector_name: Optional[str],
        force: bool,
        override_cache: bool,
        center_method: str,
    ) -> None:
        """
        Execute DPA clustering with specified parameters.

        Parameters
        ----------
        selection_name : str
            Name of feature selection
        Z : float
            Z-score threshold
        metric : str
            Distance metric
        affinity : str
            Affinity type
        density_algo : str
            Density algorithm
        k_max : int
            Maximum k
        D_thr : float
            Density threshold
        dim_algo : str
            Dimensionality algorithm
        blockAn : bool
            Block analysis
        block_ratio : int
            Block ratio
        frac : float
            Data fraction
        halos : bool
            Include halos
        method : str
            Clustering method
        sample_fraction : float
            Sampling fraction
        knn_neighbors : int
            K-NN neighbors
        use_decomposed : bool
            Use decomposed data
        cluster_name : str, optional
            Cluster name
        data_selector_name : str, optional
            Data selector name
        force : bool
            Force recalculation
        override_cache : bool
            Override cache
        center_method : str
            Center calculation method

        Returns
        -------
        None
        """
        cluster_type = DPA(
            Z=Z,
            metric=metric,
            affinity=affinity,
            density_algo=density_algo,
            k_max=k_max,
            D_thr=D_thr,
            dim_algo=dim_algo,
            blockAn=blockAn,
            block_ratio=block_ratio,
            frac=frac,
            halos=halos,
            method=method,
            sample_fraction=sample_fraction,
            knn_neighbors=knn_neighbors,
            force=force,
        )
        return self._manager.add_clustering(
            self._pipeline_data,
            selection_name,
            cluster_type,
            use_decomposed=use_decomposed,
            cluster_name=cluster_name,
            data_selector_name=data_selector_name,
            force=force,
            override_cache=override_cache,
            center_method=center_method,
        )
