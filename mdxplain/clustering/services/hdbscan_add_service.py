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

"""Service for adding HDBSCAN clustering with flexible center method selection."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager.cluster_manager import ClusterManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..cluster_type import HDBSCAN


class HDBSCANAddService:
    """
    Service for adding HDBSCAN clustering.

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
    >>> pipeline.clustering.add.hdbscan("features", min_cluster_size=10)

    >>> # Explicit center method selection
    >>> pipeline.clustering.add.hdbscan.with_median_centers("features", min_cluster_size=10)
    >>> pipeline.clustering.add.hdbscan.with_density_peak_centers("features", min_cluster_size=10)
    """

    def __init__(self, manager: ClusterManager, pipeline_data: PipelineData) -> None:
        """
        Initialize HDBSCAN service.

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
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
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
        Add HDBSCAN clustering algorithm.

        HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
        with Noise) extends DBSCAN by converting it into a hierarchical clustering
        algorithm and extracting clusters based on the stability of clusters.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        min_cluster_size : int, default=5
            Minimum size of clusters
        min_samples : int, optional
            Number of samples in a neighborhood for a point to be considered
            a core point. Defaults to min_cluster_size
        cluster_selection_epsilon : float, default=0.0
            Distance threshold for cluster extraction
        cluster_selection_method : str, default="eom"
            Method for cluster selection ('eom' or 'leaf')
        method : str, default="standard"
            Clustering method: 'standard', 'approximate_predict', or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction of data to sample for sampling-based methods
        knn_neighbors : int, default=5
            Number of neighbors for k-NN classifier in knn_sampling method
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
            Adds HDBSCAN clustering results to pipeline data

        Examples
        --------
        >>> # Basic HDBSCAN clustering
        >>> pipeline.clustering.add.hdbscan("my_features", min_cluster_size=10)

        >>> # HDBSCAN with custom parameters
        >>> pipeline.clustering.add.hdbscan(
        ...     "pca_features",
        ...     min_cluster_size=15,
        ...     min_samples=20,
        ...     cluster_selection_epsilon=0.5,
        ...     cluster_name="stable_clusters"
        ... )

        >>> # HDBSCAN with data selector
        >>> pipeline.clustering.add.hdbscan(
        ...     "torsion_features",
        ...     min_cluster_size=8,
        ...     data_selector_name="active_conformations"
        ... )

        Notes
        -----
        **Center Method:** Uses centroid (medoid) by default. The centroid is the
        actual data point closest to the cluster mean, ensuring the cluster center
        represents a real conformational state. Use .with_xxx_centers() for alternatives.

        HDBSCAN is more stable than DBSCAN across different parameter settings
        and provides a hierarchy of clusters. It's particularly good at finding
        clusters of varying densities.
        """
        return self._execute(
            selection_name, min_cluster_size, min_samples,
            cluster_selection_epsilon, cluster_selection_method, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="centroid"
        )

    def with_centroid_centers(
        self,
        selection_name: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
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
        Add HDBSCAN with centroid (medoid) centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        min_cluster_size : int, default=5
            Minimum cluster size
        min_samples : int, optional
            Minimum samples for core point
        cluster_selection_epsilon : float, default=0.0
            Cluster selection threshold
        cluster_selection_method : str, default="eom"
            Cluster selection method
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
            selection_name, min_cluster_size, min_samples,
            cluster_selection_epsilon, cluster_selection_method, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="centroid"
        )

    def with_mean_centers(
        self,
        selection_name: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
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
        Add HDBSCAN with mean centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        min_cluster_size : int, default=5
            Minimum cluster size
        min_samples : int, optional
            Minimum samples for core point
        cluster_selection_epsilon : float, default=0.0
            Cluster selection threshold
        cluster_selection_method : str, default="eom"
            Cluster selection method
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
            selection_name, min_cluster_size, min_samples,
            cluster_selection_epsilon, cluster_selection_method, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="mean"
        )

    def with_median_centers(
        self,
        selection_name: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
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
        Add HDBSCAN with median centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        min_cluster_size : int, default=5
            Minimum cluster size
        min_samples : int, optional
            Minimum samples for core point
        cluster_selection_epsilon : float, default=0.0
            Cluster selection threshold
        cluster_selection_method : str, default="eom"
            Cluster selection method
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
            selection_name, min_cluster_size, min_samples,
            cluster_selection_epsilon, cluster_selection_method, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="median"
        )

    def with_density_peak_centers(
        self,
        selection_name: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
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
        Add HDBSCAN with density peak centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        min_cluster_size : int, default=5
            Minimum cluster size
        min_samples : int, optional
            Minimum samples for core point
        cluster_selection_epsilon : float, default=0.0
            Cluster selection threshold
        cluster_selection_method : str, default="eom"
            Cluster selection method
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
            selection_name, min_cluster_size, min_samples,
            cluster_selection_epsilon, cluster_selection_method, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="density_peak"
        )

    def with_median_centroid_centers(
        self,
        selection_name: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
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
        Add HDBSCAN with median centroid centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        min_cluster_size : int, default=5
            Minimum cluster size
        min_samples : int, optional
            Minimum samples for core point
        cluster_selection_epsilon : float, default=0.0
            Cluster selection threshold
        cluster_selection_method : str, default="eom"
            Cluster selection method
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
            selection_name, min_cluster_size, min_samples,
            cluster_selection_epsilon, cluster_selection_method, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="median_centroid"
        )

    def with_rmsd_centroid_centers(
        self,
        selection_name: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
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
        Add HDBSCAN with RMSD centroid centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        min_cluster_size : int, default=5
            Minimum cluster size
        min_samples : int, optional
            Minimum samples for core point
        cluster_selection_epsilon : float, default=0.0
            Cluster selection threshold
        cluster_selection_method : str, default="eom"
            Cluster selection method
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
            selection_name, min_cluster_size, min_samples,
            cluster_selection_epsilon, cluster_selection_method, method,
            sample_fraction, knn_neighbors, use_decomposed, cluster_name,
            data_selector_name, force, override_cache, center_method="rmsd_centroid"
        )

    def _execute(
        self,
        selection_name: str,
        min_cluster_size: int,
        min_samples: Optional[int],
        cluster_selection_epsilon: float,
        cluster_selection_method: str,
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
        Execute HDBSCAN clustering with specified parameters.

        Parameters
        ----------
        selection_name : str
            Name of feature selection
        min_cluster_size : int
            Minimum cluster size
        min_samples : int, optional
            Minimum samples
        cluster_selection_epsilon : float
            Cluster selection epsilon
        cluster_selection_method : str
            Cluster selection method
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
        cluster_type = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
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
