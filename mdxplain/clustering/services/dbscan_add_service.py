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

"""Service for adding DBSCAN clustering with flexible center method selection."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager.cluster_manager import ClusterManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..cluster_type import DBSCAN


class DBSCANAddService:
    """
    Service for adding DBSCAN clustering.

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
    >>> pipeline.clustering.add.dbscan("features", eps=0.5, min_samples=5)

    >>> # Explicit center method selection
    >>> pipeline.clustering.add.dbscan.with_median_centers("features", eps=0.5)
    >>> pipeline.clustering.add.dbscan.with_density_peak_centers("features", eps=0.5)
    """

    def __init__(self, manager: ClusterManager, pipeline_data: PipelineData) -> None:
        """
        Initialize DBSCAN service.

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
        eps: float = 0.5,
        min_samples: int = 5,
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
        Add DBSCAN clustering algorithm.

        DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        groups together points that are closely packed while marking points
        in low-density regions as outliers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        eps : float, default=0.5
            Maximum distance between two samples for one to be considered
            in the neighborhood of the other
        min_samples : int, default=5
            Minimum number of samples in a neighborhood for a point to be
            considered as a core point
        method : str, default="standard"
            Clustering method: 'standard' or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction of data to sample for 'knn_sampling' method
        knn_neighbors : int, default=5
            Number of neighbors for k-NN sampling method
        use_decomposed : bool, default=True
            Whether to use decomposed data if available, otherwise use raw features
        cluster_name : str, optional
            Name for the clustering result. If None, uses algorithm-based name
        data_selector_name : str, optional
            Name of data selector to apply before clustering
        force : bool, default=False
            Force recalculation or override automatic method selection
        override_cache : bool, default=False
            Override cache settings for this clustering

        Returns
        -------
        None
            Adds DBSCAN clustering results to pipeline data

        Examples
        --------
        >>> # Basic DBSCAN clustering
        >>> pipeline.clustering.add.dbscan("my_features", eps=0.5, min_samples=5)

        >>> # DBSCAN with custom parameters
        >>> pipeline.clustering.add.dbscan(
        ...     "distance_features",
        ...     eps=1.0,
        ...     min_samples=10,
        ...     cluster_name="loose_clustering"
        ... )

        >>> # DBSCAN on raw features instead of decomposed
        >>> pipeline.clustering.add.dbscan(
        ...     "contact_features",
        ...     eps=0.3,
        ...     use_decomposed=False,
        ...     data_selector_name="folded_conformations"
        ... )

        Notes
        -----
        **Center Method:** Uses centroid (medoid) by default. The centroid is the
        actual data point closest to the cluster mean, ensuring the cluster center
        represents a real conformational state. Use .with_xxx_centers() for alternatives.

        DBSCAN automatically determines the number of clusters and can find
        clusters of arbitrary shape. Points that don't belong to any cluster
        are labeled as noise (cluster label -1).
        """
        return self._execute(
            selection_name, eps, min_samples, method, sample_fraction,
            knn_neighbors, use_decomposed, cluster_name, data_selector_name,
            force, override_cache, center_method="centroid"
        )

    def with_centroid_centers(
        self,
        selection_name: str,
        eps: float = 0.5,
        min_samples: int = 5,
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
        Add DBSCAN with centroid (medoid) centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        eps : float, default=0.5
            Maximum distance between samples in neighborhood
        min_samples : int, default=5
            Minimum samples for core point
        method : str, default="standard"
            Clustering method: 'standard' or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction for knn_sampling method
        knn_neighbors : int, default=5
            Neighbors for k-NN sampling
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
            selection_name, eps, min_samples, method, sample_fraction,
            knn_neighbors, use_decomposed, cluster_name, data_selector_name,
            force, override_cache, center_method="centroid"
        )

    def with_mean_centers(
        self,
        selection_name: str,
        eps: float = 0.5,
        min_samples: int = 5,
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
        Add DBSCAN with mean centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        eps : float, default=0.5
            Maximum distance between samples in neighborhood
        min_samples : int, default=5
            Minimum samples for core point
        method : str, default="standard"
            Clustering method: 'standard' or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction for knn_sampling method
        knn_neighbors : int, default=5
            Neighbors for k-NN sampling
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
            selection_name, eps, min_samples, method, sample_fraction,
            knn_neighbors, use_decomposed, cluster_name, data_selector_name,
            force, override_cache, center_method="mean"
        )

    def with_median_centers(
        self,
        selection_name: str,
        eps: float = 0.5,
        min_samples: int = 5,
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
        Add DBSCAN with median centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        eps : float, default=0.5
            Maximum distance between samples in neighborhood
        min_samples : int, default=5
            Minimum samples for core point
        method : str, default="standard"
            Clustering method: 'standard' or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction for knn_sampling method
        knn_neighbors : int, default=5
            Neighbors for k-NN sampling
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
            selection_name, eps, min_samples, method, sample_fraction,
            knn_neighbors, use_decomposed, cluster_name, data_selector_name,
            force, override_cache, center_method="median"
        )

    def with_density_peak_centers(
        self,
        selection_name: str,
        eps: float = 0.5,
        min_samples: int = 5,
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
        Add DBSCAN with density peak centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        eps : float, default=0.5
            Maximum distance between samples in neighborhood
        min_samples : int, default=5
            Minimum samples for core point
        method : str, default="standard"
            Clustering method: 'standard' or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction for knn_sampling method
        knn_neighbors : int, default=5
            Neighbors for k-NN sampling
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
            selection_name, eps, min_samples, method, sample_fraction,
            knn_neighbors, use_decomposed, cluster_name, data_selector_name,
            force, override_cache, center_method="density_peak"
        )

    def with_median_centroid_centers(
        self,
        selection_name: str,
        eps: float = 0.5,
        min_samples: int = 5,
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
        Add DBSCAN with median centroid centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        eps : float, default=0.5
            Maximum distance between samples in neighborhood
        min_samples : int, default=5
            Minimum samples for core point
        method : str, default="standard"
            Clustering method: 'standard' or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction for knn_sampling method
        knn_neighbors : int, default=5
            Neighbors for k-NN sampling
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
            selection_name, eps, min_samples, method, sample_fraction,
            knn_neighbors, use_decomposed, cluster_name, data_selector_name,
            force, override_cache, center_method="median_centroid"
        )

    def with_rmsd_centroid_centers(
        self,
        selection_name: str,
        eps: float = 0.5,
        min_samples: int = 5,
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
        Add DBSCAN with RMSD centroid centers.

        Parameters
        ----------
        selection_name : str
            Name of feature selection to cluster
        eps : float, default=0.5
            Maximum distance between samples in neighborhood
        min_samples : int, default=5
            Minimum samples for core point
        method : str, default="standard"
            Clustering method: 'standard' or 'knn_sampling'
        sample_fraction : float, default=0.1
            Fraction for knn_sampling method
        knn_neighbors : int, default=5
            Neighbors for k-NN sampling
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
            selection_name, eps, min_samples, method, sample_fraction,
            knn_neighbors, use_decomposed, cluster_name, data_selector_name,
            force, override_cache, center_method="rmsd_centroid"
        )

    def _execute(
        self,
        selection_name: str,
        eps: float,
        min_samples: int,
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
        Execute DBSCAN clustering with specified parameters.

        Parameters
        ----------
        selection_name : str
            Name of feature selection
        eps : float
            DBSCAN epsilon parameter
        min_samples : int
            DBSCAN min_samples parameter
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
        cluster_type = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            method=method,
            sample_fraction=sample_fraction,
            knn_neighbors=knn_neighbors,
            force=force
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
