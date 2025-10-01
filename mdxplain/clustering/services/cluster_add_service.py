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

"""Factory for adding clustering algorithms with simplified syntax."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.cluster_manager import ClusterManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..cluster_type import DBSCAN, HDBSCAN, DPA


class ClusterAddService:
    """
    Service for adding clustering algorithms without explicit type instantiation.
    
    This service provides an intuitive interface for adding clustering algorithms
    without requiring users to import and instantiate cluster types directly.
    All cluster type parameters are combined with manager.add parameters.
    
    Examples
    --------
    >>> pipeline.clustering.add.dbscan("my_features", eps=0.5, min_samples=5)
    >>> pipeline.clustering.add.hdbscan("my_features", min_cluster_size=10)
    >>> pipeline.clustering.add.dpa("my_features", Z=2.0)
    """
    
    def __init__(self, manager: ClusterManager, pipeline_data: PipelineData) -> None:
        """
        Initialize factory with manager and pipeline data.
        
        Parameters
        ----------
        manager : ClusterManager
            Cluster manager instance
        pipeline_data : PipelineData
            Pipeline data container (injected by AutoInjectProxy)
            
        Returns
        -------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    def dbscan(
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
            Fraction of data to sample for 'knn_sampling' method.
        knn_neighbors : int, default=5
            Number of neighbors for k-NN sampling method.
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
        DBSCAN automatically determines the number of clusters and can find
        clusters of arbitrary shape. Points that don't belong to any cluster
        are labeled as noise (cluster label -1).
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
        )
    
    def hdbscan(
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
        
        >>> # HDBSCAN with robust linkage
        >>> pipeline.clustering.add.hdbscan(
        ...     "torsion_features",
        ...     min_cluster_size=8,
        ...     alpha=0.5,
        ...     data_selector_name="active_conformations"
        ... )
        
        Notes
        -----
        HDBSCAN is more stable than DBSCAN across different parameter settings
        and provides a hierarchy of clusters. It's particularly good at finding
        clusters of varying densities.
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
        )
    
    def dpa(
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
        DPA is in practice easy to use for molecular dynamics data and
        does not require extensive parameter tuning. It can find clusters
        of varying shapes and densities without prior knowledge of the
        data distribution. Just vary the Z parameter to adjust the number
        of clusters.
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
        )
