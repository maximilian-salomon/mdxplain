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
Helper functions for calculating cluster centers.

Provides cluster center calculation from data and labels using various
methods (centroid, mean, median, density_peak).
"""

from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


class CenterCalculationHelper:
    """
    Helper for calculating cluster centers.

    Calculates cluster centers from data and labels using various methods.
    Automatically handles memory-efficient chunk-wise processing for large datasets.

    Available calculation methods:
    - centroid: Representative point (medoid - closest to mean)
    - mean: Average of all cluster members
    - median: Coordinate-wise median (robust to outliers)
    - density_peak: Point with highest local density
    - median_centroid: Medoid from median (robust centroid)
    - rmsd_centroid: Centroid using RMSD metric (structural)
    """

    VALID_METHODS = ["centroid", "mean", "median", "density_peak", "median_centroid", "rmsd_centroid"]

    @staticmethod
    def calculate_centers(
        data: np.ndarray,
        labels: np.ndarray,
        center_method: str = "centroid",
        chunk_size: int = 1000,
        use_memmap: bool = False,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate cluster centers from data and labels.

        Parameters
        ----------
        data : numpy.ndarray
            Original data used for clustering, shape (n_samples, n_features)
        labels : numpy.ndarray
            Cluster labels, shape (n_samples,)
        center_method : str, default="centroid"
            Method for calculating centers:
            - "centroid": Representative point (medoid - closest to mean)
            - "mean": Average of all points
            - "median": Coordinate-wise median (robust to outliers)
            - "density_peak": Point with highest local density
            - "median_centroid": Medoid from median (more robust than centroid)
            - "rmsd_centroid": Centroid using RMSD metric (structural comparisons)
        chunk_size : int, default=1000
            Chunk size for memory-safe processing
        use_memmap : bool, default=False
            Whether to use chunk-wise processing (memory-safe for large data)
        max_memory_gb : float, default=2.0
            Memory threshold for density_peak sampling (other methods ignore this)

        Returns
        -------
        Optional[numpy.ndarray]
            Array of cluster centers (n_clusters, n_features) or None

        Raises
        ------
        ValueError
            If center_method is not valid

        Notes
        -----
        Memory-safe processing:
        - centroid, mean, median, median_centroid, rmsd_centroid: Always memory-safe
        - density_peak: Uses sampling if cluster exceeds max_memory_gb

        Examples
        --------
        >>> # Standard usage
        >>> centers = calculate_centers(data, labels, "centroid")

        >>> # Memory-safe processing for large data
        >>> centers = calculate_centers(data, labels, "centroid",
        ...                             chunk_size=1000, use_memmap=True, max_memory_gb=2.0)
        """
        dispatch = {
            "centroid": CenterCalculationHelper._calculate_centroids,
            "mean": CenterCalculationHelper._calculate_means,
            "median": CenterCalculationHelper._calculate_medians,
            "density_peak": CenterCalculationHelper._calculate_density_peaks,
            "median_centroid": CenterCalculationHelper._calculate_median_centroids,
            "rmsd_centroid": CenterCalculationHelper._calculate_rmsd_centroids,
        }

        if center_method not in dispatch:
            raise ValueError(
                f"Invalid center_method: '{center_method}'. "
                f"Valid options: {list(dispatch.keys())}"
            )

        return dispatch[center_method](data, labels, chunk_size, use_memmap, max_memory_gb)

    @staticmethod
    def _calculate_centroids(
        data: np.ndarray,
        labels: np.ndarray,
        chunk_size: int,
        use_memmap: bool,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate centroids (medoids) for all clusters.

        Parameters
        ----------
        data : numpy.ndarray
            Original data
        labels : numpy.ndarray
            Cluster labels
        chunk_size : int
            Chunk size for memory-safe processing
        use_memmap : bool
            Whether to use chunk-wise processing
        max_memory_gb : float, default=2.0
            Not used by this method (kept for API consistency)

        Returns
        -------
        Optional[numpy.ndarray]
            Centroid for each cluster, shape (n_clusters, n_features)

        Notes
        -----
        Single pass over data processes all clusters simultaneously.
        max_memory_gb parameter is not used by this method (kept for API consistency).
        """
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return None

        n_clusters = len(unique_labels)
        n_features = data.shape[1]

        if not use_memmap:
            return CenterCalculationHelper._calculate_centroids_generic_fast(
                data, labels, unique_labels, metric='euclidean'
            )

        means = CenterCalculationHelper._compute_means_chunked(
            data, labels, unique_labels, chunk_size, n_clusters, n_features
        )
        centers = CenterCalculationHelper._find_centroids_generic_chunked(
            data, labels, unique_labels, chunk_size, means, n_clusters, n_features, metric='euclidean'
        )
        return centers

    @staticmethod
    def _calculate_centroids_generic_fast(data, labels, unique_labels, metric='euclidean'):
        """
        Calculate centroids for all clusters without memmap using specified metric.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset
        labels : numpy.ndarray
            Cluster labels
        unique_labels : numpy.ndarray
            Unique cluster labels
        metric : str, default='euclidean'
            Distance metric ('euclidean' or 'rmsd')

        Returns
        -------
        numpy.ndarray
            Centroids for all clusters
        """
        n_clusters = len(unique_labels)
        n_features = data.shape[1]
        centers = np.zeros((n_clusters, n_features), dtype=data.dtype)

        for cluster_idx, label in enumerate(unique_labels):
            mask = labels == label
            cluster_data = data[mask]
            mean = cluster_data.mean(axis=0)

            if metric == 'rmsd':
                dists = np.sqrt(np.mean((cluster_data - mean)**2, axis=1))
            else:
                dists = np.linalg.norm(cluster_data - mean, axis=1)

            centers[cluster_idx] = cluster_data[np.argmin(dists)]

        return centers

    @staticmethod
    def _compute_means_chunked(data, labels, unique_labels, chunk_size, n_clusters, n_features):
        """
        Compute means for all clusters in chunked fashion.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset
        labels : numpy.ndarray
            Cluster labels
        unique_labels : numpy.ndarray
            Unique cluster labels
        chunk_size : int
            Chunk size
        n_clusters : int
            Number of clusters
        n_features : int
            Number of features

        Returns
        -------
        numpy.ndarray
            Mean vectors for all clusters
        """
        means = np.zeros((n_clusters, n_features), dtype=data.dtype)
        counts = np.zeros(n_clusters, dtype=np.int64)

        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            chunk_data = data[start:end]
            chunk_labels = labels[start:end]

            for cluster_idx, label in enumerate(unique_labels):
                mask = chunk_labels == label
                if np.any(mask):
                    means[cluster_idx] += chunk_data[mask].sum(axis=0)
                    counts[cluster_idx] += np.sum(mask)

        return means / counts[:, np.newaxis]

    @staticmethod
    def _find_centroids_generic_chunked(
        data, labels, unique_labels, chunk_size, means, n_clusters, n_features, metric='euclidean'
    ):
        """
        Find centroids in chunked fashion using specified metric.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset
        labels : numpy.ndarray
            Cluster labels
        unique_labels : numpy.ndarray
            Unique cluster labels
        chunk_size : int
            Chunk size
        means : numpy.ndarray
            Mean vectors for each cluster
        n_clusters : int
            Number of clusters
        n_features : int
            Number of features
        metric : str, default='euclidean'
            Distance metric ('euclidean' or 'rmsd')

        Returns
        -------
        numpy.ndarray
            Centroids for all clusters
        """
        centers = np.zeros((n_clusters, n_features), dtype=data.dtype)
        best_dists = np.full(n_clusters, np.inf, dtype=data.dtype)

        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            CenterCalculationHelper._update_centroids_from_chunk(
                data[start:end], labels[start:end], unique_labels, means,
                centers, best_dists, metric
            )

        return centers

    @staticmethod
    def _update_centroids_from_chunk(
        chunk_data, chunk_labels, unique_labels, means, centers, best_dists, metric
    ):
        """
        Update centroids from a single chunk.

        Parameters
        ----------
        chunk_data : np.ndarray
            Data chunk
        chunk_labels : np.ndarray
            Labels for chunk
        unique_labels : np.ndarray
            Unique cluster labels
        means : np.ndarray
            Mean vectors for each cluster
        centers : np.ndarray
            Current best centroids (updated in-place)
        best_dists : np.ndarray
            Current best distances (updated in-place)
        metric : str
            Distance metric ('euclidean' or 'rmsd')

        Returns
        -------
        None
        """
        for cluster_idx, label in enumerate(unique_labels):
            mask = chunk_labels == label
            if np.any(mask):
                cluster_chunk = chunk_data[mask]

                if metric == 'rmsd':
                    dists = np.sqrt(np.mean((cluster_chunk - means[cluster_idx])**2, axis=1))
                else:
                    dists = np.linalg.norm(cluster_chunk - means[cluster_idx], axis=1)

                min_idx = np.argmin(dists)
                if dists[min_idx] < best_dists[cluster_idx]:
                    best_dists[cluster_idx] = dists[min_idx]
                    centers[cluster_idx] = cluster_chunk[min_idx]

    @staticmethod
    def _calculate_means(
        data: np.ndarray,
        labels: np.ndarray,
        chunk_size: int,
        use_memmap: bool,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate centers as mean of cluster members.

        Parameters
        ----------
        data : numpy.ndarray
            Original data
        labels : numpy.ndarray
            Cluster labels
        chunk_size : int
            Chunk size for memory-safe processing
        use_memmap : bool
            Whether to use chunk-wise processing
        max_memory_gb : float, default=2.0
            Not used by this method (kept for API consistency)

        Returns
        -------
        Optional[numpy.ndarray]
            Mean for each cluster, shape (n_clusters, n_features)

        Notes
        -----
        Single-pass over data, processes all clusters simultaneously.
        """
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return None

        n_clusters = len(unique_labels)
        n_features = data.shape[1]

        if not use_memmap:
            means = np.zeros((n_clusters, n_features), dtype=data.dtype)
            for cluster_idx, label in enumerate(unique_labels):
                mask = labels == label
                means[cluster_idx] = data[mask].mean(axis=0)
            return means

        return CenterCalculationHelper._compute_means_chunked(
            data, labels, unique_labels, chunk_size, n_clusters, n_features
        )

    @staticmethod
    def _calculate_medians(
        data: np.ndarray,
        labels: np.ndarray,
        chunk_size: int,
        use_memmap: bool,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate centers as coordinate-wise median.

        Parameters
        ----------
        data : numpy.ndarray
            Original data
        labels : numpy.ndarray
            Cluster labels
        chunk_size : int
            Chunk size for memory-safe processing
        use_memmap : bool
            Whether to use chunk-wise processing
        max_memory_gb : float, default=2.0
            Not used by this method (kept for API consistency)

        Returns
        -------
        Optional[numpy.ndarray]
            Median for each cluster, shape (n_clusters, n_features)

        Notes
        -----
        Processes one feature at a time for all clusters simultaneously.
        """
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return None

        n_clusters = len(unique_labels)
        n_features = data.shape[1]

        if not use_memmap:
            medians = np.zeros((n_clusters, n_features), dtype=data.dtype)
            for cluster_idx, label in enumerate(unique_labels):
                mask = labels == label
                medians[cluster_idx] = np.median(data[mask], axis=0)
            return medians

        return CenterCalculationHelper._compute_medians_chunked(
            data, labels, unique_labels, chunk_size, n_clusters, n_features
        )

    @staticmethod
    def _compute_medians_chunked(
        data: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray,
        chunk_size: int,
        n_clusters: int,
        n_features: int
    ) -> np.ndarray:
        """
        Compute medians for all clusters feature-by-feature.

        Parameters
        ----------
        data : np.ndarray
            Full dataset
        labels : np.ndarray
            Cluster labels
        unique_labels : np.ndarray
            Unique cluster labels
        chunk_size : int
            Chunk size for processing (unused, kept for API consistency)
        n_clusters : int
            Number of clusters
        n_features : int
            Number of features

        Returns
        -------
        np.ndarray
            Medians for all clusters, shape (n_clusters, n_features)
        """
        medians = np.zeros((n_clusters, n_features), dtype=data.dtype)

        for feat_idx in range(n_features):
            feature_column = data[:, feat_idx]

            for cluster_idx, label in enumerate(unique_labels):
                mask = labels == label
                cluster_values = feature_column[mask]
                if len(cluster_values) > 0:
                    medians[cluster_idx, feat_idx] = np.median(cluster_values)

        return medians

    @staticmethod
    def _calculate_density_peaks(
        data: np.ndarray,
        labels: np.ndarray,
        chunk_size: int,
        use_memmap: bool,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate density peaks with memory-aware sampling.

        Small clusters (< max_memory_gb): Exact k-NN on all points
        Large clusters: k-NN on 50k random sample

        Parameters
        ----------
        data : numpy.ndarray
            Original data
        labels : numpy.ndarray
            Cluster labels
        chunk_size : int
            Not used by this method (kept for API consistency)
        use_memmap : bool
            Not used by this method (kept for API consistency)
        max_memory_gb : float, default=2.0
            Memory threshold for full cluster loading

        Returns
        -------
        Optional[numpy.ndarray]
            Density peak for each cluster, shape (n_clusters, n_features)

        Notes
        -----
        Memory requirement check:
        cluster_size x n_features x dtype.itemsize vs. max_memory_gb

        Uses direct random sampling (np.random.choice) for large clusters.
        Good approximation for unimodal clusters (typical in MD).
        """
        unique_labels = np.unique(labels[labels >= 0])
        centers = []

        for label in unique_labels:
            cluster_points = CenterCalculationHelper._load_cluster_for_density(
                data, labels, label, max_memory_gb
            )
            center = CenterCalculationHelper._find_density_peak_from_cluster(cluster_points)
            centers.append(center)

        return np.array(centers) if centers else None

    @staticmethod
    def _load_cluster_for_density(data, labels, label, max_memory_gb):
        """
        Load cluster data for density calculation with memory-aware sampling.

        Parameters
        ----------
        data : np.ndarray
            Full dataset
        labels : np.ndarray
            Cluster labels
        label : int
            Cluster label to load
        max_memory_gb : float
            Memory threshold

        Returns
        -------
        np.ndarray
            Cluster points (all or sampled)
        """
        cluster_size = np.sum(labels == label)
        n_features = data.shape[1]
        bytes_needed = cluster_size * n_features * data.dtype.itemsize
        gb_needed = bytes_needed / (1024**3)

        if gb_needed <= max_memory_gb:
            mask = labels == label
            return data[mask]

        print(f"  Cluster {label}: {cluster_size:,} points ({gb_needed:.2f} GB)")
        print(f"  Exceeds {max_memory_gb:.1f} GB limit, using 50k sample")
        max_sample = min(50000, cluster_size)
        cluster_indices = np.where(labels == label)[0]
        sample_indices = np.random.choice(cluster_indices, max_sample, replace=False)
        return data[sample_indices]

    @staticmethod
    def _find_density_peak_from_cluster(cluster_points):
        """
        Find density peak in cluster using k-NN.

        Parameters
        ----------
        cluster_points : np.ndarray
            Cluster data points

        Returns
        -------
        np.ndarray
            Density peak point
        """
        if len(cluster_points) < 8:
            cluster_mean = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - cluster_mean, axis=1)
            return cluster_points[np.argmin(distances)]

        n_neighbors = min(8, len(cluster_points))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(cluster_points)
        distances, _ = nbrs.kneighbors(cluster_points)
        epsilon = np.median(distances[:, -1])

        nbrs_count = NearestNeighbors(radius=epsilon)
        nbrs_count.fit(cluster_points)
        neighbor_counts = [
            len(nbrs_count.radius_neighbors([point], return_distance=False)[0])
            for point in cluster_points
        ]

        peak_idx = np.argmax(neighbor_counts)
        return cluster_points[peak_idx]

    @staticmethod
    def _calculate_median_centroids(
        data: np.ndarray,
        labels: np.ndarray,
        chunk_size: int,
        use_memmap: bool,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate median centroids using two-pass algorithm.

        Pass 1: Compute median
        Pass 2: Find point closest to median

        More robust to outliers than regular centroid.

        Parameters
        ----------
        data : numpy.ndarray
            Original data
        labels : numpy.ndarray
            Cluster labels
        chunk_size : int
            Chunk size for memory-safe processing
        use_memmap : bool
            Whether to use chunk-wise processing
        max_memory_gb : float, default=2.0
            Not used by this method (kept for API consistency)

        Returns
        -------
        Optional[numpy.ndarray]
            Median centroid for each cluster, shape (n_clusters, n_features)

        Notes
        -----
        Two passes over data for all clusters simultaneously.

        Examples
        --------
        >>> centers = CenterCalculationHelper._calculate_median_centroids(
        ...     data, labels, chunk_size=1000, use_memmap=True
        ... )
        """
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return None

        n_clusters = len(unique_labels)
        n_features = data.shape[1]

        if not use_memmap:
            return CenterCalculationHelper._calculate_median_centroids_fast(
                data, labels, unique_labels
            )

        medians = CenterCalculationHelper._compute_medians_chunked(
            data, labels, unique_labels, chunk_size, n_clusters, n_features
        )
        centers = CenterCalculationHelper._find_centroids_generic_chunked(
            data, labels, unique_labels, chunk_size, medians, n_clusters, n_features, metric='euclidean'
        )
        return centers

    @staticmethod
    def _calculate_median_centroids_fast(
        data: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray
    ) -> np.ndarray:
        """
        Calculate median centroids for all clusters without memmap.

        Parameters
        ----------
        data : np.ndarray
            Full dataset
        labels : np.ndarray
            Cluster labels
        unique_labels : np.ndarray
            Unique cluster labels

        Returns
        -------
        np.ndarray
            Median centroids for all clusters, shape (n_clusters, n_features)
        """
        n_clusters = len(unique_labels)
        n_features = data.shape[1]
        centers = np.zeros((n_clusters, n_features), dtype=data.dtype)

        for cluster_idx, label in enumerate(unique_labels):
            mask = labels == label
            cluster_data = data[mask]
            median = np.median(cluster_data, axis=0)
            dists = np.linalg.norm(cluster_data - median, axis=1)
            centers[cluster_idx] = cluster_data[np.argmin(dists)]

        return centers

    @staticmethod
    def _calculate_rmsd_centroids(
        data: np.ndarray,
        labels: np.ndarray,
        chunk_size: int,
        use_memmap: bool,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate RMSD centroids using two-pass algorithm.

        Pass 1: Compute mean
        Pass 2: Find point with minimum RMSD to mean

        Uses RMSD (Root Mean Square Deviation) metric:
        RMSD = sqrt(mean((point - center)²))

        Parameters
        ----------
        data : numpy.ndarray
            Original data
        labels : numpy.ndarray
            Cluster labels
        chunk_size : int
            Chunk size for memory-safe processing
        use_memmap : bool
            Whether to use chunk-wise processing
        max_memory_gb : float, default=2.0
            Not used by this method (kept for API consistency)

        Returns
        -------
        Optional[numpy.ndarray]
            RMSD centroid for each cluster, shape (n_clusters, n_features)

        Notes
        -----
        Two passes over data for all clusters simultaneously.

        Examples
        --------
        >>> centers = CenterCalculationHelper._calculate_rmsd_centroids(
        ...     data, labels, chunk_size=1000, use_memmap=True
        ... )
        """
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return None

        n_clusters = len(unique_labels)
        n_features = data.shape[1]

        if not use_memmap:
            return CenterCalculationHelper._calculate_centroids_generic_fast(
                data, labels, unique_labels, metric='rmsd'
            )

        means = CenterCalculationHelper._compute_means_chunked(
            data, labels, unique_labels, chunk_size, n_clusters, n_features
        )
        centers = CenterCalculationHelper._find_centroids_generic_chunked(
            data, labels, unique_labels, chunk_size, means, n_clusters, n_features, metric='rmsd'
        )
        return centers
