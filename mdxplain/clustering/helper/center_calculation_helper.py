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
        >>> # Fast processing for small data
        >>> centers = calculate_centers(data, labels, "centroid")

        >>> # Memory-safe processing for large data
        >>> centers = calculate_centers(data, labels, "centroid",
        ...                             chunk_size=1000, use_memmap=True, max_memory_gb=2.0)
        """
        # Validate user method
        if center_method not in CenterCalculationHelper.VALID_METHODS:
            raise ValueError(
                f"Invalid center_method: '{center_method}'. "
                f"Valid options: {CenterCalculationHelper.VALID_METHODS}"
            )

        # Calculate centers based on method
        if center_method == "centroid":
            centers = CenterCalculationHelper._calculate_centroids(
                data, labels, chunk_size, use_memmap, max_memory_gb
            )
        elif center_method == "mean":
            centers = CenterCalculationHelper._calculate_means(
                data, labels, chunk_size, use_memmap, max_memory_gb
            )
        elif center_method == "median":
            centers = CenterCalculationHelper._calculate_medians(
                data, labels, chunk_size, use_memmap, max_memory_gb
            )
        elif center_method == "density_peak":
            centers = CenterCalculationHelper._calculate_density_peaks(
                data, labels, chunk_size, use_memmap, max_memory_gb
            )
        elif center_method == "median_centroid":
            centers = CenterCalculationHelper._calculate_median_centroids(
                data, labels, chunk_size, use_memmap, max_memory_gb
            )
        elif center_method == "rmsd_centroid":
            centers = CenterCalculationHelper._calculate_rmsd_centroids(
                data, labels, chunk_size, use_memmap, max_memory_gb
            )

        return centers

    @staticmethod
    def _iterate_cluster_data(
        data: np.ndarray,
        labels: np.ndarray,
        label: int,
        chunk_size: int,
        use_memmap: bool
    ):
        """
        Generator for cluster-specific data (chunk-wise or complete).

        Yields cluster data either all at once (fast path) or chunk by chunk
        (memory-safe path) depending on use_memmap flag.

        Parameters
        ----------
        data : np.ndarray
            Full dataset
        labels : np.ndarray
            Cluster labels
        label : int
            Current cluster label to extract
        chunk_size : int
            Chunk size for memmap processing
        use_memmap : bool
            Whether to use chunk-wise processing

        Yields
        ------
        np.ndarray
            Cluster data (complete or as chunk)

        Examples
        --------
        >>> # Process all data at once
        >>> for chunk in _iterate_cluster_data(data, labels, 0, 1000, False):
        ...     # chunk contains all points from cluster 0

        >>> # Process chunk by chunk
        >>> for chunk in _iterate_cluster_data(data, labels, 0, 1000, True):
        ...     # chunk contains up to 1000 points from cluster 0
        """
        if not use_memmap:
            # Fast path: All data at once
            mask = labels == label
            yield data[mask]
        else:
            # Memory-safe path: Chunk by chunk
            for start in range(0, len(data), chunk_size):
                end = min(start + chunk_size, len(data))
                chunk_mask = labels[start:end] == label
                if np.any(chunk_mask):
                    yield data[start:end][chunk_mask]

    @staticmethod
    def _compute_mean_for_label(
        data: np.ndarray,
        labels: np.ndarray,
        label: int,
        chunk_size: int,
        use_memmap: bool
    ) -> np.ndarray:
        """
        Helper to compute mean for a single cluster label.

        Parameters
        ----------
        data : np.ndarray
            Full dataset
        labels : np.ndarray
            Cluster labels
        label : int
            Cluster label to compute mean for
        chunk_size : int
            Chunk size for processing
        use_memmap : bool
            Whether to use chunk-wise processing

        Returns
        -------
        np.ndarray
            Mean vector for the cluster
        """
        sum_vec = np.zeros(data.shape[1])
        count = 0

        for chunk in CenterCalculationHelper._iterate_cluster_data(
            data, labels, label, chunk_size, use_memmap
        ):
            sum_vec += np.sum(chunk, axis=0)
            count += len(chunk)

        return sum_vec / count if count > 0 else sum_vec

    @staticmethod
    def _compute_median_for_label(
        data: np.ndarray,
        labels: np.ndarray,
        label: int,
        chunk_size: int,
        use_memmap: bool
    ) -> np.ndarray:
        """
        Compute median using feature-by-feature streaming.

        Processes one feature at a time to avoid loading entire cluster.
        Always memory-safe, always exact.

        Parameters
        ----------
        data : np.ndarray
            Full dataset
        labels : np.ndarray
            Cluster labels
        label : int
            Cluster label to compute median for
        chunk_size : int
            Chunk size for processing
        use_memmap : bool
            Whether to use chunk-wise processing

        Returns
        -------
        np.ndarray
            Exact median vector for the cluster

        Notes
        -----
        Memory usage: Only cluster_size x dtype.itemsize per feature
        Example: 1M points, float32 → 4 MB per feature (always OK)
        This is exact, not an approximation!
        """
        n_features = data.shape[1]
        median = np.zeros(n_features, dtype=data.dtype)

        # Process each feature independently
        for feat_idx in range(n_features):
            feature_values = []

            # Collect ALL values for this ONE feature
            for chunk in CenterCalculationHelper._iterate_cluster_data(
                data, labels, label, chunk_size, use_memmap
            ):
                feature_values.extend(chunk[:, feat_idx].tolist())

            # Compute exact median for this feature
            median[feat_idx] = np.median(feature_values)

        return median

    @staticmethod
    def _calculate_centroids(
        data: np.ndarray,
        labels: np.ndarray,
        chunk_size: int,
        use_memmap: bool,
        max_memory_gb: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Calculate centroids (medoids) using two-pass algorithm.

        Pass 1: Compute mean
        Pass 2: Find point closest to mean

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
        Two-pass iteration required for medoid calculation.
        Memory-safe but requires 2x I/O when using memmap.
        """
        unique_labels = np.unique(labels[labels >= 0])
        centers = []

        for label in unique_labels:
            # Pass 1: Calculate mean using helper
            mean = CenterCalculationHelper._compute_mean_for_label(
                data, labels, label, chunk_size, use_memmap
            )

            # Pass 2: Find closest real point to mean
            best_dist = float('inf')
            best_point = None

            for chunk in CenterCalculationHelper._iterate_cluster_data(
                data, labels, label, chunk_size, use_memmap
            ):
                dists = np.linalg.norm(chunk - mean, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_dist:
                    best_dist = dists[min_idx]
                    best_point = chunk[min_idx].copy()

            centers.append(best_point)

        return np.array(centers) if centers else None

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

        Single-pass algorithm with chunk-wise accumulation.
        Mean may not be an actual data point.

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
        Single-pass iteration, always memory-safe.
        """
        unique_labels = np.unique(labels[labels >= 0])
        centers = []

        for label in unique_labels:
            mean = CenterCalculationHelper._compute_mean_for_label(
                data, labels, label, chunk_size, use_memmap
            )
            centers.append(mean)

        return np.array(centers) if centers else None

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

        Uses feature-by-feature streaming for exact median computation.
        More robust to outliers than mean.

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
        Processes one feature at a time, always memory-safe and exact.
        """
        unique_labels = np.unique(labels[labels >= 0])
        centers = []

        for label in unique_labels:
            median = CenterCalculationHelper._compute_median_for_label(
                data, labels, label, chunk_size, use_memmap
            )
            centers.append(median)

        return np.array(centers) if centers else None

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
            # Check memory requirement
            cluster_size = np.sum(labels == label)
            n_features = data.shape[1]
            dtype_size = data.dtype.itemsize

            bytes_needed = cluster_size * n_features * dtype_size
            gb_needed = bytes_needed / (1024**3)

            if gb_needed <= max_memory_gb:
                # Small cluster: Load all
                mask = labels == label
                cluster_points = data[mask]
            else:
                # Large cluster: Random sample 50k points
                print(f"  Cluster {label}: {cluster_size:,} points ({gb_needed:.2f} GB)")
                print(f"  Exceeds {max_memory_gb:.1f} GB limit, using 50k sample")

                max_sample = min(50000, cluster_size)

                # Get cluster indices
                cluster_indices = np.where(labels == label)[0]

                # Sample random indices
                sample_indices = np.random.choice(cluster_indices, max_sample, replace=False)

                # Load only sampled points
                cluster_points = data[sample_indices]

            # k-NN density calculation (same for both paths)
            if len(cluster_points) < 8:
                # Fallback for tiny clusters
                cluster_mean = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - cluster_mean, axis=1)
                centers.append(cluster_points[np.argmin(distances)])
                continue

            # Estimate epsilon (median distance to 7th neighbor)
            n_neighbors = min(8, len(cluster_points))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(cluster_points)
            distances, _ = nbrs.kneighbors(cluster_points)
            epsilon = np.median(distances[:, -1])

            # Count neighbors for each point
            nbrs_count = NearestNeighbors(radius=epsilon)
            nbrs_count.fit(cluster_points)
            neighbor_counts = [
                len(nbrs_count.radius_neighbors([point], return_distance=False)[0])
                for point in cluster_points
            ]

            # Point with most neighbors is density peak
            peak_idx = np.argmax(neighbor_counts)
            centers.append(cluster_points[peak_idx])

        return np.array(centers) if centers else None

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

        Pass 1: Compute median (feature-by-feature, always exact)
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
        Median computation is feature-by-feature (memory-safe).
        Finding closest point requires second iteration.

        Examples
        --------
        >>> # More robust to outliers than centroid
        >>> centers = CenterCalculationHelper._calculate_median_centroids(
        ...     data, labels, chunk_size=1000, use_memmap=True
        ... )
        """
        unique_labels = np.unique(labels[labels >= 0])
        centers = []

        for label in unique_labels:
            # Pass 1: Calculate median using helper
            median = CenterCalculationHelper._compute_median_for_label(
                data, labels, label, chunk_size, use_memmap
            )

            # Pass 2: Find closest real point to median
            best_dist = float('inf')
            best_point = None

            for chunk in CenterCalculationHelper._iterate_cluster_data(
                data, labels, label, chunk_size, use_memmap
            ):
                dists = np.linalg.norm(chunk - median, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_dist:
                    best_dist = dists[min_idx]
                    best_point = chunk[min_idx].copy()

            centers.append(best_point)

        return np.array(centers) if centers else None

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

        Uses RMSD (Root Mean Square Deviation) metric instead of Euclidean distance:
        RMSD = sqrt(mean((point - center)²))

        More appropriate for structural comparisons in molecular dynamics.

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
        Two-pass iteration required (mean, then best RMSD).
        RMSD metric is particularly useful for molecular dynamics trajectories
        where structural similarity is more important than Euclidean distance.

        Examples
        --------
        >>> # Better for structural comparisons
        >>> centers = CenterCalculationHelper._calculate_rmsd_centroids(
        ...     data, labels, chunk_size=1000, use_memmap=True
        ... )
        """
        unique_labels = np.unique(labels[labels >= 0])
        centers = []

        for label in unique_labels:
            # Pass 1: Calculate mean using helper
            mean = CenterCalculationHelper._compute_mean_for_label(
                data, labels, label, chunk_size, use_memmap
            )

            # Pass 2: Find point with smallest RMSD to mean
            best_rmsd = float('inf')
            best_point = None

            for chunk in CenterCalculationHelper._iterate_cluster_data(
                data, labels, label, chunk_size, use_memmap
            ):
                # RMSD instead of Euclidean distance
                rmsds = np.sqrt(np.mean((chunk - mean)**2, axis=1))
                min_idx = np.argmin(rmsds)
                if rmsds[min_idx] < best_rmsd:
                    best_rmsd = rmsds[min_idx]
                    best_point = chunk[min_idx].copy()

            centers.append(best_point)

        return np.array(centers) if centers else None
