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
Abstract base class for clustering calculators.

Defines the interface that all clustering calculators must implement
for consistency across different clustering methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import warnings

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from ....utils.data_utils import DataUtils


class CalculatorBase(ABC):
    """
    Abstract base class for clustering calculators.

    Defines the interface that all clustering calculators (DBSCAN, HDBSCAN,
    DPA) must implement for consistency across different clustering methods.

    Examples
    --------
    >>> class MyCalculator(CalculatorBase):
    ...     def __init__(self, cache_path="./cache"):
    ...         super().__init__(cache_path)
    ...     def compute(self, data, **kwargs):
    ...         # Implement computation logic
    ...         return cluster_labels, metadata
    """

    def __init__(
        self, 
        cache_path: str = "./cache", 
        max_memory_gb: float = 2.0,
        chunk_size: int = 1000,
        use_memmap: bool = False
    ) -> None:
        """
        Initialize the clustering calculator.

        Parameters
        ----------
        cache_path : str, optional
            Path for cache files
        max_memory_gb : float, optional
            Maximum memory threshold in GB for standard clustering methods
        chunk_size : int, optional
            Chunk size for processing large datasets. Default is 1000.
        use_memmap : bool, optional
            Whether to use memory mapping for large datasets. Default is False.

        Returns
        -------
        None
            Initializes calculator with specified configuration

        Examples
        --------
        >>> # Basic initialization
        >>> calc = MyCalculator()

        >>> # With custom parameters
        >>> calc = MyCalculator(cache_path="./my_cache/", max_memory_gb=4.0, 
        ...                     chunk_size=5000, use_memmap=True)
        """
        self.cache_path = cache_path
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * (1024**3)
        self.chunk_size = chunk_size
        self.use_memmap = use_memmap

    @abstractmethod
    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute clustering of input data.

        This method performs the actual clustering computation
        and returns the cluster labels along with metadata about the
        clustering process.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix to cluster, shape (n_samples, n_features)
        \*\*kwargs : dict
            Additional parameters specific to the clustering method

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            
            - cluster_labels: Cluster labels for each sample (n_samples,)
            - metadata: Dictionary with clustering information including hyperparameters, number of clusters, silhouette score, etc.

        Examples
        --------
        >>> # Compute clustering
        >>> calc = MyCalculator()
        >>> data = np.random.rand(100, 50)
        >>> labels, metadata = calc.compute(data, eps=0.5, min_samples=5)
        >>> print(f"Number of clusters: {metadata['n_clusters']}")
        >>> print(f"Silhouette score: {metadata['silhouette_score']}")
        """
        pass

    def _validate_input_data(self, data: np.ndarray) -> None:
        """
        Validate input data for clustering.

        Parameters
        ----------
        data : numpy.ndarray
            Input data to validate

        Returns
        -------
        None
            Validates input data format and shape

        Raises
        ------
        ValueError
            If data format is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array")
        if data.shape[0] < 2:
            raise ValueError("Input data must have at least 2 samples")
        if data.shape[1] < 1:
            raise ValueError("Input data must have at least 1 feature")

    def _prepare_metadata(self, hyperparameters: Dict[str, Any], original_shape: Tuple[int, int], n_clusters: int, n_noise: int = 0) -> Dict[str, Any]:
        """
        Prepare base metadata dictionary.

        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters used for clustering
        original_shape : tuple
            Shape of original input data
        n_clusters : int
            Number of clusters found
        n_noise : int, optional
            Number of noise points (for algorithms that identify noise)

        Returns
        -------
        dict
            Base metadata dictionary with common information
        """
        return {
            "hyperparameters": hyperparameters,
            "original_shape": original_shape,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cache_path": self.cache_path,
        }

    def _count_clusters(self, labels: np.ndarray) -> int:
        """
        Count number of clusters (excluding noise).

        Parameters
        ----------
        labels : numpy.ndarray
            Cluster labels (-1 indicates noise)

        Returns
        -------
        int
            Number of clusters found
        """
        unique_labels = np.unique(labels)
        # Exclude noise label (-1) from count
        return len(unique_labels[unique_labels != -1])

    def _count_noise_points(self, labels: np.ndarray, noise_cluster: int = -1) -> int:
        """
        Count number of noise points.

        Parameters
        ----------
        labels : numpy.ndarray
            Cluster labels (-1 indicates noise)
        noise_cluster : int
            The value or label which is assigned to noise data points
        Returns
        -------
        int
            Number of noise points
        """
        return np.sum(labels == noise_cluster)

    def _compute_silhouette_score(self, data: np.ndarray, labels: np.ndarray, sample_size: int = 50000) -> Optional[float]:
        """
        Compute silhouette score for clustering quality assessment.
        
        Uses sampling for large datasets to avoid O(n²) performance issues.

        Parameters
        ----------
        data : numpy.ndarray
            Original data used for clustering
        labels : numpy.ndarray
            Cluster labels
        sample_size : int, optional
            Maximum number of samples to use for silhouette calculation.
            Default is 50000 to balance accuracy and performance.

        Returns
        -------
        float or None
            Silhouette score, or None if cannot be computed
        """
        # Remove noise points for silhouette calculation
        non_noise_mask = labels != -1

        if np.sum(non_noise_mask) < 2:
            return None

        # Check if we have at least 2 different clusters
        unique_labels = np.unique(labels[non_noise_mask])
        if len(unique_labels) < 2:
            return None

        # Count non-noise points without copying data
        n_non_noise = np.sum(non_noise_mask)
        
        # Use sampling for large datasets to avoid O(n²) performance issues
        if n_non_noise > sample_size:
            # Get indices of non-noise points without copying data
            non_noise_indices = np.where(non_noise_mask)[0]
            
            # Sample indices randomly for silhouette calculation
            sample_indices = np.random.choice(len(non_noise_indices), sample_size, replace=False)
            final_indices = non_noise_indices[sample_indices]
            
            # Get sampled data and labels using indices (memory-safe)
            sampled_data = data[final_indices]
            sampled_labels = labels[final_indices]
            
            # Ensure we still have at least 2 clusters in the sample
            unique_sampled_labels = np.unique(sampled_labels)
            if len(unique_sampled_labels) < 2:
                return None
                
            return silhouette_score(sampled_data, sampled_labels)
        else:
            # Use full non-noise data for small datasets (memory-safe for small data)
            non_noise_data = data[non_noise_mask]
            non_noise_labels = labels[non_noise_mask]
            return silhouette_score(non_noise_data, non_noise_labels)

    def _calculate_sample_size(
        self, 
        n_samples: int, 
        sample_fraction: float, 
        min_samples: int = 50000, 
        max_samples: int = 100000
    ) -> int:
        """
        Calculate sample size with configurable constraints for sampling-based methods.

        Parameters
        ----------
        n_samples : int
            Total number of samples in dataset
        sample_fraction : float
            Fraction of data to sample (typically 0.1 for 10%)
        min_samples : int, optional
            Minimum sample size. Default is 50k for HDBSCAN/DBSCAN
        max_samples : int, optional
            Maximum sample size. Default is 100k for HDBSCAN/DBSCAN

        Returns
        -------
        int
            Sample size between min_samples and max_samples, or n_samples if smaller
        """
        if n_samples <= min_samples:
            return n_samples  
        
        sample_size = int(n_samples * sample_fraction)
        return max(min_samples, min(max_samples, sample_size))

    def _validate_memory_and_dimensionality(self, data: np.ndarray, parameters: Dict[str, Any]) -> None:
        """
        Validate memory and dimensionality constraints for clustering algorithms.

        Parameters
        ----------
        data : numpy.ndarray
            Input data to check
        parameters : dict
            Algorithm parameters including force flag and method

        Raises
        ------
        ValueError
            If constraints are violated and force=False
        """
        n_samples, n_features = data.shape
        force = parameters["force"]
        method = parameters["method"]
        
        # Memory check: only for standard method
        if method == "standard":
            data_size_gb = data.nbytes / (1024**3)
            if data_size_gb > self.max_memory_gb:
                message = (
                    f"Dataset size ({data_size_gb:.1f}GB) exceeds {self.max_memory_gb:.1f}GB threshold.\n"
                    f"Consider using alternative methods:\n"
                    f"  • method='approximate_predict' (recommended for HDBSCAN)\n"
                    f"  • method='precomputed' (exact but slow for DBSCAN)\n"
                    f"  • method='knn_sampling' (fallback for all)\n"
                    f"  • force=True to override this check (not recommended for large datasets)\n"
                )
                if force:
                    warnings.warn(message)
                else:
                    raise ValueError(message)
        
        # Dimensionality check: >250 features
        if n_features > 250:
            message = (
                f"High dimensionality ({n_features} features) detected. "
                f"Curse of Dimensionality - clustering may not be meaningful. "
                f"Use force=True to do it anyway. Clustering needs at least a bigger "
                f"bunch of samples to work properly, on huge matrices it might be slow "
                f"or run out of memory."
            )
            if force:
                warnings.warn(message)
            else:
                raise ValueError(message)
        
        # Auto-fallback check for sampling methods
        if method != "standard":
            sample_size = parameters["sample_size"]
            if n_samples <= sample_size and not force:
                print(
                    f"Info: Dataset has only {n_samples:,} samples (<= {sample_size:,}). "
                    f"Using standard method instead of {method}. "
                    f"With force=True you can override this automatic behavior."
                )
                parameters["method"] = "standard"
                # Re-validate with standard method
                self._validate_memory_and_dimensionality(data, parameters)

    def _prepare_labels_storage(self, n_samples: int, algorithm: str, method: str) -> np.ndarray:
        """
        Prepare storage for cluster labels (memmap or numpy array).

        Parameters
        ----------
        n_samples : int
            Number of samples to store labels for
        algorithm : str
            Algorithm name (e.g., 'dbscan', 'hdbscan', 'dpa')
        method : str
            Method name (e.g., 'standard', 'knn_sampling', 'precomputed')

        Returns
        -------
        numpy.ndarray or numpy.memmap
            Storage array for cluster labels
        """
        if self.use_memmap:
            filename = f"{algorithm}_{method}_labels.dat"
            path = DataUtils.get_cache_file_path(filename, self.cache_path)
            return np.memmap(path, dtype=np.int32, mode='w+', shape=(n_samples,))
        else:
            return np.empty(n_samples, dtype=np.int32)

    def _finalize_labels(self, labels: np.ndarray, algorithm: str, method: str) -> np.ndarray:
        """
        Convert labels to final format (memmap or numpy array).

        Parameters
        ----------
        labels : numpy.ndarray
            Computed cluster labels
        algorithm : str
            Algorithm name for memmap filename
        method : str
            Method name for memmap filename

        Returns
        -------
        numpy.ndarray or numpy.memmap
            Final labels in requested format
        """
        if self.use_memmap and not isinstance(labels, np.memmap):
            filename = f"{algorithm}_{method}_labels.dat"
            path = DataUtils.get_cache_file_path(filename, self.cache_path)
            memmap_labels = np.memmap(path, dtype=np.int32, mode='w+', shape=labels.shape)
            memmap_labels[:] = labels
            return memmap_labels
        return labels

    def _perform_knn_sampling(
        self, 
        data: np.ndarray, 
        parameters: Dict[str, Any], 
        sample_indices: np.ndarray, 
        sample_labels: np.ndarray,
        algorithm: str,
        noise_label: int = -1
    ) -> np.ndarray:
        """
        Generic k-NN sampling implementation for all clustering algorithms.

        Parameters
        ----------
        data : numpy.ndarray
            Full dataset to cluster
        parameters : dict
            Algorithm parameters including knn_neighbors and sample_size
        sample_indices : numpy.ndarray
            Indices of sampled points
        sample_labels : numpy.ndarray
            Cluster labels for sampled points
        algorithm : str
            Algorithm name for memmap filename
        noise_label : int, optional
            Label for noise points (default -1, DPA uses 0 with halos=True)

        Returns
        -------
        numpy.ndarray
            Full cluster labels (memmap or numpy array)
        """
        n_samples = data.shape[0]
        sample_size = parameters["sample_size"]
        
        print(f"Using knn_sampling with {sample_size:,} samples "
              f"({sample_size/n_samples*100:.1f}% of {n_samples:,} total)")
        
        # Prepare storage (memmap or numpy array) and initialize with noise
        full_labels = self._prepare_labels_storage(n_samples, algorithm, "knn_sampling")
        full_labels[:] = noise_label
        full_labels[sample_indices] = sample_labels
        
        # Use k-NN for non-sampled points (only for non-noise clusters)
        non_noise_mask = sample_labels != noise_label
        if np.sum(non_noise_mask) > 0:
            non_noise_sample_data = data[sample_indices[non_noise_mask]]
            non_noise_sample_labels = sample_labels[non_noise_mask]
            
            # Find remaining points to classify
            remaining_mask = np.ones(n_samples, dtype=bool)
            remaining_mask[sample_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) > 0:
                # Fit k-NN classifier
                n_neighbors = min(parameters["knn_neighbors"], len(non_noise_sample_data))
                knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn_classifier.fit(non_noise_sample_data, non_noise_sample_labels)
                
                # Process remaining points in chunks (direct memmap/array writing)
                for start in tqdm(range(0, len(remaining_indices), self.chunk_size), 
                                  desc="k-NN prediction", unit="chunks"):
                    end = min(start + self.chunk_size, len(remaining_indices))
                    chunk_indices = remaining_indices[start:end]
                    
                    # k-NN prediction for chunk and write directly
                    chunk_labels = knn_classifier.predict(data[chunk_indices])
                    full_labels[chunk_indices] = chunk_labels
        
        return full_labels
