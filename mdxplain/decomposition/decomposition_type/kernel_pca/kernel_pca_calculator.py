# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Kiro AI (Claude Sonnet 4.0).
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
KernelPCA calculator for nonlinear dimensionality reduction.

Implements KernelPCA computation with support for incremental kernel
computation for large datasets using sklearn's KernelPCA.
"""

from typing import Dict, Tuple, Any

import numpy as np
from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel

from ..interfaces.calculator_base import CalculatorBase
from ....utils.data_utils import DataUtils


class KernelPCACalculator(CalculatorBase):
    """
    Calculator for Kernel Principal Component Analysis (KernelPCA) decomposition.

    Implements KernelPCA computation with support for various kernels and
    incremental kernel computation for large datasets. Uses sklearn's KernelPCA
    with precomputed kernels for memory-efficient processing.

    Examples:
    ---------
    >>> # Standard KernelPCA with RBF kernel
    >>> calc = KernelPCACalculator()
    >>> data = np.random.rand(1000, 100)
    >>> transformed, metadata = calc.compute(data, n_components=10, kernel='rbf')

    >>> # Incremental KernelPCA for large datasets
    >>> calc = KernelPCACalculator(use_memmap=True, chunk_size=200)
    >>> large_data = np.random.rand(10000, 500)
    >>> transformed, metadata = calc.compute(large_data, n_components=50)
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 10000) -> None:
        """
        Initialize KernelPCA calculator.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping and incremental kernel computation
        cache_path : str, optional
            Path for memory-mapped cache files
        chunk_size : int, optional
            Size of chunks for incremental kernel computation

        Returns:
        --------
        None
            Initializes KernelPCA calculator with specified configuration

        Examples:
        ---------
        >>> # Standard KernelPCA
        >>> calc = KernelPCACalculator()

        >>> # Incremental KernelPCA for large datasets
        >>> calc = KernelPCACalculator(use_memmap=True, chunk_size=1000)
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self._cache_prefix = "kernel_pca"

    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute KernelPCA decomposition of input data.

        Performs Kernel Principal Component Analysis on the input data matrix,
        using either standard or incremental kernel computation based on the
        configuration settings.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to decompose, shape (n_samples, n_features)
        **kwargs : dict
            KernelPCA parameters:
            - n_components : int, required
                Number of components to keep
            - gamma : float, optional
                RBF kernel coefficient (default: 1.0 / n_features)
            - use_nystrom : bool, optional
                Whether to use Nyström approximation (default: False)
            - n_landmarks : int, optional
                Number of landmarks for Nyström approximation (default: 10000)
            - random_state : int, optional
                Random state for reproducible results

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - transformed_data: KernelPCA-transformed data (n_samples, n_components)
            - metadata: Dictionary with KernelPCA information including kernel
              parameters and hyperparameters

        Examples:
        ---------
        >>> # Compute KernelPCA with RBF kernel
        >>> calc = KernelPCACalculator()
        >>> data = np.random.rand(500, 100)
        >>> transformed, metadata = calc.compute(data, n_components=10, kernel='rbf')
        >>> print(f"Kernel: {metadata['hyperparameters']['kernel']}")

        Raises:
        -------
        ValueError
            If input data is invalid or n_components is too large
        """
        self._validate_input_data(data)
        hyperparameters = self._extract_hyperparameters(data, kwargs)

        if hyperparameters["use_nystrom"]:
            return self._compute_nystrom_kernel_pca(data, hyperparameters)
        elif self.use_memmap:
            return self._compute_incremental_kernel_pca(data, hyperparameters)
        else:
            return self._compute_standard_kernel_pca(data, hyperparameters)

    def _extract_hyperparameters(self, data: np.ndarray, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate KernelPCA hyperparameters.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data for parameter validation
        kwargs : dict
            Input parameters to extract and validate

        Returns:
        --------
        dict
            Validated hyperparameters
        """
        n_components = kwargs.get("n_components", data.shape[0])
        if n_components is None:
            raise ValueError("n_components must be specified")

        gamma = kwargs.get("gamma", 1.0 / data.shape[0])
        use_nystrom = kwargs.get("use_nystrom", False)
        n_landmarks = kwargs.get("n_landmarks", 10000)
        random_state = kwargs.get("random_state", None)

        # Validate n_components
        if n_components is not None and n_components > data.shape[0]:
            raise ValueError(
                f"n_components ({n_components}) cannot be larger than the sample-number {data.shape[0]:}."
            )

        # Validate n_landmarks for Nyström
        if use_nystrom:
            n_landmarks = min(n_landmarks, data.shape[0])
            if n_landmarks < n_components:
                raise ValueError(
                    f"n_landmarks ({n_landmarks}) must be >= n_components ({n_components}) for Nyström approximation."
                )

        return {
            "n_components": n_components,
            "kernel": "rbf",
            "gamma": gamma,
            "use_nystrom": use_nystrom,
            "n_landmarks": n_landmarks,
            "random_state": random_state,
        }

    def _compute_standard_kernel_pca(self, data: np.ndarray, hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute standard KernelPCA using sklearn.decomposition.KernelPCA.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix
        hyperparameters : dict
            KernelPCA hyperparameters

        Returns:
        --------
        tuple
            Tuple of (transformed_data, metadata)
        """
        kpca = KernelPCA(
            n_components=hyperparameters["n_components"],
            kernel=hyperparameters["kernel"],
            gamma=hyperparameters["gamma"],
            random_state=hyperparameters["random_state"],
            copy_X=False,
            n_jobs=-1,
        )

        transformed_data = kpca.fit_transform(data)

        metadata = self._prepare_metadata(hyperparameters, data.shape)
        metadata.update(
            {
                "method": "standard_kernel_pca",
            }
        )

        return transformed_data, metadata

    def _compute_chunk_wise_rbf_kernel(self, data: np.ndarray, gamma: float) -> np.ndarray:
        """
        Compute RBF kernel matrix chunk-wise without loading all data into RAM.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix
        gamma : float
            RBF kernel coefficient

        Returns:
        --------
        numpy.ndarray
            Complete RBF kernel matrix computed chunk-wise
        """
        n_samples = data.shape[0]

        # Create kernel matrix as memmap if use_memmap=True
        kernel_matrix = self._create_kernel_memmap(n_samples)

        # Use half of the chunk size, cause we need to use two chunks of size chunk_size
        used_chunk_size = int(np.floor(self.chunk_size / 2))

        # Compute kernel matrix chunk-wise
        for row_start in range(0, n_samples, used_chunk_size):
            row_end = min(row_start + used_chunk_size, n_samples)
            chunk_i = data[row_start:row_end]

            for col_start in range(0, n_samples, used_chunk_size):
                col_end = min(col_start + used_chunk_size, n_samples)
                chunk_j = data[col_start:col_end]

                # Compute RBF kernel block
                block = rbf_kernel(chunk_i, chunk_j, gamma=gamma)
                kernel_matrix[row_start:row_end, col_start:col_end] = block

        return kernel_matrix

    def _compute_kernel_statistics_from_matrix(self, kernel_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute kernel statistics from precomputed kernel matrix for centering.

        Parameters:
        -----------
        kernel_matrix : numpy.ndarray
            Precomputed kernel matrix
        
        Returns:
        --------
        tuple
            Tuple containing (row_means, col_means, grand_mean) for kernel centering
        """
        n_samples = kernel_matrix.shape[0]
        row_sums = np.zeros(n_samples, dtype=np.float64)
        col_sums = np.zeros(n_samples, dtype=np.float64)
        
        # First pass: compute row sums
        for i in range(0, n_samples, self.chunk_size):
            end = min(i + self.chunk_size, n_samples)
            row_sums[i:end] = kernel_matrix[i:end].sum(axis=1)
        
        # Second pass: compute column sums
        for j in range(0, n_samples, self.chunk_size):
            end = min(j + self.chunk_size, n_samples)
            col_sums[j:end] = kernel_matrix[:, j:end].sum(axis=0)
        
        row_means = row_sums / n_samples
        col_means = col_sums / n_samples
        grand_mean = row_sums.sum() / (n_samples * n_samples)
        
        return row_means, col_means, grand_mean

    def _center_kernel_inplace(self, kernel_matrix: np.ndarray, row_means: np.ndarray, col_means: np.ndarray, grand_mean: float) -> None:
        """
        Center kernel matrix in-place using precomputed statistics.

        Parameters:
        -----------
        kernel_matrix : numpy.ndarray
            Kernel matrix to center (modified in-place)
        row_means : numpy.ndarray
            Row means for centering
        col_means : numpy.ndarray
            Column means for centering  
        grand_mean : float
            Grand mean for centering

        Returns:
        --------
        None
            Modifies kernel_matrix in-place
        """
        n_samples = kernel_matrix.shape[0]
        
        for i in range(0, n_samples, self.chunk_size):
            i_end = min(i + self.chunk_size, n_samples)
            
            # Apply centering formula: K_centered = K - row_means - col_means + grand_mean
            kernel_matrix[i:i_end] = (kernel_matrix[i:i_end] - 
                                    row_means[i:i_end, np.newaxis] - 
                                    col_means[np.newaxis, :] + 
                                    grand_mean)

    def _compute_incremental_kernel_pca(self, data: np.ndarray, hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute incremental KernelPCA with chunk-wise centered kernel matrix.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix
        hyperparameters : dict
            KernelPCA hyperparameters

        Returns:
        --------
        tuple
            Tuple of (transformed_data, metadata)
        """
        kernel_matrix = self._compute_chunk_wise_rbf_kernel(
            data, hyperparameters["gamma"]
        )

        row_means, col_means, grand_mean = self._compute_kernel_statistics_from_matrix(
            kernel_matrix
        )

        self._center_kernel_inplace(kernel_matrix, row_means, col_means, grand_mean)

        ipca = IncrementalPCA(
            n_components=hyperparameters["n_components"],
            batch_size=self.chunk_size,
            whiten=True,
            copy=False,
        )

        # Get principal components from IncrementalPCA
        principal_components = ipca.fit_transform(kernel_matrix)
        
        # Use the eigenvalues from IncrementalPCA, scaled by (n_samples-1)
        # IncrementalPCA uses explained_variance = eigenvals / (n_samples - 1)
        n_samples = kernel_matrix.shape[0]
        kernel_eigenvals = ipca.explained_variance_ * (n_samples - 1)
        
        # Scale to match sklearn KernelPCA: eigenvector * sqrt(eigenvalue)
        transformed_data = principal_components * np.sqrt(kernel_eigenvals)

        # Store transformed_data as memmap when use_memmap=True
        if self.use_memmap:
            memmap_path = DataUtils.get_cache_file_path(
                f"{self._cache_prefix}.dat", self.cache_path
            )
            transformed_memmap = np.memmap(
                memmap_path,
                dtype=transformed_data.dtype,
                mode="w+",
                shape=transformed_data.shape,
            )
            transformed_memmap[:] = transformed_data
            transformed_data = transformed_memmap

        metadata = self._prepare_metadata(hyperparameters, data.shape)
        metadata.update(
            {
                "method": "incremental_kernel_pca",
                "n_chunks": int(np.ceil(data.shape[0] / self.chunk_size)),
            }
        )

        return transformed_data, metadata

    def _create_kernel_memmap(self, n_samples: int) -> np.ndarray:
        """
        Create memmap for kernel matrix storage.

        Parameters:
        -----------
        n_samples : int
            Number of samples (kernel matrix is n_samples x n_samples)

        Returns:
        --------
        numpy.memmap
            Memory-mapped kernel matrix
        """
        memmap_path = DataUtils.get_cache_file_path(
            f"{self._cache_prefix}_kernel_matrix.dat", self.cache_path
        )

        # RBF kernel values are always float64
        kernel_matrix = np.memmap(
            memmap_path, dtype=float, mode="w+", shape=(n_samples, n_samples)
        )

        return kernel_matrix

    def _compute_nystrom_kernel_pca(self, data: np.ndarray, hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Nyström approximation KernelPCA with IncrementalPCA.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix
        hyperparameters : dict
            KernelPCA hyperparameters

        Returns:
        --------
        tuple
            Tuple of (transformed_data, metadata)
        """
        # Nyström approximation
        nystroem = Nystroem(
            kernel="rbf",
            gamma=hyperparameters["gamma"],
            n_components=hyperparameters["n_landmarks"],
            random_state=hyperparameters["random_state"],
        )

        # Transform data to approximate kernel features
        kernel_features = nystroem.fit_transform(data)

        # Incremental PCA on kernel features
        ipca = IncrementalPCA(
            n_components=hyperparameters["n_components"],
            batch_size=self.chunk_size,
            whiten=True,
            copy=False,
        )

        # Get principal components from IncrementalPCA
        principal_components = ipca.fit_transform(kernel_features)
        
        # Use the eigenvalues from IncrementalPCA, scaled by (n_samples-1)
        # IncrementalPCA uses explained_variance = eigenvals / (n_samples - 1)
        n_samples = kernel_features.shape[0]
        kernel_eigenvals = ipca.explained_variance_ * (n_samples - 1)
        
        # Scale to match sklearn KernelPCA: eigenvector * sqrt(eigenvalue)
        transformed_data = principal_components * np.sqrt(kernel_eigenvals)
        
        # Store transformed_data as memmap when use_memmap=True
        if self.use_memmap:
            memmap_path = DataUtils.get_cache_file_path(
                f"{self._cache_prefix}_nystrom_approximation.dat", self.cache_path
            )
            transformed_memmap = np.memmap(
                memmap_path,
                dtype=transformed_data.dtype,
                mode="w+",
                shape=transformed_data.shape,
            )
            transformed_memmap[:] = transformed_data
            transformed_data = transformed_memmap

        metadata = self._prepare_metadata(hyperparameters, data.shape)
        metadata.update(
            {
                "method": "nystrom_kernel_pca",
                "n_landmarks": hyperparameters["n_landmarks"],
                "approximation": "nystrom",
            }
        )

        return transformed_data, metadata
