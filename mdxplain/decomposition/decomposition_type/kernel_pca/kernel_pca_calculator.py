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
from joblib import Parallel, delayed
from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.linalg import eigh
from tqdm import tqdm

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

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000, use_parallel: bool = False, n_jobs: int = -1, min_chunk_size: int = 1000) -> None:
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
        use_parallel : bool, default=False
            Whether to use parallel processing for matrix-vector multiplication
        n_jobs : int, default=-1
            Number of parallel jobs (-1 for all available CPU cores)
        min_chunk_size : int, default=1000
            Minimum chunk size per parallel process to avoid overhead

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
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs
        self.min_chunk_size = min_chunk_size

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
        for row_start in tqdm(range(0, n_samples, used_chunk_size), desc="Computing kernel matrix rows", unit="chunks"):
            row_end = min(row_start + used_chunk_size, n_samples)
            chunk_i = data[row_start:row_end]

            for col_start in tqdm(range(0, n_samples, used_chunk_size), desc=f"Computing kernel row {row_start//used_chunk_size + 1}", unit="col_chunks", leave=False):
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
        for i in tqdm(range(0, n_samples, self.chunk_size), desc="Computing row sums", unit="chunks"):
            end = min(i + self.chunk_size, n_samples)
            row_sums[i:end] = kernel_matrix[i:end].sum(axis=1)
        
        # Second pass: compute column sums
        for j in tqdm(range(0, n_samples, self.chunk_size), desc="Computing col sums", unit="chunks"):
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
        
        for i in tqdm(range(0, n_samples, self.chunk_size), desc="Centering kernel matrix", unit="chunks"):
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
        n_samples = kernel_matrix.shape[0]

        row_means, col_means, grand_mean = self._compute_kernel_statistics_from_matrix(
            kernel_matrix
        )
        self._center_kernel_inplace(kernel_matrix, row_means, col_means, grand_mean)
        
        if self.use_parallel:
            parallel_chunk_size = max(self.min_chunk_size, self.chunk_size // self.n_jobs)
            matvec_func = lambda v: self._parallel_chunked_matvec(v, kernel_matrix, parallel_chunk_size)
        else:
            matvec_func = lambda v: self._chunked_matvec(v, kernel_matrix, self.chunk_size)
        
        # Create LinearOperator with progress tracking
        kernel_operator = self._create_progress_linear_operator(
            (n_samples, n_samples), matvec_func, kernel_matrix.dtype
        )

        print(f"Starting eigendecomposition for {hyperparameters['n_components']} components...")
        eigenvals, eigenvecs = eigs(
            kernel_operator,
            k=hyperparameters["n_components"],
            which='LM',  # Largest magnitude
            tol=1e-6
        )
        print(f"Eigendecomposition completed after {kernel_operator.call_count} matrix-vector products")

        # Sort eigenvalues and eigenvectors in descending order
        # Take only real parts (should be real due to symmetry)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals.real[idx]
        eigenvecs = eigenvecs.real[:, idx]

        # Filter positive eigenvalues for numerical stability
        lambdas = np.abs(eigenvals)
        positive_idx = lambdas > 1e-10
        lambdas = lambdas[positive_idx]
        eigenvecs = eigenvecs[:, positive_idx]

        # Transform: scale eigenvectors by sqrt(eigenvalues) (sklearn KernelPCA style)
        transformed_data = eigenvecs * np.sqrt(lambdas)

        # Store transformed_data as memmap when use_memmap=True
        if self.use_memmap:
            memmap_path = DataUtils.get_cache_file_path(
                f"{self._cache_prefix}_iterative.dat", self.cache_path
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
                "method": "iterative_kernel_pca",
                "eigenvalues": lambdas,
                "n_positive_eigenvalues": len(lambdas),
                "n_chunks": int(np.ceil(data.shape[0] / self.chunk_size)),
            }
        )

        return transformed_data, metadata

    def _chunked_matvec(self, v: np.ndarray, kernel_matrix: np.memmap, chunk_size: int) -> np.ndarray:
        """
        Performs matrix-vector product for a memmapped matrix in chunks.

        Parameters:
        -----------
        v : numpy.ndarray
            Input vector to multiply
        kernel_matrix : numpy.memmap
            Memory-mapped kernel matrix
        chunk_size : int
            Size of chunks to process at a time

        Returns:
        --------
        numpy.ndarray
            Resulting vector from the multiplication
        """
        n_samples = kernel_matrix.shape[0]
        result = np.zeros(n_samples, dtype=v.dtype)
        
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            
            # Load only one chunk of rows into RAM
            matrix_chunk = kernel_matrix[i:end, :] 
            
            # Perform multiplication and add to the result vector
            result[i:end] = matrix_chunk @ v
            
        return result

    def _parallel_chunked_matvec(self, v: np.ndarray, kernel_matrix: np.memmap, parallel_chunk_size: int) -> np.ndarray:
        """
        Performs matrix-vector product in parallel for a memmapped matrix in chunks.

        Parameters:
        -----------
        v : numpy.ndarray
            Input vector to multiply
        kernel_matrix : numpy.memmap
            Memory-mapped kernel matrix
        parallel_chunk_size : int
            Size of chunks to process at a time for parallel processing

        Returns:
        --------
        numpy.ndarray
            Resulting vector from the multiplication
        """
        n_samples = kernel_matrix.shape[0]
        
        def process_chunk(start_index):
            end_index = min(start_index + parallel_chunk_size, n_samples)
            matrix_chunk = kernel_matrix[start_index:end_index, :]
            return matrix_chunk @ v

        chunk_starts = range(0, n_samples, parallel_chunk_size)
        
        chunk_results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_chunk)(i) for i in chunk_starts
        )
        
        return np.concatenate(chunk_results)

    def _create_progress_linear_operator(self, shape, matvec_func, dtype):
        """
        Create LinearOperator with progress tracking for eigendecomposition.

        Parameters:
        -----------
        shape : tuple
            Shape of the linear operator
        matvec_func : callable
            Matrix-vector multiplication function
        dtype : numpy.dtype
            Data type of the operator

        Returns:
        --------
        LinearOperator
            LinearOperator with call counting for progress tracking
        """
        class ProgressLinearOperator(LinearOperator):
            def __init__(self, shape, dtype):
                super().__init__(shape=shape, dtype=dtype)
                self.call_count = 0
                self.last_report = 0

            def _matvec(self, v):
                self.call_count += 1
                if self.call_count - self.last_report >= 10:  # Report every 10 calls
                    print(f"  Matrix-vector products: {self.call_count}")
                    self.last_report = self.call_count
                return matvec_func(v)

        return ProgressLinearOperator(shape, dtype)
    
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
        Compute Nyström approximation KernelPCA with chunk-wise processing.

        Always uses chunk-wise processing for memory efficiency, regardless of data size.
        Fits Nyström on a sample, then processes data in chunks using partial_fit.

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
        n_samples = data.shape[0]
        n_landmarks = hyperparameters["n_landmarks"]
        n_components = hyperparameters["n_components"]

        nystroem = Nystroem(
            kernel="rbf",
            gamma=hyperparameters["gamma"],
            n_components=n_landmarks,
            random_state=hyperparameters["random_state"]
        )
        nystroem.fit(data)  # Only fit, don't transform yet
        
        # Step 2: IncrementalPCA for features (this is correct - PCA on features!)
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=self.chunk_size,
            whiten=True,
            copy=False
        )
        
        # Step 3: Chunk-wise transform and partial_fit
        for start in tqdm(range(0, n_samples, self.chunk_size), desc="Nystroem partial fitting", unit="chunks"):
            end = min(start + self.chunk_size, n_samples)
            data_chunk = data[start:end]

            # Transform chunk to Kernel-Features (n_landmarks dimensional)
            kernel_features_chunk = nystroem.transform(data_chunk)

            # Partial fit PCA on features (not on kernel matrix!)
            ipca.partial_fit(kernel_features_chunk)
        
        # Step 4: Final transform chunk-wise
        result = self._create_array_or_memmap(
            shape=(n_samples, n_components),
            dtype=np.float32,
            filename=DataUtils.get_cache_file_path(
                f"{self._cache_prefix}_nystrom.dat", self.cache_path
            )
        )

        for start in tqdm(range(0, n_samples, self.chunk_size), desc="Nystroem final transform", unit="chunks"):
            end = min(start + self.chunk_size, n_samples)
            data_chunk = data[start:end]

            kernel_features_chunk = nystroem.transform(data_chunk)
            result[start:end] = ipca.transform(kernel_features_chunk)

        metadata = self._prepare_metadata(hyperparameters, data.shape)
        metadata.update(
            {
                "method": "nystrom_kernel_pca",
                "n_landmarks": n_landmarks,
                "approximation": "nystrom",
                "explained_variance_ratio": ipca.explained_variance_ratio_.tolist() if hasattr(ipca, 'explained_variance_ratio_') else None,
            }
        )

        return result, metadata
