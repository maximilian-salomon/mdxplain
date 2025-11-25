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
ContactKernelPCA calculator for binary contact matrix dimensionality reduction.

Implements specialized KernelPCA computation for binary contact matrices using
Hamming distance-based kernel that is equivalent to RBF kernel for binary data.
"""

from typing import Any, Dict, Tuple

import numpy as np

from mdxplain.utils.progress_utils import ProgressUtils

from ..kernel_pca.kernel_pca_calculator import KernelPCACalculator


class ContactKernelPCACalculator(KernelPCACalculator):
    """
    Calculator for Contact Kernel Principal Component Analysis.

    Specialized KernelPCA calculator for binary contact matrices that uses
    a Hamming distance-based kernel. The Hamming kernel is equivalent to
    the RBF kernel for binary data since (x-y)² equals the Hamming distance
    for binary vectors.

    This calculator inherits from KernelPCACalculator and specializes the
    kernel computation for binary contact data while maintaining the same
    interface and incremental computation capabilities.

    Examples
    --------
    >>> # ContactKernelPCA for binary contact matrices
    >>> calc = ContactKernelPCACalculator()
    >>> binary_data = np.random.choice([0, 1], size=(1000, 100))
    >>> transformed, metadata = calc.compute(binary_data, n_components=10)

    >>> # Incremental ContactKernelPCA for large contact matrices
    >>> calc = ContactKernelPCACalculator(use_memmap=True, chunk_size=200)
    >>> large_binary = np.random.choice([0, 1], size=(10000, 500))
    >>> transformed, metadata = calc.compute(large_binary, n_components=50)

    >>> # Nyström approximation for very large contact matrices
    >>> calc = ContactKernelPCACalculator(use_memmap=True, chunk_size=200)
    >>> very_large_binary = np.random.choice([0, 1], size=(100000, 1000))
    >>> transformed, metadata = calc.compute(very_large_binary, n_components=50, use_nystrom=True, n_landmarks=5000)
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000, use_parallel: bool = False, n_jobs: int = -1, min_chunk_size: int = 1000) -> None:
        """
        Initialize ContactKernelPCA calculator.

        Parameters
        ----------
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

        Returns
        -------
        None
            Initializes ContactKernelPCA calculator with specified configuration

        Examples
        --------
        >>> # Standard ContactKernelPCA
        >>> calc = ContactKernelPCACalculator()

        >>> # Incremental ContactKernelPCA for large datasets
        >>> calc = ContactKernelPCACalculator(use_memmap=True, chunk_size=1000)
        """
        super().__init__(use_memmap, cache_path, chunk_size, use_parallel, n_jobs, min_chunk_size)
        self._cache_prefix = "contact_kernel_pca"

    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute ContactKernelPCA decomposition of binary contact data.

        Performs specialized Kernel Principal Component Analysis on binary
        contact matrices using the Hamming distance-based kernel that is
        equivalent to RBF kernel for binary data.

        Parameters
        ----------
        data : numpy.ndarray
            Binary contact matrix to decompose, shape (n_samples, n_features)
            Values should be 0 or 1
        kwargs : dict
            ContactKernelPCA parameters:

            - n_components : int, str, or None, default="auto"
                Number of components. Options:
                - int: Specific number of components
                - "auto": Automatic selection via elbow detection (5% of features)
                - None: Uses min(n_samples, n_features)
            - gamma : float, str, or None, default="scale"
                RBF kernel coefficient. Options:
                - float: Specific gamma value
                - "scale": 1.0 / (n_features * variance)
                - "auto": 1.0 / n_features
            - use_nystrom : bool, optional
                Whether to use Nyström approximation (default: False)
            - n_landmarks : int, optional
                Number of landmarks for Nyström approximation (default: 10000)
            - random_state : int, optional
                Random state for reproducible results

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:

            - transformed_data: ContactKernelPCA-transformed data (n_samples, n_components)
            - metadata: Dictionary with ContactKernelPCA information including
              kernel parameters, eigenvalues, and hyperparameters

        Examples
        --------
        >>> # Compute ContactKernelPCA for binary contact matrix
        >>> calc = ContactKernelPCACalculator()
        >>> contacts = np.random.choice([0, 1], size=(500, 100))
        >>> transformed, metadata = calc.compute(contacts, n_components=10)
        >>> print(f"Kernel: {metadata['hyperparameters']['kernel']}")

        Raises
        ------
        ValueError
            If input data is invalid, not binary, or n_components is too large
        """
        self._validate_binary_data(data)
        # Call parent compute which includes auto-selection logic
        return super().compute(data, **kwargs)

    def _validate_binary_data(self, data: np.ndarray) -> None:
        """
        Validate that input data is binary (contains only 0s and 1s).

        Parameters
        ----------
        data : numpy.ndarray
            Input data to validate

        Returns
        -------
        None
            Validates that data is binary

        Raises
        ------
        ValueError
            If data is not binary
        """
        self._validate_input_data(data)

        if self.chunk_size is None:
            if not np.all((data == 0) | (data == 1)):
                raise ValueError(
                    "ContactKernelPCA requires binary data. Probably your selection not only contains contacts."
                )
        else:
            # Chunk-wise validation with early exit
            for i in ProgressUtils.iterate(
                range(0, data.size, self.chunk_size),
                desc="Validating binary data",
                unit="chunks",
            ):
                chunk = data.flat[i : i + self.chunk_size]
                if not np.all((chunk == 0) | (chunk == 1)):
                    raise ValueError(
                        "ContactKernelPCA requires binary data. "
                        "Probably your selection not only contains contacts."
                    )

    def _extract_hyperparameters(self, data: np.ndarray, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate ContactKernelPCA hyperparameters.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for parameter validation
        kwargs : dict
            Input parameters to extract and validate

        Returns
        -------
        dict
            Validated hyperparameters with forced Hamming kernel
        """
        # Get base hyperparameters from parent class
        hyperparameters = super()._extract_hyperparameters(data, kwargs)

        # Use RBF kernel (equivalent to Hamming for binary data)
        # Store descriptive name in metadata but use actual kernel name for sklearn
        hyperparameters["kernel"] = "rbf"
        hyperparameters["kernel_description"] = "rbf on binary data (Hamming distance)"

        # Set default gamma for binary data if not specified
        if "gamma" not in kwargs:
            hyperparameters["gamma"] = 1.0

        # Add contact-specific metadata
        hyperparameters["contact_kernel"] = True
        hyperparameters["binary_data"] = True

        return hyperparameters
