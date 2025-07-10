# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
Abstract base class for decomposition calculators.

Defines the interface that all decomposition calculators must implement
for consistency across different dimensionality reduction methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class CalculatorBase(ABC):
    """
    Abstract base class for decomposition calculators.

    Defines the interface that all decomposition calculators (PCA, KernelPCA,
    ContactKernelPCA) must implement for consistency across different
    dimensionality reduction methods.

    Examples:
    ---------
    >>> class MyCalculator(CalculatorBase):
    ...     def __init__(self, use_memmap=False, cache_path="./cache", chunk_size=10000):
    ...         super().__init__(use_memmap, cache_path, chunk_size)
    ...     def compute(self, data, **kwargs):
    ...         # Implement computation logic
    ...         return transformed_data, metadata
    """

    def __init__(self, use_memmap=False, cache_path="./cache", chunk_size=10000):
        """
        Initialize the decomposition calculator.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Path for memory-mapped cache files
        chunk_size : int, optional
            Size of chunks for incremental processing

        Returns:
        --------
        None
            Initializes calculator with specified configuration

        Examples:
        ---------
        >>> # Basic initialization
        >>> calc = MyCalculator()

        >>> # With memory mapping
        >>> calc = MyCalculator(
        ...     use_memmap=True,
        ...     cache_path="./cache/decomp.dat",
        ...     chunk_size=1000
        ... )
        """
        self.use_memmap = use_memmap
        self.cache_path = cache_path
        self.chunk_size = chunk_size

    @abstractmethod
    def compute(self, data, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute decomposition of input data.

        This method performs the actual dimensionality reduction computation
        and returns the transformed data along with metadata about the
        transformation process.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data matrix to decompose, shape (n_samples, n_features)
        **kwargs : dict
            Additional parameters specific to the decomposition method

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - transformed_data: Decomposed data matrix (n_samples, n_components)
            - metadata: Dictionary with transformation information including
              hyperparameters, explained variance, components, etc.

        Examples:
        ---------
        >>> # Compute decomposition
        >>> calc = MyCalculator()
        >>> data = np.random.rand(100, 50)
        >>> transformed, metadata = calc.compute(data, n_components=10)
        >>> print(f"Transformed shape: {transformed.shape}")
        >>> print(f"Explained variance: {metadata['explained_variance_ratio']}")
        """
        pass

    def _validate_input_data(self, data):
        """
        Validate input data for decomposition.

        Parameters:
        -----------
        data : numpy.ndarray
            Input data to validate

        Returns:
        --------
        None
            Validates input data format and shape

        Raises:
        -------
        ValueError
            If data format is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array")
        if data.shape[0] < 2:
            raise ValueError("Input data must have at least 2 samples")
        if data.shape[1] < 2:
            raise ValueError("Input data must have at least 2 feature")

    def _prepare_metadata(self, hyperparameters, original_shape):
        """
        Prepare base metadata dictionary.

        Parameters:
        -----------
        hyperparameters : dict
            Hyperparameters used for decomposition
        original_shape : tuple
            Shape of original input data

        Returns:
        --------
        dict
            Base metadata dictionary with common information
        """
        return {
            "hyperparameters": hyperparameters,
            "original_shape": original_shape,
            "use_memmap": self.use_memmap,
            "chunk_size": self.chunk_size,
            "cache_path": self.cache_path,
        }
