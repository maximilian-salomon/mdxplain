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
Abstract base class defining the interface for all decomposition types.

Defines the interface that all decomposition types (PCA, KernelPCA, etc.)
must implement for consistency across different dimensionality reduction methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional

import numpy as np

from .decomposition_type_meta import DecompositionTypeMeta


class DecompositionTypeBase(ABC, metaclass=DecompositionTypeMeta):
    """
    Abstract base class for all decomposition types.

    Defines the interface that all decomposition types (PCA, KernelPCA, etc.)
    must implement. Each decomposition type encapsulates computation logic
    for a specific type of dimensionality reduction analysis.

    Examples
    --------
    >>> class MyDecomposition(DecompositionTypeBase):
    ...     @classmethod
    ...     def get_type_name(cls) -> str:
    ...         return 'my_decomposition'
    ...     def init_calculator(self, **kwargs):
    ...         self.calculator = MyCalculator(**kwargs)
    ...     def compute(self, data, **kwargs):
    ...         return self.calculator.compute(data, **kwargs)
    """

    def __init__(self) -> None:
        """
        Initialize the decomposition type.

        Sets up the decomposition type instance with an empty calculator that
        will be initialized later through init_calculator().

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> # Create decomposition type instance
        >>> decomp = MyDecomposition()
        >>> print(f"Type: {decomp.get_type_name()}")
        """
        self.calculator = None

    @classmethod
    @abstractmethod
    def get_type_name(cls) -> str:
        """
        Return unique string identifier for this decomposition type.

        Used as the key for storing decomposition results in TrajectoryData
        dictionaries and for type identification.

        Parameters
        ----------
        cls : type
            The decomposition type class

        Returns
        -------
        str
            Unique string identifier (e.g., 'pca', 'kernel_pca')

        Examples
        --------
        >>> print(PCA.get_type_name())
        'pca'
        >>> print(KernelPCA.get_type_name())
        'kernel_pca'
        """
        pass

    @abstractmethod
    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the calculator instance for this decomposition type.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for efficient handling of large datasets
        cache_path : str, optional
            Directory path for cache files
        chunk_size : int, optional
            Number of samples to process per chunk for incremental computation

        Returns
        -------
        None
            Sets self.calculator to initialized calculator instance

        Examples
        --------
        >>> # Basic initialization
        >>> pca = PCA()
        >>> pca.init_calculator()

        >>> # With memory mapping for large datasets
        >>> pca.init_calculator(
        ...     use_memmap=True,
        ...     cache_path='./cache/',
        ...     chunk_size=1000
        ... )
        """
        pass

    @abstractmethod
    def compute(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute decomposition using the initialized calculator.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix to decompose, shape (n_samples, n_features)

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            
            - transformed_data: Decomposed data matrix (n_samples, n_components)
            - metadata: Dictionary with transformation information including
              hyperparameters, explained variance, components, etc.

        Examples
        --------
        >>> # Compute PCA decomposition
        >>> pca = PCA()
        >>> pca.init_calculator()
        >>> data = np.random.rand(100, 50)
        >>> transformed, metadata = pca.compute(data, n_components=10)
        >>> print(f"Transformed shape: {transformed.shape}")

        Raises
        ------
        ValueError
            If calculator is not initialized or input data is invalid
        """
        pass

    def get_required_feature_type(self) -> Optional[str]:
        """
        Return required feature type for this decomposition method.

        Some decomposition methods require specific feature types (e.g., 
        DiffusionMaps requires 'coordinates'). If a specific feature type 
        is required, the DecompositionManager will validate that the 
        FeatureSelector contains only features of this type.

        Parameters
        ----------
        None

        Returns
        -------
        Optional[str]
            Required feature type name, or None if no specific type required

        Examples
        --------
        >>> # Most decompositions work with any feature type
        >>> pca = PCA()
        >>> print(pca.get_required_feature_type())
        None

        >>> # DiffusionMaps requires coordinate features
        >>> diffmaps = DiffusionMaps()
        >>> print(diffmaps.get_required_feature_type())
        'coordinates'
        """
        return None
