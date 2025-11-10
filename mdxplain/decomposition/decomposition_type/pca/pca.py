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
PCA decomposition type implementation for molecular dynamics analysis.

PCA decomposition type implementation with standard and incremental
Principal Component Analysis for dimensionality reduction of feature matrices.
"""

from typing import Dict, Tuple, Optional, Any, Union

import numpy as np

from ..interfaces.decomposition_type_base import DecompositionTypeBase
from .pca_calculator import PCACalculator


class PCA(DecompositionTypeBase):
    """
    Principal Component Analysis decomposition type.

    Implements PCA for dimensionality reduction of feature matrices from
    molecular dynamics trajectories. Supports both standard and incremental
    computation for large datasets.

    This is a linear dimensionality reduction method that finds the directions
    of maximum variance in the data and projects the data onto these directions.

    Uses sklearn's PCA and IncrementalPCA under the hood.

    Examples
    --------
    >>> # Basic PCA decomposition via DecompositionManager
    >>> from mdxplain.decomposition import decomposition_type
    >>> decomp_manager = DecompositionManager()
    >>> decomp_manager.add(
    ...     traj_data, "feature_selection", decomposition_type.PCA(n_components=10)
    ... )

    >>> # PCA with incremental computation for large datasets
    >>> pca = decomposition_type.PCA()
    >>> pca.init_calculator(use_memmap=True, chunk_size=1000)
    >>> transformed, metadata = pca.compute(large_data, n_components=50)
    """

    def __init__(self, n_components: Union[int, str, None] = "auto", random_state: Optional[int] = None, offset: Union[int, float] = 0) -> None:
        """
        Initialize PCA decomposition type with parameters.

        Creates a PCA instance with specified parameters that will be
        used during computation.

        Parameters
        ----------
        n_components : int, str, or None, default="auto"
            Number of components to keep. Options:
            - int: Specific number of components
            - "auto": Automatic selection via elbow detection (5% of features) [DEFAULT]
            - None: Uses min(n_samples, n_features)
        random_state : int, optional
            Random state for reproducible results
        offset : int or float, default=0
            Adjustment to auto-selected component count (only applies when n_components="auto"):

            - int: Direct addition/subtraction (e.g., -2 selects 2 fewer)
            - float: Percentage adjustment (e.g., -0.5 selects 50% fewer)

        Returned Metadata
        -----------------
        hyperparameters : dict
            Dictionary containing all PCA parameters used
        original_shape : tuple
            Shape of the input data (n_samples, n_features)
        use_memmap : bool
            Whether memory mapping was used
        chunk_size : int
            Chunk size used for incremental processing
        cache_path : str
            Path used for caching results
        explained_variance_ratio : array
            Percentage of variance explained by each component
        explained_variance : array
            Variance explained by each component
        method : str
            Method used ('standard_pca' or 'incremental_pca')
        n_chunks : int
            Number of chunks used (only for incremental_pca)

        Examples
        --------
        >>> # Create PCA instance with parameters
        >>> pca = PCA(n_components=10, random_state=42)
        >>> print(f"Type: {pca.get_type_name()}")
        'pca'
        """
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.offset = offset

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the type name for PCA decomposition.

        Returns the unique string identifier for PCA decomposition type
        used for storing results and type identification.

        Parameters
        ----------
        cls : type
            The PCA class

        Returns
        -------
        str
            String identifier 'pca'

        Examples
        --------
        >>> print(PCA.get_type_name())
        'pca'
        >>> # Can also be used via class directly
        >>> print(decomposition_type.PCA.get_type_name())
        'pca'
        """
        return "pca"

    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the PCA calculator with specified configuration.

        Sets up the PCA calculator with options for memory mapping and
        incremental computation for large datasets.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use incremental computation for large datasets
        cache_path : str, optional
            Path for cache files (not used for PCA but kept for interface consistency)
        chunk_size : int, optional
            Number of samples to process per chunk for incremental computation

        Returns
        -------
        None
            Sets self.calculator to initialized PCACalculator instance

        Examples
        --------
        >>> # Basic initialization
        >>> pca = PCA()
        >>> pca.init_calculator()

        >>> # With incremental computation for large datasets
        >>> pca.init_calculator(use_memmap=True, chunk_size=1000)

        >>> # With custom chunk size
        >>> pca.init_calculator(chunk_size=500)
        """
        self.calculator = PCACalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute PCA decomposition of input data.

        Performs Principal Component Analysis on the input feature matrix
        using the initialized calculator and the parameters provided during
        initialization.

        Parameters
        ----------
        data : numpy.ndarray
            Input feature matrix to decompose, shape (n_samples, n_features)

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:

            - transformed_data: PCA-transformed data matrix (n_samples, n_components)
            - metadata: Dictionary with PCA information including:
            
              * hyperparameters: Used parameters
              * explained_variance_ratio: Variance explained by each component
              * components: Principal components (eigenvectors)
              * explained_variance: Variance explained by each component
              * mean: Mean of the original data
              * method: 'standard_pca' or 'incremental_pca'

        Examples
        --------
        >>> # Compute PCA with predefined parameters
        >>> pca = PCA(n_components=10, random_state=42)
        >>> pca.init_calculator()
        >>> data = np.random.rand(1000, 100)
        >>> transformed, metadata = pca.compute(data)
        >>> print(f"Transformed shape: {transformed.shape}")

        >>> # Incremental PCA for large datasets
        >>> pca = PCA(n_components=50)
        >>> pca.init_calculator(use_memmap=True, chunk_size=200)
        >>> large_data = np.random.rand(10000, 500)
        >>> transformed, metadata = pca.compute(large_data)

        Raises
        ------
        ValueError
            If calculator is not initialized, input data is invalid,
            or n_components is too large
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        return self.calculator.compute(
            data, n_components=self.n_components, random_state=self.random_state, offset=self.offset
        )
