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
KernelPCA decomposition type implementation for molecular dynamics analysis.

KernelPCA decomposition type implementation with nonlinear dimensionality
reduction using various kernel functions for feature matrices.
"""

from typing import Dict, Tuple

import numpy as np

from ..interfaces.decomposition_type_base import DecompositionTypeBase
from .kernel_pca_calculator import KernelPCACalculator


class KernelPCA(DecompositionTypeBase):
    """
    Kernel Principal Component Analysis decomposition type.

    Implements KernelPCA for nonlinear dimensionality reduction of feature
    matrices from molecular dynamics trajectories with RBF kernel.

    This is a nonlinear dimensionality reduction method that maps the data
    to a higher-dimensional space via a kernel function and then applies
    PCA in that space.

    Examples:
    ---------
    >>> # Basic KernelPCA decomposition via DecompositionManager
    >>> from mdxplain.decomposition import decomposition_type
    >>> decomp_manager = DecompositionManager()
    >>> decomp_manager.add(
    ...     traj_data, "feature_selection", decomposition_type.KernelPCA(n_components=10, gamma=0.1)
    ... )

    >>> # KernelPCA with incremental computation for large datasets
    >>> kpca = decomposition_type.KernelPCA()
    >>> kpca.init_calculator(use_memmap=True, chunk_size=1000)
    >>> transformed, metadata = kpca.compute(large_data, n_components=50)

    >>> # KernelPCA with Nyström approximation for very large datasets
    >>> kpca = decomposition_type.KernelPCA(use_nystrom=True, n_landmarks=5000)
    >>> kpca.init_calculator()
    >>> transformed, metadata = kpca.compute(very_large_data, n_components=50)
    """

    def __init__(
        self,
        n_components=None,
        gamma=None,
        use_nystrom=False,
        n_landmarks=10000,
        random_state=None,
    ):
        """
        Initialize KernelPCA decomposition type with RBF kernel.

        Creates a KernelPCA instance that always uses RBF kernel with
        specified parameters.

        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If None, keeps min(n_samples, n_features)
        gamma : float, optional
            RBF kernel coefficient. If None, uses 1.0 / n_features
        use_nystrom : bool, default=False
            Whether to use Nyström approximation for large datasets
        n_landmarks : int, default=10000
            Number of landmarks for Nyström approximation
        random_state : int, optional
            Random state for reproducible results

        Returns:
        --------
        None

        Examples:
        ---------
        >>> # Create KernelPCA instance with RBF kernel
        >>> kpca = KernelPCA(n_components=10, gamma=0.1)
        >>> print(f"Type: {kpca.get_type_name()}")
        'kernel_pca'

        >>> # Create KernelPCA with Nyström approximation
        >>> kpca = KernelPCA(n_components=50, use_nystrom=True, n_landmarks=5000)
        """
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.use_nystrom = use_nystrom
        self.n_landmarks = n_landmarks
        self.random_state = random_state

    @classmethod
    def get_type_name(cls):
        """
        Get the type name for KernelPCA decomposition.

        Returns the unique string identifier for KernelPCA decomposition type
        used for storing results and type identification.

        Parameters:
        -----------
        cls : type
            The KernelPCA class

        Returns:
        --------
        str
            String identifier 'kernel_pca'

        Examples:
        ---------
        >>> print(KernelPCA.get_type_name())
        'kernel_pca'
        >>> # Can also be used via class directly
        >>> print(decomposition_type.KernelPCA.get_type_name())
        'kernel_pca'
        """
        return "kernel_pca"

    def init_calculator(self, use_memmap=False, cache_path="./cache", chunk_size=10000):
        """
        Initialize the KernelPCA calculator with specified configuration.

        Sets up the KernelPCA calculator with options for memory mapping and
        incremental kernel computation for large datasets.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use incremental kernel computation for large datasets
        cache_path : str, optional
            Path for cache files when using memory mapping
        chunk_size : int, optional
            Number of samples to process per chunk for incremental computation

        Returns:
        --------
        None
            Sets self.calculator to initialized KernelPCACalculator instance

        Examples:
        ---------
        >>> # Basic initialization
        >>> kpca = KernelPCA()
        >>> kpca.init_calculator()

        >>> # With incremental computation for large datasets
        >>> kpca.init_calculator(use_memmap=True, chunk_size=1000)

        >>> # With custom cache path
        >>> kpca.init_calculator(
        ...     use_memmap=True,
        ...     cache_path="./cache/kernel_pca.dat"
        ... )
        """
        self.calculator = KernelPCACalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, data) -> Tuple[np.ndarray, Dict]:
        """
        Compute KernelPCA decomposition of input data using RBF kernel.

        Performs Kernel Principal Component Analysis on the input feature matrix
        using the initialized calculator with RBF kernel and the parameters
        provided during initialization.

        Parameters:
        -----------
        data : numpy.ndarray
            Input feature matrix to decompose, shape (n_samples, n_features)

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - transformed_data: KernelPCA-transformed data matrix (n_samples, n_components)
            - metadata: Dictionary with KernelPCA information including:
              * hyperparameters: Used parameters
              * method: 'standard_kernel_pca' or 'incremental_kernel_pca'
              * optional: n_landmarks: Number of landmarks used in Nyström approximation


        Examples:
        ---------
        >>> # Compute KernelPCA with RBF kernel
        >>> kpca = KernelPCA(n_components=10, gamma=0.1)
        >>> kpca.init_calculator()
        >>> data = np.random.rand(1000, 100)
        >>> transformed, metadata = kpca.compute(data)
        >>> print(f"Transformed shape: {transformed.shape}")
        >>> print(f"Kernel: {metadata['hyperparameters']['kernel']}")
        'rbf'

        >>> # Incremental KernelPCA for large datasets
        >>> kpca = KernelPCA(n_components=50, gamma=0.01)
        >>> kpca.init_calculator(use_memmap=True, chunk_size=200)
        >>> large_data = np.random.rand(10000, 500)
        >>> transformed, metadata = kpca.compute(large_data)

        Raises:
        -------
        ValueError
            If calculator is not initialized, input data is invalid,
            or n_components is too large
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        return self.calculator.compute(
            data,
            n_components=self.n_components,
            gamma=self.gamma,
            use_nystrom=self.use_nystrom,
            n_landmarks=self.n_landmarks,
            random_state=self.random_state,
        )
