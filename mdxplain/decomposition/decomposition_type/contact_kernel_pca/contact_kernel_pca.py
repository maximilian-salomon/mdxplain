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
ContactKernelPCA decomposition type for binary contact matrix analysis.

ContactKernelPCA decomposition type implementation specialized for binary
contact matrices using Hamming distance-based kernel.
"""

from typing import Dict, Tuple

import numpy as np

from ..interfaces.decomposition_type_base import DecompositionTypeBase
from .contact_kernel_pca_calculator import ContactKernelPCACalculator


class ContactKernelPCA(DecompositionTypeBase):
    """
    Contact Kernel Principal Component Analysis decomposition type.

    Specialized KernelPCA implementation for binary contact matrices from
    molecular dynamics trajectories. Uses a Hamming distance-based kernel
    that is mathematically equivalent to the RBF kernel for binary data.

    The Hamming kernel computes the distance between binary vectors where
    Hamming distance equals (x-y)² for binary vectors. This makes it
    equivalent to the RBF kernel while being conceptually appropriate
    for binary contact data.

    Examples:
    ---------
    >>> # Basic ContactKernelPCA via DecompositionManager
    >>> from mdxplain.decomposition import decomposition_type
    >>> decomp_manager = DecompositionManager()
    >>> decomp_manager.add(
    ...     traj_data, "contact_selection",
    ...     decomposition_type.ContactKernelPCA(n_components=10, gamma=1.0)
    ... )

    >>> # ContactKernelPCA with incremental computation
    >>> ckpca = decomposition_type.ContactKernelPCA()
    >>> ckpca.init_calculator(use_memmap=True, chunk_size=1000)
    >>> transformed, metadata = ckpca.compute(contact_matrix, n_components=20)
    """

    def __init__(
        self,
        n_components=None,
        gamma=1.0,
        use_nystrom=False,
        n_landmarks=10000,
        random_state=None,
    ):
        """
        Initialize ContactKernelPCA decomposition type for contact data.

        Creates a ContactKernelPCA instance specialized for binary contact
        matrices using Hamming/RBF kernel with specified parameters.

        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If None, keeps min(n_samples, n_features)
        gamma : float, default=1.0
            Kernel coefficient for Hamming/RBF kernel on binary data
        use_nystrom : bool, default=False
            Whether to use Nyström approximation for large datasets
        n_landmarks : int, default=10000
            Number of landmarks for Nyström approximation
        random_state : int, optional
            Random state for reproducible results
            
        Returned Metadata:
        ------------------
        hyperparameters : dict
            Dictionary containing all Contact Kernel PCA parameters used
        original_shape : tuple
            Shape of the input data (n_samples, n_features)
        use_memmap : bool
            Whether memory mapping was used
        chunk_size : int
            Chunk size used for processing
        cache_path : str
            Path used for caching results
        method : str
            Method used ('standard_kernel_pca', 'nystrom_kernel_pca', or 'chunk_wise_kernel_pca')
        approximation : str
            Approximation method used ('nystrom' when Nyström approximation is enabled)
        n_chunks : int
            Number of chunks used for incremental processing
        n_landmarks : int
            Number of landmarks used for Nyström approximation (when applicable)

        Examples:
        ---------
        >>> # Create ContactKernelPCA instance for contact data
        >>> ckpca = ContactKernelPCA(n_components=15, gamma=1.0)
        >>> print(f"Type: {ckpca.get_type_name()}")
        'contact_kernel_pca'

        >>> # Create ContactKernelPCA with Nyström approximation
        >>> ckpca = ContactKernelPCA(n_components=50, use_nystrom=True, n_landmarks=5000)
        """
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.use_nystrom = use_nystrom
        self.n_landmarks = n_landmarks
        self.random_state = random_state
        self.calculator = None

    @classmethod
    def get_type_name(cls):
        """
        Get the type name for ContactKernelPCA decomposition.

        Returns the unique string identifier for ContactKernelPCA decomposition
        type used for storing results and type identification.

        Parameters:
        -----------
        cls : type
            The ContactKernelPCA class

        Returns:
        --------
        str
            String identifier 'contact_kernel_pca'

        Examples:
        ---------
        >>> print(ContactKernelPCA.get_type_name())
        'contact_kernel_pca'
        >>> # Can also be used via class directly
        >>> print(decomposition_type.ContactKernelPCA.get_type_name())
        'contact_kernel_pca'
        """
        return "contact_kernel_pca"

    def init_calculator(self, use_memmap=False, cache_path="./cache", chunk_size=10000):
        """
        Initialize the ContactKernelPCA calculator with specified configuration.

        Sets up the ContactKernelPCA calculator with options for memory mapping
        and incremental kernel computation for large binary contact matrices.

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
            Sets self.calculator to initialized ContactKernelPCACalculator instance

        Examples:
        ---------
        >>> # Basic initialization
        >>> ckpca = ContactKernelPCA()
        >>> ckpca.init_calculator()

        >>> # With incremental computation for large contact matrices
        >>> ckpca.init_calculator(use_memmap=True, chunk_size=1000)

        >>> # With custom cache path
        >>> ckpca.init_calculator(
        ...     use_memmap=True,
        ...     cache_path="./cache/contact_kernel_pca.dat"
        ... )
        """
        self.calculator = ContactKernelPCACalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, data) -> Tuple[np.ndarray, Dict]:
        """
        Compute ContactKernelPCA decomposition of binary contact data.

        Performs specialized Kernel Principal Component Analysis on binary
        contact matrices using the initialized calculator with Hamming/RBF
        kernel and the parameters provided during initialization.

        Parameters:
        -----------
        data : numpy.ndarray
            Binary contact matrix to decompose, shape (n_samples, n_features)
            Values must be 0 or 1 representing contact states

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - transformed_data: ContactKernelPCA-transformed data matrix
              (n_samples, n_components)
            - metadata: Dictionary with ContactKernelPCA information including:
              * hyperparameters: Used parameters (kernel 'rbf' equivalent to Hamming)
              * contact_kernel: True (indicates contact-specific kernel)
              * binary_data: True (indicates binary data validation)
              * method: 'standard_kernel_pca' or 'incremental_kernel_pca'
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
