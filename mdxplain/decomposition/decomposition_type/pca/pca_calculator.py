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
PCA calculator for dimensionality reduction of molecular dynamics data.

Implements PCA computation with support for incremental processing
for large datasets using sklearn's PCA and IncrementalPCA.
"""

from typing import Dict, Tuple, Any

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from ..interfaces.calculator_base import CalculatorBase
from ....utils.data_utils import DataUtils


class PCACalculator(CalculatorBase):
    """
    Calculator for Principal Component Analysis (PCA) decomposition.

    Implements PCA computation with support for both standard and incremental
    processing for large datasets. Uses sklearn's PCA for standard computation
    and sklearn's IncrementalPCA for chunk-wise processing when memory mapping is enabled.

    Examples
    --------
    >>> # Standard PCA computation
    >>> calc = PCACalculator()
    >>> data = np.random.rand(1000, 100)
    >>> transformed, metadata = calc.compute(data, n_components=10)

    >>> # Incremental PCA for large datasets
    >>> calc = PCACalculator(use_memmap=True, chunk_size=200)
    >>> large_data = np.random.rand(10000, 500)
    >>> transformed, metadata = calc.compute(large_data, n_components=50)
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize PCA calculator.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping and incremental computation
        cache_path : str, optional
            Path for memory-mapped cache files (not used for PCA)
        chunk_size : int, optional
            Size of chunks for incremental PCA processing

        Returns
        -------
        None
            Initializes PCA calculator with specified configuration

        Examples
        --------
        >>> # Standard PCA
        >>> calc = PCACalculator()

        >>> # Incremental PCA for large datasets
        >>> calc = PCACalculator(use_memmap=True, chunk_size=1000)
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self._cache_prefix = "pca"

    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute PCA decomposition of input data.

        Performs Principal Component Analysis on the input data matrix,
        using either standard PCA or incremental PCA based on the
        configuration settings.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix to decompose, shape (n_samples, n_features)
        \*\*kwargs : dict
            PCA parameters:

            - n_components : int, optional
                Number of components to keep (default: min(n_samples, n_features))
            - random_state : int, optional
                Random state for reproducible results

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            
            - transformed_data: PCA-transformed data (n_samples, n_components)
            - metadata: Dictionary with PCA information including components,
              explained variance ratio, and hyperparameters

        Examples
        --------
        >>> # Compute PCA with 10 components
        >>> calc = PCACalculator()
        >>> data = np.random.rand(500, 100)
        >>> transformed, metadata = calc.compute(data, n_components=10)
        >>> print(f"Explained variance: {metadata['explained_variance_ratio']}")

        Raises
        ------
        ValueError
            If input data is invalid or n_components is too large
        """
        self._validate_input_data(data)
        hyperparameters = self._extract_hyperparameters(data, kwargs)

        if self.use_memmap:
            return self._compute_incremental_pca(data, hyperparameters)
        else:
            return self._compute_standard_pca(data, hyperparameters)

    def _extract_hyperparameters(self, data: np.ndarray, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate PCA hyperparameters.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for parameter validation
        kwargs : dict
            Input parameters to extract and validate

        Returns
        -------
        dict
            Validated hyperparameters
        """
        n_components = kwargs.get("n_components", data.shape[1])
        random_state = kwargs.get("random_state", None)

        max_components = data.shape[1]
        if n_components is not None and n_components > data.shape[0]:
            raise ValueError(
                f"n_components ({n_components}) cannot be larger than "
                f"min(n_samples, n_features) = {max_components}"
            )

        return {
            "n_components": n_components,
            "random_state": random_state,
        }

    def _compute_standard_pca(self, data: np.ndarray, hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute standard PCA using sklearn.decomposition.PCA.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix
        hyperparameters : dict
            PCA hyperparameters

        Returns
        -------
        tuple
            Tuple of (transformed_data, metadata)
        """
        pca = PCA(
            n_components=hyperparameters["n_components"],
            random_state=hyperparameters["random_state"],
            copy=False,
            whiten=True,
        )

        transformed_data = pca.fit_transform(data)

        metadata = self._prepare_metadata(hyperparameters, data.shape)
        metadata.update(
            {
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "explained_variance": pca.explained_variance_,
                "method": "standard_pca",
            }
        )

        return transformed_data, metadata

    def _compute_incremental_pca(self, data: np.ndarray, hyperparameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute incremental PCA using sklearn.decomposition.IncrementalPCA.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix
        hyperparameters : dict
            PCA hyperparameters

        Returns
        -------
        tuple
            Tuple of (transformed_data, metadata)
        """
        ipca = IncrementalPCA(
            n_components=hyperparameters["n_components"],
            batch_size=self.chunk_size,
            whiten=True,
            copy=False,
        )

        transformed_data = ipca.fit_transform(data)

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
                "explained_variance_ratio": ipca.explained_variance_ratio_,
                "explained_variance": ipca.explained_variance_,
                "method": "incremental_pca",
                "n_chunks": int(np.ceil(data.shape[0] / self.chunk_size)),
            }
        )

        return transformed_data, metadata
