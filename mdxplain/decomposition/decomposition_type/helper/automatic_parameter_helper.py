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

"""Automatic parameter calculation helper for decomposition methods."""

import warnings
from typing import Union

import numpy as np
from kneed import KneeLocator


class AutomaticParameterHelper:
    """
    Helper for automatic parameter calculation in decomposition methods.

    Provides methods for automatic calculation of n_components via elbow
    detection, gamma parameters for kernel methods, and variance computation
    for large memory-mapped datasets.

    Examples
    --------
    >>> # Elbow detection for PCA
    >>> helper = AutomaticParameterHelper()
    >>> variance_ratios = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    >>> n = helper.find_elbow(variance_ratios, max_components=5)

    >>> # Gamma calculation for KernelPCA
    >>> data = np.random.rand(1000, 100)
    >>> gamma = helper.calculate_gamma_scale(data)

    >>> # Variance for large datasets
    >>> large_data = np.memmap('data.dat', dtype='float64', shape=(10000, 500))
    >>> var = helper.compute_variance_chunked(large_data, chunk_size=1000)
    """

    @staticmethod
    def find_elbow(
        values: np.ndarray,
        sensitivity: float = 1.0,
        max_components: int = None,
        offset: Union[int, float] = 0,
    ) -> int:
        """
        Find elbow point in decreasing curve using kneed algorithm.

        Parameters
        ----------
        values : numpy.ndarray
            Decreasing values (variance ratios or eigenvalues)
        sensitivity : float, default=1.0
            Kneed sensitivity parameter (S)
        max_components : int, optional
            Maximum number of components for warning
        offset : int or float, default=0
            Adjustment to the elbow position:

            - int: Direct addition/subtraction of components
              (e.g., -2 selects 2 fewer components, +3 selects 3 more)
            - float: Percentage-based adjustment
              (e.g., -0.5 selects 50% fewer, 0.5 selects 50% more)

        Returns
        -------
        int
            Optimal number of components (1-indexed)

        Examples
        --------
        >>> helper = AutomaticParameterHelper()
        >>> variances = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        >>> n = helper.find_elbow(variances)
        >>> print(f"Optimal: {n}")

        >>> # Select 2 fewer components than elbow
        >>> n = helper.find_elbow(variances, offset=-2)

        >>> # Select 50% fewer components than elbow
        >>> n = helper.find_elbow(variances, offset=-0.5)
        """
        x = np.arange(1, len(values) + 1)
        y = values

        knee_locator = KneeLocator(
            x, y, curve="convex", direction="decreasing", S=sensitivity
        )

        if knee_locator.elbow is None:
            fallback = min(100, len(values) // 10)
            warnings.warn(
                f"No elbow detected. Using fallback: {fallback} components",
                UserWarning,
            )
            return fallback

        n_selected = int(knee_locator.elbow)
        n_selected = AutomaticParameterHelper._apply_offset(n_selected, offset)

        if max_components is not None:
            threshold = int(0.9 * max_components)
            if n_selected >= threshold:
                warnings.warn(
                    f"Selected {n_selected} components is in 90% quantile "
                    f"(>= {threshold}/{max_components}). "
                    "This may indicate issues with data or decomposition.",
                    UserWarning,
                )

        print(f"{n_selected} components are selected, this is the elbow in the scree-plot.")

        return n_selected

    @staticmethod
    def _apply_offset(n_components: int, offset: Union[int, float]) -> int:
        """
        Apply offset adjustment to component count.

        Parameters
        ----------
        n_components : int
            Base number of components
        offset : int or float
            Offset adjustment value

        Returns
        -------
        int
            Adjusted component count (minimum 1)

        Examples
        --------
        >>> helper = AutomaticParameterHelper()
        >>> helper._apply_offset(10, -2)
        8
        >>> helper._apply_offset(10, -0.5)
        5
        """
        if isinstance(offset, int):
            n_components += offset
        elif isinstance(offset, float):
            n_components = int(n_components * (1 + offset))

        return max(1, n_components)

    @staticmethod
    def calculate_gamma_scale(
        data: np.ndarray, use_memmap: bool = False, chunk_size: int = 1000
    ) -> float:
        """
        Calculate gamma parameter using scale method.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix
        use_memmap : bool, default=False
            Whether to use chunked variance calculation
        chunk_size : int, default=1000
            Chunk size for memmap variance calculation

        Returns
        -------
        float
            Gamma parameter value

        Examples
        --------
        >>> helper = AutomaticParameterHelper()
        >>> data = np.random.rand(1000, 50)
        >>> gamma = helper.calculate_gamma_scale(data)
        """
        n_features = data.shape[1]

        if use_memmap:
            variance = AutomaticParameterHelper.compute_variance_chunked(
                data, chunk_size
            )
        else:
            variance = float(data.var())

        return 1.0 / (n_features * variance)

    @staticmethod
    def calculate_gamma_auto(n_features: int) -> float:
        """
        Calculate gamma parameter using auto method.

        Parameters
        ----------
        n_features : int
            Number of features in the dataset

        Returns
        -------
        float
            Gamma parameter value

        Examples
        --------
        >>> helper = AutomaticParameterHelper()
        >>> gamma = helper.calculate_gamma_auto(100)
        >>> print(gamma)
        0.01
        """
        return 1.0 / n_features

    @staticmethod
    def compute_variance_chunked(
        data: np.ndarray, chunk_size: int = 1000
    ) -> float:
        """
        Compute mean variance using chunk-wise processing.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix
        chunk_size : int, default=1000
            Number of samples to process per chunk

        Returns
        -------
        float
            Mean variance across all features

        Examples
        --------
        >>> helper = AutomaticParameterHelper()
        >>> data = np.random.rand(10000, 100)
        >>> var = helper.compute_variance_chunked(data, chunk_size=1000)
        """
        n_samples, n_features = data.shape
        mean = np.zeros(n_features, dtype=np.float64)

        for i in range(0, n_samples, chunk_size):
            chunk = data[i : i + chunk_size]
            mean += chunk.sum(axis=0)
        mean /= n_samples

        variance = np.zeros(n_features, dtype=np.float64)
        for i in range(0, n_samples, chunk_size):
            chunk = data[i : i + chunk_size]
            variance += ((chunk - mean) ** 2).sum(axis=0)
        variance /= n_samples

        return float(variance.mean())
