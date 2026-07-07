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

"""
Abstract base class for feature importance calculators.

Defines the interface that all feature importance calculators must implement
for consistency across different ML algorithms. Also provides shared helpers
(input validation, memory-based stratified subsampling) and a BLAS/OpenMP
thread-limiting policy for parallel estimators such as Random Forest.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


class CalculatorBase(ABC):
    """
    Abstract base class for feature importance calculators.

    Defines the interface that all feature importance calculators (e.g. DecisionTree,
    RandomForest, SVM) must implement for consistency across different
    ML algorithms used in feature importance analysis.

    In addition to the abstract ``compute`` contract, this base class provides
    shared helpers used by the concrete calculators:

    - Input validation (``_validate_input_data``)
    - Memory-based stratified subsampling (``_calculate_max_samples``,
      ``_apply_stratified_sampling``)
    - BLAS/OpenMP thread limiting for parallel estimators
      (``_limit_threadpools``)

    Examples
    --------
    >>> class MyCalculator(CalculatorBase):
    ...     def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000, max_memory_gb: float = 6.0):
    ...         super().__init__(use_memmap, cache_path, chunk_size, max_memory_gb)
    ...     def compute(self, X, y, **kwargs):
    ...         # Implement ML algorithm logic
    ...         return result_dict
    """

    def __init__(
        self,
        use_memmap: bool = False,
        cache_path: str = "./cache",
        chunk_size: int = 2000,
        max_memory_gb: float = 6.0,
        max_blas_threads: Optional[int] = 1,
        auto_limit_blas: bool = True,
    ):
        """
        Initialize the calculator with configuration options.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, default="./cache"
            Path for cache files (for future use with large models)
        chunk_size : int, default=2000
            Chunk size for processing large datasets
        max_memory_gb : float, default=6.0
            Maximum memory in GB for dataset processing.
            Datasets exceeding this limit will be automatically sampled
            to prevent memory errors during model training.
        max_blas_threads : int or None, default=1
            Preferred BLAS/OpenMP thread limit; set auto_limit_blas=False to
            disable thread limiting, or None to fall back to a safe default.
        auto_limit_blas : bool, default=True
            Apply a safe thread policy: use BLAS=1 when n_jobs != 1,
            otherwise use max_blas_threads (fallback 2 when None). This avoids
            thread oversubscription for parallel estimators such as Random
            Forest.

        Returns
        -------
        None
            Initializes calculator with given configuration
        """
        self.use_memmap = use_memmap
        self.cache_path = cache_path
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.max_blas_threads = max_blas_threads
        self.auto_limit_blas = auto_limit_blas

    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute feature importance from features and labels.

        This is the main method that must be implemented by all calculator types.
        It should train the ML model and return feature importance scores
        along with the trained model and metadata.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        y : np.ndarray
            Target labels with shape (n_samples,)
        kwargs : dict
            Additional keyword arguments specific to the ML algorithm

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:

            - 'importances': np.ndarray of feature importance scores
            - 'model': Trained ML model object
            - 'metadata': Dict with additional information (scores, parameters, etc.)

        Examples
        --------
        >>> result = calculator.compute(X, y, max_depth=5, random_state=42)
        >>> importance_scores = result['importances']
        >>> trained_model = result['model']
        >>> analysis_metadata = result['metadata']
        """
        pass

    @staticmethod
    def _uses_parallel_jobs(n_jobs: Optional[int]) -> bool:
        """
        Determine whether parallel jobs are requested.

        Parameters
        ----------
        n_jobs : int or None
            Number of requested parallel jobs

        Returns
        -------
        bool
            True if n_jobs requests more than one worker

        Examples
        --------
        >>> CalculatorBase._uses_parallel_jobs(-1)
        True
        >>> CalculatorBase._uses_parallel_jobs(1)
        False
        """
        if n_jobs is None:
            return False
        return n_jobs != 1

    def _resolve_blas_limit(self, n_jobs: Optional[int]) -> Optional[int]:
        """
        Resolve the effective BLAS/OpenMP thread limit for a fit.

        Applies the configured thread policy: with auto_limit_blas enabled,
        parallel estimators (n_jobs != 1) use a single BLAS thread to avoid
        oversubscription, otherwise the preferred max_blas_threads is used.

        Parameters
        ----------
        n_jobs : int or None
            Number of requested parallel jobs

        Returns
        -------
        int or None
            The BLAS thread limit to apply, or None for no limiting

        Examples
        --------
        >>> calc = MyCalculator()
        >>> calc._resolve_blas_limit(-1)
        1
        """
        if not self.auto_limit_blas:
            return self.max_blas_threads
        if self._uses_parallel_jobs(n_jobs):
            return 1
        if self.max_blas_threads and self.max_blas_threads > 0:
            return self.max_blas_threads
        return 2

    def _limit_threadpools(
        self, n_jobs: Optional[int]
    ) -> AbstractContextManager:
        """
        Create a context manager that limits BLAS/OpenMP threadpools.

        Wrapping a model fit in the returned context manager prevents thread
        oversubscription (parallel workers x BLAS threads) for parallel
        estimators such as Random Forest.

        Parameters
        ----------
        n_jobs : int or None
            Number of requested parallel jobs

        Returns
        -------
        contextlib.AbstractContextManager
            Context manager that enforces thread limits when enabled, or a
            no-op context manager otherwise

        Examples
        --------
        >>> calc = MyCalculator()
        >>> with calc._limit_threadpools(n_jobs=-1):
        ...     model.fit(X, y)
        """
        limit = self._resolve_blas_limit(n_jobs)
        if limit is None or limit < 1:
            return nullcontext()
        return threadpool_limits(limits=limit)

    @staticmethod
    def _validate_input_data(X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate input data for model training.

        Checks data dimensions, shapes, and presence of NaN values, raising
        clear error messages for invalid data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to validate
        y : np.ndarray
            Target labels to validate

        Returns
        -------
        None
            Validation passed

        Raises
        ------
        ValueError
            If input data has invalid shape or contains NaN values

        Examples
        --------
        >>> X = np.random.rand(100, 10)
        >>> y = np.random.choice([0, 1], 100)
        >>> CalculatorBase._validate_input_data(X, y)  # Passes
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )
        CalculatorBase._check_no_nan(X, y)

    @staticmethod
    def _check_no_nan(X: np.ndarray, y: np.ndarray) -> None:
        """
        Raise if the feature matrix or labels contain NaN values.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to check
        y : np.ndarray
            Target labels to check

        Returns
        -------
        None
            No NaN values present

        Raises
        ------
        ValueError
            If input data contains NaN values

        Examples
        --------
        >>> CalculatorBase._check_no_nan(np.zeros((2, 2)), np.zeros(2))
        """
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values")

    @staticmethod
    def _calculate_max_samples(X: np.ndarray, max_memory_gb: float) -> int:
        """
        Calculate maximum samples based on memory limit.

        Estimates the number of samples that fit within the memory limit
        based on feature count and the array dtype size.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        max_memory_gb : float
            Maximum memory in GB

        Returns
        -------
        int
            Maximum number of samples that fit in memory limit

        Examples
        --------
        >>> X = np.random.rand(100000, 1000)
        >>> max_samples = CalculatorBase._calculate_max_samples(X, 6.0)
        >>> print(f"Can use {max_samples} samples")
        """
        n_samples, n_features = X.shape
        bytes_per_sample = n_features * X.dtype.itemsize
        samples_per_gb = (1024**3) / bytes_per_sample
        max_samples = int(samples_per_gb * max_memory_gb)
        return min(max_samples, n_samples)

    @staticmethod
    def _apply_stratified_sampling(
        X: np.ndarray,
        y: np.ndarray,
        max_samples: int,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply stratified sampling if data exceeds memory limit.

        Uses sklearn train_test_split with stratify to preserve class
        distribution while reducing dataset size.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        y : np.ndarray
            Target labels with shape (n_samples,)
        max_samples : int
            Maximum number of samples allowed
        random_state : int, optional
            Random state for reproducible sampling

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (X_sampled, y_sampled) arrays

        Examples
        --------
        >>> X = np.random.rand(100000, 100)
        >>> y = np.random.choice([0, 1], 100000)
        >>> X_s, y_s = CalculatorBase._apply_stratified_sampling(
        ...     X, y, 50000, random_state=42
        ... )
        >>> print(f"Reduced from {X.shape[0]} to {X_s.shape[0]} samples")
        """
        if X.shape[0] <= max_samples:
            return X, y

        CalculatorBase._print_sampling_warning(X, max_samples)
        X_sample, _, y_sample, _ = train_test_split(
            X,
            y,
            train_size=max_samples,
            stratify=y,
            random_state=random_state,
        )
        return X_sample, y_sample

    @staticmethod
    def _print_sampling_warning(X: np.ndarray, max_samples: int) -> None:
        """
        Print a warning describing the applied memory-based subsampling.

        Informs the user that the dataset exceeds the memory limit and how to
        reduce memory usage via feature reduction.

        Parameters
        ----------
        X : np.ndarray
            Original feature matrix before sampling
        max_samples : int
            Maximum number of samples allowed by the memory constraint

        Returns
        -------
        None
            Prints the warning to the console

        Examples
        --------
        >>> X = np.random.rand(100000, 100)
        >>> CalculatorBase._print_sampling_warning(X, 50000)
        """
        bytes_used = X.shape[0] * X.shape[1] * X.dtype.itemsize
        gb_used = bytes_used / (1024**3)

        print("\n  ⚠️  WARNING: Dataset exceeds memory limit!")
        print(
            f"  Dataset: {X.shape[0]:,} samples × {X.shape[1]:,} features "
            f"({X.dtype.name} = {gb_used:.2f} GB)"
        )
        print(f"  Memory limit allows: {max_samples:,} samples")
        print("  Applying stratified sampling to fit memory constraint...\n")
        print(
            "  💡 TIP: Reduce memory usage AND improve performance with "
            "Feature Reduction:"
        )
        print(
            "      1. Pre-Selection: "
            "pipeline.feature.reduce.distances.cv(threshold_min=0.1)"
        )
        print(
            "      2. Post-Selection: "
            "pipeline.feature_selector.add.distances.with_cv_reduction(...)"
        )
        print(
            "      Available metrics: cv, std, variance, range, transitions, "
            "min, max, mean, mad\n"
        )
