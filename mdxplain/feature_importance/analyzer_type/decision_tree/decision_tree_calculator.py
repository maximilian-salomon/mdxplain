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
Decision tree calculator for feature importance analysis.

This module implements the actual Decision Tree classifier computation
using scikit-learn, following the calculator pattern similar to decomposition.
"""

from typing import Dict, Any
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ..interfaces.calculator_base import CalculatorBase


class DecisionTreeCalculator(CalculatorBase):
    """
    Calculator for Decision Tree feature importance analysis.

    Implements the actual Decision Tree classifier using scikit-learn,
    to compute feature importance scores. Handles training, prediction,
    and importance extraction.

    Examples
    --------
    >>> calculator = DecisionTreeCalculator()
    >>> X = np.random.rand(1000, 50)
    >>> y = np.random.choice([0, 1], 1000)
    >>> result = calculator.compute(X, y, max_depth=5, random_state=42)
    >>> importance_scores = result['importances']
    >>> trained_model = result['model']
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000, max_memory_gb: float = 6.0):
        """
        Initialize Decision Tree calculator.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, default="./cache"
            Path for cache files (reserved for future use)
        chunk_size : int, default=10000
            Chunk size for processing large datasets (reserved for future use)
        max_memory_gb : float, default=6.0
            Maximum memory in GB for dataset processing.
            Datasets exceeding this limit will be stratified sampled
            to prevent memory errors during DecisionTree training.

        Returns
        -------
        None
            Initializes DecisionTreeCalculator instance
        """
        super().__init__(use_memmap, cache_path, chunk_size, max_memory_gb)

    @staticmethod
    def _validate_input_data(X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate input data for Decision Tree training.

        Checks data dimensions, shapes, and presence of NaN values.
        Raises clear error messages for invalid data.

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
        >>> DecisionTreeCalculator._validate_input_data(X, y)  # Passes
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values")

    @staticmethod
    def _calculate_max_samples(X: np.ndarray, max_memory_gb: float) -> int:
        """
        Calculate maximum samples based on memory limit.

        Estimates the number of samples that fit within the memory limit
        based on feature count and float64 precision (8 bytes per value).

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
        >>> max_samples = DecisionTreeCalculator._calculate_max_samples(X, 6.0)
        >>> print(f"Can use {max_samples} samples")
        """
        n_samples, n_features = X.shape
        bytes_per_sample = n_features * X.dtype.itemsize
        samples_per_gb = (1024**3) / bytes_per_sample
        max_samples = int(samples_per_gb * max_memory_gb)
        return min(max_samples, n_samples)

    @staticmethod
    def _apply_stratified_sampling(
        X: np.ndarray, y: np.ndarray, max_samples: int, random_state: int = None
    ) -> tuple:
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
        tuple
            Tuple of (X_sampled, y_sampled) arrays

        Examples
        --------
        >>> X = np.random.rand(100000, 100)
        >>> y = np.random.choice([0, 1], 100000)
        >>> X_sample, y_sample = DecisionTreeCalculator._apply_stratified_sampling(
        ...     X, y, 50000, random_state=42
        ... )
        >>> print(f"Reduced from {X.shape[0]} to {X_sample.shape[0]} samples")
        """
        if X.shape[0] <= max_samples:
            return X, y

        # Calculate memory usage
        bytes_used = X.shape[0] * X.shape[1] * X.dtype.itemsize
        gb_used = bytes_used / (1024**3)

        print(f"\n  ⚠️  WARNING: Dataset exceeds memory limit!")
        print(f"  Dataset: {X.shape[0]:,} samples × {X.shape[1]:,} features ({X.dtype.name} = {gb_used:.2f} GB)")
        print(f"  Memory limit allows: {max_samples:,} samples")
        print(f"  Applying stratified sampling to fit memory constraint...\n")
        print(f"  💡 TIP: Reduce memory usage AND improve performance with Feature Reduction:")
        print(f"      1. Pre-Selection: pipeline.feature.reduce.distances.cv(threshold_min=0.1)")
        print(f"      2. Post-Selection: pipeline.feature_selector.add.distances.with_cv_reduction(...)")
        print(f"      Available metrics: cv, std, variance, range, transitions, min, max, mean, mad\n")

        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=random_state
        )

        return X_sample, y_sample

    @staticmethod
    def _build_training_metadata(
        dt_classifier: DecisionTreeClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dt_params: Dict[str, Any],
        X_original: np.ndarray,
        max_samples: int,
        max_memory_gb: float
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary with training metrics and model info.

        Calculates training accuracy, classification report, and collects
        model statistics for comprehensive analysis metadata including
        sampling information.

        Parameters
        ----------
        dt_classifier : DecisionTreeClassifier
            Trained Decision Tree classifier
        X_train : np.ndarray
            Feature matrix used for training (potentially sampled)
        y_train : np.ndarray
            Target labels used for training (potentially sampled)
        dt_params : Dict[str, Any]
            Dictionary of Decision Tree hyperparameters
        X_original : np.ndarray
            Original feature matrix before sampling
        max_samples : int
            Maximum samples allowed by memory constraint
        max_memory_gb : float
            Maximum memory in GB

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with training metrics and model info

        Examples
        --------
        >>> metadata = DecisionTreeCalculator._build_training_metadata(
        ...     classifier, X_train, y_train, params, X_orig, 50000, 6.0
        ... )
        >>> print(metadata["train_accuracy"])
        >>> print(metadata["sampling"]["sampled"])
        """
        # Calculate training metrics
        y_pred = dt_classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)

        # Get classification report as dict
        try:
            class_report = classification_report(y_train, y_pred, output_dict=True, zero_division=0)
        except Exception:
            # Fallback if classification_report fails
            class_report = {"accuracy": train_accuracy}

        # Build comprehensive metadata
        return {
            'algorithm': 'decision_tree',
            'hyperparameters': dt_params,
            'train_accuracy': train_accuracy,
            'classification_report': class_report,
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'tree_depth': dt_classifier.get_depth(),
            'tree_n_leaves': dt_classifier.get_n_leaves(),
            'sampling': {
                'original_samples': X_original.shape[0],
                'used_samples': X_train.shape[0],
                'sampled': X_original.shape[0] > max_samples,
                'max_memory_gb': max_memory_gb
            }
        }

    def compute(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute feature importance using Decision Tree classifier.

        Trains a Decision Tree classifier on the provided data and extracts
        feature importance scores based on impurity reduction at each split.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        y : np.ndarray
            Target labels with shape (n_samples,)
        kwargs : dict
            Decision Tree parameters (criterion, max_depth, random_state, etc.)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            
            - 'importances': np.ndarray of feature importance scores
            - 'model': Trained DecisionTreeClassifier instance
            - 'metadata': Dict with training metrics and parameters

        Raises
        ------
        ValueError
            If input data has invalid shape or contains NaN values

        Examples
        --------
        >>> result = calculator.compute(X, y, max_depth=5, random_state=42)
        >>> importance_scores = result['importances']
        >>> accuracy = result['metadata']['train_accuracy']
        """
        # Validate input data using helper
        self._validate_input_data(X, y)

        # Extract Decision Tree parameters
        dt_params = {
            'criterion': kwargs.get('criterion', 'gini'),
            'splitter': kwargs.get('splitter', 'best'),
            'max_depth': kwargs.get('max_depth', None),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'min_weight_fraction_leaf': kwargs.get('min_weight_fraction_leaf', 0.0),
            'max_features': kwargs.get('max_features', None),
            'random_state': kwargs.get('random_state', None),
            'max_leaf_nodes': kwargs.get('max_leaf_nodes', None),
            'min_impurity_decrease': kwargs.get('min_impurity_decrease', 0.0),
            'class_weight': kwargs.get('class_weight', None),
            'ccp_alpha': kwargs.get('ccp_alpha', 0.0),
        }

        # Determine max_samples: user override or auto-calculated
        user_max_samples = kwargs.get('max_samples', None)
        if user_max_samples is not None:
            # User explicitly set max_samples -> use it directly
            max_samples = user_max_samples
        else:
            # Auto-calculate based on memory limit
            max_samples = self._calculate_max_samples(X, self.max_memory_gb)

        # Apply stratified sampling if needed
        X_train, y_train = self._apply_stratified_sampling(
            X, y, max_samples, dt_params['random_state']
        )

        # Create and train Decision Tree classifier on potentially sampled data
        dt_classifier = DecisionTreeClassifier(**dt_params)
        dt_classifier.fit(X_train, y_train)

        # Build training metadata using helper (includes sampling info)
        metadata = self._build_training_metadata(
            dt_classifier, X_train, y_train, dt_params, X, max_samples, self.max_memory_gb
        )

        return {
            'importances': dt_classifier.feature_importances_,
            'model': dt_classifier,
            'metadata': metadata
        }
