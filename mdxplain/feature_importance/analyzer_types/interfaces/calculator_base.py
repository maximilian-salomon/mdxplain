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
for consistency across different ML algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class CalculatorBase(ABC):
    """
    Abstract base class for feature importance calculators.

    Defines the interface that all feature importance calculators (e.g. DecisionTree,
    RandomForest, SVM) must implement for consistency across different
    ML algorithms used in feature importance analysis.

    Examples
    --------
    >>> class MyCalculator(CalculatorBase):
    ...     def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000):
    ...         super().__init__(use_memmap, cache_path, chunk_size)
    ...     def compute(self, X, y, **kwargs):
    ...         # Implement ML algorithm logic
    ...         return result_dict
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000):
        """
        Initialize the calculator with configuration options.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, default="./cache"
            Path for cache files (for future use with large models)
        chunk_size : int, default=10000
            Chunk size for processing large datasets

        Returns
        -------
        None
            Initializes calculator with given configuration
        """
        self.use_memmap = use_memmap
        self.cache_path = cache_path
        self.chunk_size = chunk_size

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
        \*\*kwargs : dict
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