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
Base class for feature importance analyzer types.

This module provides the abstract base class for all feature importance
analyzers, following the same pattern as DecompositionTypeBase.
All analyzer implementations must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from .analyzer_type_meta import AnalyzerTypeMeta


class AnalyzerTypeBase(ABC, metaclass=AnalyzerTypeMeta):
    """
    Abstract base class for feature importance analyzer types.

    This class defines the interface that all feature importance analyzers
    must implement. It follows the same pattern as DecompositionTypeBase,
    providing a consistent interface for different ML algorithms.

    All analyzer types must implement:
    - compute(): Main analysis method
    - get_type_name(): Unique identifier string
    - get_params(): Parameter dictionary

    Examples
    --------
    >>> class CustomAnalyzer(AnalyzerTypeBase):
    ...     def compute(self, X, y):
    ...         # Implementation here
    ...         pass
    ...     
    ...     @classmethod
    ...     def get_type_name(cls) -> str:
    ...         return "custom"
    """

    def __init__(self) -> None:
        """
        Initialize analyzer type base class.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Initializes base analyzer type
        """
        self.calculator = None

    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute feature importance from data and labels.

        This is the main method that must be implemented by all analyzer types.
        It should train the ML model and return feature importance scores
        along with the trained model and metadata.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        y : np.ndarray
            Target labels with shape (n_samples,)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'importances': np.ndarray of feature importance scores
            - 'model': Trained ML model object
            - 'metadata': Dict with additional information

        Examples
        --------
        >>> result = analyzer.compute(X, y)
        >>> importance_scores = result['importances']
        >>> trained_model = result['model']
        >>> analysis_metadata = result['metadata']
        """
        pass

    @classmethod
    @abstractmethod
    def get_type_name(cls) -> str:
        """
        Get the unique type name for this analyzer.

        Returns the string identifier used to identify this analyzer type
        in the pipeline system. This name is used for storage and retrieval.

        Parameters
        ----------
        None but cls : Type[AnalyzerTypeBase]
            The class itself

        Returns
        -------
        str
            Unique string identifier for this analyzer type

        Examples
        --------
        >>> print(DecisionTree.get_type_name())
        'decision_tree'
        """
        pass

    @abstractmethod
    def init_calculator(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize the analyzer calculator with specified configuration.

        Sets up the analyzer calculator with options for memory mapping and
        chunk processing for large datasets.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, default="./cache"
            Path for cache files
        chunk_size : int, default=10000
            Number of samples to process per chunk

        Returns
        -------
        None
            Sets self.calculator to initialized calculator instance

        Examples
        --------
        >>> analyzer = DecisionTree()
        >>> analyzer.init_calculator(use_memmap=True, chunk_size=1000)
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters used by this analyzer.

        Returns a dictionary of all parameters used by this analyzer instance.
        This is used for metadata storage and reproducibility.

        Parameters
        ----------
        None but self : AnalyzerTypeBase
            The analyzer instance

        Returns
        -------
        Dict[str, Any]
            Dictionary of analyzer parameters

        Examples
        --------
        >>> params = analyzer.get_params()
        >>> print(f"Max depth: {params.get('max_depth', 'unlimited')}")
        """
        # Default implementation returns empty dict
        # Subclasses should override to return actual parameters
        return {}
