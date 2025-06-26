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
Abstract base class defining the interface for all feature types.

Defines the interface that all feature types (distances, contacts, angles, etc.)
must implement for consistency across different molecular dynamics features.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class FeatureTypeBase(ABC):
    """
    Abstract base class for all molecular dynamics feature types.

    Defines the interface that all feature types (distances, contacts, angles, etc.)
    must implement. Each feature type encapsulates computation logic and dependency
    management for a specific type of molecular dynamics analysis.

    Examples:
    ---------
    >>> class MyFeature(FeatureTypeBase):
    ...     def get_dependencies(self):
    ...         return ['distances']  # Depends on distance features
    ...     def __str__(self):
    ...         return 'my_feature'
    ...     def init_calculator(self, **kwargs):
    ...         self.calculator = MyCalculator(**kwargs)
    ...     def compute(self, input_data, feature_names):
    ...         return self.calculator.compute(input_data)
    """

    def __init__(self):
        """
        Initialize the feature type.

        Sets up the feature type instance with an empty calculator that will
        be initialized later through init_calculator().
        """
        self.calculator = None

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Get list of feature type dependencies that must be computed first.

        Returns:
        --------
        List[str]
            List of feature type names (e.g., ['distances']) that this feature
            type depends on and must be computed before this feature

        Examples:
        ---------
        >>> # Contacts depend on distances
        >>> contacts = ContactsFeature()
        >>> print(contacts.get_dependencies())
        ['distances']

        >>> # Distances have no dependencies
        >>> distances = DistancesFeature()
        >>> print(distances.get_dependencies())
        []
        """
        pass

    @classmethod
    @abstractmethod
    def get_type_name(cls) -> str:
        """
        Return unique string identifier for this feature type.

        Used as the key for storing features in TrajectoryData dictionaries
        and for dependency resolution.

        Returns:
        --------
        str
            Unique string identifier (e.g., 'distances', 'contacts')
        """
        pass

    @abstractmethod
    def init_calculator(self, use_memmap=False, cache_path=None, chunk_size=None):
        """
        Initialize the calculator instance for this feature type.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for efficient handling of large datasets
        cache_path : str, optional
            Directory path for cache files
        chunk_size : int, optional
            Number of frames to process per chunk

        Returns:
        --------
        None
            Sets self.calculator to initialized calculator instance
        """
        pass

    @abstractmethod
    def compute(
        self, input_data, feature_names, labels=None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute features using the initialized calculator.

        Parameters:
        -----------
        input_data : Any, optional
            Input data for computation (trajectories, distances, etc.)
        feature_names : list, optional
            Names for input features (used by dependent features)
        labels : list, optional
            Residue labels for generating feature names

        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            Tuple containing (computed_features, feature_names) where
            computed_features is the calculated data matrix and
            feature_names is list of feature labels
        """
        pass

    def get_input(self):
        """
        Get the input feature type that this feature depends on.

        Returns:
        --------
        str or None
            Name of the primary input feature type, or None for base features

        Examples:
        ---------
        >>> contacts = ContactsFeature()
        >>> print(contacts.get_input())
        'distances'

        >>> distances = DistancesFeature()
        >>> print(distances.get_input())
        None
        """
        return None
