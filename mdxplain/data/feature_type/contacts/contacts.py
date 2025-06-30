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
Contact feature type implementation for molecular dynamics analysis.

Contact feature type implementation with distance-based contact detection
for analyzing molecular dynamics trajectories.
"""

from typing import List

from ..distances.distances import Distances
from ..interfaces.feature_type_base import FeatureTypeBase
from .contact_calculator import ContactCalculator
from .reduce_contact_metrics import ReduceContactMetrics


class Contacts(FeatureTypeBase):
    """
    Contact feature type for detecting atomic/residue contacts based on distance cutoffs.

    Converts distance data into binary contact maps by applying distance cutoffs.
    A contact is defined as a pair being within the specified cutoff distance.
    Depends on distance features as input data.

    This feature type enables analysis of contact formation/breaking patterns,
    contact frequencies, and structural stability through contact persistence.

    Examples:
    ---------
    >>> # Basic contact calculation with default cutoff
    >>> contacts = Contacts()
    >>> contacts.init_calculator()
    >>> contact_data, names = contacts.compute(distance_data, feature_names)

    >>> # Contact calculation with custom cutoff
    >>> contacts = Contacts(cutoff=3.5)  # 3.5 Angstrom cutoff
    >>> contacts.init_calculator()
    >>> contact_data, names = contacts.compute(distance_data, feature_names)

    >>> # With memory mapping for large datasets
    >>> contacts = Contacts(cutoff=4.0)
    >>> contacts.init_calculator(use_memmap=True, cache_path='./cache/')
    >>> contact_data, names = contacts.compute(distance_data, feature_names)
    """

    ReduceMetrics = ReduceContactMetrics
    """Available reduce metrics for contact features."""

    def __init__(self, cutoff=4.5):
        """
        Initialize contact feature type with distance cutoff parameter.

        Parameters:
        -----------
        cutoff : float, default=4.5
            Distance cutoff in Angstrom for contact determination. Pairs with
            distances <= cutoff are considered in contact (1), others not (0).

        Examples:
        ---------
        >>> # Default cutoff (4.5 Angstrom)
        >>> contacts = Contacts()

        >>> # Custom cutoff for closer contacts
        >>> contacts = Contacts(cutoff=3.5)

        >>> # Longer range contacts
        >>> contacts = Contacts(cutoff=6.0)
        """
        super().__init__()
        self.cutoff = cutoff

    def init_calculator(self, use_memmap=False, cache_path=None, chunk_size=None):
        """
        Initialize the contact calculator with specified configuration.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files when using memory mapping
        chunk_size : int, optional
            Number of frames to process per chunk (None for automatic sizing)

        Examples:
        ---------
        >>> # Basic initialization
        >>> contacts.init_calculator()

        >>> # With memory mapping for large datasets
        >>> contacts.init_calculator(use_memmap=True, cache_path='./cache/')

        >>> # With custom chunk size
        >>> contacts.init_calculator(chunk_size=1000)
        """
        self.calculator = ContactCalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, input_data, feature_names, labels=None):
        """
        Compute binary contact maps from distance data using distance cutoff.

        Parameters:
        -----------
        input_data : numpy.ndarray
            Distance matrix data from distance feature type (n_frames, n_pairs)
        feature_names : list
            Feature names from distance calculations (pair identifiers)
        labels : list, optional
            Residue labels (not used by contacts - uses existing feature_names)

        Returns:
        --------
        tuple[numpy.ndarray, list]
            Tuple containing (contact_matrix, feature_names) where contact_matrix
            is binary (0/1) indicating contact presence and feature_names are
            the same pair identifiers from input

        Examples:
        ---------
        >>> # Compute contacts from distance data
        >>> contacts = Contacts(cutoff=4.0)
        >>> contacts.init_calculator()
        >>> contact_data, names = contacts.compute(distance_data, pair_names)
        >>> print(f"Contact matrix shape: {contact_data.shape}")
        >>> print(f"Contact frequency: {contact_data.mean():.3f}")
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )
        return (
            self.calculator.compute(
                input_data=input_data,
                cutoff=self.cutoff,
            ),
            feature_names,
        )

    def get_dependencies(self) -> List[str]:
        """
        Get list of feature type dependencies for contact calculations.

        Returns:
        --------
        List[str]
            List containing 'distances' as contacts require distance data

        Examples:
        ---------
        >>> contacts = Contacts()
        >>> print(contacts.get_dependencies())
        ['distances']
        """
        return [Distances.get_type_name()]

    @classmethod
    def get_type_name(cls):  # noqa: vulture
        """
        Return unique string identifier for the contact feature type.

        Returns:
        --------
        str
            String identifier 'contacts' used as key in feature dictionaries

        Examples:
        ---------
        >>> print(Contacts.get_type_name())
        'contacts'
        """
        return "contacts"

    def get_input(self):
        """
        Get the primary input feature type that contacts depend on.

        Returns:
        --------
        str
            String identifier 'distances' indicating contacts use distance data

        Examples:
        ---------
        >>> contacts = Contacts()
        >>> print(contacts.get_input())
        'distances'
        """
        return Distances.get_type_name()
