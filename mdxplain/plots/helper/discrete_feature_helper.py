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
Helper for discrete feature data preparation.

Provides data conversion for discrete features to prepare them for
visualization by mapping categorical values to integer positions.
"""

import numpy as np
from typing import Dict


class DiscreteFeatureHelper:
    """
    Helper for discrete feature data preparation.

    Provides data conversion for discrete features to prepare them for
    visualization by mapping categorical values to integer positions.

    Examples
    --------
    >>> # Convert character data to positions
    >>> data = np.array(['H', 'E', 'C', 'H'])
    >>> mapping = {'H': 0, 'E': 1, 'C': 2}
    >>> positions = DiscreteFeatureHelper.prepare_discrete_data(data, mapping)
    >>> print(positions)
    [0 1 2 0]
    """

    @staticmethod
    def prepare_discrete_data(
        data: np.ndarray,
        value_to_position: Dict
    ) -> np.ndarray:
        """
        Convert discrete data to integer positions for plotting.

        Maps categorical data values (integers or characters) to sequential
        integer positions suitable for visualization.

        Parameters
        ----------
        data : np.ndarray
            Original data (integers or character strings)
        value_to_position : dict
            Mapping from data values to plot positions

        Returns
        -------
        np.ndarray
            Data converted to integer positions

        Examples
        --------
        >>> # Integer data (no conversion needed)
        >>> data = np.array([0, 1, 2, 0, 1])
        >>> mapping = {0: 0, 1: 1, 2: 2}
        >>> positions = DiscreteFeatureHelper.prepare_discrete_data(data, mapping)
        >>> print(positions)
        [0 1 2 0 1]

        >>> # Character data (needs conversion)
        >>> data = np.array(['H', 'E', 'C', 'H', 'E'])
        >>> mapping = {'H': 0, 'E': 1, 'C': 2}
        >>> positions = DiscreteFeatureHelper.prepare_discrete_data(data, mapping)
        >>> print(positions)
        [0 1 2 0 1]

        Notes
        -----
        Integer encoding data is already in position format and returned as-is.
        Character encoding data is mapped through value_to_position dictionary.
        """
        if data.dtype.kind in ['U', 'S']:  # String/char data
            # Map characters to integer positions
            return np.array([value_to_position.get(v, 0) for v in data])
        else:  # Integer data
            # Already in position format
            return data.astype(int)
