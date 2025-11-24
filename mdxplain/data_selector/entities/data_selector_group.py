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
Data selector group entity.

This module provides the DataSelectorGroup class for organizing
multiple data selectors under a common group name.
"""

from typing import List


class DataSelectorGroup:
    """
    Simple collection of DataSelector names under a group name.

    This class serves as a lightweight container to organize
    multiple related data selectors into named groups for easier
    management and reference in comparisons.

    Attributes
    ----------
    name : str
        Name of the group
    selector_names : List[str]
        List of data selector names in this group

    Examples
    --------
    >>> group = DataSelectorGroup("clusters")
    >>> group.selector_names = ["cluster_0", "cluster_1", "cluster_2"]
    >>> print(group.name)
    clusters
    >>> print(len(group.selector_names))
    3
    """

    def __init__(self, name: str):
        """
        Initialize a data selector group.

        Parameters
        ----------
        name : str
            Name for the group

        Returns
        -------
        None
            Initializes DataSelectorGroup instance

        Examples
        --------
        >>> group = DataSelectorGroup("my_group")
        >>> group.name
        'my_group'
        >>> group.selector_names
        []
        """
        self.name = name
        self.selector_names: List[str] = []
