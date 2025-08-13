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
Metaclass for cluster types to enable string representation of classes.

Allows cluster_type.DBSCAN to return "dbscan" directly
instead of requiring instantiation or method calls.
"""

from abc import ABCMeta


class ClusterTypeMeta(ABCMeta):
    """
    Metaclass for cluster types to enable string representation of classes.

    This allows cluster_type.DBSCAN to return "dbscan" directly
    instead of requiring instantiation or method calls.

    Inherits from ABCMeta to be compatible with ABC classes.
    """

    def __repr__(cls):
        """
        Return string representation of the cluster type class.

        Parameters:
        -----------
        cls : type
            The cluster type class

        Returns:
        --------
        str
            String representation of the cluster type class
        """
        return cls.get_type_name()

    def __str__(cls):
        """
        Return string representation for use as dict keys, etc.

        Parameters:
        -----------
        cls : type
            The cluster type class

        Returns:
        --------
        str
            String representation of the cluster type class
        """
        return cls.get_type_name()

    def __hash__(cls):
        """
        Make classes hashable for use as dict keys.

        Parameters:
        -----------
        cls : type
            The cluster type class

        Returns:
        --------
        int
            Hash value of the cluster type class
        """
        return hash(cls.get_type_name())
