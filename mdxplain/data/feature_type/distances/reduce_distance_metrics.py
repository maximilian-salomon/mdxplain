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
Statistical metrics for distance feature analysis and filtering.

Provides metric identifiers for statistical analysis and filtering
of distance features based on variability and transition patterns.
"""


class ReduceDistanceMetrics:
    """
    Available statistical metrics for distance feature analysis and filtering.

    Provides metric identifiers for filtering of distance features.

    Examples:
    ---------
    >>> metrics = ReduceDistanceMetrics()
    >>> print(metrics.CV)         # 'cv'
    >>> print(metrics.STD)        # 'std'
    >>> print(metrics.TRANSITIONS) # 'transitions'
    """

    @property
    def CV(self) -> str:
        """
        Coefficient of variation metric - relative variability (std/mean).

        Returns:
        --------
        str
            Metric identifier 'cv' for coefficient of variation analysis
        """
        return "cv"

    @property
    def STD(self) -> str:
        """
        Standard deviation metric - absolute variability measure.

        Returns:
        --------
        str
            Metric identifier 'std' for standard deviation analysis
        """
        return "std"

    @property
    def VARIANCE(self) -> str:
        """
        Variance metric - squared standard deviation.

        Returns:
        --------
        str
            Metric identifier 'variance' for variance analysis
        """
        return "variance"

    @property
    def RANGE(self) -> str:
        """
        Range metric - difference between maximum and minimum values.

        Returns:
        --------
        str
            Metric identifier 'range' for range analysis
        """
        return "range"

    @property
    def TRANSITIONS(self) -> str:
        """
        Transition analysis metric - detects conformational changes over time.

        Returns:
        --------
        str
            Metric identifier 'transitions' for transition analysis
        """
        return "transitions"
