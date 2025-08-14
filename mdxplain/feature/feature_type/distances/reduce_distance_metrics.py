# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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
    >>> print(ReduceDistanceMetrics.CV)         # 'cv'
    >>> print(ReduceDistanceMetrics.STD)        # 'std'
    >>> print(ReduceDistanceMetrics.TRANSITIONS) # 'transitions'
    """

    CV = "cv"
    """Coefficient of variation metric - relative variability (std/mean)."""

    STD = "std"
    """Standard deviation metric - absolute variability measure."""

    VARIANCE = "variance"
    """Variance metric - squared standard deviation."""

    RANGE = "range"
    """Range metric - difference between maximum and minimum values."""

    TRANSITIONS = "transitions"
    """Transition analysis metric - detects conformational changes over time."""

    MIN = "min"
    """Minimum distance metric - minimum distance value."""

    MAD = "mad"
    """Median absolute deviation metric - median absolute deviation."""

    MEAN = "mean"
    """Mean distance metric - average distance value."""
