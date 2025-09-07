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
Reduce metrics for torsion angle feature type.

Statistical metrics and reduction methods for torsion angle data including
circular statistics, angular distributions, and conformational analysis.
"""

from enum import Enum


class ReduceTorsionsMetrics(Enum):
    """
    Available reduction metrics for torsion angle feature analysis.

    Statistical metrics for analyzing torsion angle data including circular
    statistics, conformational flexibility, and angular distribution analysis.

    Examples:
    ---------
    >>> # Standard deviation (use with caution for circular data)
    >>> metric = ReduceTorsionsMetrics.STD

    >>> # Circular statistics for proper angular analysis
    >>> metric = ReduceTorsionsMetrics.CIRCULAR_MEAN
    >>> metric = ReduceTorsionsMetrics.CIRCULAR_VARIANCE

    >>> # Angular range analysis
    >>> metric = ReduceTorsionsMetrics.ANGULAR_RANGE
    """

    STD = "std"
    """Standard deviation of torsion angles (use with caution for circular data)"""

    VARIANCE = "variance"
    """Variance of torsion angles (use with caution for circular data)"""

    MEAN = "mean"
    """Mean torsion angle (use with caution for circular data)"""

    MIN = "min"
    """Minimum torsion angle value"""

    MAX = "max"
    """Maximum torsion angle value"""

    MAD = "mad"
    """Median absolute deviation of torsion angles"""

    RANGE = "range"
    """Range accounting for circularity"""

    TRANSITIONS = "transitions"
    """Number of angular transitions exceeding threshold"""
