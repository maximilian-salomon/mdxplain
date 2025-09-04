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
Reduce metrics for coordinates feature type.

Statistical metrics and reduction methods for coordinate data analysis
including structural variability, mobility, and positional stability metrics.
"""

from enum import Enum


class ReduceCoordinatesMetrics(Enum):
    """
    Available reduction metrics for coordinates feature analysis.

    Statistical metrics for analyzing coordinate data including position
    variability, structural stability, and atomic mobility measures.

    Examples:
    ---------
    >>> # Standard deviation of positions
    >>> metric = ReduceCoordinatesMetrics.STD

    >>> # Root mean square fluctuation
    >>> metric = ReduceCoordinatesMetrics.RMSF

    >>> # Coefficient of variation
    >>> metric = ReduceCoordinatesMetrics.CV
    """

    STD = "std"
    """Standard deviation of coordinates across frames"""

    VARIANCE = "variance" 
    """Variance of coordinates across frames"""

    CV = "cv"
    """Coefficient of variation (std/mean) for coordinates"""

    RANGE = "range"
    """Range (max - min) of coordinates"""

    MAD = "mad"
    """Median absolute deviation of coordinates"""

    MEAN = "mean"
    """Mean position for each coordinate"""

    MIN = "min"
    """Minimum position for each coordinate"""

    MAX = "max"
    """Maximum position for each coordinate"""

    RMSF = "rmsf"
    """Root mean square fluctuation per atom relative to mean position"""

    TRANSITIONS = "transitions"
    """Number of positional transitions exceeding threshold"""