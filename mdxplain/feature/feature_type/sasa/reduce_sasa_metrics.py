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
Reduce metrics for SASA feature type.

Statistical metrics and reduction methods for SASA data analysis
including surface area dynamics, burial patterns, and solvent exposure.
"""

from enum import Enum


class ReduceSASAMetrics(Enum):
    """
    Available reduction metrics for SASA feature analysis.

    Statistical metrics for analyzing SASA data including surface area
    variability, burial dynamics, and solvent accessibility patterns.

    Examples:
    ---------
    >>> # Standard deviation of SASA values
    >>> metric = ReduceSASAMetrics.STD

    >>> # Burial fraction below threshold
    >>> metric = ReduceSASAMetrics.BURIAL_FRACTION

    >>> # Dynamic range of surface area
    >>> metric = ReduceSASAMetrics.RANGE
    """

    STD = "std"
    """Standard deviation of SASA across frames"""

    VARIANCE = "variance"
    """Variance of SASA across frames"""

    CV = "cv"
    """Coefficient of variation (std/mean) for SASA"""

    RANGE = "range"
    """Range (max - min) of SASA values"""

    MAD = "mad"
    """Median absolute deviation of SASA"""

    MEAN = "mean"
    """Mean SASA for each residue/atom"""

    MIN = "min"
    """Minimum SASA for each residue/atom"""

    MAX = "max"
    """Maximum SASA for each residue/atom"""

    BURIAL_FRACTION = "burial_fraction"
    """Fraction of time below burial threshold"""

    EXPOSURE_FRACTION = "exposure_fraction"
    """Fraction of time above exposure threshold"""

    TRANSITIONS = "transitions"
    """Number of SASA transitions exceeding threshold"""