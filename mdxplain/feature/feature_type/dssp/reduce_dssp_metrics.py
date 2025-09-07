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
Reduce metrics for DSSP feature type.

Statistical metrics and reduction methods for DSSP secondary structure data
including structural stability, transition dynamics, and class distributions.
"""

from enum import Enum


class ReduceDSSPMetrics(Enum):
    """
    Available reduction metrics for DSSP feature analysis.

    Statistical metrics for analyzing DSSP secondary structure data including
    structural stability, transition patterns, and class distribution analysis.

    Examples:
    ---------
    >>> # Standard deviation of DSSP assignments
    >>> metric = ReduceDSSPMetrics.STD

    >>> # Secondary structure stability
    >>> metric = ReduceDSSPMetrics.CLASS_STABILITY

    >>> # Transition frequency analysis
    >>> metric = ReduceDSSPMetrics.TRANSITION_FREQUENCY
    """
    CLASS_FREQUENCIES = "class_frequencies"
    """Frequency of each secondary structure class per residue"""

    TRANSITION_FREQUENCY = "transition_frequency"
    """Frequency of secondary structure transitions per residue"""

    CLASS_STABILITY = "class_stability"
    """Secondary structure stability (1 - transition_frequency)"""

    DOMINANT_CLASS = "dominant_class"
    """Most frequent secondary structure class per residue"""

    TRANSITIONS = "transitions"
    """Number of secondary structure class transitions"""
