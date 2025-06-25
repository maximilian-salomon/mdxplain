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
Statistical metrics for contact feature analysis and filtering.

Provides metric identifiers for reducing contact data based on frequency
and stability patterns in molecular dynamics trajectories.
"""


class ReduceContactMetrics:
    """
    Available statistical metrics for contact feature analysis and filtering.

    Provides metric identifiers for reducing contact data.
    So we can use them to filter the contact data.

    Examples:
    ---------
    >>> metrics = ReduceContactMetrics()
    >>> print(metrics.FREQUENCY)  # 'frequency'
    >>> print(metrics.STABILITY)  # 'stability'
    """

    @property
    def FREQUENCY(self) -> str:
        """
        Contact frequency metric - fraction of frames where pair is in contact.

        Returns:
        --------
        str
            Metric identifier 'frequency' for contact frequency analysis
        """
        return "frequency"

    @property
    def STABILITY(self) -> str:
        """
        Contact stability metric - measure of contact persistence over time.

        Returns:
        --------
        str
            Metric identifier 'stability' for contact stability analysis
        """
        return "stability"
