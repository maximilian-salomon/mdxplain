# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0).
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
Container for bound analysis methods.

This file provides the BoundStats class that serves as a container
for dynamically bound analysis methods on feature data objects.
"""


class BoundStats:
    """Container for bound analysis methods.
    
    This class serves as a dynamic container that holds bound analysis methods
    which are created by the FeatureBindingHelper. Analysis methods are bound
    to feature data at runtime and automatically use the appropriate data
    (original or reduced) based on the feature state.

    It is more like a namespace for methods than a traditional class.
    
    Examples
    --------
    >>> # BoundStats is typically created automatically
    >>> feature_data.analysis.compute_mean()  # Uses bound method
    >>> feature_data.analysis.compute_std()   # Uses bound method
    """
    pass