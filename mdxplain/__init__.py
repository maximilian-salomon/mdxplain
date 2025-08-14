# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance diverse Coding AI tools => See Readme for details.
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
mdxplain - A Python toolkit for molecular dynamics trajectory analysis.

This package provides tools for analyzing molecular dynamics trajectories with focus
on feature extraction, dimensionality reduction, and machine learning applications.
"""

from . import trajectory
from . import feature
from . import feature_selection
from . import decomposition
from . import clustering
from . import pipeline

__all__ = [
    "trajectory",
    "feature", 
    "feature_selection",
    "decomposition",
    "clustering",
    "pipeline"
]
