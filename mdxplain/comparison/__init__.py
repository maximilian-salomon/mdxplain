# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
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
Comparison module for creating ML-ready data comparisons.

This module provides functionality to create comparisons between different
data selections (created by DataSelector) for machine learning analysis.
It supports various comparison modes like pairwise, one-vs-rest, multiclass,
and binary comparisons.

Main Components:
- ComparisonData: Entity storing comparison definitions and sub-comparisons
- ComparisonManager: Manager for creating and managing comparisons

Examples:
---------
>>> from mdxplain.comparison import ComparisonManager
>>> manager = ComparisonManager()
>>> manager.create_comparison(
...     pipeline_data, "folded_vs_unfolded", "binary",
...     "key_features", ["folded_frames", "unfolded_frames"]
... )
"""

from .entities.comparison_data import ComparisonData
from .managers.comparison_manager import ComparisonManager

__all__ = [
    "ComparisonData",
    "ComparisonManager",
]