# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# Utility modules for MD analysis
#
# Contains core utility functions for data handling and processing.
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

"""Utility functions and helper classes for mdxplain package."""

from .data_utils import DataUtils
from .color_utils import ColorUtils
from .top_features_utils import TopFeaturesUtils

__all__ = [
    "DataUtils",
    "ColorUtils",
    "TopFeaturesUtils",
]
