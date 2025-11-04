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
Data selector module for row-based trajectory frame selection.

This module provides the DataSelector functionality that serves as the
counterpart to FeatureSelector. While FeatureSelector chooses columns (features),
DataSelector chooses rows (frames) based on tags, clusters, or combinations.

Main Components:
- DataSelectorData: Entity storing frame indices and selection criteria
- DataSelectorManager: Manager for creating and managing row selections
"""

from .entities.data_selector_data import DataSelectorData
from .manager.data_selector_manager import DataSelectorManager

__all__ = [
    "DataSelectorData",
    "DataSelectorManager",
]