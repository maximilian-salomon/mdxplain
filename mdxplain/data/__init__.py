# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# Data loading utilities for MD analysis
#
# Contains trajectory loaders and data import utilities.
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
Data handling and trajectory management for mdxplain package.

This module provides classes for loading, managing, and processing molecular dynamics
trajectory data, including feature extraction and data serialization capabilities.
"""

from .entities.trajectory_data import TrajectoryData
from .managers.feature_manager import FeatureManager
from .managers.feature_selector import FeatureSelector
from .managers.trajcetory_manager import TrajectoryManager

__all__ = ["TrajectoryData", "TrajectoryManager", "FeatureManager", "FeatureSelector"]
