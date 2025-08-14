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

"""Trajectory helper utilities."""

from .trajectory_loader import TrajectoryLoader
from .nomenclature import Nomenclature
from .consistency_checker import TrajectoryConsistencyChecker
from .selection_resolver import TrajectorySelectionResolver
from .keyword_manager import KeywordManager
from .trajectory_processor import TrajectoryProcessor
from .trajectory_validation_helper import TrajectoryValidationHelper

__all__ = [
    "TrajectoryLoader",
    "Nomenclature",
    "TrajectoryConsistencyChecker",
    "TrajectorySelectionResolver",
    "KeywordManager",
    "TrajectoryProcessor",
    "TrajectoryValidationHelper",
]
