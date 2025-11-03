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
Helper classes for comparison management.

This module contains helper classes that provide static methods for
comparison validation and sub-comparison creation. These helpers
separate concerns and make the ComparisonManager more focused.
"""

from .comparison_validation_helper import ComparisonValidationHelper
from .sub_comparison_creation_helper import SubComparisonCreationHelper

__all__ = [
    "ComparisonValidationHelper",
    "SubComparisonCreationHelper",
]