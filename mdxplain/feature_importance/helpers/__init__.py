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
Helper classes for feature importance operations.

This package contains helper classes that extract common logic from
FeatureImportanceManager methods to improve code organization and reusability.

Notes
-----
All helpers are internal to the feature_importance module and should be
imported directly where needed. No exports to avoid circular import issues
when helpers/__init__.py is executed during direct helper imports.
"""

# No imports or exports - all helpers are internal
# Import directly: from .helpers.analysis_runner_helper import AnalysisRunnerHelper
__all__ = []