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
Analyzer types for feature importance analysis.

This module provides various ML algorithm implementations for feature
importance analysis, following the same pattern as decomposition_types.

Available Analyzers:
- DecisionTree: Decision tree classifier for feature importance

Examples:
---------
>>> from mdxplain.feature_importance import analyzer_types
>>> analyzer = analyzer_types.DecisionTree(max_depth=5, random_state=42)
>>> pipeline.feature_importance.add(
...     "my_comparison", analyzer, "tree_analysis"
... )
"""

from .decision_tree.decision_tree import DecisionTree

__all__ = ["DecisionTree"]