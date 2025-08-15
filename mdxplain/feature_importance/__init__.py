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
Feature importance analysis module for ML-based feature importance evaluation.

This module provides functionality to analyze feature importance using various
machine learning algorithms. It works with comparisons created by the
ComparisonManager to identify which features are most important for
distinguishing between different data groups.

Main Components:
- FeatureImportanceData: Entity storing feature importance results
- FeatureImportanceManager: Manager for running importance analyses
- analyzer_types: Various ML algorithms for importance analysis

Examples:
---------
>>> from mdxplain.feature_importance import FeatureImportanceManager, analyzer_types
>>> manager = FeatureImportanceManager()
>>> manager.add(
...     pipeline_data, "folded_vs_unfolded",
...     analyzer_types.DecisionTree(max_depth=5), "tree_analysis"
... )
"""

from .entities.feature_importance_data import FeatureImportanceData
from .managers.feature_importance_manager import FeatureImportanceManager
from . import analyzer_types

__all__ = [
    "FeatureImportanceData",
    "FeatureImportanceManager",
    "analyzer_types",
]