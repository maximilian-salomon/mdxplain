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
Violin plot data preparer.

Thin wrapper around BaseFeatureImportancePlotDataPreparer providing
violin-plot-specific data preparation. Inherits all core functionality
from the base class.
"""

from __future__ import annotations

from ....helper.base_feature_importance_plot_data_preparer import BaseFeatureImportancePlotDataPreparer


class ViolinDataPreparer(BaseFeatureImportancePlotDataPreparer):
    """
    Data preparer for violin plots.

    Inherits all functionality from BaseFeatureImportancePlotDataPreparer.
    Provides data preparation for violin plot visualizations from feature
    importance analysis or manual feature selection.

    Can be extended in the future with violin-specific customizations
    by overriding base class methods.

    Examples
    --------
    >>> # Feature Importance mode
    >>> data, metadata, colors, cutoff = ViolinDataPreparer.prepare_from_feature_importance(
    ...     pipeline_data, "tree_analysis", n_top=10
    ... )

    >>> # Manual Selection mode
    >>> data, metadata, colors, cutoff = ViolinDataPreparer.prepare_from_manual_selection(
    ...     pipeline_data, "my_selector", ["cluster_0", "cluster_1"]
    ... )

    Notes
    -----
    This class currently inherits all methods from BaseFeatureImportancePlotDataPreparer
    without modifications. Future violin-specific enhancements can be added here.
    """
    pass
