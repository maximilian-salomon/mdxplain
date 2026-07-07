# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Opus 4.8).
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

"""Unit tests for the importance bar plotter."""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from mdxplain.plots.plot_type.importance_bar.importance_bar_plotter import (
    ImportanceBarPlotter,
)

_TOP_FEATURES = [
    {
        "feature_index": 1,
        "importance_score": 0.5,
        "feature_type": "contacts",
        "feature_name": "A-B",
    },
    {
        "feature_index": 4,
        "importance_score": 0.3,
        "feature_type": "torsions",
        "feature_name": "phi",
    },
]

_PATCH_TARGET = (
    "mdxplain.plots.plot_type.importance_bar.importance_bar_plotter."
    "TopFeaturesHelper.get_top_features_with_names"
)


class _FiDataStub:
    """Minimal FeatureImportanceData stub for plotter tests."""

    def __init__(self, metadata, analyzer_type="random_forest"):
        self._metadata = metadata
        self.analyzer_type = analyzer_type

    def list_comparisons(self):
        return ["cluster_0_vs_rest"]

    def get_comparison(self, comparison_name):
        return None, self._metadata


class _PipelineStub:
    """Minimal pipeline_data stub exposing feature_importance_data."""

    def __init__(self, fi_data):
        self.feature_importance_data = {"analysis": fi_data}


def _make_plotter(metadata, analyzer_type="random_forest"):
    """Create a plotter wired to a stubbed pipeline."""
    pipeline_data = _PipelineStub(_FiDataStub(metadata, analyzer_type))
    return ImportanceBarPlotter(pipeline_data, cache_dir="./cache")


def test_extract_std_with_importance_std():
    """Standard deviations are read from the nested analysis metadata."""
    metadata = {
        "analysis_metadata": {"importance_std": [0.0, 0.1, 0.2, 0.3, 0.4]}
    }
    errors = ImportanceBarPlotter._extract_std(metadata, [1, 4])
    assert errors == [0.1, 0.4]


def test_extract_std_without_importance_std_returns_none():
    """Missing standard deviations yield None (plain bars)."""
    assert ImportanceBarPlotter._extract_std({}, [0, 1]) is None
    assert (
        ImportanceBarPlotter._extract_std(
            {"analysis_metadata": {"importance_std": []}}, [0]
        )
        is None
    )


def test_feature_label_combines_type_and_name():
    """Feature labels combine feature type and feature name."""
    label = ImportanceBarPlotter._feature_label(_TOP_FEATURES[0])
    assert label == "contacts: A-B"


def test_feature_label_appends_merged_count():
    """A filter representative shows its merged-neighbour count."""
    label = ImportanceBarPlotter._feature_label(_TOP_FEATURES[0], {1: 3})
    assert label == "contacts: A-B (+3)"


def _errorbar_containers(ax):
    """Return the error-bar containers drawn on an axes."""
    return [c for c in ax.containers if "Errorbar" in type(c).__name__]


def test_plot_with_std_draws_error_bars():
    """An analysis with a stored std draws error bars on the bars."""
    metadata = {
        "analysis_metadata": {
            "algorithm": "random_forest",
            "importance_method": "shap",
            "importance_std": [0.0, 0.05, 0.0, 0.0, 0.02],
        }
    }
    plotter = _make_plotter(metadata)
    with patch(_PATCH_TARGET, return_value=_TOP_FEATURES):
        fig = plotter.plot("analysis", n_top=2)

    ax = fig.get_axes()[0]
    assert len(ax.patches) == 2
    assert len(_errorbar_containers(ax)) == 1
    plt.close(fig)


def test_plot_without_std_draws_no_error_bars():
    """An analysis without a stored std draws plain bars only."""
    metadata = {"analysis_metadata": {"algorithm": "decision_tree"}}
    plotter = _make_plotter(metadata, analyzer_type="decision_tree")
    with patch(_PATCH_TARGET, return_value=_TOP_FEATURES):
        fig = plotter.plot("analysis", n_top=2)

    ax = fig.get_axes()[0]
    assert len(ax.patches) == 2
    assert len(_errorbar_containers(ax)) == 0
    plt.close(fig)


def test_plot_missing_analysis_raises():
    """Plotting an unknown analysis name raises a ValueError."""
    plotter = _make_plotter({"algorithm": "random_forest"})
    with pytest.raises(ValueError):
        plotter.plot("does_not_exist")
