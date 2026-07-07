# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex.
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

"""Unit tests for feature importance manager helpers."""

from unittest.mock import patch

import numpy as np
import pytest

from mdxplain.comparison.entities.comparison_data import ComparisonData
from mdxplain.feature_importance.entities.feature_importance_data import (
    FeatureImportanceData,
)
from mdxplain.feature_importance.manager.feature_importance_manager import (
    FeatureImportanceManager,
)
from mdxplain.pipeline.entities.pipeline_data import PipelineData


class _TreeModel:
    """Minimal stub model exposing a tree_ attribute."""

    tree_ = object()


class _EnsembleModel:
    """Minimal stub model exposing an estimators_ attribute."""

    estimators_ = [object(), object()]


def _make_fi_data(model, comparison_name="cluster_0_vs_rest"):
    """Build a FeatureImportanceData holding one comparison with a model."""
    fi_data = FeatureImportanceData("analysis")
    fi_data.analyzer_type = "decision_tree"
    fi_data.comparison_name = "comparison"
    fi_data.add_comparison_result(
        np.zeros(6),
        {"comparison": comparison_name, "model": model, "labels": (7, 9)},
    )
    return fi_data


def _make_pipeline_data(analyzer_type):
    """Build a PipelineData holding one analysis and its comparison."""
    pipeline_data = PipelineData()
    fi_data = FeatureImportanceData("analysis")
    fi_data.analyzer_type = analyzer_type
    fi_data.comparison_name = "cmp"
    pipeline_data.feature_importance_data["analysis"] = fi_data
    pipeline_data.comparison_data["cmp"] = ComparisonData(
        "cmp", "one_vs_rest", "selector"
    )
    return pipeline_data


def test_get_split_rules_for_comparison_passes_target_label():
    """Split-rule extraction should forward the stored target label."""
    manager = FeatureImportanceManager()
    fi_data = _make_fi_data(_TreeModel())
    top_features = [
        {"feature_index": 2},
        {"feature_index": 5},
    ]

    with patch(
        "mdxplain.feature_importance.manager.feature_importance_manager."
        "RepresentativeFinderHelper._extract_tree_rules",
        return_value={2: {"threshold": 1.0, "weight": 0.5}},
    ) as mock_extract:
        split_rules = manager._get_split_rules_for_comparison(
            fi_data,
            "cluster_0_vs_rest",
            top_features,
        )

    assert split_rules == {2: {"threshold": 1.0, "weight": 0.5}}
    mock_extract.assert_called_once()
    args = mock_extract.call_args.args
    assert args[1] == [2, 5]
    assert args[2] == 7


def test_is_tree_based_model_detects_tree_and_ensemble():
    """Single trees and ensembles are recognized as tree-based."""
    assert FeatureImportanceManager._is_tree_based_model(_TreeModel())
    assert FeatureImportanceManager._is_tree_based_model(_EnsembleModel())
    assert not FeatureImportanceManager._is_tree_based_model(object())


def test_get_split_rules_skips_non_tree_model():
    """Non-tree models yield no split rules and skip extraction."""
    manager = FeatureImportanceManager()
    fi_data = _make_fi_data(object())

    with patch(
        "mdxplain.feature_importance.manager.feature_importance_manager."
        "RepresentativeFinderHelper._extract_tree_rules",
    ) as mock_extract:
        split_rules = manager._get_split_rules_for_comparison(
            fi_data,
            "cluster_0_vs_rest",
            [{"feature_index": 1}],
        )

    assert split_rules == {}
    mock_extract.assert_not_called()


def test_validate_representative_analysis_accepts_random_forest():
    """Random Forest analyses are accepted for representative frames."""
    manager = FeatureImportanceManager()
    pipeline_data = _make_pipeline_data("random_forest")

    fi_data, comp_data = manager._validate_representative_analysis(
        pipeline_data, "analysis"
    )

    assert fi_data.analyzer_type == "random_forest"
    assert comp_data is pipeline_data.comparison_data["cmp"]


def test_validate_representative_analysis_rejects_unknown_analyzer():
    """Unsupported analyzer types are rejected with a clear error."""
    manager = FeatureImportanceManager()
    pipeline_data = _make_pipeline_data("svm")

    with pytest.raises(ValueError):
        manager._validate_representative_analysis(pipeline_data, "analysis")
