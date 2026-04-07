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

from mdxplain.feature_importance.manager.feature_importance_manager import (
    FeatureImportanceManager,
)


class _DummyFeatureImportanceData:
    """Minimal stub exposing get_comparison for split-rule tests."""

    def get_comparison(self, comparison_name: str):
        return None, {
            "comparison": comparison_name,
            "model": object(),
            "labels": (7, 9),
        }


def test_get_split_rules_for_comparison_passes_target_label():
    """Split-rule extraction should forward the stored target label."""
    manager = FeatureImportanceManager()
    fi_data = _DummyFeatureImportanceData()
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
