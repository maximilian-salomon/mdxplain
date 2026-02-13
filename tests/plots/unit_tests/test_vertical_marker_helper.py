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

"""Unit tests for shared vertical marker helper utilities."""

import pytest

from mdxplain.plots.helper.vertical_marker_helper import VerticalMarkerHelper


class TestVerticalMarkerHelper:
    """Validation and resolution tests for VerticalMarkerHelper."""

    def test_validate_markers_rejects_invalid_key(self):
        """Bool keys must be rejected to avoid ambiguous selector semantics."""
        with pytest.raises(ValueError, match="Invalid key"):
            VerticalMarkerHelper.validate_markers(
                marker_positions={True: [1.0, 2.0]},
                key_context="selectors"
            )

    def test_validate_labels_rejects_unknown_keys(self):
        """Label dictionaries must only contain keys present in markers."""
        with pytest.raises(ValueError, match="unknown key"):
            VerticalMarkerHelper.validate_labels(
                marker_positions={"a": [1.0]},
                marker_labels={"b": "event"}
            )

    def test_resolve_markers_deduplicates_by_position_and_color(self):
        """Markers with same x/color are collapsed into one draw instruction."""
        resolved = VerticalMarkerHelper.resolve_markers(
            marker_positions={"a": [1.0, 2.0], "b": 1.0},
            marker_labels="event",
            color_resolver=lambda _: ["#ff0000"]
        )

        assert len(resolved) == 2
        assert (1.0, "#ff0000", "event") in resolved
        assert (2.0, "#ff0000", "event") in resolved

    def test_resolve_markers_raises_for_label_length_mismatch(self):
        """Per-position labels must match the number of marker positions."""
        with pytest.raises(ValueError, match="list length must match"):
            VerticalMarkerHelper.resolve_markers(
                marker_positions={"a": [1.0, 2.0]},
                marker_labels={"a": ["first"] * 3},
                color_resolver=lambda _: ["#00ff00"]
            )

    def test_build_legend_handles_uses_unique_labels(self):
        """Legend handles are unique by label even for repeated marker labels."""
        handles = VerticalMarkerHelper.build_legend_handles(
            resolved_markers=[
                (1.0, "#ff0000", "event"),
                (2.0, "#00ff00", "event"),
                (3.0, "#0000ff", "other"),
                (4.0, "#123456", None),
            ]
        )

        labels = [handle.get_label() for handle in handles]
        assert labels == ["event", "other"]
