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
        legend_entries = {}
        resolved = VerticalMarkerHelper.resolve_markers(
            marker_positions={"a": [1.0, 2.0], "b": 1.0},
            marker_labels="event",
            color_resolver=lambda _: ["#ff0000"],
            legend_entries=legend_entries
        )

        assert set(resolved) == {
            (1.0, "#ff0000", "event"),
            (2.0, "#ff0000", "event")
        }
        assert legend_entries == {"event": "#ff0000"}

    def test_validate_labels_rejects_non_string_per_key_values(self):
        """Per-key labels must be plain strings."""
        with pytest.raises(ValueError, match="must be a string"):
            VerticalMarkerHelper.validate_labels(
                marker_positions={"a": [1.0, 2.0]},
                marker_labels={"a": ["first"]}  # type: ignore[arg-type]
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

    def test_validate_label_colors_accepts_per_label_mapping(self):
        """Legend colors can be overridden per label."""
        VerticalMarkerHelper.validate_label_colors(
            marker_labels={"a": "release"},
            label_colors={"release": "#224466"}
        )

    def test_resolve_markers_prefers_label_color_override(self):
        """Legend color overrides should win over marker-derived colors."""
        legend_entries = {}
        resolved = VerticalMarkerHelper.resolve_markers(
            marker_positions={"a": [1.0, 2.0]},
            marker_labels={"a": "release"},
            color_resolver=lambda _: ["#99ccff"],
            legend_entries=legend_entries,
            label_colors={"release": "#224466"}
        )

        assert set(resolved) == {
            (1.0, "#99ccff", "release"),
            (2.0, "#99ccff", "release")
        }
        assert legend_entries == {"release": "#224466"}

    def test_validate_label_colors_rejects_unknown_label(self):
        """Color overrides must reference known marker labels."""
        with pytest.raises(ValueError, match="unknown label"):
            VerticalMarkerHelper.validate_label_colors(
                marker_labels={"a": "release"},
                label_colors={"other": "#224466"}
            )

    def test_build_legend_handles_uses_pre_resolved_entries(self):
        """Legend handles should be constructible without scanning marker tuples."""
        handles = VerticalMarkerHelper.build_legend_handles(
            resolved_markers=[],
            legend_entries={"release": "#224466", "other": "#556677"}
        )

        assert [h.get_label() for h in handles] == ["release", "other"]
        assert [h.get_color() for h in handles] == ["#224466", "#556677"]
