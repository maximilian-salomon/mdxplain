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

"""Shared helpers for vertical marker validation and legend preparation."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from matplotlib.lines import Line2D

MarkerKey = Union[int, str]
MarkerPositions = Optional[Dict[MarkerKey, Union[float, List[float], Tuple[float, ...]]]]
MarkerLabels = Optional[Union[str, Dict[MarkerKey, str]]]
MarkerLabelColors = Optional[Union[str, Dict[str, str]]]
ResolvedMarker = Tuple[float, str, Optional[str]]
LegendEntries = Dict[str, str]


class VerticalMarkerHelper:
    """Utility methods for marker parsing, validation, and legend entries."""

    @staticmethod
    def validate_markers(marker_positions: MarkerPositions, key_context: str) -> None:
        """
        Validate marker dictionary structure.

        Parameters
        ----------
        marker_positions : dict or None
            Marker position dictionary.
        key_context : str
            Human-readable key context for error messages.

        Returns
        -------
        None
            Validation helper; returns only when values are valid.

        Raises
        ------
        ValueError
            If `marker_positions` is not a dictionary, contains invalid keys,
            or contains invalid position values.
        """
        if marker_positions is None:
            return

        if not isinstance(marker_positions, dict):
            raise ValueError(
                "vertical_markers must be a dictionary mapping "
                f"{key_context} to numeric x-positions."
            )

        for key, raw_positions in marker_positions.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise ValueError(
                    f"vertical_markers keys must be {key_context} "
                    f"(int/str). Invalid key: {key!r}."
                )
            VerticalMarkerHelper.normalize_positions(raw_positions, key)

    @staticmethod
    def validate_labels(marker_positions: MarkerPositions, marker_labels: MarkerLabels) -> None:
        """
        Validate marker label structure.

        Parameters
        ----------
        marker_positions : dict or None
            Marker position dictionary.
        marker_labels : str or dict or None
            Marker legend labels.

        Returns
        -------
        None
            Validation helper; returns only when values are valid.

        Raises
        ------
        ValueError
            If label structure is invalid, references unknown marker keys, or
            contains non-string label values.
        """
        if marker_labels is None:
            return

        if isinstance(marker_labels, str):
            return

        if not isinstance(marker_labels, dict):
            raise ValueError(
                "vertical_marker_labels must be None, a string, or a dictionary."
            )

        marker_positions = marker_positions or {}
        for key, label in marker_labels.items():
            if key not in marker_positions:
                raise ValueError(
                    "vertical_marker_labels contains unknown key "
                    f"{key!r}. Keys must match vertical_markers."
                )
            if not isinstance(label, str):
                raise ValueError(
                    f"vertical_marker_labels[{key!r}] must be a string. "
                    f"Got: {label!r}."
                )

    @staticmethod
    def validate_label_colors(marker_labels: MarkerLabels, label_colors: MarkerLabelColors) -> None:
        """
        Validate optional label-to-color overrides.

        Parameters
        ----------
        marker_labels : str or dict or None
            Marker legend labels.
        label_colors : str or dict or None
            Legend color override:
            - str: one shared color for all marker legend labels
            - dict[label] = color: per-label legend colors

        Returns
        -------
        None
            Validation helper; returns only when values are valid.

        Raises
        ------
        ValueError
            If color override structure is invalid, contains non-string values,
            or references labels that are not present in `marker_labels`.
        """
        if label_colors is None:
            return

        if isinstance(label_colors, str):
            return

        if not isinstance(label_colors, dict):
            raise ValueError(
                "vertical_marker_label_colors must be None, a string, or a "
                "dictionary mapping label text to color strings."
            )

        known_labels = VerticalMarkerHelper._collect_known_labels(marker_labels)
        if not known_labels:
            raise ValueError(
                "vertical_marker_label_colors requires vertical_marker_labels "
                "to define at least one label."
            )

        for label, color in label_colors.items():
            if not isinstance(label, str):
                raise ValueError(
                    "vertical_marker_label_colors keys must be label strings. "
                    f"Invalid key: {label!r}."
                )
            if not isinstance(color, str):
                raise ValueError(
                    "vertical_marker_label_colors values must be color strings. "
                    f"Invalid value for label {label!r}: {color!r}."
                )
            if label not in known_labels:
                raise ValueError(
                    "vertical_marker_label_colors contains unknown label "
                    f"{label!r}. Known labels: {sorted(known_labels)}."
                )

    @staticmethod
    def normalize_positions(raw_positions: Union[float, List[float], Tuple[float, ...]], selector_key: MarkerKey) -> List[float]:
        """
        Normalize one marker value entry to a list of numeric x-positions.

        Parameters
        ----------
        raw_positions : float or List[float] or Tuple[float, ...]
            Marker position definition for one key.
        selector_key : int or str
            Key used for error context in validation messages.

        Returns
        -------
        List[float]
            Normalized marker positions as float values.

        Raises
        ------
        ValueError
            If `raw_positions` is neither numeric nor a list/tuple of numerics.
        """
        if isinstance(raw_positions, (int, float)) and not isinstance(raw_positions, bool):
            return [float(raw_positions)]

        if isinstance(raw_positions, (list, tuple)):
            values: List[float] = []
            for value in raw_positions:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        "vertical_markers values must be numeric or lists of numeric values. "
                        f"Invalid value for key {selector_key!r}: {value!r}."
                    )
                values.append(float(value))
            return values

        raise ValueError(
            "vertical_markers values must be numeric or lists of numeric values. "
            f"Invalid value for key {selector_key!r}: {raw_positions!r}."
        )

    @staticmethod
    def resolve_markers(
        marker_positions: MarkerPositions,
        marker_labels: MarkerLabels,
        color_resolver: Callable[[MarkerKey], List[str]],
        legend_entries: Optional[LegendEntries] = None,
        label_colors: MarkerLabelColors = None
    ) -> List[ResolvedMarker]:
        """
        Resolve marker dictionaries to unique `(x, color, label)` tuples.

        Parameters
        ----------
        marker_positions : MarkerPositions
            Marker definition dictionary mapping keys to x-position values.
        marker_labels : MarkerLabels
            Marker labels as shared string or per-key string dictionary.
        color_resolver : Callable[[int or str], List[str]]
            Callback that resolves one marker key to one or many colors.
        legend_entries : Dict[str, str], optional
            Optional output mapping populated with unique legend entries in
            first-seen order (`label -> color`).
        label_colors : str or dict or None, optional
            Optional legend color override:
            shared color string or dictionary (`label -> color`).

        Returns
        -------
        List[ResolvedMarker]
            Resolved marker tuples for plotting and legend generation.

        Notes
        -----
        Markers are deduplicated by `(x_position, color)`. When duplicates are
        encountered, labeled variants take precedence over unlabeled ones.
        """
        if not marker_positions:
            return []

        marker_map: Dict[Tuple[float, str], Optional[str]] = {}
        if legend_entries is not None:
            legend_entries.clear()

        for key, raw_positions in marker_positions.items():
            x_positions = VerticalMarkerHelper.normalize_positions(raw_positions, key)
            label = VerticalMarkerHelper._resolve_label_for_key(marker_labels, key)

            colors: List[str] = []
            for color in color_resolver(key):
                if color and color not in colors:
                    colors.append(color)

            for color in colors:
                for x_pos in x_positions:
                    marker_key = (x_pos, color)
                    existing_label = marker_map.get(marker_key)
                    if existing_label is None and label is not None:
                        marker_map[marker_key] = label
                    elif marker_key not in marker_map:
                        marker_map[marker_key] = label

                    VerticalMarkerHelper._update_legend_entries(
                        legend_entries=legend_entries,
                        label=label,
                        marker_color=color,
                        label_colors=label_colors
                    )

        return [(x_pos, color, label) for (x_pos, color), label in marker_map.items()]

    @staticmethod
    def build_legend_handles(
        resolved_markers: List[ResolvedMarker],
        legend_entries: Optional[LegendEntries] = None,
        line_width: float = 1.5,
        alpha: float = 0.85
    ) -> List[Line2D]:
        """
        Build unique legend handles for resolved markers that have labels.

        Parameters
        ----------
        resolved_markers : List[ResolvedMarker]
            Normalized marker tuples.
        legend_entries : Dict[str, str], optional
            Optional pre-resolved legend entries (`label -> color`).
            When provided, handles are built from this mapping directly.
        line_width : float, default=1.5
            Legend line width for marker handles.
        alpha : float, default=0.85
            Legend line transparency.

        Returns
        -------
        List[Line2D]
            Legend handles with unique labels in first-seen order.
        """
        handles: List[Line2D] = []
        if legend_entries is not None:
            for label, color in legend_entries.items():
                if not label:
                    continue
                handles.append(
                    Line2D(
                        [0], [0],
                        color=color,
                        linestyle="--",
                        linewidth=line_width,
                        alpha=alpha,
                        label=label
                    )
                )
            return handles

        seen_labels = set()
        for _, color, label in resolved_markers:
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    linestyle="--",
                    linewidth=line_width,
                    alpha=alpha,
                    label=label
                )
            )
        return handles

    @staticmethod
    def _resolve_label_for_key(marker_labels: MarkerLabels, key: MarkerKey) -> Optional[str]:
        """
        Resolve one marker label for a marker key.

        Parameters
        ----------
        marker_labels : str or dict or None
            Marker label definition.
        key : int or str
            Marker key to resolve.

        Returns
        -------
        Optional[str]
            Label for the marker key, or `None` when no label is defined.

        Raises
        ------
        ValueError
            If `marker_labels` has an invalid structure.
        """
        if marker_labels is None:
            return None
        if isinstance(marker_labels, str):
            return marker_labels
        if not isinstance(marker_labels, dict):
            raise ValueError(
                "vertical_marker_labels must be None, a string, or a dictionary."
            )

        label = marker_labels.get(key)
        if label is None:
            return None
        if not isinstance(label, str):
            raise ValueError(
                f"vertical_marker_labels[{key!r}] must be a string. Got: {label!r}."
            )
        return label

    @staticmethod
    def _collect_known_labels(marker_labels: MarkerLabels) -> Set[str]:
        """
        Collect the set of labels defined by `marker_labels`.

        Parameters
        ----------
        marker_labels : str or dict or None
            Marker label definition.

        Returns
        -------
        Set[str]
            Unique labels found in the marker label definition.
        """
        if marker_labels is None:
            return set()
        if isinstance(marker_labels, str):
            return {marker_labels}
        if not isinstance(marker_labels, dict):
            return set()
        return {label for label in marker_labels.values() if isinstance(label, str)}

    @staticmethod
    def _resolve_legend_color(label: str, marker_color: str, label_colors: MarkerLabelColors) -> str:
        """
        Resolve legend color for one marker label.

        Parameters
        ----------
        label : str
            Marker label text.
        marker_color : str
            Marker line color.
        label_colors : str or dict or None
            Optional legend color override.

        Returns
        -------
        str
            Color used for the legend handle.
        """
        if isinstance(label_colors, str):
            return label_colors
        if isinstance(label_colors, dict):
            return label_colors.get(label, marker_color)
        return marker_color

    @staticmethod
    def _update_legend_entries(
        legend_entries: Optional[LegendEntries],
        label: Optional[str],
        marker_color: str,
        label_colors: MarkerLabelColors
    ) -> None:
        """
        Update pre-resolved legend entries for one marker occurrence.

        Parameters
        ----------
        legend_entries : dict or None
            Output mapping for legend entries (`label -> color`).
        label : str or None
            Marker label text for current occurrence.
        marker_color : str
            Marker line color for current occurrence.
        label_colors : str or dict or None
            Optional legend color override.

        Returns
        -------
        None
            Updates the legend map in place.
        """
        if label is None or legend_entries is None or label in legend_entries:
            return
        legend_entries[label] = VerticalMarkerHelper._resolve_legend_color(
            label=label,
            marker_color=marker_color,
            label_colors=label_colors
        )
