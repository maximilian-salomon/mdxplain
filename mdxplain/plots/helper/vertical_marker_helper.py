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

"""
Shared helpers for vertical guide marker validation and legend preparation.

This module centralizes marker parsing for multiple plot types and defines
shared type aliases:

- ``MarkerKey``: dictionary key used for marker definitions (selector/tag key)
- ``MarkerPositions``: optional mapping from key to one or many x-positions
- ``MarkerLabels``: optional global/per-key/per-position marker labels
- ``ResolvedMarker``: normalized marker tuple ``(x_position, color, label)``
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

from matplotlib.lines import Line2D

# This are only type aliases, no actual code
MarkerKey = Union[int, str]
MarkerPositions = Optional[
    Dict[MarkerKey, Union[float, List[float], Tuple[float, ...]]]
]
MarkerLabels = Optional[
    Union[
        str,
        Dict[MarkerKey, Union[str, List[str], Tuple[str, ...]]]
    ]
]
ResolvedMarker = Tuple[float, str, Optional[str]]


class VerticalMarkerHelper:
    """
    Utility methods for cross-plot vertical marker handling.

    The helper keeps validation and normalization logic in one place so
    Time-Series and Density plots behave consistently for marker inputs.
    """

    @staticmethod
    def validate_markers(
        marker_positions: MarkerPositions,
        key_context: str
    ) -> None:
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
    def validate_labels(
        marker_positions: MarkerPositions,
        marker_labels: MarkerLabels
    ) -> None:
        """
        Validate marker label structure.

        Parameters
        ----------
        marker_positions : dict or None
            Marker position dictionary.
        marker_labels : str, dict, or None
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
                "vertical_marker_labels must be None, a string, "
                "or a dictionary."
            )

        marker_positions = marker_positions or {}
        for key, label_spec in marker_labels.items():
            if key not in marker_positions:
                raise ValueError(
                    "vertical_marker_labels contains unknown key "
                    f"{key!r}. Keys must match vertical_markers."
                )

            if isinstance(label_spec, str):
                continue

            if isinstance(label_spec, (list, tuple)):
                if len(label_spec) == 0:
                    raise ValueError(
                        f"vertical_marker_labels for key {key!r} cannot be empty."
                    )
                for label in label_spec:
                    if not isinstance(label, str):
                        raise ValueError(
                            "vertical_marker_labels list entries must be strings. "
                            f"Invalid label for key {key!r}: {label!r}."
                        )
                continue

            raise ValueError(
                "vertical_marker_labels values must be strings or lists/tuples "
                f"of strings. Invalid value for key {key!r}: {label_spec!r}."
            )

    @staticmethod
    def normalize_positions(
        raw_positions: Union[float, List[float], Tuple[float, ...]],
        selector_key: MarkerKey
    ) -> List[float]:
        """
        Normalize one marker value entry to a list of numeric x-positions.

        Parameters
        ----------
        raw_positions : float or List[float] or Tuple[float, ...]
            Marker position definition for one key.
        selector_key : MarkerKey
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
        color_resolver: Callable[[MarkerKey], List[str]]
    ) -> List[ResolvedMarker]:
        """
        Resolve marker dictionaries to unique `(x, color, label)` tuples.

        Parameters
        ----------
        marker_positions : MarkerPositions
            Marker definition dictionary mapping keys to x-position values.
        marker_labels : MarkerLabels
            Marker label definition as global/per-key/per-position labels.
        color_resolver : Callable[[MarkerKey], List[str]]
            Callback that resolves one key to one or many colors.

        Returns
        -------
        List[ResolvedMarker]
            Resolved marker tuples for plotting and legend generation.

        Raises
        ------
        ValueError
            Propagates validation errors for marker positions/labels.

        Notes
        -----
        Markers are deduplicated by `(x_position, color)`. When duplicates are
        encountered, labeled variants take precedence over unlabeled ones.
        """
        if not marker_positions:
            return []

        marker_map: Dict[Tuple[float, str], Optional[str]] = {}

        for key, raw_positions in marker_positions.items():
            x_positions = VerticalMarkerHelper.normalize_positions(raw_positions, key)
            labels = VerticalMarkerHelper._resolve_labels_for_positions(
                marker_labels, key, len(x_positions)
            )

            colors = []
            for color in color_resolver(key):
                if color and color not in colors:
                    colors.append(color)

            for color in colors:
                for x_pos, label in zip(x_positions, labels):
                    marker_key = (x_pos, color)
                    existing_label = marker_map.get(marker_key)
                    if existing_label is None and label is not None:
                        marker_map[marker_key] = label
                    elif marker_key not in marker_map:
                        marker_map[marker_key] = label

        return [
            (x_pos, color, label)
            for (x_pos, color), label in marker_map.items()
        ]

    @staticmethod
    def build_legend_handles(
        resolved_markers: List[ResolvedMarker],
        line_width: float = 1.5,
        alpha: float = 0.85
    ) -> List[Line2D]:
        """
        Build unique legend handles for resolved markers that have labels.

        Parameters
        ----------
        resolved_markers : List[ResolvedMarker]
            Normalized marker tuples.
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
    def _resolve_labels_for_positions(
        marker_labels: MarkerLabels,
        key: MarkerKey,
        n_positions: int
    ) -> List[Optional[str]]:
        """
        Resolve labels for one marker key to `n_positions` entries.

        Parameters
        ----------
        marker_labels : MarkerLabels
            Label definition structure.
        key : MarkerKey
            Marker key to resolve.
        n_positions : int
            Number of x-positions for this key.

        Returns
        -------
        List[Optional[str]]
            One label per marker position (or None when unlabeled).

        Raises
        ------
        ValueError
            If per-position label count does not match `n_positions`.
        """
        if marker_labels is None:
            return [None] * n_positions

        if isinstance(marker_labels, str):
            return [marker_labels] * n_positions

        if key not in marker_labels:
            return [None] * n_positions

        label_spec = marker_labels[key]
        if isinstance(label_spec, str):
            return [label_spec] * n_positions

        if len(label_spec) == 1 and n_positions > 1:
            return [label_spec[0]] * n_positions

        if len(label_spec) != n_positions:
            raise ValueError(
                "vertical_marker_labels list length must match the number of "
                f"positions for key {key!r}. Got {len(label_spec)} labels for "
                f"{n_positions} positions."
            )

        return [str(value) for value in label_spec]
