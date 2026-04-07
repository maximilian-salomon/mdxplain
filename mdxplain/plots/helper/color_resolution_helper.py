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
Shared helper for resolving user color configuration.

Handles the common `colors` behavior used by multiple plot types:

- None -> use defaults
- str -> build colors from matplotlib colormap
- dict -> merge explicit overrides on defaults
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
from matplotlib import cm, colors as mcolors

from .color_mapping_helper import ColorMappingHelper


class ColorResolutionHelper:
    """
    Shared resolver for label-to-color mappings.

    Provides deterministic color assignment across plot types for the
    common `colors` API:

    - `None`: keep default mapping
    - `str`: sample colors from a matplotlib colormap
    - `dict`: merge explicit overrides onto defaults
    """

    @staticmethod
    def resolve_label_colors(
        labels: List[str],
        colors: Optional[Union[str, Dict[str, str]]] = None,
        default_colors: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Resolve colors for labels.

        Parameters
        ----------
        labels : List[str]
            Labels to colorize in deterministic order.
        colors : str or Dict[str, str], optional
            User-provided color configuration:
            - None: use defaults
            - str: matplotlib colormap name
            - dict: explicit mapping merged onto defaults
        default_colors : Dict[str, str], optional
            Optional base mapping used when colors is None or dict.
            Missing labels fall back to automatic palette colors.

        Returns
        -------
        Dict[str, str]
            Label -> color mapping.

        Raises
        ------
        ValueError
            If colors is not None/str/dict or if colormap name is invalid.
        """
        if not labels:
            return {}

        resolved_defaults = ColorResolutionHelper._resolve_default_colors(
            labels, default_colors
        )

        if colors is None:
            return resolved_defaults

        if isinstance(colors, str):
            return ColorResolutionHelper.build_colors_from_colormap(labels, colors)

        if isinstance(colors, dict):
            resolved = dict(resolved_defaults)
            for label, color in colors.items():
                if label in resolved:
                    resolved[label] = color
            return resolved

        raise ValueError(
            "Invalid colors argument. Expected None, matplotlib colormap name "
            "(str), or Dict[str, str]."
        )

    @staticmethod
    def build_colors_from_colormap(
        labels: List[str],
        colormap: str
    ) -> Dict[str, str]:
        """
        Build deterministic label colors from a matplotlib colormap.

        Parameters
        ----------
        labels : List[str]
            Labels to colorize in deterministic order.
        colormap : str
            Matplotlib colormap name.

        Returns
        -------
        Dict[str, str]
            Mapping from label to hex color.

        Raises
        ------
        ValueError
            If the colormap name is unknown to matplotlib.
        """
        if not labels:
            return {}

        try:
            cmap = cm.get_cmap(colormap)
        except ValueError as exc:
            raise ValueError(f"Unknown colormap '{colormap}'.") from exc

        if len(labels) == 1:
            sample_points = [0.5]
        else:
            sample_points = np.linspace(0.0, 1.0, len(labels))

        return {
            label: mcolors.to_hex(cmap(point))
            for label, point in zip(labels, sample_points)
        }

    @staticmethod
    def _resolve_default_colors(
        labels: List[str],
        default_colors: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Resolve default color mapping for labels.

        Parameters
        ----------
        labels : List[str]
            Labels that must receive a color.
        default_colors : Dict[str, str], optional
            Preferred defaults by label. May be partial.

        Returns
        -------
        Dict[str, str]
            Complete label-to-color mapping where missing labels are
            backfilled with deterministic cluster palette colors.

        Notes
        -----
        This method guarantees that every label in `labels` appears in
        the returned mapping.
        """
        if default_colors is None:
            palette = ColorMappingHelper.get_cluster_colors(len(labels))
            return {label: palette[i] for i, label in enumerate(labels)}

        resolved: Dict[str, str] = {}
        missing_labels: List[str] = []
        for label in labels:
            color = default_colors.get(label)
            if color is None:
                missing_labels.append(label)
            else:
                resolved[label] = color

        if missing_labels:
            fallback_palette = ColorMappingHelper.get_cluster_colors(len(missing_labels))
            for i, label in enumerate(missing_labels):
                resolved[label] = fallback_palette[i]

        return resolved
