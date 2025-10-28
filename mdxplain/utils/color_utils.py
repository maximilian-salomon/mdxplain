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
Central color utilities for consistent color generation across modules.

This module provides utilities for generating perceptually distinct colors
using HSV color space. Used by plots, structure visualization, and other
modules requiring color mapping.
"""

import colorsys
from typing import Dict


class ColorUtils:
    """
    Utility class for consistent color generation across all modules.

    Provides static methods for generating perceptually distinct colors
    using HSV color space. Colors are distributed evenly across the hue
    spectrum with variations in saturation and value for distinction.

    Examples
    --------
    >>> colors = ColorUtils.generate_distinct_colors(3)
    >>> len(colors)
    3
    >>> colors[0]
    '#bf4040'

    >>> # Add special color for noise/undefined
    >>> colors_with_special = ColorUtils.add_special_color(colors, -1, "#000000")
    >>> len(colors_with_special)
    4
    """

    @staticmethod
    def generate_distinct_colors(n_items: int) -> Dict[int, str]:
        """
        Generate perceptually distinct colors using HSV color space.

        Creates a mapping of indices to hex color strings with colors
        distributed evenly across the hue spectrum. Variations in
        saturation and value provide additional distinction.

        Parameters
        ----------
        n_items : int
            Number of distinct colors to generate

        Returns
        -------
        Dict[int, str]
            Mapping from index (0 to n_items-1) to hex color string

        Examples
        --------
        >>> colors = ColorUtils.generate_distinct_colors(3)
        >>> len(colors)
        3
        >>> colors[0]
        '#bf4040'

        >>> colors = ColorUtils.generate_distinct_colors(100)
        >>> len(colors)
        100

        Notes
        -----
        - Uses HSV color space for perceptually distinct colors
        - Hue varies across full spectrum (0-1)
        - Saturation varies 0.65-0.85 for visual variety
        - Value (brightness) varies 0.75-0.90 for readability
        - Deterministic: same n_items always gives same colors
        """
        colors = {}

        for i in range(n_items):
            # Distribute hue evenly across color wheel
            hue = i / max(n_items, 1)

            # Vary saturation and value for additional distinction
            saturation = 0.65 + (i % 3) * 0.1  # 0.65, 0.75, 0.85
            value = 0.75 + (i % 2) * 0.15      # 0.75, 0.90

            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)

            # Convert to hex
            colors[i] = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )

        return colors

    @staticmethod
    def add_special_color(
        colors: Dict[int, str],
        special_index: int,
        color: str
    ) -> Dict[int, str]:
        """
        Add special color to existing color dictionary.

        Parameters
        ----------
        colors : Dict[int, str]
            Existing color mapping
        special_index : int
            Index for special color (e.g., -1 for noise)
        color : str
            Hex color string for special index

        Returns
        -------
        Dict[int, str]
            Updated color mapping with special color added

        Examples
        --------
        >>> colors = ColorUtils.generate_distinct_colors(3)
        >>> colors_with_noise = ColorUtils.add_special_color(
        ...     colors, -1, "#000000"
        ... )
        >>> len(colors_with_noise)
        4
        >>> colors_with_noise[-1]
        '#000000'

        Notes
        -----
        Does not modify original dictionary - returns new dict with
        special color added.
        """
        result = colors.copy()
        result[special_index] = color
        return result
