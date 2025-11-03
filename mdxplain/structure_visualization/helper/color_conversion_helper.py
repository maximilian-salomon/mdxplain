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
Color conversion utilities for structure visualization.

This module provides utilities for converting between color formats
(HEX, RGB, PyMOL) and mixing colors for overlapping features.
"""

from typing import List


class ColorConversionHelper:
    """
    Helper class for color format conversions and mixing.

    Provides static methods for converting HEX colors to various formats
    (RGB tuples, PyMOL format) and for mixing multiple colors via RGB
    averaging. Used by both NGLView and PyMOL visualizations.

    Examples
    --------
    >>> # Convert HEX to RGB
    >>> rgb = ColorConversionHelper.hex_to_rgb("#ff0000")
    >>> print(rgb)
    (255, 0, 0)

    >>> # Mix colors
    >>> mixed = ColorConversionHelper.mix_hex_colors(["#ff0000", "#0000ff"])
    >>> print(mixed)
    '#7f007f'

    >>> # Convert to PyMOL format
    >>> pymol_color = ColorConversionHelper.hex_to_pymol_rgb("#ff0000")
    >>> print(pymol_color)
    '0xff0000'
    """

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """
        Convert HEX color to RGB tuple.

        Parameters
        ----------
        hex_color : str
            HEX color string (e.g., "#ff0000" or "ff0000")

        Returns
        -------
        tuple
            RGB values (r, g, b) as integers (0-255)

        Examples
        --------
        >>> ColorConversionHelper.hex_to_rgb("#ff0000")
        (255, 0, 0)
        >>> ColorConversionHelper.hex_to_rgb("00ff00")
        (0, 255, 0)

        Notes
        -----
        Leading '#' is automatically stripped if present.
        """
        hex_clean = hex_color.lstrip('#')
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
        return (r, g, b)

    @staticmethod
    def hex_to_pymol_rgb(hex_color: str) -> str:
        """
        Convert HEX color to PyMOL RGB format.

        PyMOL uses hexadecimal color format with '0x' prefix instead
        of '#'. This method performs the conversion.

        Parameters
        ----------
        hex_color : str
            HEX color string (e.g., "#ff0000" or "ff0000")

        Returns
        -------
        str
            PyMOL RGB format (e.g., "0xff0000")

        Examples
        --------
        >>> ColorConversionHelper.hex_to_pymol_rgb("#ff0000")
        '0xff0000'
        >>> ColorConversionHelper.hex_to_pymol_rgb("00ff00")
        '0x00ff00'

        Notes
        -----
        Leading '#' is automatically stripped if present before
        adding PyMOL's '0x' prefix.
        """
        hex_clean = hex_color.lstrip('#')
        return f"0x{hex_clean}"

    @staticmethod
    def calculate_average_rgb(rgb_values: List[tuple]) -> tuple:
        """
        Calculate average RGB values from multiple RGB tuples.

        Averages RGB components independently using integer division.
        Used as intermediate step in color mixing.

        Parameters
        ----------
        rgb_values : List[tuple]
            List of RGB tuples, each containing (r, g, b) integers

        Returns
        -------
        tuple
            Average (r, g, b) values as integers

        Examples
        --------
        >>> rgb_list = [(255, 0, 0), (0, 0, 255)]
        >>> ColorConversionHelper.calculate_average_rgb(rgb_list)
        (127, 0, 127)

        Notes
        -----
        Uses integer division (//) for averaging, which may cause
        small rounding differences compared to float averaging.
        """
        n = len(rgb_values)
        avg_r = sum(rgb[0] for rgb in rgb_values) // n
        avg_g = sum(rgb[1] for rgb in rgb_values) // n
        avg_b = sum(rgb[2] for rgb in rgb_values) // n
        return (avg_r, avg_g, avg_b)

    @staticmethod
    def mix_hex_colors(hex_colors: List[str]) -> str:
        """
        Calculate RGB average of multiple HEX colors.

        Mixes multiple colors by converting to RGB, averaging each
        component independently, and converting back to HEX. Used
        for visualizing overlapping features with blended colors.

        Parameters
        ----------
        hex_colors : List[str]
            List of HEX color strings (e.g., ["#ff0000", "#0000ff"])

        Returns
        -------
        str
            Mixed color as HEX string (#RRGGBB)

        Examples
        --------
        >>> # Mix red and blue to get purple
        >>> mixed = ColorConversionHelper.mix_hex_colors(
        ...     ["#ff0000", "#0000ff"]
        ... )
        >>> print(mixed)
        '#7f007f'

        >>> # Mix three colors
        >>> mixed = ColorConversionHelper.mix_hex_colors(
        ...     ["#ff0000", "#00ff00", "#0000ff"]
        ... )

        >>> # Empty list returns gray
        >>> ColorConversionHelper.mix_hex_colors([])
        '#808080'

        Notes
        -----
        - Empty list returns neutral gray (#808080)
        - Integer division may cause small rounding differences
        - All colors weighted equally in averaging
        """
        if not hex_colors:
            return "#808080"

        rgb_values = [
            ColorConversionHelper.hex_to_rgb(color)
            for color in hex_colors
        ]
        avg_r, avg_g, avg_b = ColorConversionHelper.calculate_average_rgb(
            rgb_values
        )

        return f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"
