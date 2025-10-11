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
Helper class for color mapping in plots.

Provides consistent color schemes across all visualizations, including
dynamically generated cluster colors and colormap selection for landscapes.
"""

import colorsys
from typing import Dict


# Color for noise points (cluster label -1)
NOISE_COLOR = "#000000"


class ColorMappingHelper:
    """
    Helper class for color mapping in plot visualizations.

    Provides methods for generating distinct colors for any number of clusters
    using HSV color space, and selecting appropriate colormaps for landscapes.

    Examples
    --------
    >>> # Get colors for 5 clusters
    >>> colors = ColorMappingHelper.get_cluster_colors(5)
    >>> colors[0]
    '#bf4040'

    >>> # Get colormap for energy landscape
    >>> cmap = ColorMappingHelper.get_landscape_colormap(energy_values=True)
    >>> cmap
    'viridis_r'
    """

    @staticmethod
    def get_cluster_colors(n_clusters: int, include_noise: bool = True) -> Dict[int, str]:
        """
        Get color mapping for cluster labels with unlimited distinct colors.

        Generates perceptually distinct colors using HSV color space.
        Colors are distributed evenly across the hue spectrum with
        variations in saturation and value for additional distinction.

        Parameters
        ----------
        n_clusters : int
            Number of clusters (excluding noise)
        include_noise : bool, default=True
            Include color for noise label (-1)

        Returns
        -------
        Dict[int, str]
            Mapping from cluster label to hex color

        Examples
        --------
        >>> # Generate colors for 3 clusters
        >>> colors = ColorMappingHelper.get_cluster_colors(3)
        >>> len(colors)
        4  # includes noise
        >>> -1 in colors
        True

        >>> # Without noise
        >>> colors = ColorMappingHelper.get_cluster_colors(2, include_noise=False)
        >>> len(colors)
        2

        >>> # Works for any number of clusters
        >>> colors = ColorMappingHelper.get_cluster_colors(100)
        >>> len(colors)
        101

        Notes
        -----
        - Uses HSV color space for perceptually distinct colors
        - Hue varies across full spectrum (0-1)
        - Saturation varies 0.65-0.85 for visual variety
        - Value (brightness) varies 0.75-0.90 for readability
        - Deterministic: same n_clusters always gives same colors
        """
        colors = {}

        if include_noise:
            colors[-1] = NOISE_COLOR

        for i in range(n_clusters):
            # Distribute hue evenly across color wheel
            hue = i / max(n_clusters, 1)

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
    def get_landscape_colormap(energy_values: bool = False) -> str:
        """
        Get appropriate colormap for landscape plot.

        Parameters
        ----------
        energy_values : bool, default=False
            If True, returns inverted colormap (low energy = dark)

        Returns
        -------
        str
            Matplotlib colormap name

        Examples
        --------
        >>> ColorMappingHelper.get_landscape_colormap()
        'viridis'

        >>> ColorMappingHelper.get_landscape_colormap(energy_values=True)
        'viridis_r'

        Notes
        -----
        Viridis is perceptually uniform and colorblind-safe.
        For energy landscapes, inverted version is used so that
        low energy (stable states) appear darker.
        """
        if energy_values:
            return "viridis_r"  # Inverted: dark = low energy
        return "viridis"
