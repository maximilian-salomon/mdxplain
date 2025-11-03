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
cluster-specific color mapping, colormap selection, and cluster ID parsing.
Uses central ColorUtils for color generation to avoid code redundancy.
"""

import re
from typing import Dict, List

from ...utils.color_utils import ColorUtils


# Color for noise points in clustering (cluster label -1)
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

        Generates perceptually distinct colors using central color_utils.
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
        - Uses central ColorUtils.generate_distinct_colors() for consistency
        - Hue varies across full spectrum (0-1)
        - Saturation varies 0.65-0.85 for visual variety
        - Value (brightness) varies 0.75-0.90 for readability
        - Deterministic: same n_clusters always gives same colors
        - Noise color (index -1) is black (#000000) when include_noise=True
        """
        colors = ColorUtils.generate_distinct_colors(n_clusters)
        if include_noise:
            return ColorUtils.add_special_color(colors, -1, NOISE_COLOR)
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

    @staticmethod
    def parse_cluster_id(name: str) -> int:
        """
        Parse cluster ID from name string (case-insensitive).

        Searches for pattern "cluster_X" or "Cluster X" anywhere in the
        name. Returns first cluster ID found, -1 if none.

        Parameters
        ----------
        name : str
            Name to parse (e.g., comparison name, DataSelector name)

        Returns
        -------
        int
            Cluster ID (>=0) if found, -1 if no cluster pattern found

        Examples
        --------
        >>> ColorMappingHelper.parse_cluster_id("cluster_0_vs_rest")
        0
        >>> ColorMappingHelper.parse_cluster_id("Cluster_5_vs_rest")
        5
        >>> ColorMappingHelper.parse_cluster_id("cluster 3 analysis")
        3
        >>> ColorMappingHelper.parse_cluster_id("my_cluster_7_vs_cluster_2")
        7
        >>> ColorMappingHelper.parse_cluster_id("folded")
        -1

        Notes
        -----
        Uses regex pattern r'cluster[_\ |bsol| s]*(\ |bsol| d+)' with re.IGNORECASE:
        
        - Matches "cluster" (case-insensitive)
        - Followed by optional underscore or space
        - Followed by one or more digits

        Returns first match found. Useful for distinguishing cluster-based
        from non-cluster entities across all visualization types.
        """
        match = re.search(r'cluster[_\s]*(\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return -1

    @staticmethod
    def create_data_selector_color_mapping(
        data_selector_names: List[str]
    ) -> Dict[str, str]:
        """
        Create color mapping for DataSelectors with cluster consistency.

        Cluster-based DataSelectors retain their exact cluster colors.
        Non-cluster DataSelectors get free colors from the same palette.
        Ensures color consistency across all visualizations.

        Parameters
        ----------
        data_selector_names : List[str]
            List of DataSelector names (e.g., ["cluster_0", "cluster_1", "folded"])

        Returns
        -------
        Dict[str, str]
            Mapping of data_selector_name -> color_hex

        Examples
        --------
        >>> selector_names = ["cluster_1", "cluster_3", "folded", "unfolded"]
        >>> colors = ColorMappingHelper.create_data_selector_color_mapping(selector_names)
        >>> # cluster_1 gets color_1, cluster_3 gets color_3
        >>> # folded gets color_0, unfolded gets color_2 (free colors)

        Notes
        -----
        Algorithm:
        1. Parse all DataSelector names to extract cluster IDs
        2. Determine max_cluster_id and total DataSelectors
        3. Generate sufficient colors from get_cluster_colors()
        4. Assign cluster-DataSelectors their exact cluster color
        5. Assign non-cluster DataSelectors free colors (not used by clusters)

        Ensures consistency with cluster plots across all visualizations.
        Used in violin plots, comparison visualizations, etc.
        """
        sorted_names = sorted(data_selector_names)

        # Phase 1: Identify cluster DataSelectors and their IDs
        cluster_selectors = {}  # {selector_name: cluster_id}
        non_cluster_selectors = []
        max_cluster_id = -1

        for selector_name in sorted_names:
            cluster_id = ColorMappingHelper.parse_cluster_id(selector_name)
            if cluster_id >= 0:
                cluster_selectors[selector_name] = cluster_id
                max_cluster_id = max(max_cluster_id, cluster_id)
            else:
                non_cluster_selectors.append(selector_name)

        # Phase 2: Generate enough colors (without noise color)
        n_colors_needed = max(max_cluster_id + 1, len(sorted_names))
        all_colors = ColorMappingHelper.get_cluster_colors(
            n_colors_needed, include_noise=False
        )

        # Phase 3: Assign colors
        selector_colors = {}
        used_color_indices = set()

        # 3a: Cluster DataSelectors get their exact cluster color
        for selector_name, cluster_id in cluster_selectors.items():
            selector_colors[selector_name] = all_colors[cluster_id]
            used_color_indices.add(cluster_id)

        # 3b: Non-cluster DataSelectors get free colors
        free_color_indices = [
            i for i in range(n_colors_needed)
            if i not in used_color_indices
        ]

        for i, selector_name in enumerate(non_cluster_selectors):
            color_idx = free_color_indices[i]
            selector_colors[selector_name] = all_colors[color_idx]

        return selector_colors
