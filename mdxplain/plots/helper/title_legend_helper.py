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
Helper for plot title and legend management.

Provides shared utilities for consistent title wrapping, positioning,
and legend creation across density and violin plots.
"""

import textwrap
from typing import Optional, Dict
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


class TitleLegendHelper:
    """
    Helper class for plot title and legend operations.

    Provides static methods for wrapping long titles, adding figure-level
    titles with consistent positioning, and creating unified legends for
    DataSelector colors.

    Examples
    --------
    >>> # Wrap long title
    >>> wrapped = TitleLegendHelper.wrap_title("Very Long Feature Name", 20)
    >>> print(wrapped)
    Very Long Feature
    Name

    >>> # Add figure title
    >>> TitleLegendHelper.add_title(fig, "My Plot", title_y=0.98)

    >>> # Add DataSelector legend
    >>> colors = {"cluster_0": "#FF0000", "cluster_1": "#00FF00"}
    >>> TitleLegendHelper.add_legend(fig, colors, "Clusters", None)
    """

    @staticmethod
    def wrap_title(title: str, max_chars_per_line: int = 40) -> str:
        """
        Wrap long titles to multiple lines.

        Splits title into multiple lines if it exceeds max_chars_per_line,
        breaking at word boundaries to maintain readability.

        Parameters
        ----------
        title : str
            Original title text
        max_chars_per_line : int, default=40
            Maximum characters per line before wrapping

        Returns
        -------
        str
            Title with line breaks inserted

        Examples
        --------
        >>> wrapped = TitleLegendHelper.wrap_title("Very long feature name", 20)
        >>> print(wrapped)
        Very long feature
        name

        >>> short = TitleLegendHelper.wrap_title("Short", 20)
        >>> print(short)
        Short

        Notes
        -----
        Uses textwrap.wrap() which breaks at word boundaries.
        Empty titles return empty string.
        """
        lines = textwrap.wrap(title, width=max_chars_per_line)
        return '\n'.join(lines)

    @staticmethod
    def estimate_title_height(
        title: str,
        max_chars_per_line: int = 80,
        fontsize: int = 18
    ) -> tuple:
        """
        Wrap title and estimate height in inches.

        Wraps title text to multiple lines and estimates the vertical space
        required in inches based on number of lines and fontsize.

        Parameters
        ----------
        title : str
            Original title text
        max_chars_per_line : int, default=80
            Maximum characters per line before wrapping
        fontsize : int, default=18
            Font size in points (standard matplotlib suptitle default)

        Returns
        -------
        wrapped_title : str
            Title with line breaks inserted
        height_inches : float
            Estimated height in inches required for the wrapped title

        Examples
        --------
        >>> wrapped, height = TitleLegendHelper.estimate_title_height(
        ...     "Short Title"
        ... )
        >>> print(wrapped)
        Short Title
        >>> print(f"{height:.3f} inches")
        0.200 inches

        >>> wrapped, height = TitleLegendHelper.estimate_title_height(
        ...     "Very Long Title That Will Be Wrapped Into Multiple Lines",
        ...     max_chars_per_line=30
        ... )
        >>> lines = wrapped.count('\\n') + 1
        >>> print(f"Lines: {lines}, Height: {height:.3f} inches")
        Lines: 3, Height: 0.600 inches

        Notes
        -----
        Height calculation:
        - Converts fontsize from points to inches: fontsize / 72
        - Multiplies by 0.8 to account for line spacing factor
        - Total height = n_lines * (fontsize / 72) * 0.8

        This is an approximation. Actual rendering may vary slightly
        depending on matplotlib backend and font metrics.
        """
        wrapped = TitleLegendHelper.wrap_title(title, max_chars_per_line)
        n_lines = len(wrapped.split('\n'))
        # Convert fontsize to inches: points / 72
        # Apply spacing factor of 0.8 for line height
        height_inches = n_lines * (fontsize / 72) * 0.8
        return wrapped, height_inches

    @staticmethod
    def compute_title_max_chars_from_width(subplot_width_inches: float) -> int:
        """
        Compute maximum characters per line based on subplot width.

        Calculates appropriate title wrapping length based on actual
        measured subplot width in inches. Uses conservative estimate
        of character width to ensure titles fit within subplot bounds.

        Parameters
        ----------
        subplot_width_inches : float
            Actual subplot width in inches (measured from GridSpec position)

        Returns
        -------
        int
            Maximum characters per line, clamped to range [40, 150]

        Examples
        --------
        >>> # Narrow subplot (5 inches wide)
        >>> max_chars = TitleLegendHelper.compute_title_max_chars_from_width(5.0)
        >>> print(max_chars)
        40

        >>> # Medium subplot (10 inches wide)
        >>> max_chars = TitleLegendHelper.compute_title_max_chars_from_width(10.0)
        >>> print(max_chars)
        80

        >>> # Wide subplot (15 inches wide)
        >>> max_chars = TitleLegendHelper.compute_title_max_chars_from_width(15.0)
        >>> print(max_chars)
        120

        Notes
        -----
        Character width estimation:
        - Assumes ~8 characters per inch at typical font sizes (conservative)
        - Includes safety margin to prevent text overflow
        - Clamps result to reasonable range [40, 150] characters

        This estimate works well for proportional fonts at standard
        matplotlib title font sizes (14-18pt).
        """
        # Conservative estimate: ~8 chars per inch at typical font sizes
        # Includes safety margin to prevent overflow
        chars_per_inch = 8
        max_chars = int(subplot_width_inches * chars_per_inch)

        # Clamp to reasonable range
        return max(40, min(150, max_chars))

    @staticmethod
    def add_title(
        fig: Figure,
        title: Optional[str],
        title_y: float = 0.98,
        default_title: str = "Feature Importance Plot"
    ) -> None:
        """
        Add figure-level title with consistent positioning.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to add title to
        title : str, optional
            Custom title text. If None, uses default_title.
        title_y : float, default=0.98
            Y-position of title (relative to figure height, 0-1)
        default_title : str, default="Feature Importance Plot"
            Fallback title when title is None

        Returns
        -------
        None
            Modifies fig in place by adding suptitle

        Examples
        --------
        >>> fig = plt.figure()
        >>> TitleLegendHelper.add_title(fig, "My Analysis")

        >>> # With default title
        >>> TitleLegendHelper.add_title(fig, None)

        >>> # Custom positioning
        >>> TitleLegendHelper.add_title(fig, "Title", title_y=0.95)

        Notes
        -----
        Title is added using fig.suptitle() with:
        - fontsize=18
        - fontweight='bold'
        - y=title_y for vertical positioning
        """
        title_text = title if title else default_title
        fig.suptitle(title_text, fontsize=18, fontweight='bold', y=title_y)

    @staticmethod
    def add_legend(
        fig: Figure,
        data_selector_colors: Dict[str, str],
        legend_title: Optional[str],
        legend_labels: Optional[Dict[str, str]],
        contact_threshold: Optional[float] = None,
        legend_x: float = 0.98,
        legend_y: float = 0.94
    ) -> None:
        """
        Add figure-wide legend for DataSelectors and contact threshold.

        Creates unified legend showing DataSelector names with their colors
        and optionally a contact threshold line. Legend is positioned in
        figure coordinates for consistent placement.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to add legend to
        data_selector_colors : Dict[str, str]
            Mapping of DataSelector name to color hex code
        legend_title : str, optional
            Custom legend title. If None, uses "DataSelectors".
        legend_labels : Dict[str, str], optional
            Custom display names for DataSelectors.
            Maps original names to display names.
            Example: {"cluster_0": "Inactive", "cluster_1": "Active"}
        contact_threshold : float, optional
            Contact threshold value in Angstrom. If provided, adds a
            dashed red line entry to legend showing the threshold.
        legend_x : float, default=0.98
            X-position of legend (in figure coordinates, 0-1)
        legend_y : float, default=0.94
            Y-position of legend (in figure coordinates, 0-1)

        Returns
        -------
        None
            Modifies fig in place by adding legend

        Examples
        --------
        >>> colors = {"cluster_0": "#FF0000", "cluster_1": "#00FF00"}
        >>> TitleLegendHelper.add_legend(fig, colors, None, None)

        >>> # With custom labels and threshold
        >>> labels = {"cluster_0": "Active", "cluster_1": "Inactive"}
        >>> TitleLegendHelper.add_legend(
        ...     fig, colors, "States", labels,
        ...     contact_threshold=4.5
        ... )

        >>> # Custom positioning
        >>> TitleLegendHelper.add_legend(
        ...     fig, colors, "Clusters", None,
        ...     contact_threshold=None,
        ...     legend_x=0.95, legend_y=0.90
        ... )

        Notes
        -----
        Legend is created with:
        - Sorted DataSelector names (alphabetical)
        - Colored patches (alpha=0.7)
        - Optional contact threshold line (red, dashed)
        - fontsize=14, title_fontsize=16, title in bold
        - framealpha=0.9 for visibility
        - loc="upper left" with bbox_to_anchor for precise positioning
        """
        legend_handles = []

        # Add DataSelector handles
        for selector_name in sorted(data_selector_colors.keys()):
            display_name = (
                legend_labels.get(selector_name, selector_name)
                if legend_labels
                else selector_name
            )
            patch = Patch(
                facecolor=data_selector_colors[selector_name],
                alpha=0.7,
                label=display_name,
            )
            legend_handles.append(patch)

        # Add contact threshold handle if present
        if contact_threshold is not None:
            threshold_line = Line2D(
                [0], [0],
                color='red',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f'Contact Threshold ({contact_threshold:.1f} Å)'
            )
            legend_handles.append(threshold_line)

        legend_title_text = legend_title if legend_title else "DataSelectors"

        fig.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(legend_x, legend_y),
            fontsize=14,
            title=legend_title_text,
            title_fontproperties={'weight': 'bold', 'size': 16},
            framealpha=0.9,
        )
