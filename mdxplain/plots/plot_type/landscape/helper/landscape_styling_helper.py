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
Helper for styling landscape plots.

Provides methods for finalizing figures, adding legends, setting axis labels,
and saving landscape visualizations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import LayoutCalculatorHelper


class LandscapeStylingHelper:
    """
    Helper class for landscape plot styling operations.

    Provides static methods for finalizing figures, adding legends,
    and styling axes in landscape visualizations.

    Examples
    --------
    >>> # Finalize figure
    >>> LandscapeStylingHelper.finalize_figure(
    ...     fig, axes, n_plots, n_rows, n_cols, title,
    ...     "pca", "dbscan", energy_values=True
    ... )

    >>> # Add legend
    >>> LandscapeStylingHelper.add_central_legend(
    ...     fig, cluster_colors, "all", fig_width=12.0, right_inch=0.8
    ... )
    """

    @staticmethod
    def finalize_figure(
        fig: Figure,
        axes: np.ndarray,
        n_plots: int,
        n_rows: int,
        n_cols: int,
        title: Optional[str],
        decomposition_name: str,
        clustering_name: Optional[str],
        energy_values: bool,
        title_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None,
    ) -> None:
        """
        Remove unused subplots and set overall title.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to finalize
        axes : numpy.ndarray
            Array of subplot axes
        n_plots : int
            Number of actual plots
        n_rows : int
            Number of subplot rows
        n_cols : int
            Number of subplot columns
        title : Optional[str]
            Custom title or None
        decomposition_name : str
            Name of decomposition
        clustering_name : Optional[str]
            Name of clustering if used
        energy_values : bool
            Whether energy landscape is shown
        title_fontsize : int, optional
            Font size for the figure title.
        tick_fontsize : int, optional
            Font size for the tick labels.

        Returns
        -------
        None
            Modifies figure in-place

        Examples
        --------
        >>> LandscapeStylingHelper.finalize_figure(
        ...     fig, axes, 4, 2, 2, None, "pca", "dbscan", True
        ... )
        """
        total_subplots = n_rows * n_cols
        for idx in range(n_plots, total_subplots):
            row, col = LayoutCalculatorHelper.get_subplot_position(
                idx, n_rows, n_cols
            )
            fig.delaxes(axes[row, col])

        for ax in axes.flatten():
            ax.tick_params(axis='both', labelsize=tick_fontsize or 10)

        if title:
            fig.suptitle(title, fontsize=title_fontsize or 14, fontweight='bold')
        else:
            landscape_type = "Energy" if energy_values else "Density"
            base_title = f"{decomposition_name.upper()} {landscape_type} Landscape"

            if clustering_name:
                full_title = (
                    f"{base_title} with Cluster Representation of {clustering_name}"
                )
            else:
                full_title = base_title

            fig.suptitle(full_title, fontsize=title_fontsize or 14, fontweight='bold')

    @staticmethod
    def add_central_legend(
        fig: Figure,
        cluster_colors: Dict[int, str],
        show_clusters: Union[str, List[int]],
        fig_width: float,
        right_inch: float,
        legend_fontsize: Optional[int] = None
    ) -> None:
        """
        Add central legend for all displayed clusters.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to add legend to
        cluster_colors : Dict[int, str]
            Color mapping for all clusters
        show_clusters : Union[str, List[int]]
            Which clusters are displayed
        fig_width : float
            Figure width in inches
        right_inch : float
            Right margin in inches
        legend_fontsize : int, optional
            Font size for the legend entries.

        Returns
        -------
        None
            Modifies figure in-place

        Examples
        --------
        >>> LandscapeStylingHelper.add_central_legend(
        ...     fig, colors, "all", 12.0, 0.8
        ... )
        """
        if show_clusters == "all":
            cluster_ids = sorted([k for k in cluster_colors.keys() if k >= 0])
        else:
            cluster_ids = sorted(show_clusters)

        handles = []
        labels = []
        for cluster_id in cluster_ids:
            handles.append(
                plt.Line2D([0], [0], color=cluster_colors[cluster_id], lw=4)
            )
            labels.append(f'Cluster {cluster_id}')

        # Position legend with dynamic gap based on font size
        base_gap_inch = 0.1
        # Add extra gap for larger font sizes
        font_size = legend_fontsize or 10
        extra_gap = (font_size - 10) * 0.05  # 0.05 inch per font size point above 10
        legend_gap_inch = base_gap_inch + extra_gap

        # Calculate where plots end (right edge after subplots_adjust)
        plot_right = 1 - (right_inch / fig_width)
        # Legend starts with dynamic gap after plot edge
        legend_left_inch = plot_right * fig_width + legend_gap_inch
        legend_x = legend_left_inch / fig_width

        fig.legend(
            handles, labels,
            loc='center left',
            bbox_to_anchor=(legend_x, 0.5),
            frameon=True,
            fontsize=legend_fontsize or 10
        )

    @staticmethod
    def set_axis_labels(
        ax,
        dim_x: int,
        dim_y: int,
        xaxis_label: Optional[str],
        yaxis_label: Optional[str],
        xlabel_fontsize: Optional[int] = None,
        ylabel_fontsize: Optional[int] = None,
    ) -> None:
        """
        Set x and y axis labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to set labels on
        dim_x : int
            X-axis dimension index
        dim_y : int
            Y-axis dimension index
        xaxis_label : Optional[str]
            Custom X-axis label
        yaxis_label : Optional[str]
            Custom Y-axis label
        xlabel_fontsize : int, optional
            Font size for the x-axis label.
        ylabel_fontsize : int, optional
            Font size for the y-axis label.

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> LandscapeStylingHelper.set_axis_labels(
        ...     ax, 0, 1, "PC1", "PC2"
        ... )
        """
        xlabel = xaxis_label if xaxis_label else f"Component {dim_x}"
        ylabel = yaxis_label if yaxis_label else f"Component {dim_y}"

        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize or 12)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize or 12)

    @staticmethod
    def set_axis_limits(
        ax,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float]
    ) -> None:
        """
        Set axis limits.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to set limits on
        xlim : Tuple[float, float]
            X-axis limits (auto-calculated if not provided to plot())
        ylim : Tuple[float, float]
            Y-axis limits (auto-calculated if not provided to plot())

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> LandscapeStylingHelper.set_axis_limits(ax, (-5, 5), (-3, 3))
        """
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
