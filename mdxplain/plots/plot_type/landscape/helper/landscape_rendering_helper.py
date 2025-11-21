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
Helper for rendering landscape plot elements.

Provides methods for rendering backgrounds, scatter plots, cluster regions,
and cluster centers in landscape visualizations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from . import EnergyCalculatorHelper
from ....helper.color_mapping_helper import ColorMappingHelper


class LandscapeRenderingHelper:
    """
    Helper class for landscape plot rendering operations.

    Provides static methods for rendering various landscape plot elements
    including backgrounds, scatter plots, and cluster visualizations.

    Examples
    --------
    >>> # Render energy background
    >>> LandscapeRenderingHelper.plot_energy_background(
    ...     ax, data_x, data_y, bins=50, temperature=310.15,
    ...     xlim=(-5, 5), ylim=(-5, 5)
    ... )

    >>> # Render cluster centers
    >>> LandscapeRenderingHelper.plot_centers(
    ...     ax, centers, cluster_ids, dim_x=0, dim_y=1,
    ...     cluster_colors=colors, marker='X', size=200
    ... )
    """

    @staticmethod
    def plot_energy_background(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        temperature: float,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        contour_label_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None
    ) -> None:
        """
        Plot free energy landscape background using KDE.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        bins : int
            Number of contour levels
        temperature : float
            Temperature in Kelvin
        xlim : Tuple[float, float]
            X-axis limits for grid calculation
        ylim : Tuple[float, float]
            Y-axis limits for grid calculation
        contour_label_fontsize : int, optional
            Font size for colorbar label (default: 10)
        tick_fontsize : int, optional
            Font size for the tick labels.

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> LandscapeRenderingHelper.plot_energy_background(
        ...     ax, data_x, data_y, 50, 310.15, (-5, 5), (-5, 5)
        ... )
        """
        # KDE-based energy calculation over extended grid
        X, Y, energy = EnergyCalculatorHelper.calculate_kde_energy_landscape(
            data_x, data_y, bins, temperature, xlim, ylim
        )

        vmin, vmax = EnergyCalculatorHelper.get_energy_range(energy)
        cmap = ColorMappingHelper.get_landscape_colormap(energy_values=True)

        # Plot with contourf
        cf = ax.contourf(
            X,
            Y,
            energy,
            levels=bins,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8
        )

        # Add colorbar
        cbar = plt.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=tick_fontsize or 10, pad=10 + (tick_fontsize - 10) * 0.5 if tick_fontsize else 5)

        labelpad = 15 + (contour_label_fontsize - 10) * 1.5 if contour_label_fontsize else 15
        
        cbar.set_label(
            'Free Energy Δ (kcal/mol)',
            rotation=270,
            labelpad=labelpad,
            fontsize=contour_label_fontsize or 10
        )

    @staticmethod
    def plot_density_background(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float]
    ) -> None:
        """
        Plot probability density background using KDE.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        bins : int
            Number of contour levels
        xlim : Tuple[float, float]
            X-axis limits for grid calculation
        ylim : Tuple[float, float]
            Y-axis limits for grid calculation

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> LandscapeRenderingHelper.plot_density_background(
        ...     ax, data_x, data_y, 50, (-5, 5), (-5, 5)
        ... )
        """
        # Get KDE grid over extended limits (no transformation)
        X, Y, density = EnergyCalculatorHelper.calculate_kde_grid(
            data_x, data_y, bins, xlim, ylim
        )

        cmap = ColorMappingHelper.get_landscape_colormap(energy_values=False)

        # Plot with contourf (matches energy plot styling)
        cf = ax.contourf(
            X,
            Y,
            density,
            levels=bins,
            cmap=cmap,
            alpha=0.8
        )

        # Add colorbar
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label('Probability Density', rotation=270, labelpad=15)

    @staticmethod
    def create_scatter(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        labels: Optional[np.ndarray],
        cluster_colors: Optional[Dict[int, str]],
        alpha: float,
        data_scatter: bool
    ) -> None:
        """
        Create scatter plot - clustered or gray.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        labels : Optional[numpy.ndarray]
            Cluster labels for each point (None for gray scatter)
        cluster_colors : Optional[Dict[int, str]]
            Color mapping for clusters
        alpha : float
            Point transparency
        data_scatter : bool
            Whether to show gray scatter when labels is None

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> # Clustered scatter
        >>> LandscapeRenderingHelper.create_scatter(
        ...     ax, data_x, data_y, labels, colors, alpha=0.6, data_scatter=True
        ... )

        >>> # Gray scatter
        >>> LandscapeRenderingHelper.create_scatter(
        ...     ax, data_x, data_y, None, None, alpha=0.3, data_scatter=True
        ... )
        """
        if labels is not None:
            # Clustered scatter
            df = pd.DataFrame({'x': data_x, 'y': data_y, 'cluster': labels})
            sns.scatterplot(
                data=df,
                x='x',
                y='y',
                hue='cluster',
                palette=cluster_colors,
                ax=ax,
                s=1,
                alpha=alpha,
                legend=False
            )
        elif data_scatter:
            # Gray scatter
            sns.scatterplot(
                x=data_x,
                y=data_y,
                ax=ax,
                color='gray',
                s=1,
                alpha=alpha,
                legend=False
            )

    @staticmethod
    def plot_cluster_voronoi(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        labels: np.ndarray,
        cluster_colors: Dict[int, str],
        bins: int,
        alpha: float = 0.3
    ) -> None:
        """
        Plot cluster regions as transparent filled contours using Voronoi assignment.

        Uses Voronoi-style nearest-neighbor assignment to create
        cluster regions on a grid.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        labels : numpy.ndarray
            Cluster labels for each point
        cluster_colors : Dict[int, str]
            Color mapping for clusters
        bins : int
            Grid resolution for cluster regions
        alpha : float, default=0.3
            Transparency for cluster regions

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> LandscapeRenderingHelper.plot_cluster_voronoi(
        ...     ax, data_x, data_y, labels, colors, bins=50, alpha=0.3
        ... )
        """
        # Create grid (same resolution as KDE)
        x_grid = np.linspace(data_x.min(), data_x.max(), bins)
        y_grid = np.linspace(data_y.min(), data_y.max(), bins)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Nearest neighbor assignment
        tree = cKDTree(np.vstack([data_x, data_y]).T)
        _, indices = tree.query(np.vstack([X.ravel(), Y.ravel()]).T)
        cluster_grid = labels[indices].reshape(X.shape)

        # Plot each cluster as transparent region
        for cluster_id in np.unique(labels):
            if cluster_id < 0:  # Skip noise
                continue
            mask = cluster_grid == cluster_id
            color = cluster_colors[cluster_id]
            cluster_masked = np.where(mask, cluster_id, np.nan)

            ax.contourf(
                X,
                Y,
                cluster_masked,
                levels=[cluster_id - 0.5, cluster_id + 0.5],
                colors=[color],
                alpha=alpha
            )

    @staticmethod
    def plot_cluster_density_contours(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        labels: np.ndarray,
        cluster_colors: Dict[int, str],
        bins: int,
        percentile_levels: List[int] = [20, 40, 60, 80],
        contour_label_fontsize: Optional[int] = None
    ) -> None:
        """
        Plot cluster density contours with percentile labels.

        Uses KDE to create smooth density contours for each cluster
        with labels showing percentage of points enclosed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        labels : numpy.ndarray
            Cluster labels for each point
        cluster_colors : Dict[int, str]
            Color mapping for clusters
        bins : int
            Grid resolution for KDE evaluation
        percentile_levels : List[int], default=[20, 40, 60, 80]
            Percentile levels for contour lines
        contour_label_fontsize : int, optional
            Font size for contour labels (default: 10)

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> LandscapeRenderingHelper.plot_cluster_density_contours(
        ...     ax, data_x, data_y, labels, colors, 50, [20, 40, 60, 80]
        ... )
        """
        # Create grid for KDE evaluation
        x_grid = np.linspace(data_x.min(), data_x.max(), bins)
        y_grid = np.linspace(data_y.min(), data_y.max(), bins)
        X, Y = np.meshgrid(x_grid, y_grid)

        for cluster_id in np.unique(labels):
            if cluster_id < 0:
                continue

            mask = labels == cluster_id
            cluster_points = np.vstack([data_x[mask], data_y[mask]])

            if cluster_points.shape[1] < 3:
                continue

            kde = gaussian_kde(cluster_points)
            point_densities = kde(cluster_points)
            levels_to_plot = np.percentile(point_densities, percentile_levels)

            density_grid = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

            CS = ax.contour(
                X, Y, density_grid,
                levels=levels_to_plot,
                colors=[cluster_colors[cluster_id]],
                linewidths=2
            )

            fmt = {
                level: f'{100-pct:.0f}%'
                for level, pct in zip(levels_to_plot, percentile_levels)
            }
            ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=contour_label_fontsize or 10)

    @staticmethod
    def plot_centers(
        ax,
        centers: np.ndarray,
        cluster_ids: List[int],
        dim_x: int,
        dim_y: int,
        cluster_colors: Dict[int, str],
        marker: str,
        size: int
    ) -> None:
        """
        Plot cluster centers.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        centers : numpy.ndarray
            Cluster center coordinates
        cluster_ids : List[int]
            Cluster IDs corresponding to centers
        dim_x : int
            X-axis dimension index
        dim_y : int
            Y-axis dimension index
        cluster_colors : Dict[int, str]
            Color mapping for clusters
        marker : str
            Marker style
        size : int
            Marker size

        Returns
        -------
        None
            Modifies axes in-place

        Examples
        --------
        >>> LandscapeRenderingHelper.plot_centers(
        ...     ax, centers, [0, 1, 2], 0, 1, colors, 'X', 200
        ... )
        """
        for idx, cluster_id in enumerate(cluster_ids):
            center_x = centers[idx, dim_x]
            center_y = centers[idx, dim_y]
            color = cluster_colors[cluster_id]

            ax.scatter(
                center_x, center_y,
                c=color,
                s=size,
                marker=marker,
                edgecolors='black',
                linewidths=1.5,
                zorder=10
            )
