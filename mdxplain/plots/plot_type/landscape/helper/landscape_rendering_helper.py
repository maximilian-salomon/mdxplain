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
        use_kde: bool = False,
        mask_empty_bins: bool = True,
        cmap=None,
        contour_label_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None
    ) -> None:
        """
        Plot free energy landscape background.

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
        use_kde : bool, default=False
            Use KDE smoothing for background. Histogram (default) shows
            actual observations and preserves energy barriers.
        mask_empty_bins : bool, default=True
            Mask bins without observations (white/transparent). If False,
            empty bins are filled with the maximum energy for continuous color.
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
        X, Y, energy = LandscapeRenderingHelper._calculate_energy_grid(
            data_x, data_y, bins, temperature, xlim, ylim, use_kde
        )

        vmin, vmax = EnergyCalculatorHelper.get_energy_range(energy)
        energy_plot = LandscapeRenderingHelper._prepare_energy_plot_data(
            energy, mask_empty_bins, vmin, vmax
        )
        if cmap is None:
            cmap = ColorMappingHelper.get_landscape_colormap(energy_values=True)

        cf = ax.contourf(X, Y, energy_plot, levels=bins, cmap=cmap,
                         vmin=vmin, vmax=vmax, alpha=0.8)

        LandscapeRenderingHelper._add_energy_colorbar(
            ax, cf, contour_label_fontsize, tick_fontsize
        )

    @staticmethod
    def _calculate_energy_grid(
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        temperature: float,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        use_kde: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate energy grid using KDE or histogram.

        Parameters
        ----------
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        bins : int
            Number of bins
        temperature : float
            Temperature in Kelvin
        xlim : Tuple[float, float]
            X-axis limits
        ylim : Tuple[float, float]
            Y-axis limits
        use_kde : bool
            Use KDE smoothing

        Returns
        -------
        X : numpy.ndarray
            X grid
        Y : numpy.ndarray
            Y grid
        energy : numpy.ndarray
            Energy values
        """
        if use_kde:
            import warnings
            warnings.warn(
                "KDE smoothing can filter out small energy barriers and distort "
                "the free energy landscape. Histograms show actual observations. "
                "Use KDE only if you know what you do.",
                UserWarning
            )
            return EnergyCalculatorHelper.calculate_kde_energy_landscape(
                data_x, data_y, bins, temperature, xlim, ylim
            )
        return EnergyCalculatorHelper.calculate_histogram_energy_landscape(
            data_x, data_y, bins, temperature, xlim, ylim
        )

    @staticmethod
    def _prepare_energy_plot_data(
        energy: np.ndarray,
        mask_empty_bins: bool,
        vmin: float,
        vmax: float
    ) -> np.ndarray:
        """
        Prepare energy data for plotting with masking or filling.

        Parameters
        ----------
        energy : numpy.ndarray
            Raw energy values
        mask_empty_bins : bool
            Whether to mask invalid values
        vmin : float
            Minimum energy value
        vmax : float
            Maximum energy value

        Returns
        -------
        numpy.ndarray
            Prepared energy data
        """
        if mask_empty_bins:
            return np.ma.masked_invalid(energy)
        return np.nan_to_num(energy, nan=vmax, posinf=vmax, neginf=vmin)

    @staticmethod
    def _add_energy_colorbar(
        ax,
        cf,
        contour_label_fontsize: Optional[int],
        tick_fontsize: Optional[int]
    ) -> None:
        """
        Add colorbar to energy plot with font size adjustments.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to add colorbar to
        cf
            Contour plot object
        contour_label_fontsize : Optional[int]
            Font size for colorbar label
        tick_fontsize : Optional[int]
            Font size for tick labels

        Returns
        -------
        None
        """
        cbar = plt.colorbar(cf, ax=ax)
        tick_size = tick_fontsize or 10
        cbar.ax.tick_params(
            labelsize=tick_size,
            pad=10 + (tick_fontsize - 10) * 0.5 if tick_fontsize else 5
        )

        label_size = contour_label_fontsize or 10
        labelpad = 15 + (contour_label_fontsize - 10) * 1.5 if contour_label_fontsize else 15

        cbar.set_label(
            'Free Energy Δ (kcal/mol)',
            rotation=270,
            labelpad=labelpad,
            fontsize=label_size
        )

    @staticmethod
    def plot_density_background(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        use_kde: bool = False,
        mask_empty_bins: bool = True,
        cmap=None,
        contour_label_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None
    ) -> None:
        """
        Plot probability density background.

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
        use_kde : bool, default=False
            Use KDE smoothing for background. Histogram (default) shows
            actual observations.
        mask_empty_bins : bool, default=True
            Mask bins without observations (white/transparent). If False,
            empty bins are filled with the maximum density color for continuity.
        contour_label_fontsize : int, optional
            Font size for colorbar label (default: 10)
        tick_fontsize : int, optional
            Font size for the tick labels.
        use_kde : bool, default=False
            Use KDE smoothing for background. Histogram (default) shows
            actual observations.

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
        X, Y, density = LandscapeRenderingHelper._calculate_density_grid(
            data_x, data_y, bins, xlim, ylim, use_kde
        )

        density_plot = LandscapeRenderingHelper._prepare_density_plot_data(
            density, mask_empty_bins
        )

        vmin, vmax = LandscapeRenderingHelper._calculate_density_color_bounds(
            density_plot
        )

        if cmap is None:
            cmap = ColorMappingHelper.get_landscape_colormap(energy_values=False)

        cf = ax.contourf(X, Y, density_plot, levels=bins, cmap=cmap,
                         vmin=vmin, vmax=vmax, alpha=0.8)

        LandscapeRenderingHelper._add_density_colorbar(
            ax, cf, contour_label_fontsize, tick_fontsize
        )

    @staticmethod
    def _calculate_density_grid(
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        use_kde: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate density grid using KDE or histogram.

        Parameters
        ----------
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        bins : int
            Number of bins
        xlim : Tuple[float, float]
            X-axis limits
        ylim : Tuple[float, float]
            Y-axis limits
        use_kde : bool
            Use KDE smoothing

        Returns
        -------
        X : numpy.ndarray
            X grid
        Y : numpy.ndarray
            Y grid
        density : numpy.ndarray
            Density values
        """
        if use_kde:
            return EnergyCalculatorHelper.calculate_kde_grid(
                data_x, data_y, bins, xlim, ylim
            )
        return EnergyCalculatorHelper.calculate_histogram_grid(
            data_x, data_y, bins, xlim, ylim
        )

    @staticmethod
    def _prepare_density_plot_data(
        density: np.ndarray,
        mask_empty_bins: bool
    ) -> np.ndarray:
        """
        Prepare density data for plotting with masking or filling.

        Parameters
        ----------
        density : numpy.ndarray
            Raw density values
        mask_empty_bins : bool
            Whether to mask empty/invalid values

        Returns
        -------
        numpy.ndarray
            Prepared density data
        """
        if mask_empty_bins:
            return np.ma.masked_where(
                (density <= 0) | ~np.isfinite(density),
                density
            )
        return np.nan_to_num(density, nan=0.0, neginf=0.0)

    @staticmethod
    def _calculate_density_color_bounds(
        density_plot: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate color scale bounds for density plot.

        Parameters
        ----------
        density_plot : numpy.ndarray
            Density data (may be masked)

        Returns
        -------
        vmin : float
            Minimum value for color scale
        vmax : float
            Maximum value for color scale
        """
        if np.ma.isMaskedArray(density_plot):
            finite_vals = density_plot.compressed()
        else:
            finite_vals = density_plot[np.isfinite(density_plot)]

        if finite_vals.size == 0:
            return 0.0, 1.0
        return finite_vals.min(), finite_vals.max()

    @staticmethod
    def _add_density_colorbar(
        ax,
        cf,
        contour_label_fontsize: Optional[int],
        tick_fontsize: Optional[int]
    ) -> None:
        """
        Add colorbar to density plot with font size adjustments.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to add colorbar to
        cf
            Contour plot object
        contour_label_fontsize : Optional[int]
            Font size for colorbar label
        tick_fontsize : Optional[int]
            Font size for tick labels

        Returns
        -------
        None
        """
        cbar = plt.colorbar(cf, ax=ax)
        tick_size = tick_fontsize or 10
        cbar.ax.tick_params(
            labelsize=tick_size,
            pad=10 + (tick_fontsize - 10) * 0.5 if tick_fontsize else 5
        )

        label_size = contour_label_fontsize or 10
        labelpad = 15 + (contour_label_fontsize - 10) * 1.5 if contour_label_fontsize else 15

        cbar.set_label(
            'Probability Density',
            rotation=270,
            labelpad=labelpad,
            fontsize=label_size
        )

    @staticmethod
    def create_scatter(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        labels: Optional[np.ndarray],
        cluster_colors: Optional[Dict[int, str]],
        alpha: float,
        data_scatter: bool,
        frame_tag_map: Optional[Dict[int, str]] = None,
        tag_colors: Optional[Dict[str, str]] = None,
        unselected_indices: Optional[List[int]] = None,
        scatter_size: int = 1
    ) -> None:
        """
        Create scatter plot - clustered, tag-based, or gray.

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
        frame_tag_map : Optional[Dict[int, str]], default=None
            Mapping from frame index to tag (for tag-based coloring)
        tag_colors : Optional[Dict[str, str]], default=None
            Mapping from tag to color (for tag-based coloring)
        unselected_indices : Optional[List[int]], default=None
            Indices of unselected points to plot in gray
        scatter_size : int, default=1
            Size of scatter points in matplotlib units

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

        >>> # Tag-based scatter
        >>> LandscapeRenderingHelper.create_scatter(
        ...     ax, data_x, data_y, None, None, 0.6, True,
        ...     frame_tag_map={0: "biased", 1: "unbiased"},
        ...     tag_colors={"biased": "#1f77b4"}
        ... )
        """
        LandscapeRenderingHelper._plot_unselected_if_needed(
            ax, data_x, data_y, unselected_indices, alpha, scatter_size
        )

        if frame_tag_map is not None:
            LandscapeRenderingHelper._plot_tag_colored_scatter(
                ax, data_x, data_y, frame_tag_map, tag_colors, alpha, scatter_size
            )
        elif labels is not None:
            LandscapeRenderingHelper._plot_cluster_colored_scatter(
                ax, data_x, data_y, labels, cluster_colors, alpha, scatter_size
            )
        elif data_scatter:
            LandscapeRenderingHelper._plot_gray_scatter(
                ax, data_x, data_y, alpha, scatter_size
            )

    @staticmethod
    def _plot_unselected_if_needed(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        unselected_indices: Optional[List[int]],
        alpha: float,
        scatter_size: int = 1
    ) -> None:
        """
        Plot unselected points in gray if requested.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        unselected_indices : Optional[List[int]]
            Indices of unselected points
        alpha : float
            Base alpha for colored points
        scatter_size : int, default=1
            Size of scatter points

        Returns
        -------
        None
        """
        if unselected_indices is not None and len(unselected_indices) > 0:
            ax.scatter(
                data_x[unselected_indices],
                data_y[unselected_indices],
                color='gray',
                s=scatter_size,
                alpha=alpha * 0.4,
                zorder=1
            )

    @staticmethod
    def _plot_tag_colored_scatter(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        frame_tag_map: Dict[int, str],
        tag_colors: Dict[str, str],
        alpha: float,
        scatter_size: int = 1
    ) -> None:
        """
        Plot scatter with tag-based coloring.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        frame_tag_map : Dict[int, str]
            Frame index to tag mapping
        tag_colors : Dict[str, str]
            Tag to color mapping
        alpha : float
            Point transparency
        scatter_size : int, default=1
            Size of scatter points

        Returns
        -------
        None
        """
        indices = list(frame_tag_map.keys())
        tags = [frame_tag_map[i] for i in indices]
        df = pd.DataFrame({
            'x': data_x[indices],
            'y': data_y[indices],
            'tag': tags
        })
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='tag',
            palette=tag_colors,
            ax=ax,
            s=scatter_size,
            alpha=alpha,
            legend=False,
            zorder=2
        )

    @staticmethod
    def _plot_cluster_colored_scatter(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        labels: np.ndarray,
        cluster_colors: Dict[int, str],
        alpha: float,
        scatter_size: int = 1
    ) -> None:
        """
        Plot scatter with cluster-based coloring.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        labels : numpy.ndarray
            Cluster labels
        cluster_colors : Dict[int, str]
            Cluster to color mapping
        alpha : float
            Point transparency
        scatter_size : int, default=1
            Size of scatter points

        Returns
        -------
        None
        """
        df = pd.DataFrame({'x': data_x, 'y': data_y, 'cluster': labels})
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='cluster',
            palette=cluster_colors,
            ax=ax,
            s=scatter_size,
            alpha=alpha,
            legend=False,
            zorder=2
        )

    @staticmethod
    def _plot_gray_scatter(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        alpha: float,
        scatter_size: int = 1
    ) -> None:
        """
        Plot gray scatter points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        alpha : float
            Point transparency
        scatter_size : int, default=1
            Size of scatter points

        Returns
        -------
        None
        """
        sns.scatterplot(
            x=data_x,
            y=data_y,
            ax=ax,
            color='gray',
            s=scatter_size,
            alpha=alpha,
            legend=False,
            zorder=2
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
    def plot_tag_density_contours(
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        frame_tag_map: Dict[int, str],
        tag_colors: Optional[Dict[str, str]],
        bins: int,
        percentile_levels: List[int] = [20, 40, 60, 80],
        contour_label_fontsize: Optional[int] = None
    ) -> None:
        """
        Plot tag density contours with percentile labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        frame_tag_map : Dict[int, str]
            Mapping from frame index to tag
        tag_colors : Optional[Dict[str, str]]
            Tag -> color mapping
        bins : int
            Grid resolution for KDE evaluation
        percentile_levels : List[int], default=[20, 40, 60, 80]
            Percentile levels for contour lines
        contour_label_fontsize : int, optional
            Font size for contour labels
        """
        if not frame_tag_map:
            return
        if tag_colors is None:
            return

        indices = list(frame_tag_map.keys())
        tags = [frame_tag_map[i] for i in indices]

        x_vals = data_x[indices]
        y_vals = data_y[indices]

        # Create grid for KDE evaluation
        x_grid = np.linspace(data_x.min(), data_x.max(), bins)
        y_grid = np.linspace(data_y.min(), data_y.max(), bins)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Ordered unique tags
        seen = set()
        ordered_tags = []
        for tag in tags:
            if tag in seen:
                continue
            seen.add(tag)
            ordered_tags.append(tag)

        for tag in ordered_tags:
            mask = [t == tag for t in tags]
            tag_points = np.vstack([x_vals[mask], y_vals[mask]])

            if tag_points.shape[1] < 3:
                continue

            kde = gaussian_kde(tag_points)
            point_densities = kde(tag_points)
            levels_to_plot = np.percentile(point_densities, percentile_levels)

            density_grid = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

            color = tag_colors.get(tag, "#808080")
            CS = ax.contour(
                X, Y, density_grid,
                levels=levels_to_plot,
                colors=[color],
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
