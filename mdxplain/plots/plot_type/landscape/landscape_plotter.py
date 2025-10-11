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
Landscape plotter for decomposition data visualization.

Creates 2D landscape plots with optional clustering overlay,
cluster centers, and energy transformation.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from matplotlib.figure import Figure
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from .helper import EnergyCalculatorHelper, LayoutCalculatorHelper
from ...helper.color_mapping_helper import ColorMappingHelper
from ...helper.validation_helper import ValidationHelper
from ....utils.data_utils import DataUtils


class LandscapePlotter:
    """
    Plotter for decomposition landscape visualizations.

    Creates 2D projections of decomposition data
    with optional clustering overlay and energy transformation.

    Examples
    --------
    >>> # Basic landscape plot
    >>> plotter = LandscapePlotter(pipeline_data)
    >>> plotter.plot(
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1]
    ... )

    >>> # With clustering and energy
    >>> plotter.plot(
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1, 2, 3],
    ...     clustering_name="dbscan",
    ...     show_centers=True,
    ...     energy_values=True
    ... )
    """

    def __init__(self, pipeline_data, cache_dir: str = "./cache") -> None:
        """
        Initialize landscape plotter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        cache_dir : str, default="./cache"
            Directory for saving plot files

        Returns
        -------
        None
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = cache_dir

    def plot(
        self,
        decomposition_name: str,
        dimensions: List[int],
        clustering_name: Optional[str] = None,
        show_centers: bool = False,
        energy_values: bool = False,
        bins: int = 50,
        temperature: float = 310.15,
        alpha: float = 0.6,
        cluster_contour: bool = True,
        cluster_contour_voronoi: bool = False,
        data_scatter: bool = True,
        show_clusters: Union[str, List[int]] = "all",
        center_marker: str = 'X',
        center_size: int = 200,
        title: Optional[str] = None,
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        subplot_size: float = 6.0,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300
    ) -> Figure:
        """
        Create landscape plot(s) for decomposition data.

        Parameters
        ----------
        decomposition_name : str
            Name of decomposition to plot
        dimensions : List[int]
            Dimension indices to plot (must be even number)
        clustering_name : Optional[str], default=None
            Name of clustering for overlay
        show_centers : bool, default=False
            Show cluster centers (requires clustering_name)
        energy_values : bool, default=False
            Show free energy landscape instead of density
        bins : int, default=50
            Number of bins for histogram/energy calculation
        temperature : float, default=300.0
            Temperature in Kelvin for energy calculation
        alpha : float, default=0.6
            Transparency for scatter/contour overlays
        cluster_contour : bool, default=False
            Show clusters as transparent contours instead of scatter points
        cluster_contour_voronoi : bool, default=True
            Use Voronoi-style contours (True) or KDE-based density contours (False).
            Only applies when cluster_contour=True
        data_scatter : bool, default=True
            Show gray scatter points when no clustering
        show_clusters : Union[str, List[int]], default="all"
            Which clusters to display: "all" or list of cluster IDs.
            Colors remain consistent regardless of selection
        center_marker : str, default='X'
            Marker style for cluster centers
        center_size : int, default=200
            Marker size for cluster centers
        title : Optional[str], default=None
            Custom title (overrides auto-generated)
        xaxis_label : Optional[str], default=None
            Custom X-axis label (default: "Component {dim_x}")
        yaxis_label : Optional[str], default=None
            Custom Y-axis label (default: "Component {dim_y}")
        xlim : Optional[Tuple[float, float]], default=None
            X-axis limits. If None, auto-calculated with 20% padding beyond data range
        ylim : Optional[Tuple[float, float]], default=None
            Y-axis limits. If None, auto-calculated with 20% padding beyond data range
        subplot_size : float, default=4.0
            Size of each subplot in inches
        save_fig : bool, default=False
            Save figure to file
        filename : Optional[str], default=None
            Custom filename (overrides auto-generated)
        file_format : str, default="png"
            File format for saving
        dpi : int, default=300
            Resolution for saved figure

        Returns
        -------
        matplotlib.figure.Figure
            Created figure object

        Raises
        ------
        ValueError
            If inputs are invalid (via validation helpers)

        Examples
        --------
        >>> # Single 2D plot
        >>> fig = plotter.plot("pca", [0, 1])

        >>> # Multi-dimensional grid
        >>> fig = plotter.plot(
        ...     "pca",
        ...     [0, 1, 2, 3],
        ...     clustering_name="dbscan",
        ...     energy_values=True,
        ...     save_fig=True
        ... )

        >>> # Custom styling with contour clusters
        >>> fig = plotter.plot(
        ...     "tica",
        ...     [0, 1],
        ...     clustering_name="dbscan",
        ...     cluster_contour=True,
        ...     xaxis_label="Slow Mode 1",
        ...     yaxis_label="Slow Mode 2",
        ...     xlim=(-5, 5),
        ...     ylim=(-3, 3)
        ... )
        """
        # Validate all inputs using atomic validation methods
        decomp_obj = ValidationHelper.validate_decomposition_exists(
            self.pipeline_data, decomposition_name
        )
        n_components = decomp_obj.data.shape[1]

        ValidationHelper.validate_dimensions_list(
            dimensions, decomposition_name, n_components
        )
        ValidationHelper.validate_dimensions_for_layout(dimensions)
        ValidationHelper.validate_show_centers_requirement(
            show_centers, clustering_name
        )

        if clustering_name:
            n_frames_decomp = decomp_obj.data.shape[0]
            ValidationHelper.validate_clustering_compatibility(
                self.pipeline_data, clustering_name,
                decomposition_name, n_frames_decomp
            )

        # Load decomposition data
        decomp_data = decomp_obj.data

        # Load clustering data if specified
        if clustering_name:
            labels, centers, cluster_colors = self._load_clustering_data(
                clustering_name, show_centers
            )
            # Filter clusters while keeping colors consistent
            labels, centers, cluster_ids = self._filter_clusters(labels, centers, show_clusters)
        else:
            labels, centers, cluster_colors, cluster_ids = None, None, None, []

        # Setup figure layout
        dim_pairs = LayoutCalculatorHelper.create_dimension_pairs(dimensions)
        fig, axes, n_plots, n_rows, n_cols, fig_width, fig_height = self._setup_figure(
            dim_pairs, subplot_size
        )

        # Plot each dimension pair
        for idx, (dim_x, dim_y) in enumerate(dim_pairs):
            row, col = LayoutCalculatorHelper.get_subplot_position(
                idx, n_rows, n_cols
            )
            self._plot_single_landscape(
                axes[row, col],
                decomp_data,
                dim_x,
                dim_y,
                labels,
                centers,
                cluster_ids,
                cluster_colors,
                energy_values,
                bins,
                temperature,
                alpha,
                cluster_contour,
                cluster_contour_voronoi,
                data_scatter,
                center_marker,
                center_size,
                xaxis_label,
                yaxis_label,
                xlim,
                ylim
            )

        # Finalize figure
        self._finalize_figure(
            fig, axes, n_plots, n_rows, n_cols,
            title, decomposition_name, clustering_name, energy_values
        )

        # Calculate dynamic whitespace (fixed absolute inches → scales with figure size)
        left_inch = 0.3    # Fixed 0.3 inch on left
        right_inch = 0.8   # Fixed 0.8 inch on right (for legend)
        top_inch = 0.5     # Fixed 0.5 inch on top (for title)
        bottom_inch = 0.3  # Fixed 0.3 inch on bottom

        left = left_inch / fig_width
        right = 1 - (right_inch / fig_width)
        top = 1 - (top_inch / fig_height)
        bottom = bottom_inch / fig_height

        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        # Add central legend if clustering is present (after subplots_adjust)
        if clustering_name:
            self._add_central_legend(fig, cluster_colors, show_clusters, fig_width, right_inch)

        # Save if requested
        if save_fig:
            self._save_figure(
                fig,
                filename,
                decomposition_name,
                dimensions,
                clustering_name,
                show_centers,
                energy_values,
                file_format,
                dpi
            )

        return fig

    def _generate_landscape_filename(
        self,
        decomposition_name: str,
        dimensions: List[int],
        clustering_name: Optional[str] = None,
        show_centers: bool = False,
        energy_values: bool = False
    ) -> str:
        """
        Generate automatic filename for landscape plot.

        Parameters
        ----------
        decomposition_name : str
            Name of decomposition
        dimensions : List[int]
            Dimensions plotted
        clustering_name : Optional[str], default=None
            Name of clustering if used
        show_centers : bool, default=False
            Whether centers are shown
        energy_values : bool, default=False
            Whether energy values are used

        Returns
        -------
        str
            Generated filename without extension
        """
        dim_str = "-".join(map(str, dimensions))
        name = f"landscape_{decomposition_name}_dim{dim_str}"

        if clustering_name:
            name += f"_{clustering_name}"
        if show_centers:
            name += "_centers"
        if energy_values:
            name += "_energy"

        return name

    def _plot_single_landscape(
        self,
        ax,
        decomp_data: np.ndarray,
        dim_x: int,
        dim_y: int,
        labels: Optional[np.ndarray],
        centers: Optional[np.ndarray],
        cluster_ids: List[int],
        cluster_colors: Optional[Dict[int, str]],
        energy_values: bool,
        bins: int,
        temperature: float,
        alpha: float,
        cluster_contour: bool,
        cluster_contour_voronoi: bool,
        data_scatter: bool,
        center_marker: str,
        center_size: int,
        xaxis_label: Optional[str],
        yaxis_label: Optional[str],
        xlim: Optional[Tuple[float, float]],
        ylim: Optional[Tuple[float, float]]
    ) -> None:
        """
        Plot single 2D landscape on given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        decomp_data : numpy.ndarray
            Decomposition data
        dim_x : int
            X-axis dimension
        dim_y : int
            Y-axis dimension
        labels : Optional[numpy.ndarray]
            Cluster labels
        centers : Optional[numpy.ndarray]
            Cluster centers
        cluster_ids : List[int]
            Cluster IDs corresponding to centers
        cluster_colors : Optional[Dict[int, str]]
            Color mapping for clusters
        energy_values : bool
            Plot energy landscape
        bins : int
            Number of bins for background and cluster contours
        temperature : float
            Temperature for energy calculation
        alpha : float
            Transparency for overlays
        cluster_contour : bool
            Show clusters as contours instead of scatter
        cluster_contour_voronoi : bool
            Use Voronoi (True) or KDE density (False) for contours
        data_scatter : bool
            Show gray scatter when no clustering
        center_marker : str
            Marker for centers
        center_size : int
            Size of center markers
        xaxis_label : Optional[str]
            Custom X-axis label
        yaxis_label : Optional[str]
            Custom Y-axis label
        xlim : Optional[Tuple[float, float]]
            X-axis limits. If None, auto-calculated with 20% padding
        ylim : Optional[Tuple[float, float]]
            Y-axis limits. If None, auto-calculated with 20% padding

        Returns
        -------
        None
            Modifies ax in place
        """
        # Extract data for this projection
        data_x = decomp_data[:, dim_x]
        data_y = decomp_data[:, dim_y]

        # Auto-calculate limits with 20% padding if not provided
        if xlim is None:
            x_range = data_x.max() - data_x.min()
            xlim = (data_x.min() - 0.2 * x_range, data_x.max() + 0.2 * x_range)
        if ylim is None:
            y_range = data_y.max() - data_y.min()
            ylim = (data_y.min() - 0.2 * y_range, data_y.max() + 0.2 * y_range)

        # Plot background (energy or density) over extended limits
        if energy_values:
            self._plot_energy_background(ax, data_x, data_y, bins, temperature, xlim, ylim)
        else:
            self._plot_density_background(ax, data_x, data_y, bins, xlim, ylim)

        # Overlay cluster visualization
        if labels is not None and cluster_contour:
            if cluster_contour_voronoi:
                self._plot_cluster_voronoi(
                    ax, data_x, data_y, labels, cluster_colors, bins, alpha
                )
            else:
                self._plot_cluster_density_contours(
                    ax, data_x, data_y, labels, cluster_colors, bins
                )
        else:
            self._create_scatter(
                ax, data_x, data_y, labels, cluster_colors, alpha, data_scatter
            )

        # Plot cluster centers if present
        if centers is not None:
            self._plot_centers(
                ax, centers, cluster_ids, dim_x, dim_y, cluster_colors,
                center_marker, center_size
            )

        # Set axis styling
        self._set_axis_labels(ax, dim_x, dim_y, xaxis_label, yaxis_label)
        self._set_axis_limits(ax, xlim, ylim)

    def _plot_energy_background(
        self,
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        temperature: float,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float]
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

        Returns
        -------
        None
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
        cbar.set_label('Free Energy (kcal/mol)', rotation=270, labelpad=15)

    def _plot_density_background(
        self,
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

    def _create_scatter(
        self,
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

    def _plot_cluster_voronoi(
        self,
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

    def _plot_cluster_density_contours(
        self,
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        labels: np.ndarray,
        cluster_colors: Dict[int, str],
        bins: int,
        percentile_levels: List[int] = [20, 40, 60, 80]
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

        Returns
        -------
        None
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
            ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

    def _plot_centers(
        self,
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

    def _load_clustering_data(
        self,
        clustering_name: str,
        show_centers: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, str]]:
        """
        Load clustering labels, centers, and color mapping.

        Parameters
        ----------
        clustering_name : str
            Name of clustering
        show_centers : bool
            Whether to load cluster centers

        Returns
        -------
        labels : numpy.ndarray
            Cluster labels for each point
        centers : Optional[numpy.ndarray]
            Cluster center coordinates if show_centers=True
        cluster_colors : Dict[int, str]
            Color mapping for clusters
        """
        cluster_obj = self.pipeline_data.cluster_data[clustering_name]
        labels = cluster_obj.labels
        centers = cluster_obj.get_centers() if show_centers else None
        n_clusters = len(np.unique(labels[labels >= 0]))
        cluster_colors = ColorMappingHelper.get_cluster_colors(n_clusters)
        return labels, centers, cluster_colors

    def _filter_clusters(
        self,
        labels: np.ndarray,
        centers: Optional[np.ndarray],
        show_clusters: Union[str, List[int]]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """
        Filter clusters based on selection while keeping colors consistent.

        Parameters
        ----------
        labels : numpy.ndarray
            Original cluster labels
        centers : Optional[numpy.ndarray]
            Original cluster centers
        show_clusters : Union[str, List[int]]
            Clusters to show: "all" or list of cluster IDs

        Returns
        -------
        filtered_labels : numpy.ndarray
            Labels with non-selected clusters set to -1 (noise)
        filtered_centers : Optional[numpy.ndarray]
            Filtered centers (only selected clusters)
        cluster_ids : List[int]
            Cluster IDs corresponding to filtered centers

        Notes
        -----
        Color mapping remains based on ALL clusters for consistency
        """
        if show_clusters == "all":
            if centers is not None:
                cluster_ids = list(range(len(centers)))
            else:
                cluster_ids = []
            return labels, centers, cluster_ids

        filtered_labels = labels.copy()
        mask = np.isin(labels, show_clusters)
        filtered_labels[~mask] = -1

        if centers is not None:
            filtered_centers = centers[list(show_clusters)]
            cluster_ids = list(show_clusters)
        else:
            filtered_centers = None
            cluster_ids = []

        return filtered_labels, filtered_centers, cluster_ids

    def _setup_figure(
        self,
        dim_pairs: List[Tuple[int, int]],
        subplot_size: float
    ) -> Tuple[Figure, np.ndarray, int, int, int, float, float]:
        """
        Create figure and axes grid for subplots.

        Parameters
        ----------
        dim_pairs : List[Tuple[int, int]]
            List of dimension pairs to plot
        subplot_size : float
            Size of each subplot in inches

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure
        axes : numpy.ndarray
            Array of subplot axes
        n_plots : int
            Number of plots
        n_rows : int
            Number of subplot rows
        n_cols : int
            Number of subplot columns
        fig_width : float
            Figure width in inches
        fig_height : float
            Figure height in inches
        """
        n_plots = len(dim_pairs)
        n_rows, n_cols = LayoutCalculatorHelper.calculate_grid_layout(n_plots)

        # Apply minimum size constraint
        min_subplot_size = 4.0
        subplot_size_final = max(subplot_size, min_subplot_size)
        subplot_height = subplot_size_final
        subplot_width = subplot_size_final * 1.3

        fig_width = n_cols * subplot_width
        fig_height = n_rows * subplot_height

        # Calculate available size per subplot (includes colorbars, labels, etc.)
        available_height_per_subplot = fig_height / n_rows
        available_width_per_subplot = fig_width / n_cols

        # Calculate dynamic spacing (fixed absolute inches → scales with available size)
        hspace_inch = 0.8  # Fixed 0.8 inch vertical spacing
        wspace_inch = 0.8  # Fixed 0.8 inch horizontal spacing
        hspace = hspace_inch / available_height_per_subplot
        wspace = wspace_inch / available_width_per_subplot

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
            gridspec_kw={'hspace': hspace, 'wspace': wspace}
        )
        return fig, axes, n_plots, n_rows, n_cols, fig_width, fig_height

    def _finalize_figure(
        self,
        fig: Figure,
        axes: np.ndarray,
        n_plots: int,
        n_rows: int,
        n_cols: int,
        title: Optional[str],
        decomposition_name: str,
        clustering_name: Optional[str],
        energy_values: bool
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

        Returns
        -------
        None
        """
        total_subplots = n_rows * n_cols
        for idx in range(n_plots, total_subplots):
            row, col = LayoutCalculatorHelper.get_subplot_position(
                idx, n_rows, n_cols
            )
            fig.delaxes(axes[row, col])

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            landscape_type = "Energy" if energy_values else "Density"
            base_title = f"{decomposition_name.upper()} {landscape_type} Landscape"

            if clustering_name:
                full_title = (
                    f"{base_title} with Cluster Representation of {clustering_name}"
                )
            else:
                full_title = base_title

            fig.suptitle(full_title, fontsize=14, fontweight='bold')

    def _add_central_legend(
        self,
        fig: Figure,
        cluster_colors: Dict[int, str],
        show_clusters: Union[str, List[int]],
        fig_width: float,
        right_inch: float
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

        Returns
        -------
        None
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

        # Position legend with fixed gap in inches from plot edge
        legend_gap_inch = 0.1  # 0.1 inch gap between plot and legend
        # Calculate where plots end (right edge after subplots_adjust)
        plot_right = 1 - (right_inch / fig_width)
        # Legend starts 0.1 inch after plot edge
        legend_left_inch = plot_right * fig_width + legend_gap_inch
        legend_x = legend_left_inch / fig_width

        fig.legend(
            handles, labels,
            loc='center left',
            bbox_to_anchor=(legend_x, 0.5),
            frameon=True,
            fontsize=10
        )

    def _save_figure(
        self,
        fig: Figure,
        filename: Optional[str],
        decomposition_name: str,
        dimensions: List[int],
        clustering_name: Optional[str],
        show_centers: bool,
        energy_values: bool,
        file_format: str,
        dpi: int
    ) -> None:
        """
        Generate filename and save figure to file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : Optional[str]
            Custom filename or None for auto-generation
        decomposition_name : str
            Name of decomposition
        dimensions : List[int]
            Dimension indices plotted
        clustering_name : Optional[str]
            Name of clustering if used
        show_centers : bool
            Whether centers are shown
        energy_values : bool
            Whether energy values are used
        file_format : str
            File format for saving
        dpi : int
            Resolution for saved figure

        Returns
        -------
        None
        """
        if filename is None:
            filename = self._generate_landscape_filename(
                decomposition_name,
                dimensions,
                clustering_name,
                show_centers,
                energy_values
            )
        if not filename.endswith(f".{file_format}"):
            filename = f"{filename}.{file_format}"
        filepath = DataUtils.get_cache_file_path(filename, self.cache_dir)
        fig.savefig(filepath, dpi=dpi, format=file_format, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")

    def _set_axis_labels(
        self,
        ax,
        dim_x: int,
        dim_y: int,
        xaxis_label: Optional[str],
        yaxis_label: Optional[str]
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

        Returns
        -------
        None
        """
        xlabel = xaxis_label if xaxis_label else f"Component {dim_x}"
        ylabel = yaxis_label if yaxis_label else f"Component {dim_y}"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _set_axis_limits(
        self,
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
        """
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
