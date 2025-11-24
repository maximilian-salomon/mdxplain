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

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from matplotlib.figure import Figure

from .helper import LayoutCalculatorHelper
from .helper.landscape_rendering_helper import LandscapeRenderingHelper
from .helper.landscape_styling_helper import LandscapeStylingHelper
from ...helper.validation_helper import ValidationHelper
from ...helper.clustering_data_helper import ClusteringDataHelper
from ...helper.svg_export_helper import SvgExportHelper
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
        show_centers: bool = True,
        energy_values: bool = True,
        use_kde: bool = False,
        mask_empty_bins: bool = True,
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
        dpi: int = 300,
        title_fontsize: Optional[int] = None,
        xlabel_fontsize: Optional[int] = None,
        ylabel_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None,
        legend_fontsize: Optional[int] = None,
        contour_label_fontsize: Optional[int] = None,
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
        energy_values : bool, default=True
            Show free energy landscape instead of density
        use_kde : bool, default=False
            Use KDE smoothing for background density estimation.

            **Default (False)**: Histogram-based - shows actual observations,
            preserves energy barriers, scientifically accurate.

            **KDE (True)**: Smooth visualization but can filter out small energy
            barriers and distort the landscape. Use only if you know what you do.
            NOT for quantitative analysis. A warning will be issued.
        mask_empty_bins : bool, default=True
            Mask bins without observations in the background (energy/density)
            as white/transparent. Set False to fill them with the maximum color
            for continuity.
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
            File format for saving (png, pdf, svg, etc.).
            When using 'svg', text elements remain editable in SVG editors.
        dpi : int, default=300
            Resolution for saved figure
        title_fontsize : int, optional
            Font size for figure title (default: 14)
        xlabel_fontsize : int, optional
            Font size for X-axis labels (default: 12)
        ylabel_fontsize : int, optional
            Font size for Y-axis labels (default: 12)
        tick_fontsize : int, optional
            Font size for axis tick labels (default: 10)
        legend_fontsize : int, optional
            Font size for legend entries (default: 10)
        contour_label_fontsize : int, optional
            Font size for energy contour labels (default: 10)

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

        # If no clustering is provided, centers cannot be shown – disable silently
        if clustering_name is None and show_centers:
            show_centers = False

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
            dim_pairs, subplot_size, tick_fontsize=tick_fontsize, xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize, clustering_name=clustering_name, legend_fontsize=legend_fontsize
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
                use_kde,
                mask_empty_bins,
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
                ylim,
                contour_label_fontsize,
                xlabel_fontsize,
                ylabel_fontsize,
                tick_fontsize
            )

        # Finalize figure
        LandscapeStylingHelper.finalize_figure(
            fig, axes, n_plots, n_rows, n_cols,
            title, decomposition_name, clustering_name, energy_values,
            title_fontsize, tick_fontsize
        )

        # Calculate dynamic whitespace (fixed absolute inches → scales with figure size)
        left_inch = 0.3    # Fixed 0.3 inch on left

        # Dynamic right margin based on legend font size
        base_right_inch = 0.8
        if clustering_name and legend_fontsize:
            # Add extra space for larger legend font sizes
            extra_right = (legend_fontsize - 10) * 0.5  # 0.5 inch per font size point above 10
            right_inch = base_right_inch + extra_right
        else:
            right_inch = base_right_inch

        top_inch = 0.5     # Fixed 0.5 inch on top (for title)
        bottom_inch = 0.3  # Fixed 0.3 inch on bottom

        left = left_inch / fig_width
        right = 1 - (right_inch / fig_width)
        top = 1 - (top_inch / fig_height)
        bottom = bottom_inch / fig_height

        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        # Add central legend if clustering is present (after subplots_adjust)
        if clustering_name:
            LandscapeStylingHelper.add_central_legend(
                fig, cluster_colors, show_clusters, fig_width, right_inch, legend_fontsize
            )

        # Save if requested
        if save_fig:
            # Configure SVG export for editable text
            SvgExportHelper.apply_svg_config_if_needed(file_format)

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
        use_kde: bool,
        mask_empty_bins: bool,
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
        ylim: Optional[Tuple[float, float]],
        contour_label_fontsize: Optional[int],
        xlabel_fontsize: Optional[int],
        ylabel_fontsize: Optional[int],
        tick_fontsize: Optional[int]
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
        use_kde : bool
            Use KDE smoothing for background (histogram is default)
        mask_empty_bins : bool
            Mask bins without observations in energy background
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
            LandscapeRenderingHelper.plot_energy_background(
                ax, data_x, data_y, bins, temperature, xlim, ylim,
                use_kde=use_kde,
                mask_empty_bins=mask_empty_bins,
                contour_label_fontsize=contour_label_fontsize,
                tick_fontsize=tick_fontsize
            )
        else:
            LandscapeRenderingHelper.plot_density_background(
                ax, data_x, data_y, bins, xlim, ylim,
                use_kde=use_kde,
                mask_empty_bins=mask_empty_bins,
                contour_label_fontsize=contour_label_fontsize,
                tick_fontsize=tick_fontsize
            )

        # Overlay cluster visualization
        if labels is not None and cluster_contour:
            if cluster_contour_voronoi:
                LandscapeRenderingHelper.plot_cluster_voronoi(
                    ax, data_x, data_y, labels, cluster_colors, bins, alpha
                )
            else:
                LandscapeRenderingHelper.plot_cluster_density_contours(
                    ax, data_x, data_y, labels, cluster_colors, bins, contour_label_fontsize=contour_label_fontsize
                )
        else:
            LandscapeRenderingHelper.create_scatter(
                ax, data_x, data_y, labels, cluster_colors, alpha, data_scatter
            )

        # Plot cluster centers if present
        if centers is not None:
            LandscapeRenderingHelper.plot_centers(
                ax, centers, cluster_ids, dim_x, dim_y, cluster_colors,
                center_marker, center_size
            )

        # Set axis styling
        LandscapeStylingHelper.set_axis_labels(ax, dim_x, dim_y, xaxis_label, yaxis_label, xlabel_fontsize,
                                               ylabel_fontsize)
        LandscapeStylingHelper.set_axis_limits(ax, xlim, ylim)

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
        labels, cluster_colors = ClusteringDataHelper.load_clustering_data(
            self.pipeline_data, clustering_name
        )
        cluster_obj = self.pipeline_data.cluster_data[clustering_name]
        centers = cluster_obj.get_centers() if show_centers else None
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
        subplot_size: float,
        tick_fontsize: Optional[int] = None,
        xlabel_fontsize: Optional[int] = None,
        ylabel_fontsize: Optional[int] = None,
        clustering_name: Optional[str] = None,
        legend_fontsize: Optional[int] = None,
    ) -> Tuple[Figure, np.ndarray, int, int, int, float, float]:
        """
        Create figure and axes grid for subplots.

        Parameters
        ----------
        dim_pairs : List[Tuple[int, int]]
            List of dimension pairs to plot
        subplot_size : float
            Size of each subplot in inches
        tick_fontsize : int, optional
            Font size for the tick labels.
        xlabel_fontsize : int, optional
            Font size for the x-axis label.
        ylabel_fontsize : int, optional
            Font size for the y-axis label.
        clustering_name : str, optional
            Name of clustering (for legend space calculation)
        legend_fontsize : int, optional
            Font size for legend entries (for space calculation)

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

        # Calculate maximum font size increase for scaling subplots
        max_font_increase = 0
        if tick_fontsize:
            max_font_increase = max(max_font_increase, tick_fontsize - 10)
        if xlabel_fontsize:
            max_font_increase = max(max_font_increase, xlabel_fontsize - 12)
        if ylabel_fontsize:
            max_font_increase = max(max_font_increase, ylabel_fontsize - 12)

        # Apply minimum size constraint
        min_subplot_size = 4.0
        subplot_size_final = max(subplot_size, min_subplot_size)

        # Add extra space for larger font sizes (0.15 inch per font point increase)
        font_scale_factor = max_font_increase * 0.15
        subplot_height = subplot_size_final + font_scale_factor
        subplot_width = (subplot_size_final + font_scale_factor) * 1.3

        fig_width = n_cols * subplot_width
        fig_height = n_rows * subplot_height

        # Add extra width for legend if clustering is present
        if clustering_name:
            base_legend_width = 2.0  # Base space for legend
            if legend_fontsize:
                # Add extra space for larger legend font sizes
                extra_legend_width = (legend_fontsize - 10) * 0.5
                fig_width += base_legend_width + extra_legend_width
            else:
                fig_width += base_legend_width

        # Calculate available size per subplot (includes colorbars, labels, etc.)
        available_height_per_subplot = fig_height / n_rows
        available_width_per_subplot = fig_width / n_cols

        # Calculate dynamic spacing (fixed absolute inches → scales with available size)
        hspace_inch = 0.8  # Fixed 0.8 inch vertical spacing
        wspace_inch = 0.8  # Fixed 0.8 inch horizontal spacing
        if tick_fontsize:
            hspace_inch += (tick_fontsize - 10) * 0.1
            wspace_inch += (tick_fontsize - 10) * 0.1
        if xlabel_fontsize:
            hspace_inch += (xlabel_fontsize - 12) * 0.1

        hspace = hspace_inch / available_height_per_subplot
        wspace = wspace_inch / available_width_per_subplot

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
            gridspec_kw={'hspace': hspace, 'wspace': wspace}
        )
        return fig, axes, n_plots, n_rows, n_cols, fig_width, fig_height

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
