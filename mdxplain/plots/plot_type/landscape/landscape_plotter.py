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
from .helper.landscape_tag_coloring_helper import LandscapeTagColoringHelper
from ...helper.validation_helper import ValidationHelper
from ...helper.clustering_data_helper import ClusteringDataHelper
from ...helper.svg_export_helper import SvgExportHelper
from ....utils.data_utils import DataUtils
from ....decomposition.entities.decomposition_data import DecompositionData


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
        bins: Union[int, str] = "auto",
        temperature: float = 310.15,
        alpha: float = 0.6,
        cluster_contour: bool = True,
        cluster_contour_voronoi: bool = False,
        data_scatter: bool = True,
        show_clusters: Union[str, List[int]] = "all",
        tag_coloring: Optional[List[str]] = None,
        scatter_show_all: bool = False,
        center_marker: str = 'X',
        center_size: int = 200,
        scatter_size: int = 1,
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
        bins : int or str, default="auto"
            Number of bins for histogram/energy calculation.
            Use "auto" to automatically determine optimal bin count using numpy's
            histogram_bin_edges algorithm.
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
        tag_coloring : Optional[List[str]], default=None
            Color scatter points by trajectory tags instead of clusters.
            Provide list of tags, e.g., ["biased", "unbiased"].
            If a frame matches multiple tags, the last tag in the list is used.
            When set, overrides cluster-based coloring from clustering_name
        scatter_show_all : bool, default=False
            Show unselected points in gray (applies to both cluster and tag mode):
            - **Cluster mode:** When show_clusters=[0,1], other clusters/noise shown in gray
            - **Tag mode:** When tag_coloring=["biased"], frames without this tag shown in gray
            - False (default): Only show selected points, hide others (current behavior)
        center_marker : str, default='X'
            Marker style for cluster centers
        center_size : int, default=200
            Marker size for cluster centers
        scatter_size : int, default=1
            Size of scatter points in matplotlib units. Applies to all scatter
            points (cluster-colored, tag-colored, gray, and unselected).
            Typical values: 1 (tiny), 5-10 (small), 20-50 (medium), 100+ (large).
            Note: Cluster centers use `center_size` parameter separately.
        title : Optional[str], default=None
            Custom title (overrides auto-generated)
        xaxis_label : Optional[str], default=None
            Custom X-axis label (default: "Component {dim_x}")
        yaxis_label : Optional[str], default=None
            Custom Y-axis label (default: "Component {dim_y}")
        xlim : Optional[Tuple[float, float]], default=None
            X-axis limits. If None, auto-calculated with 5% padding beyond data range
        ylim : Optional[Tuple[float, float]], default=None
            Y-axis limits. If None, auto-calculated with 5% padding beyond data range
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
            If inputs are invalid (via validation helper)

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
        # Disable show_centers silently if no clustering provided
        if clustering_name is None and show_centers:
            show_centers = False

        # Validate all inputs
        decomp_obj = self._validate_plot_inputs(
            decomposition_name, dimensions, clustering_name, show_centers
        )

        # Load decomposition and clustering data
        decomp_data = decomp_obj.data
        labels, centers, cluster_colors, cluster_ids = self._load_plot_data(
            clustering_name, show_centers, show_clusters
        )

        # Prepare tag-based coloring or scatter_show_all for cluster mode
        frame_tag_map, tag_colors, unselected_indices = self._prepare_tag_coloring(
            tag_coloring, scatter_show_all, decomp_obj
        )

        if unselected_indices is None:
            unselected_indices = self._prepare_cluster_scatter_show_all(
                clustering_name, scatter_show_all, labels
            )

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
                scatter_size,
                xaxis_label,
                yaxis_label,
                xlim,
                ylim,
                contour_label_fontsize,
                xlabel_fontsize,
                ylabel_fontsize,
                tick_fontsize,
                frame_tag_map,
                tag_colors,
                unselected_indices
            )

        # Finalize figure, add legend, and save if requested
        self._finalize_and_save_figure(
            fig, axes, n_plots, n_rows, n_cols, fig_width, fig_height,
            clustering_name, frame_tag_map, cluster_colors, tag_colors,
            show_clusters, legend_fontsize, title, decomposition_name,
            energy_values, title_fontsize, tick_fontsize, save_fig,
            filename, dimensions, show_centers, file_format, dpi
        )

        return fig

    def _validate_plot_inputs(
        self,
        decomposition_name: str,
        dimensions: List[int],
        clustering_name: Optional[str],
        show_centers: bool
    ) -> DecompositionData:
        """
        Validate all plot inputs.

        Parameters
        ----------
        decomposition_name : str
            Name of decomposition to plot
        dimensions : List[int]
            Dimension indices to plot
        clustering_name : Optional[str]
            Name of clustering for overlay
        show_centers : bool
            Whether to show cluster centers

        Returns
        -------
        DecompositionData
            Validated decomposition object

        Raises
        ------
        ValueError
            If any validation fails
        """
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

        return decomp_obj

    def _load_plot_data(
        self,
        clustering_name: Optional[str],
        show_centers: bool,
        show_clusters: Union[str, List[int]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[int, str]], List[int]]:
        """
        Load clustering data if specified.

        Parameters
        ----------
        clustering_name : Optional[str]
            Name of clustering
        show_centers : bool
            Whether to load cluster centers
        show_clusters : Union[str, List[int]]
            Which clusters to display

        Returns
        -------
        labels : Optional[np.ndarray]
            Cluster labels
        centers : Optional[np.ndarray]
            Cluster centers
        cluster_colors : Optional[Dict[int, str]]
            Color mapping for clusters
        cluster_ids : List[int]
            Cluster IDs corresponding to centers
        """
        if clustering_name:
            labels, centers, cluster_colors = self._load_clustering_data(
                clustering_name, show_centers
            )
            labels, centers, cluster_ids = self._filter_clusters(
                labels, centers, show_clusters
            )
        else:
            labels, centers, cluster_colors, cluster_ids = None, None, None, []

        return labels, centers, cluster_colors, cluster_ids

    def _prepare_tag_coloring(
        self,
        tag_coloring: Optional[List[str]],
        scatter_show_all: bool,
        decomp_obj: DecompositionData
    ) -> Tuple[Optional[Dict[int, str]], Optional[Dict[str, str]], Optional[List[int]]]:
        """
        Prepare tag-based coloring data.

        Parameters
        ----------
        tag_coloring : Optional[List[str]]
            List of tags for coloring, or None if not using tags
        scatter_show_all : bool
            Whether to collect unselected frame indices
        decomp_obj : DecompositionData
            Decomposition object for frame mapping

        Returns
        -------
        frame_tag_map : Optional[Dict[int, str]]
            Mapping from frame index to tag name
        tag_colors : Optional[Dict[str, str]]
            Mapping from tag name to hex color
        unselected_indices : Optional[List[int]]
            Indices of frames without matching tags
        """
        if tag_coloring is None:
            return None, None, None

        frame_tag_map, tag_colors, unselected_indices = (
            LandscapeTagColoringHelper.build_frame_tag_map(
                decomp_obj,
                self.pipeline_data.trajectory_data,
                tag_coloring,
                scatter_show_all
            )
        )
        return frame_tag_map, tag_colors, unselected_indices

    def _prepare_cluster_scatter_show_all(
        self,
        clustering_name: Optional[str],
        scatter_show_all: bool,
        labels: Optional[np.ndarray]
    ) -> Optional[List[int]]:
        """
        Prepare unselected indices for cluster mode with scatter_show_all.

        Parameters
        ----------
        clustering_name : Optional[str]
            Name of clustering, or None if not using clustering
        scatter_show_all : bool
            Whether to show unselected points in gray
        labels : Optional[np.ndarray]
            Cluster labels array

        Returns
        -------
        Optional[List[int]]
            List of indices with label == -1 (noise), or None if not applicable
        """
        if clustering_name and scatter_show_all and labels is not None:
            unselected_mask = labels == -1
            return np.where(unselected_mask)[0].tolist()
        return None

    def _determine_legend_type(
        self,
        clustering_name: Optional[str],
        frame_tag_map: Optional[Dict[int, str]]
    ) -> str:
        """
        Determine which legend type to add.

        Parameters
        ----------
        clustering_name : Optional[str]
            Name of clustering, or None if not using clustering
        frame_tag_map : Optional[Dict[int, str]]
            Frame-to-tag mapping, or None if not using tags

        Returns
        -------
        str
            Legend type: "cluster", "tag", or "none"
        """
        if frame_tag_map is not None:
            return "tag"
        if clustering_name is not None:
            return "cluster"
        return "none"

    def _finalize_and_save_figure(
        self,
        fig: Figure,
        axes: np.ndarray,
        n_plots: int,
        n_rows: int,
        n_cols: int,
        fig_width: float,
        fig_height: float,
        clustering_name: Optional[str],
        frame_tag_map: Optional[Dict[int, str]],
        cluster_colors: Optional[Dict[int, str]],
        tag_colors: Optional[Dict[str, str]],
        show_clusters: Union[str, List[int]],
        legend_fontsize: Optional[int],
        title: Optional[str],
        decomposition_name: str,
        energy_values: bool,
        title_fontsize: Optional[int],
        tick_fontsize: Optional[int],
        save_fig: bool,
        filename: Optional[str],
        dimensions: List[int],
        show_centers: bool,
        file_format: str,
        dpi: int
    ) -> None:
        """
        Finalize figure layout, add legend, and save if requested.

        Parameters
        ----------
        fig : Figure
            Figure to finalize
        axes : np.ndarray
            Array of subplot axes
        n_plots : int
            Number of plots
        n_rows : int
            Number of rows
        n_cols : int
            Number of columns
        fig_width : float
            Figure width in inches
        fig_height : float
            Figure height in inches
        clustering_name : Optional[str]
            Name of clustering
        frame_tag_map : Optional[Dict[int, str]]
            Frame-to-tag mapping
        cluster_colors : Optional[Dict[int, str]]
            Cluster colors
        tag_colors : Optional[Dict[str, str]]
            Tag colors
        show_clusters : Union[str, List[int]]
            Which clusters to show
        legend_fontsize : Optional[int]
            Legend font size
        title : Optional[str]
            Custom title
        decomposition_name : str
            Decomposition name
        energy_values : bool
            Whether energy values are used
        title_fontsize : Optional[int]
            Title font size
        tick_fontsize : Optional[int]
            Tick font size
        save_fig : bool
            Whether to save figure
        filename : Optional[str]
            Custom filename
        dimensions : List[int]
            Dimensions plotted
        show_centers : bool
            Whether centers are shown
        file_format : str
            File format
        dpi : int
            DPI for saving

        Returns
        -------
        None
        """
        LandscapeStylingHelper.finalize_figure(
            fig, axes, n_plots, n_rows, n_cols,
            title, decomposition_name, clustering_name, energy_values,
            title_fontsize, tick_fontsize
        )

        # Calculate dynamic whitespace
        left_inch = 0.3
        base_right_inch = 0.8
        if clustering_name and legend_fontsize:
            extra_right = (legend_fontsize - 10) * 0.5
            right_inch = base_right_inch + extra_right
        else:
            right_inch = base_right_inch

        top_inch = 0.5
        bottom_inch = 0.3

        left = left_inch / fig_width
        right = 1 - (right_inch / fig_width)
        top = 1 - (top_inch / fig_height)
        bottom = bottom_inch / fig_height

        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        # Add appropriate legend
        legend_type = self._determine_legend_type(clustering_name, frame_tag_map)

        if legend_type == "cluster":
            LandscapeStylingHelper.add_central_legend(
                fig, cluster_colors, show_clusters, fig_width, right_inch, legend_fontsize
            )
        elif legend_type == "tag":
            LandscapeStylingHelper.add_tag_legend(
                fig, tag_colors, fig_width, right_inch, legend_fontsize
            )

        # Save if requested
        if save_fig:
            SvgExportHelper.apply_svg_config_if_needed(file_format)
            self._save_figure(
                fig, filename, decomposition_name, dimensions,
                clustering_name, show_centers, energy_values, file_format, dpi
            )

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
        scatter_size: int,
        xaxis_label: Optional[str],
        yaxis_label: Optional[str],
        xlim: Optional[Tuple[float, float]],
        ylim: Optional[Tuple[float, float]],
        contour_label_fontsize: Optional[int],
        xlabel_fontsize: Optional[int],
        ylabel_fontsize: Optional[int],
        tick_fontsize: Optional[int],
        frame_tag_map: Optional[Dict[int, str]] = None,
        tag_colors: Optional[Dict[str, str]] = None,
        unselected_indices: Optional[List[int]] = None
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
        bins : int or str
            Number of bins for background and cluster contours.
            Use "auto" to automatically determine optimal bin count.
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
        scatter_size : int
            Size of scatter points
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
        data_x = decomp_data[:, dim_x]
        data_y = decomp_data[:, dim_y]

        xlim, ylim = self._calculate_plot_limits(data_x, data_y, xlim, ylim)
        bins = self._calculate_bins(data_x, data_y, bins)

        self._plot_background(
            ax, data_x, data_y, bins, temperature, xlim, ylim,
            energy_values, use_kde, mask_empty_bins,
            contour_label_fontsize, tick_fontsize
        )

        self._plot_cluster_overlay(
            ax, data_x, data_y, labels, cluster_colors, alpha,
            cluster_contour, cluster_contour_voronoi, bins, data_scatter,
            contour_label_fontsize, frame_tag_map, tag_colors, unselected_indices,
            scatter_size
        )

        if centers is not None:
            LandscapeRenderingHelper.plot_centers(
                ax, centers, cluster_ids, dim_x, dim_y, cluster_colors,
                center_marker, center_size
            )

        LandscapeStylingHelper.set_axis_labels(
            ax, dim_x, dim_y, xaxis_label, yaxis_label,
            xlabel_fontsize, ylabel_fontsize
        )
        LandscapeStylingHelper.set_axis_limits(ax, xlim, ylim)

    def _calculate_plot_limits(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        xlim: Optional[Tuple[float, float]],
        ylim: Optional[Tuple[float, float]]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate plot limits with 5% padding if not provided.

        Parameters
        ----------
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        xlim : Optional[Tuple[float, float]]
            Custom X-axis limits
        ylim : Optional[Tuple[float, float]]
            Custom Y-axis limits

        Returns
        -------
        xlim : Tuple[float, float]
            X-axis limits
        ylim : Tuple[float, float]
            Y-axis limits
        """
        if xlim is None:
            x_range = data_x.max() - data_x.min()
            xlim = (data_x.min() - 0.05 * x_range, data_x.max() + 0.05 * x_range)
        if ylim is None:
            y_range = data_y.max() - data_y.min()
            ylim = (data_y.min() - 0.05 * y_range, data_y.max() + 0.05 * y_range)
        return xlim, ylim

    def _calculate_bins(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: Union[int, str]
    ) -> int:
        """
        Calculate optimal bins if "auto" is specified.

        Parameters
        ----------
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        bins : int or str
            Number of bins or "auto"

        Returns
        -------
        int
            Number of bins
        """
        if bins == "auto":
            x_edges = np.histogram_bin_edges(data_x, bins="auto")
            y_edges = np.histogram_bin_edges(data_y, bins="auto")
            return max(len(x_edges), len(y_edges)) - 1
        return bins

    def _plot_background(
        self,
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        bins: int,
        temperature: float,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        energy_values: bool,
        use_kde: bool,
        mask_empty_bins: bool,
        contour_label_fontsize: Optional[int],
        tick_fontsize: Optional[int]
    ) -> None:
        """
        Plot energy or density background.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        bins : int
            Number of bins
        temperature : float
            Temperature for energy calculation
        xlim : Tuple[float, float]
            X-axis limits
        ylim : Tuple[float, float]
            Y-axis limits
        energy_values : bool
            Plot energy landscape
        use_kde : bool
            Use KDE smoothing
        mask_empty_bins : bool
            Mask empty bins
        contour_label_fontsize : Optional[int]
            Colorbar label font size
        tick_fontsize : Optional[int]
            Tick label font size

        Returns
        -------
        None
        """
        if energy_values:
            LandscapeRenderingHelper.plot_energy_background(
                ax, data_x, data_y, bins, temperature, xlim, ylim,
                use_kde=use_kde, mask_empty_bins=mask_empty_bins,
                contour_label_fontsize=contour_label_fontsize,
                tick_fontsize=tick_fontsize
            )
        else:
            LandscapeRenderingHelper.plot_density_background(
                ax, data_x, data_y, bins, xlim, ylim,
                use_kde=use_kde, mask_empty_bins=mask_empty_bins,
                contour_label_fontsize=contour_label_fontsize,
                tick_fontsize=tick_fontsize
            )

    def _plot_cluster_overlay(
        self,
        ax,
        data_x: np.ndarray,
        data_y: np.ndarray,
        labels: Optional[np.ndarray],
        cluster_colors: Optional[Dict[int, str]],
        alpha: float,
        cluster_contour: bool,
        cluster_contour_voronoi: bool,
        bins: int,
        data_scatter: bool,
        contour_label_fontsize: Optional[int],
        frame_tag_map: Optional[Dict[int, str]],
        tag_colors: Optional[Dict[str, str]],
        unselected_indices: Optional[List[int]],
        scatter_size: int
    ) -> None:
        """
        Plot cluster overlay as contours or scatter.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        data_x : numpy.ndarray
            X-axis data
        data_y : numpy.ndarray
            Y-axis data
        labels : Optional[numpy.ndarray]
            Cluster labels
        cluster_colors : Optional[Dict[int, str]]
            Cluster colors
        alpha : float
            Transparency
        cluster_contour : bool
            Show as contours
        cluster_contour_voronoi : bool
            Use Voronoi contours
        bins : int
            Number of bins for contours
        data_scatter : bool
            Show gray scatter
        contour_label_fontsize : Optional[int]
            Contour label font size
        frame_tag_map : Optional[Dict[int, str]]
            Frame-to-tag mapping
        tag_colors : Optional[Dict[str, str]]
            Tag colors
        unselected_indices : Optional[List[int]]
            Unselected frame indices
        scatter_size : int
            Size of scatter points

        Returns
        -------
        None
        """
        if labels is not None and cluster_contour:
            if cluster_contour_voronoi:
                LandscapeRenderingHelper.plot_cluster_voronoi(
                    ax, data_x, data_y, labels, cluster_colors, bins, alpha
                )
            else:
                LandscapeRenderingHelper.plot_cluster_density_contours(
                    ax, data_x, data_y, labels, cluster_colors, bins,
                    contour_label_fontsize=contour_label_fontsize
                )
        else:
            LandscapeRenderingHelper.create_scatter(
                ax, data_x, data_y, labels, cluster_colors, alpha, data_scatter,
                frame_tag_map=frame_tag_map, tag_colors=tag_colors,
                unselected_indices=unselected_indices, scatter_size=scatter_size
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
