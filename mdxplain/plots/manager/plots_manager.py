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
Plots manager for trajectory analysis visualizations.

Central manager for all plotting functionality, providing access to
decomposition-focused and clustering-focused plot methods, as well
as direct plotting capabilities.
"""

from typing import List, Optional, Tuple, Union
from matplotlib.figure import Figure

from ..services.decomposition_facade import DecompositionFacade
from ..services.clustering_facade import ClusteringFacade
from ..services.feature_importance_facade import FeatureImportanceFacade
from ..services.feature_facade import FeatureFacade
from ..plot_type.landscape import LandscapePlotter
from ..plot_type.membership import MembershipPlotter


class PlotsManager:
    """
    Manager for all plotting operations.

    Provides five access patterns:
    1. Direct: pipeline.plots.landscape(...), pipeline.plots.membership(...)
    2. Decomposition-focused: pipeline.plots.decomposition.landscape(...)
    3. Clustering-focused: pipeline.plots.clustering.landscape(...)
    4. Feature importance: pipeline.plots.feature_importance.violins(...)
    5. Manual feature: pipeline.plots.feature.violins(...)

    Examples
    --------
    >>> # Direct access
    >>> pipeline.plots.landscape(
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1]
    ... )

    >>> # Decomposition-focused
    >>> pipeline.plots.decomposition.landscape(
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1]
    ... )

    >>> # Clustering-focused (show_centers=True by default)
    >>> pipeline.plots.clustering.landscape(
    ...     clustering_name="dbscan",
    ...     decomposition_name="pca",
    ...     dimensions=[0, 1]
    ... )

    >>> # Feature importance violin plots
    >>> pipeline.plots.feature_importance.violins(
    ...     feature_importance_name="tree_analysis",
    ...     n_top=10
    ... )

    >>> # Manual feature violin plots
    >>> pipeline.plots.feature.violins(
    ...     feature_selector="my_selector",
    ...     data_selectors=["cluster_0", "cluster_1"]
    ... )
    """

    def __init__(
        self,
        use_memmap: bool = True,
        chunk_size: int = 2000,
        cache_dir: str = "./cache",
    ) -> None:
        """
        Initialize plots manager.

        Parameters
        ----------
        use_memmap : bool, default=True
            Whether to use memory mapping for large datasets
        chunk_size : int, default=2000
            Chunk size for memory-efficient processing
        cache_dir : str, default="./cache"
            Directory for cache files

        Returns
        -------
        None
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir

    @property
    def decomposition(self) -> DecompositionFacade:
        """
        Access decomposition-focused plotting methods.

        Returns
        -------
        DecompositionFacade
            Decomposition plotting interface

        Note
        ----
        Pipeline data is passed as None here because it will be
        automatically injected later when the facade methods are called.

        Examples
        --------
        >>> # Create decomposition landscape
        >>> pipeline.plots.decomposition.landscape(
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1, 2, 3]
        ... )
        """
        return DecompositionFacade(self, None)

    @property
    def clustering(self) -> ClusteringFacade:
        """
        Access clustering-focused plotting methods.

        Returns
        -------
        ClusteringFacade
            Clustering plotting interface

        Note
        ----
        Pipeline data is passed as None here because it will be
        automatically injected later when the facade methods are called.

        Examples
        --------
        >>> # Create clustering landscape
        >>> pipeline.plots.clustering.landscape(
        ...     clustering_name="dbscan",
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1],
        ...     show_centers=True
        ... )
        """
        return ClusteringFacade(self, None)

    @property
    def feature_importance(self) -> FeatureImportanceFacade:
        """
        Access feature importance visualization methods.

        Returns
        -------
        FeatureImportanceFacade
            Feature importance plotting interface

        Note
        ----
        Pipeline data is passed as None here because it will be
        automatically injected later when the facade methods are called.

        Examples
        --------
        >>> # Create violin plots for top features
        >>> pipeline.plots.feature_importance.violins(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10
        ... )

        >>> # Create density plots
        >>> pipeline.plots.feature_importance.densities(
        ...     feature_importance_name="tree_analysis",
        ...     n_top=10
        ... )
        """
        return FeatureImportanceFacade(self, None)

    @property
    def feature(self) -> FeatureFacade:
        """
        Access manual feature visualization methods.

        Returns
        -------
        FeatureFacade
            Manual feature plotting interface

        Note
        ----
        Pipeline data is passed as None here because it will be
        automatically injected later when the facade methods are called.

        Examples
        --------
        >>> # Create violin plots for selected features
        >>> pipeline.plots.feature.violins(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"]
        ... )

        >>> # Create density plots
        >>> pipeline.plots.feature.densities(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"]
        ... )

        >>> # Create time series plots
        >>> pipeline.plots.feature.time_series(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"]
        ... )
        """
        return FeatureFacade(self, None)

    def landscape(
        self,
        pipeline_data,
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
        center_marker: str = "X",
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
        Create landscape plot directly from plots manager.

        Convenience method for quick landscape visualization without
        going through decomposition or clustering facades.

        Parameters
        ----------
        decomposition_name : str
            Name of decomposition to plot (e.g., "pca", "tica")
        dimensions : List[int]
            Dimension indices to plot (must be even number).
            Paired consecutively: [0,1,2,3] → [(0,1), (2,3)]
        clustering_name : Optional[str], default=None
            Name of clustering for color overlay
        show_centers : bool, default=False
            Show cluster centers (requires clustering_name)
        energy_values : bool, default=False
            Show free energy landscape instead of density
        use_kde : bool, default=False
            Use KDE smoothing for background density estimation.

            **Default (False)**: Histogram-based - shows actual observations,
            preserves energy barriers, scientifically accurate.

            **KDE (True)**: Smooth visualization but can filter out small energy
            barriers and distort the landscape. Use only if you know what you do.
            NOT for quantitative analysis. A warning will be issued.
        mask_empty_bins : bool, default=True
            Mask unsampled bins in the background (energy/density) as white/
            transparent. Set False to fill with the maximum color for
            continuous fill.
        bins : int or str, default="auto"
            Number of bins for histogram/energy calculation
        temperature : float, default=300.0
            Temperature in Kelvin for energy calculation
        alpha : float, default=0.6
            Transparency for scatter points (0=transparent, 1=opaque)
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
            Custom overall title (overrides default)
        xaxis_label : Optional[str], default=None
            Custom X-axis label (default: "Component {dim_x}")
        yaxis_label : Optional[str], default=None
            Custom Y-axis label (default: "Component {dim_y}")
        xlim : Optional[Tuple[float, float]], default=None
            X-axis limits for all subplots
        ylim : Optional[Tuple[float, float]], default=None
            Y-axis limits for all subplots
        subplot_size : float, default=4.0
            Size of each subplot in inches
        save_fig : bool, default=False
            Save figure to file
        filename : Optional[str], default=None
            Custom filename (overrides auto-generated name)
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
            Font size for tick labels (default: 10)
        legend_fontsize : int, optional
            Font size for legend entries (default: 10)
        contour_label_fontsize : int, optional
            Font size for contour labels (default: 10)

        Returns
        -------
        matplotlib.figure.Figure
            Created figure object

        Raises
        ------
        ValueError
            If decomposition not found
        ValueError
            If dimensions invalid or odd number
        ValueError
            If show_centers=True but no clustering_name
        ValueError
            If clustering not compatible with decomposition

        Examples
        --------
        >>> # Simple landscape
        >>> fig = pipeline.plots.landscape(
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1]
        ... )

        >>> # With clustering overlay
        >>> fig = pipeline.plots.landscape(
        ...     decomposition_name="pca",
        ...     dimensions=[0, 1],
        ...     clustering_name="dbscan",
        ...     show_centers=True
        ... )

        >>> # Energy landscape with saving
        >>> fig = pipeline.plots.landscape(
        ...     decomposition_name="tica",
        ...     dimensions=[0, 1, 2, 3],
        ...     energy_values=True,
        ...     save_fig=True,
        ...     filename="tica_energy_landscape.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        This is a direct convenience method. For more specialized workflows,
        consider using:
        
        - pipeline.plots.decomposition.landscape() for decomposition focus
        - pipeline.plots.clustering.landscape() for clustering focus
        """
        plotter = LandscapePlotter(pipeline_data, cache_dir=self.cache_dir)

        return plotter.plot(
            decomposition_name=decomposition_name,
            dimensions=dimensions,
            clustering_name=clustering_name,
            show_centers=show_centers,
            energy_values=energy_values,
            use_kde=use_kde,
            mask_empty_bins=mask_empty_bins,
            bins=bins,
            temperature=temperature,
            alpha=alpha,
            cluster_contour=cluster_contour,
            cluster_contour_voronoi=cluster_contour_voronoi,
            data_scatter=data_scatter,
            show_clusters=show_clusters,
            center_marker=center_marker,
            center_size=center_size,
            title=title,
            xaxis_label=xaxis_label,
            yaxis_label=yaxis_label,
            xlim=xlim,
            ylim=ylim,
            subplot_size=subplot_size,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi,
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            contour_label_fontsize=contour_label_fontsize,
        )

    def membership(
        self,
        pipeline_data,
        clustering_name: str,
        traj_selection: Union[int, str, List, "all"] = "all",
        height_per_trajectory: float = 0.3,
        show_frame_numbers: bool = True,
        show_legend: bool = True,
        title: Optional[str] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300,
        title_fontsize: Optional[int] = None,
        xlabel_fontsize: Optional[int] = None,
        ylabel_fontsize: Optional[int] = None,
        tick_fontsize: Optional[int] = None,
        legend_fontsize: Optional[int] = None,
    ) -> Figure:
        """
        Create cluster membership timeline plot.

        Visualizes cluster assignment over time as horizontal colored bars,
        with each trajectory on a separate row. Uses efficient block-based
        rendering for performance.

        Parameters
        ----------
        clustering_name : str
            Name of clustering to visualize
        traj_selection : int, str, list, or "all", default="all"
            Trajectory selection (uses TrajectoryData.get_trajectory_indices()).
            Controls which trajectories to plot AND their order.
        height_per_trajectory : float, default=0.3
            Height in inches per trajectory bar
        show_frame_numbers : bool, default=True
            Show frame numbers on x-axis
        show_legend : bool, default=True
            Show cluster color legend
        title : Optional[str], default=None
            Custom title (auto-generated if None)
        save_fig : bool, default=False
            Save figure to file
        filename : Optional[str], default=None
            Custom filename (auto-generated if None)
        file_format : str, default="png"
            File format for saving (png, pdf, svg, etc.).
            When using 'svg', text elements remain editable in SVG editors.
        dpi : int, default=300
            Resolution for saved figure
        title_fontsize : int, optional
            Font size for title (default: 14)
        xlabel_fontsize : int, optional
            Font size for X-axis label (default: 12)
        ylabel_fontsize : int, optional
            Font size for Y-axis trajectory labels (default: 11)
        tick_fontsize : int, optional
            Font size for tick labels (default: 10)
        legend_fontsize : int, optional
            Font size for legend entries (default: 10)

        Returns
        -------
        matplotlib.figure.Figure
            Created figure object

        Raises
        ------
        ValueError
            If clustering not found or no trajectories selected

        Examples
        --------
        >>> # All trajectories in original order
        >>> fig = pipeline.plots.membership("dbscan")

        >>> # Specific trajectories in custom order
        >>> fig = pipeline.plots.membership(
        ...     "dbscan",
        ...     traj_selection=[2, 0, 5]
        ... )

        >>> # By tag selection
        >>> fig = pipeline.plots.membership(
        ...     "hdbscan",
        ...     traj_selection="tag:system_A"
        ... )

        >>> # Customized appearance with saving
        >>> fig = pipeline.plots.membership(
        ...     "dpa",
        ...     traj_selection=[0, 1, 2],
        ...     height_per_trajectory=0.5,
        ...     title="DPA Clustering Timeline",
        ...     save_fig=True,
        ...     filename="dpa_membership.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        - Uses block-based rendering: O(transitions) not O(frames)
        - Trajectory order in plot matches order in traj_selection
        - Colors consistent with other plots via ColorMappingHelper
        - Noise points (-1) shown in black
        - Best for visualizing temporal dynamics of cluster assignments
        """
        plotter = MembershipPlotter(pipeline_data, cache_dir=self.cache_dir)

        return plotter.plot(
            clustering_name=clustering_name,
            traj_selection=traj_selection,
            height_per_trajectory=height_per_trajectory,
            show_frame_numbers=show_frame_numbers,
            show_legend=show_legend,
            title=title,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi,
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
        )
