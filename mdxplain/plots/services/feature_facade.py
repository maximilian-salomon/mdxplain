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
Facade for manual feature visualization.

Provides simplified interface for creating plots based on manual
feature and DataSelector selection.
"""

from typing import List, Dict, Optional, Union
from matplotlib.figure import Figure

from ..plot_type.violin.violin_plotter import ViolinPlotter
from ..plot_type.density.density_plotter import DensityPlotter
from ..plot_type.time_series.time_series_plotter import TimeSeriesPlotter


class FeatureFacade:
    """
    Facade for manual feature visualization.

    Provides high-level interface for creating visualizations from
    manually selected features and DataSelectors. Simplifies access to
    specialized plotters while managing pipeline data and configuration.

    Examples
    --------
    >>> # Access via plots manager
    >>> facade = plots_manager.feature
    >>> fig = facade.violins("my_selector", ["cluster_0", "cluster_1"])
    >>> fig = facade.densities("my_selector", ["cluster_0", "cluster_1"])
    >>> fig = facade.time_series("my_selector", ["cluster_0", "cluster_1"])
    """

    def __init__(self, manager, pipeline_data) -> None:
        """
        Initialize feature facade.

        Parameters
        ----------
        manager : PlotsManager
            Plots manager instance
        pipeline_data : PipelineData
            Pipeline data container

        Returns
        -------
        None
            Initializes FeatureFacade instance
        """
        self.pipeline_data = pipeline_data
        self.cache_dir = manager.cache_dir

    def violins(
        self,
        feature_selector: str,
        data_selectors: List[str],
        contact_transformation: bool = True,
        max_cols: int = 4,
        long_labels: bool = False,
        contact_threshold: Optional[float] = 4.5,
        title: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[Dict[str, str]] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300
    ) -> Figure:
        """
        Create violin plots from manual feature selection.

        Visualizes the distribution of feature values showing separate
        violins for each DataSelector group with cluster-consistent colors.

        Parameters
        ----------
        feature_selector : str
            Name of feature selector
        data_selectors : List[str]
            DataSelector names to plot
        contact_transformation : bool, default=True
            If True, automatically convert contact features to distances.
            If False, plot contacts as binary values with Gaussian smoothing.
        max_cols : int, default=4
            Maximum number of columns in grid layout
        long_labels : bool, default=False
            If True, use long descriptive labels for discrete features
            (e.g., "Contact"/"Non-Contact", "Alpha helix"/"Loop").
            If False, use short labels (e.g., "C"/"NC", "H"/"C").
        contact_threshold : float, optional
            Distance threshold in Angstrom for drawing contact threshold line
        title : str, optional
            Custom plot title
        legend_title : str, optional
            Custom legend title
        legend_labels : Dict[str, str], optional
            Custom labels for DataSelectors in legend
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename
        file_format : str, default="png"
            File format for saving
        dpi : int, default=300
            Resolution for saved figure

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing violin plots

        Raises
        ------
        ValueError
            If feature selector or data selectors not found

        Examples
        --------
        >>> # Basic violin plot
        >>> fig = facade.violins(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"]
        ... )

        >>> # With custom layout
        >>> fig = facade.violins(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     max_cols=3
        ... )

        >>> # Save to file
        >>> fig = facade.violins(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     save_fig=True,
        ...     filename="features.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        - Each feature gets its own subplot with violins for all DataSelectors
        - Y-axis shows feature values with units (Distance, Angle, etc.)
        - Grid layout controlled by max_cols parameter
        """
        plotter = ViolinPlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_selector=feature_selector,
            data_selectors=data_selectors,
            contact_transformation=contact_transformation,
            max_cols=max_cols,
            long_labels=long_labels,
            contact_threshold=contact_threshold,
            title=title,
            legend_title=legend_title,
            legend_labels=legend_labels,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi
        )

    def densities(
        self,
        feature_selector: str,
        data_selectors: List[str],
        contact_transformation: bool = True,
        max_cols: int = 4,
        long_labels: bool = False,
        kde_bandwidth: Union[str, float] = "scott",
        base_sigma: float = 0.05,
        max_sigma: float = 0.12,
        alpha: float = 0.3,
        line_width: float = 2.0,
        contact_threshold: Optional[float] = 4.5,
        title: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[Dict[str, str]] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300
    ) -> Figure:
        """
        Create density plots from manual feature selection.

        Visualizes feature distributions as overlaid density curves in grid
        layout. Each feature gets one grid cell with curves for each
        DataSelector group using cluster-consistent colors.

        Parameters
        ----------
        feature_selector : str
            Name of feature selector
        data_selectors : List[str]
            DataSelector names to plot
        contact_transformation : bool, default=True
            If True, convert contact features to distances.
            If False, plot contacts with Gaussian smoothing.
        max_cols : int, default=4
            Maximum number of columns in grid layout
        long_labels : bool, default=False
            If True, use long descriptive labels for discrete features
        kde_bandwidth : str or float, default="scott"
            KDE bandwidth for continuous features
        base_sigma : float, default=0.05
            Minimum Gaussian width for binary contact features
        max_sigma : float, default=0.12
            Maximum Gaussian width for binary contact features
        alpha : float, default=0.3
            Transparency for filled density curves
        line_width : float, default=2.0
            Width of density curve contour lines
        contact_threshold : float, optional
            Distance threshold for contact threshold line
        title : str, optional
            Custom plot title
        legend_title : str, optional
            Custom legend title
        legend_labels : Dict[str, str], optional
            Custom labels for DataSelectors in legend
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename
        file_format : str, default="png"
            File format for saving
        dpi : int, default=300
            Resolution for saved figure

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing density plots

        Raises
        ------
        ValueError
            If feature selector or data selectors not found

        Examples
        --------
        >>> # Basic density plot
        >>> fig = facade.densities(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"]
        ... )

        >>> # Custom KDE bandwidth
        >>> fig = facade.densities(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     kde_bandwidth=0.5
        ... )

        >>> # Save to file
        >>> fig = facade.densities(
        ...     feature_selector="my_selector",
        ...     data_selectors=["cluster_0", "cluster_1"],
        ...     save_fig=True,
        ...     filename="densities.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        - Features displayed in grid layout with max_cols per row
        - Discrete features use Gaussian smoothing
        - Continuous features use KDE
        - Colors consistent with clustering via ColorMappingHelper
        """
        plotter = DensityPlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_selector=feature_selector,
            data_selectors=data_selectors,
            contact_transformation=contact_transformation,
            max_cols=max_cols,
            long_labels=long_labels,
            kde_bandwidth=kde_bandwidth,
            base_sigma=base_sigma,
            max_sigma=max_sigma,
            alpha=alpha,
            line_width=line_width,
            contact_threshold=contact_threshold,
            title=title,
            legend_title=legend_title,
            legend_labels=legend_labels,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi
        )

    def time_series(
        self,
        feature_selector: str,
        traj_selection: Union[int, str, List, "all"] = "all",
        use_time: bool = True,
        tags_for_coloring: Optional[List[str]] = None,
        allow_multi_tag_plotting: bool = False,
        clustering_name: Optional[str] = None,
        membership_per_feature: bool = False,
        membership_traj_selection: Union[str, int, List] = "all",
        contact_transformation: bool = True,
        max_cols: int = 2,
        long_labels: bool = False,
        subplot_height: float = 2.5,
        membership_bar_height: Optional[float] = None,
        show_legend: bool = True,
        contact_threshold: Optional[float] = 4.5,
        title: Optional[str] = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 300,
        smoothing: bool = True,
        smoothing_method: str = "savitzky",
        smoothing_window: int = 51,
        smoothing_polyorder: int = 3,
        show_unsmoothed_background: bool = True
    ) -> Figure:
        """
        Create time series plots from manual feature selection.

        Visualizes temporal evolution of selected features as line plots
        with one subplot per feature. Each trajectory is shown as a separate
        line, optionally colored by trajectory number or tags.

        Parameters
        ----------
        feature_selector : str
            Name of feature selector
        traj_selection : int, str, list, or "all", default="all"
            Trajectories to plot
        use_time : bool, default=True
            If True, use Time (ns) for x-axis. If False, use frame numbers.
        tags_for_coloring : list of str, optional
            Tags to use for trajectory coloring
        allow_multi_tag_plotting : bool, default=False
            Plot multi-tag trajectories multiple times
        clustering_name : str, optional
            Name of clustering for membership visualization
        membership_per_feature : bool, default=False
            Show membership bar below each feature subplot
        membership_traj_selection : int, str, list, or "all", default="all"
            Trajectories to include in membership visualization
        contact_transformation : bool, default=True
            Convert contact features to distances
        max_cols : int, default=2
            Maximum number of columns in grid layout
        long_labels : bool, default=False
            Use long descriptive labels for discrete features
        subplot_height : float, default=2.5
            Height per feature subplot in inches
        membership_bar_height : float, optional
            Height per trajectory in membership bar
        show_legend : bool, default=True
            Show legend for trajectory/tag colors
        contact_threshold : Optional[float], default=4.5
            Distance threshold for contact threshold line
        title : str, optional
            Custom plot title
        save_fig : bool, default=False
            Save figure to file
        filename : str, optional
            Custom filename
        file_format : str, default="png"
            File format for saving
        dpi : int, default=300
            Resolution for saved figure
        smoothing : bool, default=True
            Enable or disable data smoothing
        smoothing_method : str, default="savitzky"
            Smoothing method ("moving_average" or "savitzky")
        smoothing_window : int, default=51
            Window size for smoothing in frames
        smoothing_polyorder : int, default=3
            Polynomial order for Savitzky-Golay filter (ignored for moving_average)
        show_unsmoothed_background : bool, default=True
            Show unsmoothed data as transparent background line when smoothing is enabled

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing time series plots

        Raises
        ------
        ValueError
            If feature selector not found

        Examples
        --------
        >>> # Basic time series plot
        >>> fig = facade.time_series(
        ...     feature_selector="my_selector"
        ... )

        >>> # With cluster membership visualization
        >>> fig = facade.time_series(
        ...     feature_selector="my_selector",
        ...     clustering_name="dbscan",
        ...     membership_per_feature=True
        ... )

        >>> # Save to file
        >>> fig = facade.time_series(
        ...     feature_selector="my_selector",
        ...     save_fig=True,
        ...     filename="timeseries.pdf",
        ...     file_format="pdf"
        ... )

        Notes
        -----
        - Each feature gets its own subplot with all trajectories overlaid
        - X-axis shows either Time (ns) or frame numbers
        - Contact features automatically converted to distances (default)
        - Cluster membership shown as colored horizontal bars
        """
        plotter = TimeSeriesPlotter(self.pipeline_data, self.cache_dir)
        return plotter.plot(
            feature_selector=feature_selector,
            traj_selection=traj_selection,
            use_time=use_time,
            tags_for_coloring=tags_for_coloring,
            allow_multi_tag_plotting=allow_multi_tag_plotting,
            clustering_name=clustering_name,
            membership_per_feature=membership_per_feature,
            membership_traj_selection=membership_traj_selection,
            contact_transformation=contact_transformation,
            max_cols=max_cols,
            long_labels=long_labels,
            subplot_height=subplot_height,
            membership_bar_height=membership_bar_height,
            show_legend=show_legend,
            contact_threshold=contact_threshold,
            title=title,
            save_fig=save_fig,
            filename=filename,
            file_format=file_format,
            dpi=dpi,
            smoothing=smoothing,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_polyorder=smoothing_polyorder,
            show_unsmoothed_background=show_unsmoothed_background
        )
