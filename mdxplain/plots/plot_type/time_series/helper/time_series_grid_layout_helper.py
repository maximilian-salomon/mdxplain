# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
#
# This program is free software under GNU LGPL v3.

"""Helper for grid layout in time series plots."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData


class TimeSeriesGridLayoutHelper:
    """Helper for grid layout with membership integration."""

    @staticmethod
    def calculate_grid_dimensions(
        n_rows: int,
        clustering_name: Optional[str],
        membership_per_feature: bool,
        membership_traj_selection: Union[str, int, List],
        feature_selector_name: str,
        membership_bar_height: float,
        subplot_height: float,
        pipeline_data: PipelineData
    ) -> Tuple[int, Optional[List[float]]]:
        """
        Calculate grid dimensions with membership.

        Parameters
        ----------
        n_rows : int
            Feature rows
        clustering_name : str or None
            Clustering
        membership_per_feature : bool
            Per-feature mode
        membership_traj_selection : Union[str, int, List]
            Selection
        feature_selector_name : str
            Selector
        membership_bar_height : float
            Bar height
        subplot_height : float
            Subplot height
        pipeline_data : PipelineData
            Pipeline data

        Returns
        -------
        total_rows : int
            Total grid rows
        height_ratios : List[float] or None
            Height ratios
        """
        if not clustering_name:
            return n_rows, None

        from .time_series_membership_plot_helper import TimeSeriesMembershipPlotHelper
        membership_indices = TimeSeriesMembershipPlotHelper.get_membership_indices(
            pipeline_data, membership_traj_selection, feature_selector_name
        )
        n_traj = len(membership_indices)

        spacing_factor = 1.2
        membership_row_height_inches = n_traj * membership_bar_height * spacing_factor
        membership_height_ratio = membership_row_height_inches / subplot_height

        if membership_per_feature:
            total_rows = n_rows * 2
            height_ratios = []
            for _ in range(n_rows):
                height_ratios.extend([1.0, membership_height_ratio])
        else:
            total_rows = n_rows + 1
            height_ratios = [1.0] * n_rows + [membership_height_ratio]

        return total_rows, height_ratios

    @staticmethod
    def create_gridspec(
        fig: Figure,
        total_rows: int,
        n_cols: int,
        wspace: float,
        hspace: float,
        height_ratios: Optional[List[float]],
        width_ratios: Optional[List[float]],
        title: Optional[str]
    ) -> Tuple[GridSpec, str, float]:
        """
        Create GridSpec with title spacing.

        Parameters
        ----------
        fig : Figure
            Figure
        total_rows : int
            Rows
        n_cols : int
            Columns
        wspace : float
            Width spacing
        hspace : float
            Height spacing
        height_ratios : List[float] or None
            Height ratios
        width_ratios : List[float] or None
            Width ratios
        title : str or None
            Title

        Returns
        -------
        gs : GridSpec
            Grid
        wrapped_title : str
            Title
        top : float
            Top position
        """
        from ....helper.title_legend_helper import TitleLegendHelper

        wrapped_title = TitleLegendHelper.wrap_title(
            title or "Time Series Plot", max_chars_per_line=80
        )

        title_offset_from_top = 0.15  # Titel-Position
        gap_to_plots = 0.3  # Abstand Titel zu Plots
        legend_height = 0.4  # Platz für Legende
        title_legend_space = title_offset_from_top + gap_to_plots + legend_height
        top = 1.0 - (title_legend_space / fig.get_figheight())

        gs = GridSpec(
            total_rows, n_cols, figure=fig,
            wspace=wspace, hspace=hspace, top=top,
            height_ratios=height_ratios, width_ratios=width_ratios
        )

        return gs, wrapped_title, top
