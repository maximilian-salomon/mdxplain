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
Helper for membership bar plotting in time series.

Handles membership bar rendering, image-based replication for per-feature mode,
and legend creation for cluster membership visualization.
"""

from __future__ import annotations

from typing import List, Dict, Union, TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData
    from ..time_series_plot_config import TimeSeriesPlotConfig

from .membership_bar_renderer import MembershipBarRenderer

class TimeSeriesMembershipPlotHelper:
    """
    Helper for membership bar plotting.

    Provides membership bar plotting with two modes:
    
    - per_feature: Image-based replication (efficient for N features)
    - single: Standard bottom plot (simple for overview)

    Examples
    --------
    >>> helper = TimeSeriesMembershipPlotHelper()
    >>> helper.plot_membership_bars(
    ...     fig, gs, pipeline_data, 2, "dbscan", False, ...
    ... )
    """

    @staticmethod
    def get_membership_indices(
        pipeline_data: PipelineData,
        membership_traj_selection: Union[str, int, List],
        feature_selector_name: str
    ) -> List[int]:
        """
        Get trajectory indices for membership bars.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data
        membership_traj_selection : Union[str, int, List]
            Trajectory selection
        feature_selector_name : str
            Feature selector name

        Returns
        -------
        List[int]
            Trajectory indices with data

        Examples
        --------
        >>> indices = TimeSeriesMembershipPlotHelper.get_membership_indices(
        ...     pipeline_data, "all", "selector"
        ... )
        """
        all_indices = pipeline_data.trajectory_data.get_trajectory_indices(
            membership_traj_selection
        )

        _, frame_mapping = pipeline_data.get_selected_data(
            feature_selector_name, return_frame_mapping=True
        )
        traj_in_data = set(traj_idx for traj_idx, _ in frame_mapping.values())

        return [idx for idx in all_indices if idx in traj_in_data]

    @staticmethod
    def plot_membership_bars(
        fig: Figure,
        gs: GridSpec,
        n_rows: int,
        config: TimeSeriesPlotConfig
    ) -> None:
        """
        Plot membership bars (per-feature or single bottom plot).

        Parameters
        ----------
        fig : Figure
            Figure to plot on
        gs : GridSpec
            GridSpec layout
        n_rows : int
            Number of feature rows
        config : TimeSeriesPlotConfig
            Central configuration object

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesMembershipPlotHelper.plot_membership_bars(
        ...     fig, gs, 2, config
        ... )
        """
        membership_indices = TimeSeriesMembershipPlotHelper.get_membership_indices(
            config.pipeline_data, config.membership_traj_selection,
            config.feature_selector_name
        )

        # Skip membership plotting if no valid trajectories found in selector
        if not membership_indices:
            return

        prepared_data, cluster_colors, _ = MembershipBarRenderer.prepare_membership_data(
            config.pipeline_data, config.clustering_name, membership_indices,
            config.use_time
        )

        if config.membership_per_feature:
            TimeSeriesMembershipPlotHelper._plot_per_feature_membership(
                fig, gs, config, prepared_data, membership_indices, cluster_colors
            )
        else:
            TimeSeriesMembershipPlotHelper._plot_single_membership(
                fig, gs, n_rows, config, prepared_data, membership_indices,
                cluster_colors
            )

    @staticmethod
    def _plot_per_feature_membership(
        fig: Figure,
        gs: GridSpec,
        config: TimeSeriesPlotConfig,
        prepared_data: Dict,
        membership_indices: List[int],
        cluster_colors: Dict
    ):
        """
        Plot membership per feature using image replication.

        Parameters
        ----------
        fig : Figure
            Figure to plot on
        gs : GridSpec
            GridSpec layout
        config : TimeSeriesPlotConfig
            Central configuration object
        prepared_data : Dict
            Prepared membership data
        membership_indices : List[int]
            Trajectory indices
        cluster_colors : Dict
            Cluster color mapping

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesMembershipPlotHelper._plot_per_feature_membership(
        ...     fig, gs, config, data, indices, colors
        ... )
        """
        membership_img = TimeSeriesMembershipPlotHelper._render_membership_to_image(
            fig, gs, prepared_data, membership_indices, cluster_colors,
            config.membership_bar_height, config.use_time
        )

        traj_names = [
            config.pipeline_data.trajectory_data.trajectory_names[idx]
            for idx in membership_indices
        ]

        y_spacing = 0.3
        for i in range(len(config.all_features)):
            _, row, col, _ = config.layout[i]
            membership_row = row * 2 + 1

            # Y-labels (always at column 0)
            label_ax = fig.add_subplot(gs[membership_row, 0])
            n = len(membership_indices)
            label_ax.set_yticks([j * y_spacing for j in range(n)])
            label_ax.set_yticklabels(traj_names, fontsize=config.tick_fontsize or 10)

            min_extent = 0 - (config.membership_bar_height / 2)
            max_extent = (n - 1) * y_spacing + (config.membership_bar_height / 2)
            label_ax.set_ylim(min_extent - 0.02, max_extent + 0.02)
            label_ax.margins(y=0)
            label_ax.invert_yaxis()
            label_ax.set_xlim(0, 1)
            label_ax.set_xticks([])
            for spine in label_ax.spines.values():
                spine.set_visible(False)

            # Image (at col + 1 because label column always exists in per-feature mode)
            img_ax = fig.add_subplot(gs[membership_row, col + 1])

            # Use same extent and y-configuration as label axes for proper alignment
            first_traj_data = prepared_data[membership_indices[0]]
            x_values = first_traj_data['x_values']
            extent = [
                x_values[0],
                x_values[-1],
                max_extent + 0.02,
                min_extent - 0.02
            ]

            img_ax.imshow(membership_img, aspect='auto', extent=extent)
            img_ax.set_ylim(min_extent - 0.02, max_extent + 0.02)
            img_ax.invert_yaxis()

            # Set xlim explicitly for exact alignment with feature plots
            x_range = x_values[-1] - x_values[0]
            x_margin = x_range * 0.05
            img_ax.set_xlim(x_values[0] - x_margin, x_values[-1] + x_margin)
            img_ax.yaxis.set_visible(False)
            img_ax.spines['left'].set_visible(False)
            img_ax.spines['right'].set_visible(False)
            img_ax.spines['top'].set_visible(False)

            # Add x-label to all membership plots
            img_ax.set_xlabel("Time (ns)" if config.use_time else "Frame", fontsize=config.xlabel_fontsize or 12)

    @staticmethod
    def _plot_single_membership(
        fig: Figure,
        gs: GridSpec,
        n_rows: int,
        config: TimeSeriesPlotConfig,
        prepared_data: Dict,
        membership_indices: List[int],
        cluster_colors: Dict
    ):
        """
        Plot single membership at bottom.

        Parameters
        ----------
        fig : Figure
            Figure to plot on
        gs : GridSpec
            GridSpec layout
        n_rows : int
            Number of feature rows
        config : TimeSeriesPlotConfig
            Central configuration object
        prepared_data : Dict
            Prepared membership data
        membership_indices : List[int]
            Trajectory indices
        cluster_colors : Dict
            Cluster color mapping

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesMembershipPlotHelper._plot_single_membership(
        ...     fig, gs, 2, config, data, indices, colors
        ... )
        """
        ax = fig.add_subplot(gs[n_rows, :])

        y_spacing = 0.6
        MembershipBarRenderer.render_membership_bars_from_prepared(
            ax, prepared_data, membership_indices, cluster_colors,
            config.membership_bar_height, y_spacing
        )

        ax.set_xlabel("Time (ns)" if config.use_time else "Frame", fontsize=config.xlabel_fontsize or 12)

        TimeSeriesMembershipPlotHelper._configure_y_axis(
            ax, config.pipeline_data, membership_indices, config.membership_bar_height, y_spacing, config.tick_fontsize
        )

    @staticmethod
    def _render_membership_to_image(
        fig, gs, prepared_data, membership_indices, cluster_colors,
        bar_height, use_time
    ):
        """
        Render membership to reusable image.

        Parameters
        ----------
        fig : Figure
            Figure for dimension calculation
        gs : GridSpec
            GridSpec for column width calculation
        prepared_data : Dict
            Prepared membership data
        membership_indices : List[int]
            Trajectory indices
        cluster_colors : Dict
            Cluster color mapping
        bar_height : float
            Height of each bar
        use_time : bool
            Use time axis (True) or frames (False)

        Returns
        -------
        np.ndarray
            RGB image array

        Examples
        --------
        >>> img = TimeSeriesMembershipPlotHelper._render_membership_to_image(
        ...     fig, gs, data, indices, colors, 0.5, True
        ... )

        Notes
        -----
        Creates temporary figure at higher DPI for image-based replication.
        More efficient than redrawing membership bars for each feature.
        """
        col_width = fig.get_figwidth() / gs.ncols
        n_traj = len(membership_indices)
        spacing_factor = 1.2
        row_height = max(0.8, n_traj * bar_height * spacing_factor)
        figsize = (col_width * 0.95, row_height)
        dpi = int(fig.dpi * 1.5)

        temp_fig, temp_ax = plt.subplots(figsize=figsize, dpi=dpi)

        y_spacing = 0.3
        MembershipBarRenderer.render_membership_bars_from_prepared(
            temp_ax, prepared_data, membership_indices, cluster_colors, bar_height, y_spacing
        )

        first_traj_data = prepared_data[membership_indices[0]]
        x_values = first_traj_data['x_values']
        temp_ax.set_xlim(x_values[0], x_values[-1])

        n = len(membership_indices)
        min_extent = 0 - (bar_height / 2)
        max_extent = (n - 1) * y_spacing + (bar_height / 2)
        temp_ax.set_ylim(min_extent - 0.02, max_extent + 0.02)
        temp_ax.margins(y=0)
        temp_ax.invert_yaxis()
        temp_ax.set_yticks([])

        temp_fig.canvas.draw()

        # Extract only the axes bbox (without padding/margins)
        bbox = temp_ax.get_window_extent().transformed(temp_fig.dpi_scale_trans.inverted())
        width = int(bbox.width * temp_fig.dpi)
        x0 = int(bbox.x0 * temp_fig.dpi)
        y0 = int(bbox.y0 * temp_fig.dpi)
        y1 = int(bbox.y1 * temp_fig.dpi)

        buffer = np.asarray(temp_fig.canvas.buffer_rgba())[:, :, :3]
        total_height = buffer.shape[0]

        # Convert Y from bottom-up (matplotlib) to top-down (numpy)
        y_top = total_height - y1
        y_bottom = total_height - y0

        img = buffer[y_top:y_bottom, x0:x0+width]

        plt.close(temp_fig)
        return img

    @staticmethod
    def _configure_y_axis(ax, pipeline_data, membership_indices, bar_height, y_spacing=0.6, tick_fontsize=None):
        """
        Configure Y-axis for membership plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to configure
        pipeline_data : PipelineData
            Pipeline data for trajectory names
        membership_indices : List[int]
            Trajectory indices
        bar_height : float
            Height of each bar
        y_spacing : float, default=0.6
            Vertical spacing between bars
        tick_fontsize : int, optional
            Font size for tick labels (default: 10)

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesMembershipPlotHelper._configure_y_axis(
        ...     ax, pipeline_data, [0, 1, 2], 0.5, 0.6, 10
        ... )

        Notes
        -----
        Sets trajectory names as Y-axis labels and configures limits.
        Inverts Y-axis so first trajectory is at top.
        """
        traj_names = [
            pipeline_data.trajectory_data.trajectory_names[idx]
            for idx in membership_indices
        ]

        n = len(membership_indices)
        ax.set_yticks([i * y_spacing for i in range(n)])
        ax.set_yticklabels(traj_names, fontsize=tick_fontsize or 10)

        min_extent = 0 - (bar_height / 2)
        max_extent = (n - 1) * y_spacing + (bar_height / 2)
        ax.set_ylim(min_extent - 0.02, max_extent + 0.02)
        ax.margins(y=0)
        ax.invert_yaxis()
