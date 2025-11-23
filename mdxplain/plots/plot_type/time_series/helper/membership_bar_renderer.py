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
Helper for rendering cluster membership bars in time series plots.

Reuses logic from MembershipPlotter for efficient block-based rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict
import numpy as np
import matplotlib.pyplot as plt

from ...membership.helper.block_optimizer_helper import BlockOptimizerHelper
from ....helper.color_mapping_helper import NOISE_COLOR
from ....helper.clustering_data_helper import ClusteringDataHelper
from .time_series_data_preparer import TimeSeriesDataPreparer

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData


class MembershipBarRenderer:
    """
    Renderer for cluster membership bars in time series plots.

    Reuses MembershipPlotter's efficient block-based rendering logic
    to draw horizontal bars showing cluster membership over time.

    Examples
    --------
    >>> # Render membership bars on axis
    >>> MembershipBarRenderer.render_membership_bars(
    ...     ax, pipeline_data, "dbscan", [0, 1, 2],
    ...     bar_height=0.3, use_time=True
    ... )
    """

    @staticmethod
    def prepare_membership_data(
        pipeline_data: PipelineData,
        clustering_name: str,
        traj_indices: List[int],
        use_time: bool
    ) -> tuple:
        """
        Prepare membership data once for multiple plots.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        clustering_name : str
            Clustering name
        traj_indices : List[int]
            Trajectory indices
        use_time : bool
            Use time or frames

        Returns
        -------
        prepared_data : Dict[int, Dict]
            {traj_idx: {'blocks': [...], 'x_values': [...]}}
        cluster_colors : Dict[int, str]
            Cluster color mapping
        labels : numpy.ndarray
            All cluster labels

        Examples
        --------
        >>> data, colors, labels = MembershipBarRenderer.prepare_membership_data(
        ...     pipeline_data, "dbscan", [0, 1, 2], True
        ... )
        """
        labels, cluster_colors = ClusteringDataHelper.load_clustering_data(
            pipeline_data, clustering_name
        )

        prepared_data = {}
        for traj_idx in traj_indices:
            traj_labels = MembershipBarRenderer._get_trajectory_labels(
                pipeline_data, labels, traj_idx
            )
            blocks = BlockOptimizerHelper.labels_to_blocks(traj_labels)
            x_values = TimeSeriesDataPreparer.get_x_values(
                pipeline_data, traj_idx, use_time
            )
            prepared_data[traj_idx] = {
                'blocks': blocks,
                'x_values': x_values
            }

        return prepared_data, cluster_colors, labels

    @staticmethod
    def render_membership_bars_from_prepared(
        ax: plt.Axes,
        prepared_data: Dict[int, Dict],
        traj_indices: List[int],
        cluster_colors: Dict[int, str],
        bar_height: float = 0.15,
        y_spacing: float = 0.3
    ) -> None:
        """
        Render membership bars using pre-computed data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to render on
        prepared_data : Dict[int, Dict]
            Pre-computed data from prepare_membership_data
        traj_indices : List[int]
            Trajectory indices to render
        cluster_colors : Dict[int, str]
            Cluster color mapping
        bar_height : float, default=0.15
            Height of bars
        y_spacing : float, default=0.5
            Vertical spacing between bars

        Returns
        -------
        None

        Examples
        --------
        >>> data, colors, _ = MembershipBarRenderer.prepare_membership_data(...)
        >>> MembershipBarRenderer.render_membership_bars_from_prepared(
        ...     ax, data, [0, 1], colors, 0.3, 0.7
        ... )
        """
        for i, traj_idx in enumerate(traj_indices):
            y_pos = i * y_spacing
            traj_data = prepared_data[traj_idx]
            blocks = traj_data['blocks']
            x_values = traj_data['x_values']

            MembershipBarRenderer._render_blocks(
                ax, blocks, y_pos, x_values, cluster_colors, bar_height
            )

    @staticmethod
    def _get_trajectory_labels(
        pipeline_data: PipelineData,
        labels: np.ndarray,
        traj_idx: int
    ) -> np.ndarray:
        """
        Extract labels for trajectory.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        labels : numpy.ndarray
            All labels
        traj_idx : int
            Trajectory index

        Returns
        -------
        numpy.ndarray
            Labels for this trajectory
        """
        start_frame = sum(
            pipeline_data.trajectory_data.trajectories[i].n_frames
            for i in range(traj_idx)
        )
        end_frame = start_frame + pipeline_data.trajectory_data.trajectories[traj_idx].n_frames
        return labels[start_frame:end_frame]

    @staticmethod
    def _render_blocks(
        ax: plt.Axes,
        blocks: List[tuple],
        y_pos: int,
        x_values: np.ndarray,
        cluster_colors: Dict[int, str],
        bar_height: float
    ) -> None:
        """
        Render membership blocks.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to render on
        blocks : List[tuple]
            Block definitions
        y_pos : int
            Y position for this trajectory
        x_values : numpy.ndarray
            X-axis values
        cluster_colors : Dict[int, str]
            Color mapping
        bar_height : float
            Bar height

        Returns
        -------
        None
        """
        frame_step = (
            x_values[1] - x_values[0] if len(x_values) > 1 else 1
        )

        for start, end, cluster_id in blocks:
            color = cluster_colors.get(cluster_id, NOISE_COLOR)
            x_start = x_values[start]
            if end + 1 < len(x_values):
                x_width = x_values[end + 1] - x_start
            else:
                x_width = (x_values[-1] - x_start) + frame_step

            ax.barh(
                y=y_pos,
                width=x_width,
                left=x_start,
                height=bar_height,
                color=color,
                edgecolor='none',
                align='center'
            )
