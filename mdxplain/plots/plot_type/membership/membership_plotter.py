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
Cluster membership timeline plotter.

Creates horizontal bar visualizations showing cluster membership
over time for multiple trajectories, with efficient block-based rendering.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from matplotlib.figure import Figure

from .helper import BlockOptimizerHelper
from ...helper.color_mapping_helper import NOISE_COLOR
from ...helper.validation_helper import ValidationHelper
from ...helper.clustering_data_helper import ClusteringDataHelper
from ...helper.svg_export_helper import SvgExportHelper
from ....utils.data_utils import DataUtils


class MembershipPlotter:
    """
    Plotter for cluster membership timeline visualization.

    Creates horizontal bar plots showing cluster assignment over time
    for selected trajectories. Uses efficient block-based rendering
    to handle large trajectories with many frames.

    Examples
    --------
    >>> # Basic usage
    >>> plotter = MembershipPlotter(pipeline_data)
    >>> fig = plotter.plot("dbscan")

    >>> # Custom trajectory order
    >>> fig = plotter.plot("hdbscan", traj_selection=[2, 0, 5])

    >>> # By tag selection
    >>> fig = plotter.plot("dpa", traj_selection="tag:system_A")
    """

    def __init__(self, pipeline_data, cache_dir: str = "./cache") -> None:
        """
        Initialize membership plotter.

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
        legend_fontsize: Optional[int] = None
    ) -> Figure:
        """
        Create cluster membership timeline plot.

        Visualizes cluster assignment over time as horizontal colored bars,
        with each trajectory on a separate row.

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
            Font size for overall figure title (default: 14)
        xlabel_fontsize : int, optional
            Font size for X-axis label (default: 12)
        ylabel_fontsize : int, optional
            Font size for Y-axis trajectory labels (default: 11)
        tick_fontsize : int, optional
            Font size for X-axis tick labels (default: 10)
        legend_fontsize : int, optional
            Font size for cluster legend entries (default: 10)

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
        >>> fig = plotter.plot("dbscan")

        >>> # Specific trajectories in custom order
        >>> fig = plotter.plot("dbscan", traj_selection=[2, 0, 5])

        >>> # By tag selection
        >>> fig = plotter.plot("hdbscan", traj_selection="tag:system_A")

        >>> # Customized appearance
        >>> fig = plotter.plot(
        ...     "dpa",
        ...     traj_selection=[0, 1, 2],
        ...     height_per_trajectory=0.5,
        ...     title="DPA Clustering Membership",
        ...     save_fig=True
        ... )

        Notes
        -----
        - Uses block-based rendering for efficiency (O(transitions) not O(frames))
        - Trajectory order in plot matches order in traj_selection
        - Colors are consistent with other plots via ColorMappingHelper
        - Noise points (-1) are shown in black
        """
        # Validate inputs using atomic validation methods
        ValidationHelper.validate_clustering_exists(
            self.pipeline_data, clustering_name
        )
        traj_indices = ValidationHelper.validate_trajectory_selection(
            self.pipeline_data, traj_selection
        )

        # Get clustering data
        labels, cluster_colors = ClusteringDataHelper.load_clustering_data(
            self.pipeline_data, clustering_name
        )

        # Setup figure with fixed width
        fig, ax = self._setup_figure(
            len(traj_indices), height_per_trajectory,
            xlabel_fontsize or 12, tick_fontsize or 10
        )

        # Plot membership bars
        self._plot_trajectory_bars(
            ax, traj_indices, labels, cluster_colors, height_per_trajectory
        )

        # Set axis labels and styling
        self._set_y_labels(ax, traj_indices, ylabel_fontsize or 11)
        self._set_x_labels(ax, show_frame_numbers, xlabel_fontsize or 12, tick_fontsize or 10)

        # Apply tight layout BEFORE adding title and legend
        # Reserve right 12% for legend, bottom 5% for X-axis label
        fig.tight_layout(rect=[0, 0.05, 0.88, 1.0])

        # Add title AFTER tight_layout
        self._set_title(fig, title, clustering_name, title_fontsize or 14)

        # Add legend AFTER tight_layout
        if show_legend:
            self._add_legend(fig, cluster_colors, legend_fontsize or 10)

        # Save if requested
        if save_fig:
            # Configure SVG export for editable text
            SvgExportHelper.apply_svg_config_if_needed(file_format)

            self._save_figure(
                fig, filename, clustering_name, traj_selection, file_format, dpi
            )

        return fig

    def _setup_figure(
        self,
        n_trajectories: int,
        height_per_trajectory: float,
        xlabel_fontsize: int,
        tick_fontsize: int
    ) -> Tuple[Figure, plt.Axes]:
        """
        Setup figure with fixed width and dynamic height.

        Parameters
        ----------
        n_trajectories : int
            Number of trajectories to plot
        height_per_trajectory : float
            Height in inches per trajectory
        xlabel_fontsize : int
            Font size for X-axis label
        tick_fontsize : int
            Font size for tick labels

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure
        ax : matplotlib.axes.Axes
            Single axis for plotting

        Notes
        -----
        Uses fixed width of 12 inches for consistent timeline visualization.
        Height scales dynamically based on trajectory count and font sizes.
        """
        # Fixed width for consistent timeline appearance
        fig_width = 12.0

        # Base space for title and labels
        base_space = 0.8

        # Add extra space for larger fonts
        extra_space = 0.0
        if xlabel_fontsize > 12:
            extra_space += (xlabel_fontsize - 12) * 0.02
        if tick_fontsize > 10:
            extra_space += (tick_fontsize - 10) * 0.02

        # Dynamic height based on trajectory count + space for title and labels
        fig_height = n_trajectories * height_per_trajectory + base_space + extra_space

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        return fig, ax

    def _plot_trajectory_bars(
        self,
        ax: plt.Axes,
        traj_indices: List[int],
        labels: np.ndarray,
        cluster_colors: Dict[int, str],
        height_per_trajectory: float
    ) -> None:
        """
        Plot membership bars for all trajectories.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on
        traj_indices : List[int]
            Trajectory indices to plot
        labels : numpy.ndarray
            Cluster labels for all frames
        cluster_colors : Dict[int, str]
            Color mapping for clusters
        height_per_trajectory : float
            Height in inches per trajectory (bar height is 50% of this)

        Returns
        -------
        None
        """
        # Calculate bar height as 50% of available trajectory height
        bar_height = height_per_trajectory * 0.5

        for y_pos, traj_idx in enumerate(traj_indices):
            # Get labels for this trajectory
            traj_labels = self._get_trajectory_labels(labels, traj_idx)

            # Convert to blocks for efficiency
            blocks = BlockOptimizerHelper.labels_to_blocks(traj_labels)

            # Plot each block
            for start, end, cluster_id in blocks:
                color = cluster_colors.get(cluster_id, NOISE_COLOR)
                ax.barh(
                    y=y_pos,
                    width=end - start + 1,
                    left=start,
                    height=bar_height,
                    color=color,
                    edgecolor='none',
                    align='center'
                )

    def _get_trajectory_labels(
        self, labels: np.ndarray, traj_idx: int
    ) -> np.ndarray:
        """
        Extract labels for specific trajectory.

        Parameters
        ----------
        labels : numpy.ndarray
            Labels for all frames
        traj_idx : int
            Trajectory index

        Returns
        -------
        numpy.ndarray
            Labels for frames in this trajectory
        """
        # Calculate frame range for this trajectory
        start_frame = sum(
            len(self.pipeline_data.trajectory_data.trajectories[i])
            for i in range(traj_idx)
        )
        end_frame = start_frame + len(
            self.pipeline_data.trajectory_data.trajectories[traj_idx]
        )

        return labels[start_frame:end_frame]

    def _set_y_labels(self, ax: plt.Axes, traj_indices: List[int], ylabel_fontsize: int) -> None:
        """
        Set y-axis labels to trajectory names.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to configure
        traj_indices : List[int]
            Trajectory indices
        ylabel_fontsize : int
            Font size for trajectory labels

        Returns
        -------
        None
        """
        names = [
            self.pipeline_data.trajectory_data.trajectory_names[i]
            for i in traj_indices
        ]
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=ylabel_fontsize)
        ax.invert_yaxis()  # Top trajectory at top

    def _set_x_labels(self, ax: plt.Axes, show_frame_numbers: bool, xlabel_fontsize: int, tick_fontsize: int) -> None:
        """
        Configure x-axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to configure
        show_frame_numbers : bool
            Whether to show frame numbers
        xlabel_fontsize : int
            Font size for X-axis label
        tick_fontsize : int
            Font size for tick labels

        Returns
        -------
        None
        """
        if show_frame_numbers:
            ax.set_xlabel("Frame Number", fontsize=xlabel_fontsize)
        else:
            ax.set_xlabel("Time", fontsize=xlabel_fontsize)

        # Set tick label fontsize
        ax.tick_params(axis='x', labelsize=tick_fontsize)

    def _set_title(
        self, fig: Figure, title: Optional[str], clustering_name: str, title_fontsize: int
    ) -> None:
        """
        Set figure title positioned at top with constant 10mm spacing.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to add title to
        title : Optional[str]
            Custom title or None for auto-generation
        clustering_name : str
            Name of clustering (for auto-generation)
        title_fontsize : int
            Font size for title

        Returns
        -------
        None
        """
        title_text = title if title else (
            f"Cluster Membership Timeline - {clustering_name.upper()}"
        )

        # Calculate y position for constant 10mm spacing above figure
        # 10mm = 10/25.4 inches = 0.394 inches
        _, fig_height = fig.get_size_inches()
        title_spacing_inches = 0.394
        y_position = 1.0 + (title_spacing_inches / fig_height)

        fig.suptitle(
            title_text,
            fontsize=title_fontsize,
            fontweight='bold',
            y=y_position  # Constant 10mm spacing regardless of figure height
        )

    def _add_legend(
        self, fig: Figure, cluster_colors: Dict[int, str], legend_fontsize: int
    ) -> None:
        """
        Add cluster color legend positioned outside plot area on right.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to add legend to
        cluster_colors : Dict[int, str]
            Color mapping for clusters
        legend_fontsize : int
            Font size for legend entries

        Returns
        -------
        None
        """
        # Create legend handles
        handles = []
        labels = []

        # Sort cluster IDs (excluding noise)
        cluster_ids = sorted([k for k in cluster_colors.keys() if k >= 0])

        for cluster_id in cluster_ids:
            handles.append(
                plt.Line2D([0], [0], color=cluster_colors[cluster_id], lw=4)
            )
            labels.append(f'Cluster {cluster_id}')

        # Add noise if present
        if -1 in cluster_colors:
            handles.append(
                plt.Line2D([0], [0], color=cluster_colors[-1], lw=4)
            )
            labels.append('Noise')

        # Position legend with overlap for closer appearance
        fig.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(0.90, 0.5),
            frameon=True,
            fontsize=legend_fontsize
        )

    def _save_figure(
        self,
        fig: Figure,
        filename: Optional[str],
        clustering_name: str,
        traj_selection: Union[int, str, List, "all"],
        file_format: str,
        dpi: int
    ) -> None:
        """
        Save figure to file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : Optional[str]
            Custom filename or None for auto-generation
        clustering_name : str
            Name of clustering
        traj_selection : int, str, list, or "all"
            Trajectory selection
        file_format : str
            File format
        dpi : int
            Resolution

        Returns
        -------
        None
        """
        if filename is None:
            filename = self._generate_filename(clustering_name, traj_selection)

        if not filename.endswith(f".{file_format}"):
            filename = f"{filename}.{file_format}"

        filepath = DataUtils.get_cache_file_path(filename, self.cache_dir)
        fig.savefig(filepath, dpi=dpi, format=file_format, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")

    def _generate_filename(
        self, clustering_name: str, traj_selection: Union[int, str, List, "all"]
    ) -> str:
        """
        Generate automatic filename.

        Parameters
        ----------
        clustering_name : str
            Name of clustering
        traj_selection : int, str, list, or "all"
            Trajectory selection

        Returns
        -------
        str
            Generated filename without extension
        """
        name = f"membership_{clustering_name}"

        if traj_selection != "all":
            if isinstance(traj_selection, int):
                name += f"_traj{traj_selection}"
            elif isinstance(traj_selection, str):
                # Clean string for filename
                clean = traj_selection.replace(":", "_").replace(" ", "_")
                name += f"_{clean}"
            elif isinstance(traj_selection, list):
                name += f"_traj{len(traj_selection)}"

        return name
