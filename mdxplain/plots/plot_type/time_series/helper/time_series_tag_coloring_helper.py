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
Helper for tag-based coloring in time series plots.

Handles trajectory tag mapping, color assignment, and tag-based plotting
with proper bug fixes for auto-detection, circular logic, and legend display.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData

from ....helper.color_mapping_helper import ColorMappingHelper


class TimeSeriesTagColoringHelper:
    """
    Helper for tag-based trajectory coloring.

    Provides methods for tag mapping, color assignment, and plotting
    with tag-based coloring. Fixes all bugs from original implementation:
    - Auto-detection of tag coloring mode
    - No circular logic dependencies
    - Real trajectory names instead of "Traj 0"
    - Only shows tags that actually exist

    Examples
    --------
    >>> # Auto-detect tag coloring mode
    >>> use_tags = TimeSeriesTagColoringHelper.should_use_tag_coloring(
    ...     tags_for_coloring=["test"]
    ... )
    >>> print(use_tags)  # True

    >>> # Build tag map
    >>> tag_map = TimeSeriesTagColoringHelper.build_tag_map(
    ...     pipeline_data, [0, 1, 2], ["system_A"]
    ... )
    """

    @staticmethod
    def should_use_tag_coloring(
        tags_for_coloring: Optional[List[str]]
    ) -> bool:
        """
        Determine if tag coloring should be used (AUTO-DETECTION).

        FIX: Automatically enables tag coloring if tags_for_coloring is set.
        User doesn't need to manually set color_by_tags=True.

        Parameters
        ----------
        tags_for_coloring : List[str] or None
            Tags specified by user

        Returns
        -------
        bool
            True if tag coloring should be used

        Examples
        --------
        >>> # User sets tags → auto-enable
        >>> TimeSeriesTagColoringHelper.should_use_tag_coloring(["test"])
        True

        >>> # No tags → disable
        >>> TimeSeriesTagColoringHelper.should_use_tag_coloring(None)
        False
        """
        return tags_for_coloring is not None and len(tags_for_coloring) > 0

    @staticmethod
    def build_tag_map(
        pipeline_data: PipelineData,
        traj_indices: List[int],
        tags_for_coloring: Optional[List[str]],
        allow_multi_tag_plotting: bool = False
    ) -> Dict[int, List[str]]:
        """
        Build tag mapping for trajectories.

        FIX: No circular logic - builds map unconditionally,
        independent of color_by_tags parameter.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        traj_indices : List[int]
            Trajectory indices to map
        tags_for_coloring : List[str] or None
            Tags to filter for
        allow_multi_tag_plotting : bool, default=False
            Allow trajectories with multiple tags

        Returns
        -------
        Dict[int, List[str]]
            Mapping of traj_idx -> matching tags

        Examples
        --------
        >>> tag_map = TimeSeriesTagColoringHelper.build_tag_map(
        ...     pipeline_data, [0, 1, 2], ["system_A", "biased"]
        ... )
        >>> print(tag_map)  # {0: ["system_A"], 1: ["biased"]}
        """
        has_tags = TimeSeriesTagColoringHelper._has_tags_for_coloring(tags_for_coloring)
        if not has_tags:
            return {idx: [] for idx in traj_indices}

        return TimeSeriesTagColoringHelper._build_tag_map_with_tags(
            pipeline_data, traj_indices, tags_for_coloring, allow_multi_tag_plotting
        )

    @staticmethod
    def _has_tags_for_coloring(tags_for_coloring: Optional[List[str]]) -> bool:
        """
        Check if tags for coloring are provided.

        Parameters
        ----------
        tags_for_coloring : List[str] or None
            Tags to check

        Returns
        -------
        bool
            True if tags are provided and non-empty

        Examples
        --------
        >>> TimeSeriesTagColoringHelper._has_tags_for_coloring(["test"])
        True
        >>> TimeSeriesTagColoringHelper._has_tags_for_coloring(None)
        False
        """
        return tags_for_coloring is not None and len(tags_for_coloring) > 0

    @staticmethod
    def _build_tag_map_with_tags(
        pipeline_data: PipelineData,
        traj_indices: List[int],
        tags_for_coloring: List[str],
        allow_multi_tag_plotting: bool
    ) -> Dict[int, List[str]]:
        """
        Build tag map with provided tags.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        traj_indices : List[int]
            Trajectory indices
        tags_for_coloring : List[str]
            Tags to match
        allow_multi_tag_plotting : bool
            Allow multi-tag trajectories

        Returns
        -------
        Dict[int, List[str]]
            Tag mapping

        Examples
        --------
        >>> tag_map = TimeSeriesTagColoringHelper._build_tag_map_with_tags(
        ...     pipeline_data, [0, 1], ["test"], False
        ... )
        """
        tag_map = {}
        for traj_idx in traj_indices:
            tags = TimeSeriesTagColoringHelper._process_trajectory_for_tag_map(
                pipeline_data, traj_idx, tags_for_coloring, allow_multi_tag_plotting
            )
            if tags is not None:
                tag_map[traj_idx] = tags
        return tag_map

    @staticmethod
    def _process_trajectory_for_tag_map(
        pipeline_data: PipelineData,
        traj_idx: int,
        tags_for_coloring: List[str],
        allow_multi_tag_plotting: bool
    ) -> Optional[List[str]]:
        """
        Process single trajectory for tag map.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        traj_idx : int
            Trajectory index
        tags_for_coloring : List[str]
            Tags to match
        allow_multi_tag_plotting : bool
            Allow trajectories with multiple tags

        Returns
        -------
        List[str] or None
            Matching tags if trajectory should be included, None otherwise

        Examples
        --------
        >>> tags = TimeSeriesTagColoringHelper._process_trajectory_for_tag_map(
        ...     pipeline_data, 0, ["system_A"], False
        ... )
        """
        matching_tags = TimeSeriesTagColoringHelper._get_matching_tags(
            pipeline_data, traj_idx, tags_for_coloring
        )

        if not matching_tags:
            return None

        if allow_multi_tag_plotting or len(matching_tags) == 1:
            return matching_tags

        return None

    @staticmethod
    def _get_matching_tags(
        pipeline_data: PipelineData,
        traj_idx: int,
        tags_for_coloring: List[str]
    ) -> List[str]:
        """
        Get matching tags for trajectory.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        traj_idx : int
            Trajectory index
        tags_for_coloring : List[str]
            Tags to match

        Returns
        -------
        List[str]
            Matching tags for this trajectory

        Examples
        --------
        >>> tags = TimeSeriesTagColoringHelper._get_matching_tags(
        ...     pipeline_data, 0, ["system_A", "biased"]
        ... )
        """
        traj_tags = pipeline_data.trajectory_data.get_trajectory_tags(traj_idx)
        if traj_tags is None:
            return []
        return [t for t in traj_tags if t in tags_for_coloring]

    @staticmethod
    def prepare_tag_legend_colors(
        tag_map: Dict[int, List[str]]
    ) -> Dict[str, str]:
        """
        Prepare tag-to-color mapping for legend.

        Only includes tags that actually exist in tag_map.

        Parameters
        ----------
        tag_map : Dict[int, List[str]]
            Tag mapping from build_tag_map()

        Returns
        -------
        Dict[str, str]
            Tag -> color mapping (only existing tags)

        Examples
        --------
        >>> colors = TimeSeriesTagColoringHelper.prepare_tag_legend_colors(
        ...     {0: ["system_A"]}
        ... )
        >>> print(colors)  # {"system_A": "#color"}
        """
        # Find tags that actually exist in tag_map
        existing_tags = set()
        for tags in tag_map.values():
            existing_tags.update(tags)

        # Sort for consistent color assignment
        sorted_existing_tags = sorted(existing_tags)

        # Assign colors only to existing tags
        colors = ColorMappingHelper.get_cluster_colors(len(sorted_existing_tags))
        return {tag: colors[i] for i, tag in enumerate(sorted_existing_tags)}

    @staticmethod
    def prepare_trajectory_legend_colors(
        pipeline_data: PipelineData,
        tag_map: Dict[int, List[str]]
    ) -> Dict[str, str]:
        """
        Prepare trajectory-name-to-color mapping for legend.

        FIX: Uses real trajectory names from pipeline_data instead of "Traj 0".

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        tag_map : Dict[int, List[str]]
            Tag mapping (keys are trajectory indices)

        Returns
        -------
        Dict[str, str]
            Trajectory name -> color mapping

        Examples
        --------
        >>> colors = TimeSeriesTagColoringHelper.prepare_trajectory_legend_colors(
        ...     pipeline_data, {0: [], 1: []}
        ... )
        >>> print(colors)  # {"system_A_run1": "#color", "system_B_run1": "#color2"}
        """
        traj_indices = sorted(tag_map.keys())
        colors_list = ColorMappingHelper.get_cluster_colors(len(traj_indices), include_noise=False)

        # Use REAL trajectory names
        return {
            pipeline_data.trajectory_data.trajectory_names[idx]: colors_list[i]
            for i, idx in enumerate(traj_indices)
        }

    @staticmethod
    def plot_feature_with_tag_colors(
        ax: plt.Axes,
        pipeline_data: PipelineData,
        feat_idx: int,
        tag_map: Dict[int, List[str]],
        tag_colors: Dict[str, str],
        feature_selector_name: str,
        use_time: bool
    ) -> None:
        """
        Plot feature lines colored by tags.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        pipeline_data : PipelineData
            Pipeline data container
        feat_idx : int
            Feature index
        tag_map : Dict[int, List[str]]
            Tag mapping
        tag_colors : Dict[str, str]
            Tag color mapping
        feature_selector_name : str
            Feature selector name
        use_time : bool
            Use time (True) or frames (False)

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesTagColoringHelper.plot_feature_with_tag_colors(
        ...     ax, pipeline_data, 0, tag_map, colors, "selector", True
        ... )
        """
        for traj_idx, tags in tag_map.items():
            for tag in tags:
                color = tag_colors.get(tag, "black")
                TimeSeriesTagColoringHelper._plot_single_line(
                    ax, pipeline_data, traj_idx, feat_idx,
                    feature_selector_name, use_time, color
                )

    @staticmethod
    def plot_feature_with_trajectory_colors(
        ax: plt.Axes,
        pipeline_data: PipelineData,
        feat_idx: int,
        tag_map: Dict[int, List[str]],
        traj_colors: Dict[str, str],
        feature_selector_name: str,
        use_time: bool
    ) -> None:
        """
        Plot feature lines colored by trajectory.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        pipeline_data : PipelineData
            Pipeline data container
        feat_idx : int
            Feature index
        tag_map : Dict[int, List[str]]
            Tag mapping (keys used for trajectory indices)
        traj_colors : Dict[str, str]
            Trajectory name -> color mapping
        feature_selector_name : str
            Feature selector name
        use_time : bool
            Use time (True) or frames (False)

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesTagColoringHelper.plot_feature_with_trajectory_colors(
        ...     ax, pipeline_data, 0, tag_map, colors, "selector", True
        ... )
        """
        for traj_idx in tag_map.keys():
            traj_name = pipeline_data.trajectory_data.trajectory_names[traj_idx]
            color = traj_colors.get(traj_name, "black")
            TimeSeriesTagColoringHelper._plot_single_line(
                ax, pipeline_data, traj_idx, feat_idx,
                feature_selector_name, use_time, color
            )

    @staticmethod
    def _plot_single_line(
        ax: plt.Axes,
        pipeline_data: PipelineData,
        traj_idx: int,
        feat_idx: int,
        feature_selector_name: str,
        use_time: bool,
        color: str
    ) -> None:
        """
        Plot single trajectory line.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        pipeline_data : PipelineData
            Pipeline data container
        traj_idx : int
            Trajectory index
        feat_idx : int
            Feature index
        feature_selector_name : str
            Feature selector name
        use_time : bool
            Use time or frames
        color : str
            Line color

        Returns
        -------
        None

        Examples
        --------
        >>> TimeSeriesTagColoringHelper._plot_single_line(
        ...     ax, pipeline_data, 0, 5, "selector", True, "blue"
        ... )
        """
        trajectory = pipeline_data.trajectory_data.trajectories[traj_idx]
        matrix, frame_mapping = pipeline_data.get_selected_data(
            feature_selector_name, return_frame_mapping=True
        )

        traj_frame_indices = TimeSeriesTagColoringHelper._get_trajectory_frame_indices(
            frame_mapping, traj_idx
        )

        y_values = matrix[traj_frame_indices, feat_idx]

        if use_time:
            local_frames = [frame_mapping[i][1] for i in traj_frame_indices]
            x_values = trajectory.time[local_frames] / 1000
        else:
            x_values = np.arange(len(y_values))

        ax.plot(x_values, y_values, color=color, linewidth=1.0, alpha=0.8)

        if hasattr(matrix, '_mmap') and matrix._mmap is not None:
            matrix._mmap.close()
        del matrix

    @staticmethod
    def _get_trajectory_frame_indices(
        frame_mapping: Dict[int, tuple],
        traj_idx: int
    ) -> List[int]:
        """
        Get global frame indices for specific trajectory.

        Parameters
        ----------
        frame_mapping : Dict[int, tuple]
            Frame mapping from get_selected_data
        traj_idx : int
            Trajectory index to filter for

        Returns
        -------
        List[int]
            Global frame indices for this trajectory

        Examples
        --------
        >>> mapping = {0: (0, 5), 1: (0, 10), 2: (1, 3)}
        >>> indices = TimeSeriesTagColoringHelper._get_trajectory_frame_indices(
        ...     mapping, 0
        ... )
        >>> print(indices)  # [0, 1]
        """
        return [
            global_idx for global_idx, (t_idx, _) in frame_mapping.items()
            if t_idx == traj_idx
        ]
