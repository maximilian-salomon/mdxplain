# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.5).
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
Generic helper for trajectory tag operations.

Provides shared functionality for tag matching, color mapping,
and legend creation across different plot types.
"""

from typing import Dict, List, Optional
import matplotlib.lines as mlines

from ...trajectory.entities.trajectory_data import TrajectoryData
from .color_mapping_helper import ColorMappingHelper


class TagHelper:
    """
    Generic helper for trajectory tag operations.

    Provides shared functionality for tag matching, color mapping,
    and legend creation across different plot types.

    Examples
    --------
    >>> # Get matching tags for trajectory
    >>> matching = TagHelper.get_matching_tags(
    ...     trajectory_data, traj_idx=0, tags_for_coloring=["biased", "unbiased"]
    ... )
    >>> print(matching)  # ["biased"]

    >>> # Filter by priority (last wins)
    >>> best = TagHelper.filter_by_priority(
    ...     matching_tags=["biased", "unbiased"],
    ...     tags_for_coloring=["biased", "unbiased", "other"]
    ... )
    >>> print(best)  # "unbiased"
    """

    @staticmethod
    def get_matching_tags(
        trajectory_data: TrajectoryData,
        traj_idx: int,
        tags_for_coloring: List[str]
    ) -> List[str]:
        """
        Get matching tags for trajectory.

        Returns all tags from trajectory that appear in tags_for_coloring list.

        Parameters
        ----------
        trajectory_data : TrajectoryData
            Trajectory data container with tag information
        traj_idx : int
            Trajectory index to get tags for
        tags_for_coloring : List[str]
            List of tags to match against

        Returns
        -------
        List[str]
            Matching tags (preserves order from tags_for_coloring)

        Examples
        --------
        >>> matching = TagHelper.get_matching_tags(
        ...     trajectory_data, 0, ["biased", "unbiased"]
        ... )
        """
        traj_tags = trajectory_data.get_trajectory_tags(traj_idx)
        if traj_tags is None:
            return []
        traj_tag_set = set(traj_tags)
        return [tag for tag in tags_for_coloring if tag in traj_tag_set]

    @staticmethod
    def filter_by_priority(
        matching_tags: List[str],
        tags_for_coloring: List[str]
    ) -> Optional[str]:
        """
        Get best matching tag based on priority (last wins).

        When multiple tags match, returns the last one in tags_for_coloring list.
        This implements the "last wins" priority rule.

        Parameters
        ----------
        matching_tags : List[str]
            Tags that matched for a trajectory
        tags_for_coloring : List[str]
            Ordered list of tags (priority: last is highest)

        Returns
        -------
        str or None
            Best matching tag (last in tags_for_coloring) or None if no matches

        Examples
        --------
        >>> best = TagHelper.filter_by_priority(
        ...     ["biased", "unbiased"], ["biased", "unbiased", "other"]
        ... )
        >>> print(best)  # "unbiased"
        """
        if not matching_tags:
            return None
        matching_tag_set = set(matching_tags)
        for tag in reversed(tags_for_coloring):
            if tag in matching_tag_set:
                return tag
        return None

    @staticmethod
    def prepare_tag_colors(tags_for_coloring: List[str]) -> Dict[str, str]:
        """
        Prepare tag-to-color mapping.

        Uses ColorMappingHelper for consistent color assignment across plots.

        Parameters
        ----------
        tags_for_coloring : List[str]
            List of tags to assign colors to

        Returns
        -------
        Dict[str, str]
            Mapping from tag name to hex color string

        Examples
        --------
        >>> colors = TagHelper.prepare_tag_colors(["biased", "unbiased"])
        >>> print(colors)  # {"biased": "#1f77b4", "unbiased": "#ff7f0e"}
        """
        colors_list = ColorMappingHelper.get_cluster_colors(
            len(tags_for_coloring), include_noise=False
        )
        return {tag: colors_list[i] for i, tag in enumerate(tags_for_coloring)}

    @staticmethod
    def create_tag_legend_handles(
        tag_colors: Dict[str, str]
    ) -> List[mlines.Line2D]:
        """
        Create legend handles for tags.

        Generates matplotlib Line2D objects for each tag-color pair,
        suitable for use in figure legends.

        Parameters
        ----------
        tag_colors : Dict[str, str]
            Mapping from tag to hex color

        Returns
        -------
        List[matplotlib.lines.Line2D]
            Legend handles for each tag (sorted alphabetically)

        Examples
        --------
        >>> handles = TagHelper.create_tag_legend_handles(
        ...     {"biased": "#1f77b4", "unbiased": "#ff7f0e"}
        ... )
        """
        handles = []
        for tag, color in sorted(tag_colors.items()):
            handle = mlines.Line2D(
                [], [],
                color=color,
                marker='o',
                linestyle='None',
                markersize=8,
                label=tag
            )
            handles.append(handle)
        return handles
