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
Helper for tag-based coloring in landscape plots.

Handles frame-to-tag mapping for decomposition landscape plots,
enabling trajectory tag-based scatter point coloring with proper
legend generation. Uses TagHelper for all generic tag operations.
"""

from typing import Dict, List, Optional, Tuple

from .....decomposition.entities.decomposition_data import DecompositionData
from .....trajectory.entities.trajectory_data import TrajectoryData
from ....helper.tag_helper import TagHelper


class LandscapeTagColoringHelper:
    """
    Helper for tag-based frame coloring in landscape plots.

    Provides methods for mapping decomposition frames to trajectory tags.
    Uses TagHelper for generic tag operations (no duplication).

    Examples
    --------
    >>> # Build frame-tag mapping
    >>> frame_tag_map, tag_colors, unselected = (
    ...     LandscapeTagColoringHelper.build_frame_tag_map(
    ...         decomp_obj, trajectory_data, ["biased", "unbiased"], True
    ...     )
    ... )
    """

    @staticmethod
    def build_frame_tag_map(
        decomp_obj: DecompositionData,
        trajectory_data: TrajectoryData,
        tags_for_coloring: List[str],
        scatter_show_all: bool = False
    ) -> Tuple[Dict[int, str], Dict[str, str], Optional[List[int]]]:
        """
        Build mapping from frame indices to tags.

        Uses decomposition frame_mapping to link frames to trajectories,
        then assigns tags based on trajectory tags via TagHelper.
        When multiple tags match, the last tag in tags_for_coloring list wins.

        Parameters
        ----------
        decomp_obj : DecompositionData
            Decomposition object with frame_mapping attribute
        trajectory_data : TrajectoryData
            Trajectory data container with tag information
        tags_for_coloring : List[str]
            Tags to use for coloring (order matters - last wins)
        scatter_show_all : bool, default=False
            If True, return unselected frame indices for gray plotting

        Returns
        -------
        frame_tag_map : Dict[int, str]
            Mapping from frame_idx to tag (only frames with matching tags)
        tag_colors : Dict[str, str]
            Mapping from tag to hex color
        unselected_indices : Optional[List[int]]
            Frame indices without matching tags (None if scatter_show_all=False)

        Examples
        --------
        >>> frame_map, colors, unsel = helper.build_frame_tag_map(
        ...     decomp, traj_data, ["biased", "unbiased"], scatter_show_all=True
        ... )
        """
        tag_colors = TagHelper.prepare_tag_colors(tags_for_coloring)
        frame_tag_map, unselected_indices = LandscapeTagColoringHelper._map_frames_to_tags(
            decomp_obj, trajectory_data, tags_for_coloring, scatter_show_all
        )
        return frame_tag_map, tag_colors, unselected_indices

    @staticmethod
    def _map_frames_to_tags(
        decomp_obj: DecompositionData,
        trajectory_data: TrajectoryData,
        tags_for_coloring: List[str],
        scatter_show_all: bool
    ) -> Tuple[Dict[int, str], Optional[List[int]]]:
        """
        Map frames to tags using frame_mapping.

        Parameters
        ----------
        decomp_obj : DecompositionData
            Decomposition object
        trajectory_data : TrajectoryData
            Trajectory data
        tags_for_coloring : List[str]
            Tags to match
        scatter_show_all : bool
            Whether to collect unselected frames

        Returns
        -------
        frame_tag_map : Dict[int, str]
            Frame to tag mapping
        unselected_indices : Optional[List[int]]
            Unselected frame indices
        """
        frame_tag_map = {}
        unselected_indices = [] if scatter_show_all else None

        frame_mapping = decomp_obj.get_frame_mapping()
        n_frames = decomp_obj.data.shape[0]
        traj_tag_map = LandscapeTagColoringHelper._build_trajectory_tag_map(
            frame_mapping=frame_mapping,
            n_frames=n_frames,
            trajectory_data=trajectory_data,
            tags_for_coloring=tags_for_coloring
        )

        for frame_idx in range(n_frames):
            traj_idx, _ = frame_mapping[frame_idx]
            best_tag = traj_tag_map.get(traj_idx)

            if best_tag is not None:
                frame_tag_map[frame_idx] = best_tag
            elif scatter_show_all:
                unselected_indices.append(frame_idx)

        return frame_tag_map, unselected_indices

    @staticmethod
    def _build_trajectory_tag_map(
        frame_mapping: Dict[int, Tuple[int, int]],
        n_frames: int,
        trajectory_data: TrajectoryData,
        tags_for_coloring: List[str]
    ) -> Dict[int, str]:
        """
        Build `traj_idx -> best_tag` once and reuse for all frame assignments.

        Parameters
        ----------
        frame_mapping : Dict[int, Tuple[int, int]]
            Global frame mapping from decomposition.
        n_frames : int
            Number of selected decomposition frames.
        trajectory_data : TrajectoryData
            Trajectory metadata source.
        tags_for_coloring : List[str]
            Tag priority list (last wins).

        Returns
        -------
        Dict[int, str]
            Mapping from trajectory index to best matching tag.
        """
        unique_traj_indices = {
            frame_mapping[frame_idx][0]
            for frame_idx in range(n_frames)
        }
        traj_tag_map: Dict[int, str] = {}
        for traj_idx in unique_traj_indices:
            matching_tags = TagHelper.get_matching_tags(
                trajectory_data=trajectory_data,
                traj_idx=traj_idx,
                tags_for_coloring=tags_for_coloring
            )
            best_tag = TagHelper.filter_by_priority(
                matching_tags=matching_tags,
                tags_for_coloring=tags_for_coloring
            )
            if best_tag is not None:
                traj_tag_map[traj_idx] = best_tag
        return traj_tag_map
