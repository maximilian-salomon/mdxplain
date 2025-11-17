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
Helper for creating data selector groups from clusters and tags.

This module provides utilities to automatically create multiple
data selectors from clustering results or trajectory tags.
"""

from __future__ import annotations

from typing import List, Union, TYPE_CHECKING

from ..entities.data_selector_group import DataSelectorGroup
from .frame_selection_helper import FrameSelectionHelper

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ..manager.data_selector_manager import DataSelectorManager


class GroupCreationHelper:
    """
    Helper class for creating data selector groups.

    This class provides static methods to automatically create
    multiple data selectors from clustering results or tags,
    organizing them into named groups.

    Examples
    --------
    >>> # Create selectors for all clusters
    >>> group = GroupCreationHelper.create_cluster_selectors(
    ...     pipeline_data, manager, "clusters", "my_clustering"
    ... )
    >>> print(group.selector_names)
    ['clusters_0', 'clusters_1', 'clusters_2']
    """

    @staticmethod
    def create_cluster_selectors(
        pipeline_data: PipelineData,
        manager: DataSelectorManager,
        group_name: str,
        clustering_name: str,
        cluster_ids: Union[List[int], None] = None,
        noise_id: Union[int, None] = -1,
        force: bool = False,
    ) -> DataSelectorGroup:
        """
        Create data selectors for clusters.

        Creates one data selector per cluster using the manager.
        Noise clusters are filtered out by default.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing clustering results
        manager : DataSelectorManager
            Manager instance for creating and managing selectors
        group_name : str
            Name for the selector group
        clustering_name : str
            Name of the clustering to use for selector creation
        cluster_ids : List[int], optional
            Specific cluster IDs to include.
            If None, includes all non-noise clusters.
        noise_id : int or None, default=-1
            Cluster ID that represents noise/outliers to filter out.
            - If int: Filters out this specific cluster ID (e.g., -1 for sklearn)
            - If None: No filtering, creates selectors for ALL cluster IDs
        force : bool, default=False
            Whether to overwrite existing selectors with same names.
            If False, raises ValueError when selector already exists.

        Returns
        -------
        DataSelectorGroup
            Created group containing all generated selector names.
            Access selector names via group.selector_names attribute.

        Raises
        ------
        ValueError
            If clustering_name does not exist in pipeline_data
        ValueError
            If selector already exists and force is False

        Examples
        --------
        >>> # Create selectors for all non-noise clusters
        >>> group = GroupCreationHelper.create_cluster_selectors(
        ...     pipeline_data, manager, "clusters", "dbscan_clustering"
        ... )
        >>> print(group.selector_names)
        ['clusters_0', 'clusters_1', 'clusters_2']

        >>> # Create selectors for specific clusters only
        >>> group = GroupCreationHelper.create_cluster_selectors(
        ...     pipeline_data, manager, "folded", "clustering",
        ...     cluster_ids=[0, 1]
        ... )

        >>> # Include ALL clusters (even noise)
        >>> group = GroupCreationHelper.create_cluster_selectors(
        ...     pipeline_data, manager, "all_states", "clustering",
        ...     noise_id=None
        ... )
        """
        FrameSelectionHelper.validate_clustering_exists(
            pipeline_data, clustering_name
        )
        cluster_data = pipeline_data.cluster_data[clustering_name]
        labels = cluster_data.get_labels()
        all_cluster_ids = [int(cid) for cid in sorted(set(labels))]

        group = DataSelectorGroup(group_name)
        for cid in all_cluster_ids:
            # Skip noise if specified
            if noise_id is not None and cid == noise_id:
                continue
            # Skip if not in filter
            if cluster_ids is not None and cid not in cluster_ids:
                continue
            GroupCreationHelper._create_cluster_selector(
                pipeline_data, manager, group, group_name, clustering_name, cid, force
            )
        return group

    @staticmethod
    def _create_cluster_selector(
        pipeline_data: PipelineData,
        manager: DataSelectorManager,
        group: DataSelectorGroup,
        group_name: str,
        clustering_name: str,
        cid: int,
        force: bool,
    ) -> None:
        """
        Create selector for single cluster and add to group.

        Creates a data selector for the specified cluster using the manager,
        then adds the selector name to the group's selector list.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing cluster data
        manager : DataSelectorManager
            Manager instance for creating selectors
        group : DataSelectorGroup
            Group to add the created selector name to
        group_name : str
            Base name for the selector group
        clustering_name : str
            Name of clustering to select from
        cid : int
            Cluster ID to create selector for
        force : bool
            Whether to overwrite existing selector

        Returns
        -------
        None
            Modifies group in-place by appending selector name

        Raises
        ------
        ValueError
            If selector already exists and force is False
        """
        name = f"{group_name}_{cid}"
        GroupCreationHelper._create_or_overwrite(pipeline_data, manager, name, force)
        manager.select_by_cluster(pipeline_data, name, clustering_name, [cid])
        group.selector_names.append(name)

    @staticmethod
    def create_tag_selectors(
        pipeline_data: PipelineData,
        manager: DataSelectorManager,
        group_name: str,
        tags: Union[List[str], None] = None,
        force: bool = False,
    ) -> DataSelectorGroup:
        """
        Create data selectors for trajectory tags.

        Creates one data selector per tag using the manager.
        Each selector contains all frames from trajectories with the specified tag.
        No logic duplication - manager does all the work.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing trajectory tag information
        manager : DataSelectorManager
            Manager instance for creating and managing selectors
        group_name : str
            Name for the selector group
        tags : List[str], optional
            Specific tags to create selectors for.
            If None, creates selectors for all available tags found in trajectories.
        force : bool, default=False
            Whether to overwrite existing selectors with same names.
            If False, raises ValueError when selector already exists.

        Returns
        -------
        DataSelectorGroup
            Created group containing all generated selector names.
            Access selector names via group.selector_names attribute.

        Raises
        ------
        ValueError
            If selector already exists and force is False

        Examples
        --------
        >>> # Create selectors for specific tags
        >>> group = GroupCreationHelper.create_tag_selectors(
        ...     pipeline_data, manager, "systems",
        ...     tags=["system_A", "system_B"]
        ... )
        >>> print(group.selector_names)
        ['systems_system_A', 'systems_system_B']

        >>> # Create selectors for all available tags
        >>> group = GroupCreationHelper.create_tag_selectors(
        ...     pipeline_data, manager, "conditions", tags=None
        ... )
        >>> print(group.selector_names)
        ['conditions_wild_type', 'conditions_mutant', 'conditions_biased']
        """
        all_tags = set()
        for tag_list in pipeline_data.trajectory_data.trajectory_tags.values():
            all_tags.update(tag_list)
        tags_to_use = sorted(all_tags) if tags is None else tags

        group = DataSelectorGroup(group_name)
        for tag in tags_to_use:
            selector_name = f"{group_name}_{tag}"
            GroupCreationHelper._create_or_overwrite(
                pipeline_data, manager, selector_name, force
            )
            manager.select_by_tags(pipeline_data, selector_name, [tag], match_all=True)
            group.selector_names.append(selector_name)

        return group

    @staticmethod
    def _create_or_overwrite(
        pipeline_data: PipelineData,
        manager: DataSelectorManager,
        name: str,
        force: bool,
    ) -> None:
        """
        Create selector or overwrite if force is True.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        manager : DataSelectorManager
            Manager instance
        name : str
            Selector name
        force : bool
            Overwrite if exists

        Returns
        -------
        None
            Creates or clears selector

        Raises
        ------
        ValueError
            If selector exists and force is False
        """
        if name in pipeline_data.data_selector_data:
            if not force:
                raise ValueError(f"Selector '{name}' exists. Use force=True.")
            manager.clear_selector(pipeline_data, name)
        else:
            manager.create(pipeline_data, name)
