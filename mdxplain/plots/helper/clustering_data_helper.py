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
Helper for loading clustering data and colors.

Centralizes clustering data access logic used across multiple plotters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple
import numpy as np

from .color_mapping_helper import ColorMappingHelper

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData


class ClusteringDataHelper:
    """
    Helper for loading clustering data and colors.

    Provides centralized access to clustering labels and color mappings,
    used by multiple plotter types (time series, landscape, membership).

    Examples
    --------
    >>> labels, colors = ClusteringDataHelper.load_clustering_data(
    ...     pipeline_data, "dbscan"
    ... )
    """

    @staticmethod
    def load_clustering_data(
        pipeline_data: PipelineData,
        clustering_name: str
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Load clustering labels and color mapping.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        clustering_name : str
            Name of clustering to load

        Returns
        -------
        labels : numpy.ndarray
            Cluster labels for all frames
        cluster_colors : Dict[int, str]
            Mapping from cluster ID to color

        Examples
        --------
        >>> labels, colors = ClusteringDataHelper.load_clustering_data(
        ...     pipeline_data, "dbscan"
        ... )
        >>> print(labels.shape)  # (n_frames,)
        >>> print(colors[0])  # '#color_hex'
        """
        cluster_obj = pipeline_data.cluster_data[clustering_name]
        labels = cluster_obj.labels
        n_clusters = len(np.unique(labels[labels >= 0]))
        cluster_colors = ColorMappingHelper.get_cluster_colors(n_clusters)
        return labels, cluster_colors
