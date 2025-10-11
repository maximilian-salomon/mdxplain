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
Helper class for input validation in plots module.

Provides validation for plot parameters to ensure data consistency
and proper error messages before plot generation.
"""

from __future__ import annotations

from typing import List, Optional

from mdxplain.pipeline.entities.pipeline_data import PipelineData


class ValidationHelper:
    """
    Helper class for input validation in plots.

    Provides static methods for validating plot parameters to ensure
    data consistency and provide clear error messages before plot generation.

    Examples
    --------
    >>> # Validate landscape inputs
    >>> ValidationHelper.validate_landscape_inputs(
    ...     pipeline_data, "pca", [0, 1], "dbscan", False
    ... )

    >>> # Validate dimension layout
    >>> ValidationHelper.validate_dimensions_for_layout([0, 1, 2, 3])
    """

    @staticmethod
    def validate_landscape_inputs(
        pipeline_data: PipelineData,
        decomposition_name: str,
        dimensions: List[int],
        clustering_name: Optional[str] = None,
        show_centers: bool = False
    ) -> None:
        """
        Validate inputs for landscape plot generation.

        Checks decomposition existence, dimension validity, clustering
        requirements, and consistency between parameters.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        decomposition_name : str
            Name of decomposition to plot
        dimensions : List[int]
            Dimension indices to plot
        clustering_name : Optional[str], default=None
            Name of clustering (if overlay requested)
        show_centers : bool, default=False
            Whether to show cluster centers

        Returns
        -------
        None
            Raises ValueError if validation fails

        Raises
        ------
        ValueError
            If decomposition not found
        ValueError
            If dimensions invalid or out of range
        ValueError
            If show_centers=True but no clustering specified
        ValueError
            If clustering not found
        ValueError
            If clustering incompatible with decomposition

        Examples
        --------
        >>> # Valid inputs
        >>> ValidationHelper.validate_landscape_inputs(
        ...     pipeline_data, "pca", [0, 1], "dbscan", False
        ... )

        >>> # Missing clustering for centers
        >>> ValidationHelper.validate_landscape_inputs(
        ...     pipeline_data, "pca", [0, 1], None, show_centers=True
        ... )
        ValueError: show_centers=True requires clustering_name to be specified

        >>> # Invalid dimension
        >>> ValidationHelper.validate_landscape_inputs(
        ...     pipeline_data, "pca", [0, 15], None, False
        ... )
        ValueError: Dimension 15 out of range for decomposition 'pca' (max: 9)
        """
        # Validate decomposition exists and get components
        decomp_obj = ValidationHelper._validate_decomposition_exists(
            pipeline_data, decomposition_name
        )
        n_components = decomp_obj.data.shape[1]

        # Validate dimensions list (format, range, duplicates)
        ValidationHelper._validate_dimensions_list(
            dimensions, decomposition_name, n_components
        )

        # Validate show_centers requirement
        ValidationHelper._validate_show_centers_requirement(
            show_centers, clustering_name
        )

        # Validate clustering compatibility if specified
        if clustering_name:
            n_frames_decomp = decomp_obj.data.shape[0]
            ValidationHelper._validate_clustering_compatibility(
                pipeline_data, clustering_name,
                decomposition_name, n_frames_decomp
            )

    @staticmethod
    def _validate_decomposition_exists(
        pipeline_data: PipelineData,
        decomposition_name: str
    ):
        """
        Check if decomposition exists in pipeline data.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        decomposition_name : str
            Name of decomposition to check

        Returns
        -------
        DecompositionData
            Decomposition object if found

        Raises
        ------
        ValueError
            If decomposition not found
        """
        if decomposition_name not in pipeline_data.decomposition_data:
            available = list(pipeline_data.decomposition_data.keys())
            raise ValueError(
                f"Decomposition '{decomposition_name}' not found. "
                f"Available: {available}"
            )
        return pipeline_data.decomposition_data[decomposition_name]

    @staticmethod
    def _validate_dimensions_list(
        dimensions: List[int],
        decomposition_name: str,
        n_components: int
    ) -> None:
        """
        Validate dimensions list format, type, range, and uniqueness.

        Parameters
        ----------
        dimensions : List[int]
            Dimension indices to validate
        decomposition_name : str
            Name of decomposition (for error messages)
        n_components : int
            Number of available components in decomposition

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If dimensions empty, wrong type, out of range, or duplicates
        """
        if not dimensions:
            raise ValueError("dimensions cannot be empty")

        if not isinstance(dimensions, list):
            raise ValueError(
                f"dimensions must be a list, got {type(dimensions).__name__}"
            )

        # Check each dimension
        for dim in dimensions:
            if not isinstance(dim, int):
                raise ValueError(
                    f"All dimensions must be integers, got {type(dim).__name__}"
                )
            if dim < 0 or dim >= n_components:
                raise ValueError(
                    f"Dimension {dim} out of range for decomposition "
                    f"'{decomposition_name}' (valid range: 0-{n_components-1})"
                )

        # Check for duplicates
        if len(dimensions) != len(set(dimensions)):
            raise ValueError(
                f"Duplicate dimensions found in {dimensions}"
            )

    @staticmethod
    def _validate_show_centers_requirement(
        show_centers: bool,
        clustering_name: Optional[str]
    ) -> None:
        """
        Validate show_centers requires clustering_name.

        Parameters
        ----------
        show_centers : bool
            Whether cluster centers should be shown
        clustering_name : Optional[str]
            Name of clustering

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If show_centers=True but clustering_name not specified
        """
        if show_centers and not clustering_name:
            raise ValueError(
                "show_centers=True requires clustering_name to be specified"
            )

    @staticmethod
    def _validate_clustering_compatibility(
        pipeline_data: PipelineData,
        clustering_name: str,
        decomposition_name: str,
        n_frames_decomp: int
    ) -> None:
        """
        Validate clustering exists and matches decomposition frame count.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        clustering_name : str
            Name of clustering to validate
        decomposition_name : str
            Name of decomposition (for error messages)
        n_frames_decomp : int
            Number of frames in decomposition

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If clustering not found or frame count mismatch
        """
        if clustering_name not in pipeline_data.cluster_data:
            available = list(pipeline_data.cluster_data.keys())
            raise ValueError(
                f"Clustering '{clustering_name}' not found. "
                f"Available: {available}"
            )

        # Check frame count compatibility
        cluster_obj = pipeline_data.cluster_data[clustering_name]
        n_frames_cluster = len(cluster_obj.labels)

        if n_frames_decomp != n_frames_cluster:
            raise ValueError(
                f"Clustering '{clustering_name}' has {n_frames_cluster} frames "
                f"but decomposition '{decomposition_name}' has {n_frames_decomp} frames. "
                "They must match for overlay plotting."
            )

    @staticmethod
    def validate_dimensions_for_layout(dimensions: List[int]) -> None:
        """
        Validate that dimensions list is suitable for grid layout.

        For landscape plots, dimensions are paired consecutively.
        An even number of dimensions is required for proper grid layout.

        Parameters
        ----------
        dimensions : List[int]
            Dimension indices to plot

        Returns
        -------
        None
            Raises ValueError if validation fails

        Raises
        ------
        ValueError
            If odd number of dimensions provided

        Examples
        --------
        >>> # Valid - even number
        >>> ValidationHelper.validate_dimensions_for_layout([0, 1, 2, 3])

        >>> # Invalid - odd number
        >>> ValidationHelper.validate_dimensions_for_layout([0, 1, 2])
        ValueError: Odd number of dimensions (3) provided. Landscape plots require
        consecutive dimension pairs. Please provide an even number of dimensions.

        Notes
        -----
        Dimensions are paired consecutively: [0,1,2,3] becomes [(0,1), (2,3)]
        for a 2x1 grid layout.
        """
        if len(dimensions) % 2 != 0:
            raise ValueError(
                f"Odd number of dimensions ({len(dimensions)}) provided. "
                "Landscape plots require consecutive dimension pairs. "
                "Please provide an even number of dimensions."
            )
