# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
Matrix operations helper for feature selection system.

Provides matrix collection, merging, and processing operations for
the feature selection system in PipelineData.
"""

import numpy as np
from .selection_memmap_helper import SelectionMemmapHelper


class SelectionMatrixHelper:
    """Helper class for matrix operations in feature selection system."""

    @staticmethod
    def collect_matrices_for_selection(pipeline_data, name: str) -> list:
        """
        Collect matrices for all features in the selection.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        name : str
            Name of the selection to collect matrices for

        Returns:
        --------
        list
            List of matrices for all features in the selection

        Raises:
        -------
        ValueError
            If the selection does not exist
        """
        matrices = []

        # Get results from FeatureSelectorData object instead of separate selected_data
        selector_data = pipeline_data.selected_feature_data[name]
        all_results = selector_data.get_all_results()

        for feature_type, selection_info in all_results.items():
            feature_matrices = SelectionMatrixHelper._get_matrices_for_feature(
                pipeline_data, feature_type, selection_info, name
            )
            matrices.extend(feature_matrices)

        return matrices

    @staticmethod
    def merge_matrices(
        matrices: list, name: str, use_memmap: bool, cache_dir: str, chunk_size: int
    ) -> np.ndarray:
        """
        Merge collected matrices horizontally.

        Parameters:
        -----------
        matrices : list
            List of matrices to merge
        name : str
            Name of the selection to merge matrices for
        use_memmap : bool
            Whether to use memory mapping
        cache_dir : str
            Cache directory for memmap files
        chunk_size : int
            Chunk size for processing

        Returns:
        --------
        numpy.ndarray
            Merged matrix with selected columns from all features

        Raises:
        -------
        ValueError
            If no valid data is found for the selection
        """
        if not matrices:
            raise ValueError(f"No valid data found for selection '{name}'.")

        # Check if any matrix is memmap and preserve memmap nature
        if use_memmap:
            return SelectionMemmapHelper.memmap_hstack(
                matrices, name, cache_dir, chunk_size
            )
        else:
            return np.hstack(matrices)

    @staticmethod
    def _get_matrices_for_feature(
        pipeline_data, feature_type: str, selection_info: dict, name: str
    ) -> list:
        """
        Get matrices for a single feature type.

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object containing feature data
        feature_type : str
            Name of the feature type to get matrices for
        selection_info : dict
            Selection information for the feature type
        name : str
            Name of the selection to get matrices for

        Returns:
        --------
        list
            List of matrices for the feature type

        Raises:
        -------
        ValueError
            If the feature type does not exist
        """
        feature_data = pipeline_data.feature_data[feature_type]
        indices = selection_info["indices"]
        use_reduced_flags = selection_info["use_reduced"]

        reduced_indices, original_indices = SelectionMatrixHelper.group_indices_by_type(
            indices, use_reduced_flags
        )

        matrices = []
        matrices.extend(
            SelectionMatrixHelper._get_data_matrices(
                feature_data,
                reduced_indices,
                "reduced_data",
                "Reduced",
                feature_type,
                name,
                pipeline_data.use_memmap,
                pipeline_data.cache_dir,
                pipeline_data.chunk_size,
            )
        )
        matrices.extend(
            SelectionMatrixHelper._get_data_matrices(
                feature_data,
                original_indices,
                "data",
                "Original",
                feature_type,
                name,
                pipeline_data.use_memmap,
                pipeline_data.cache_dir,
                pipeline_data.chunk_size,
            )
        )

        return matrices

    @staticmethod
    def group_indices_by_type(indices: list, use_reduced_flags: list) -> tuple:
        """
        Group indices by whether they use reduced or original data.

        Parameters:
        -----------
        indices : list
            List of indices to group
        use_reduced_flags : list
            List of flags indicating whether the indices use reduced data

        Returns:
        --------
        tuple
            Tuple containing two lists: reduced_indices and original_indices
        """
        reduced_indices = []
        original_indices = []

        for col_idx, use_reduced in zip(indices, use_reduced_flags):
            if use_reduced:
                reduced_indices.append(col_idx)
            else:
                original_indices.append(col_idx)

        return reduced_indices, original_indices

    @staticmethod
    def _get_data_matrices(
        feature_data,
        indices: list,
        data_attr: str,
        data_type: str,
        feature_type: str,
        name: str,
        use_memmap: bool,
        cache_dir: str,
        chunk_size: int,
    ) -> list:
        """
        Get matrices from specified data attribute.

        Parameters:
        -----------
        feature_data : FeatureData
            Feature data object
        indices : list
            List of indices to get matrices for
        data_attr : str
            Attribute of the feature data object to get matrices from
        data_type : str
            Type of data to get matrices from
        feature_type : str
            Name of the feature type to get matrices for
        name : str
            Name of the selection to get matrices for
        use_memmap : bool
            Whether to use memory mapping
        cache_dir : str
            Cache directory for memmap files
        chunk_size : int
            Chunk size for processing

        Returns:
        --------
        list
            List of matrices for the feature type

        Raises:
        -------
        ValueError
            If the data attribute is not available
        """
        if not indices:
            return []

        data = getattr(feature_data, data_attr)
        if data is None:
            raise ValueError(
                f"{data_type} data not available for feature '{feature_type}' "
                f"in selection '{name}'."
            )

        if use_memmap:
            return SelectionMemmapHelper.create_memmap_selection(
                data, indices, name, data_type, feature_type, cache_dir, chunk_size
            )
        return [data[:, indices]]
