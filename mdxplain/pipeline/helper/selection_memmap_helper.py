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
Memory mapping operations helper for feature selection system.

Provides memory-mapped matrix operations for efficient handling of
large datasets in the feature selection system.
"""

from typing import List
import numpy as np
from tqdm import tqdm
from ...utils.data_utils import DataUtils


class SelectionMemmapHelper:
    """Helper class for memory mapping operations in feature selection system."""

    @staticmethod
    def create_memmap_selection(
        data: np.ndarray,
        indices: List[int],
        name: str,
        data_type: str,
        feature_type: str,
        cache_dir: str,
        chunk_size: int,
    ) -> List[np.ndarray]:
        """
        Create memory-efficient selection using chunk-wise processing.

        This method avoids loading entire columns into RAM by processing
        data in chunks and writing directly to a memmap output file.

        Parameters:
        -----------
        data : numpy.ndarray or memmap
            Source data to select from
        indices : list
            Column indices to select
        name : str
            Selection name for cache file naming
        data_type : str
            Type of data (for cache naming)
        feature_type : str
            Feature type name (for cache naming)
        cache_dir : str
            Cache directory for memmap files
        chunk_size : int
            Chunk size for processing

        Returns:
        --------
        list
            List containing the memmap-selected data matrix
        """
        n_rows, _ = data.shape
        n_cols = len(indices)

        cache_path = DataUtils.get_cache_file_path(
            f"{name}_{feature_type}_{data_type}_selection.dat", cache_dir
        )

        result = np.memmap(
            cache_path,
            dtype=data.dtype,
            mode="w+",
            shape=(n_rows, n_cols),
        )

        for row_start in tqdm(range(0, n_rows, chunk_size), desc="Selecting columns", unit="chunks"):
            row_end = min(row_start + chunk_size, n_rows)
            result[row_start:row_end, :] = data[row_start:row_end, indices]

        return [result]

    @staticmethod
    def memmap_hstack(
        matrices: list, name: str, cache_dir: str, chunk_size: int
    ) -> np.ndarray:
        """
        Horizontally stack matrices while preserving memmap nature.

        Parameters:
        -----------
        matrices : list
            List of matrices to stack (all are memmap)
        name : str
            Name of the selection for cache file naming
        cache_dir : str
            Cache directory for memmap files
        chunk_size : int
            Chunk size for processing

        Returns:
        --------
        numpy.ndarray
            Stacked matrix stored as memmap
        """
        # Calculate total shape
        total_samples = matrices[0].shape[0]
        total_features = sum(matrix.shape[1] for matrix in matrices)

        # Determine the appropriate dtype for mixed data types
        dtypes = [matrix.dtype for matrix in matrices]
        result_dtype = np.result_type(*dtypes)

        # Create cache path for the stacked matrix
        cache_path = DataUtils.get_cache_file_path(f"{name}.dat", cache_dir)

        # Create memmap for the result
        result = np.memmap(
            cache_path,
            dtype=result_dtype,
            mode="w+",
            shape=(total_samples, total_features),
        )

        # Fill the result matrix column by column, chunk by chunk
        col_start = 0
        for i, matrix in enumerate(matrices):
            col_end = col_start + matrix.shape[1]

            # Process in chunks to avoid loading entire matrix into memory
            for row_start in tqdm(range(0, total_samples, chunk_size), desc=f"Concatenating matrix {i+1}/{len(matrices)}", unit="chunks", leave=False):
                row_end = min(row_start + chunk_size, total_samples)
                result[row_start:row_end, col_start:col_end] = matrix[
                    row_start:row_end, :
                ]

            col_start = col_end

        return result

    @staticmethod
    def create_memmap_frame_selection(
        data: np.ndarray,
        frame_indices: List[int],
        name: str,
        cache_dir: str,
        chunk_size: int,
    ) -> np.ndarray:
        """
        Create memory-efficient frame selection using chunk-wise processing.

        This method avoids loading entire rows into RAM by processing
        data in chunks and writing directly to a memmap output file.
        Handles frame selection (row-wise) efficiently for large datasets.

        Parameters:
        -----------
        data : numpy.ndarray or memmap
            Source data to select from
        frame_indices : list
            Row indices to select
        name : str
            Selection name for cache file naming
        cache_dir : str
            Cache directory for memmap files
        chunk_size : int
            Chunk size for processing

        Returns:
        --------
        numpy.ndarray
            Memmap array with selected frames
        """
        n_selected_frames = len(frame_indices)
        _, n_cols = data.shape

        cache_path = DataUtils.get_cache_file_path(
            f"{name}_frame_selection.dat", cache_dir
        )

        result = np.memmap(
            cache_path,
            dtype=data.dtype,
            mode="w+",
            shape=(n_selected_frames, n_cols),
        )

        # Process in chunks to avoid loading entire data into memory
        for chunk_start in tqdm(range(0, n_selected_frames, chunk_size), desc="Creating frame selection", unit="chunks"):
            chunk_end = min(chunk_start + chunk_size, n_selected_frames)
            
            # Get indices for this chunk
            chunk_indices = frame_indices[chunk_start:chunk_end]
            
            # Copy data for this chunk
            result[chunk_start:chunk_end, :] = data[chunk_indices, :]

        return result
