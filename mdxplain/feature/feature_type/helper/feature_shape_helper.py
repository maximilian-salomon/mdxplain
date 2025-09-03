# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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
Feature array format conversion utilities.

Provides helper functions for converting between square and condensed feature
array formats. All methods are static and support memory-mapped arrays for
efficient processing of large datasets.
"""

from typing import Optional, List, Tuple
import mdtraj as md
import numpy as np


class FeatureShapeHelper:
    """
    Static utility class for feature array format conversion and shape operations.

    Handles conversion between square (NxMxM) and condensed (NxP) formats for
    distance matrices and contact maps with memory-mapped array support.
    """

    @staticmethod
    def is_memmap(array: np.ndarray) -> bool:
        """
        Check if array is memory-mapped.

        Parameters:
        -----------
        array : np.ndarray
            Array to check

        Returns:
        --------
        bool
            True if array is memory-mapped

        Examples:
        ---------
        >>> array = np.memmap("array.npy", mode="w+", shape=(100, 100))
        >>> FeatureShapeHelper.is_memmap(array)
        True

        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> FeatureShapeHelper.is_memmap(array)
        False
        """
        return hasattr(array, "filename") and array.filename is not None

    @staticmethod
    def squareform_to_condensed(square_array: np.ndarray, k: int = 0, output_path: Optional[str] = None, chunk_size: int = 2000) -> np.ndarray:
        """
        Convert square format to condensed format.

        Parameters:
        -----------
        square_array : np.ndarray
            Square format array (NxMxM)
        k : int, default=0
            Diagonal offset (0 excludes main diagonal)
        output_path : str, optional
            Path for memory-mapped output
        chunk_size : int, optional
            Chunk size for processing

        Returns:
        --------
        np.ndarray
            Condensed format array (NxP)

        Examples:
        ---------
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> FeatureShapeHelper.squareform_to_condensed(array)
        array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        >>> array = np.memmap("array.npy", mode="w+", shape=(100, 100))
        >>> FeatureShapeHelper.squareform_to_condensed(array)
        array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        >>> array = np.memmap("array.npy", mode="w+", shape=(100, 100))
        >>> FeatureShapeHelper.squareform_to_condensed(array, chunk_size=10)
        array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        Raises:
        -------
        ValueError
            If square_array is not 3D
        """
        i_indices, j_indices = np.triu_indices(square_array.shape[-1], k=k)

        if len(square_array.shape) == 3:
            return FeatureShapeHelper._convert_3d_square_to_condensed(
                square_array, i_indices, j_indices, output_path, chunk_size
            )
        else:
            return square_array[i_indices, j_indices]

    @staticmethod
    def _convert_3d_square_to_condensed(
        square_array: np.ndarray, 
        i_indices: np.ndarray, 
        j_indices: np.ndarray, 
        output_path: Optional[str], 
        chunk_size: int
    ) -> np.ndarray:
        """
        Convert 3D square array to condensed format.

        Parameters:
        -----------
        square_array : np.ndarray
            Square format array (NxMxM)
        i_indices : np.ndarray
            Indices of the first dimension of the square array
        j_indices : np.ndarray
            Indices of the second dimension of the square array
        output_path : str, optional
            Path for memory-mapped output
        chunk_size : int, optional
            Chunk size for processing

        Returns:
        --------
        np.ndarray
            Condensed format array (NxP)
        """
        n_frames = square_array.shape[0]
        n_contacts = len(i_indices)

        result = FeatureShapeHelper._create_condensed_output_array(
            output_path, square_array.dtype, n_frames, n_contacts
        )

        use_chunked_processing = FeatureShapeHelper.is_memmap(square_array)

        if use_chunked_processing:
            FeatureShapeHelper._fill_condensed_chunked(
                square_array, result, i_indices, j_indices, chunk_size, n_frames
            )
        else:
            result[:] = square_array[:, i_indices, j_indices]

        return result

    @staticmethod
    def _create_condensed_output_array(output_path: Optional[str], dtype: type, n_frames: int, n_contacts: int) -> np.ndarray:
        """
        Create output array for condensed format.

        Parameters:
        -----------
        output_path : str, optional
            Path for memory-mapped output
        dtype : dtype
            Data type of the output array
        n_frames : int
            Number of frames in the input array
        n_contacts : int
            Number of contacts in the input array

        Returns:
        --------
        np.ndarray
            Output array for condensed format
        """
        if output_path is not None:
            return np.memmap(
                output_path,
                dtype=dtype,
                mode="w+",
                shape=(n_frames, n_contacts),
            )
        else:
            return np.zeros((n_frames, n_contacts), dtype=dtype)

    @staticmethod
    def _fill_condensed_chunked(
        square_array: np.ndarray, 
        result: np.ndarray, 
        i_indices: np.ndarray, 
        j_indices: np.ndarray, 
        chunk_size: int, 
        n_frames: int
    ) -> None:
        """
        Fill condensed array using chunked processing.

        Parameters:
        -----------
        square_array : np.ndarray
            Square format array (NxMxM)
        result : np.ndarray
            Condensed format array (NxP)
        i_indices : np.ndarray
            Indices of the first dimension of the square array
        j_indices : np.ndarray
            Indices of the second dimension of the square array
        chunk_size : int
            Chunk size for processing
        n_frames : int
            Number of frames in the input array

        Returns:
        --------
        None
        """
        for i in range(0, n_frames, chunk_size):
            end_idx = min(i + chunk_size, n_frames)
            chunk = square_array[i:end_idx]
            result[i:end_idx] = chunk[:, i_indices, j_indices]

    @staticmethod
    def condensed_to_squareform(
        condensed_array: np.ndarray, 
        residue_pairs: np.ndarray, 
        n_residues: int, 
        chunk_size: int = 2000, 
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Convert condensed format to square format using MDTraj.

        Parameters:
        -----------
        condensed_array : np.ndarray
            Condensed format array (NxP)
        residue_pairs : np.ndarray
            Residue pair indices
        n_residues : int
            Number of residues
        chunk_size : int, optional
            Chunk size for processing
        output_path : str, optional
            Path for memory-mapped output

        Returns:
        --------
        np.ndarray
            Square format array (NxMxM)

        Raises:
        -------
        ValueError
            If condensed_array is not 1D or 2D
        """
        array_dims = len(condensed_array.shape)

        if array_dims == 1:
            return FeatureShapeHelper._convert_1d_to_square(
                condensed_array, residue_pairs
            )
        elif array_dims == 2:
            return FeatureShapeHelper._convert_2d_to_square(
                condensed_array, residue_pairs, n_residues, chunk_size, output_path
            )
        else:
            raise ValueError("condensed_array must be 1D or 2D")

    @staticmethod
    def _convert_1d_to_square(condensed_array: np.ndarray, residue_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Convert 1D condensed array to square format.

        Parameters:
        -----------
        condensed_array : np.ndarray
            Condensed format array (NxP)
        residue_pairs : np.ndarray
            Residue pair indices

        Returns:
        --------
        np.ndarray
            Square format array (NxMxM)
        """
        return md.geometry.squareform(condensed_array.reshape(1, -1), residue_pairs)[0]

    @staticmethod
    def _convert_2d_to_square(
        condensed_array: np.ndarray, 
        residue_pairs: np.ndarray, 
        n_residues: int, 
        chunk_size: int, 
        output_path: Optional[str]
    ) -> np.ndarray:
        """
        Convert 2D condensed array to square format.

        Parameters:
        -----------
        condensed_array : np.ndarray
            Condensed format array (NxP)
        residue_pairs : np.ndarray
            Residue pair indices
        n_residues : int
            Number of residues
        chunk_size : int, optional
            Chunk size for processing
        output_path : str, optional
            Path for memory-mapped output

        Returns:
        --------
        np.ndarray
            Square format array (NxMxM)
        """
        use_chunked_processing = FeatureShapeHelper.is_memmap(condensed_array)

        if use_chunked_processing:
            return FeatureShapeHelper._convert_2d_chunked(
                condensed_array, residue_pairs, n_residues, chunk_size, output_path
            )
        else:
            return md.geometry.squareform(condensed_array, residue_pairs)

    @staticmethod
    def _convert_2d_chunked(
        condensed_array: np.ndarray, 
        residue_pairs: np.ndarray, 
        n_residues: int, 
        chunk_size: int, 
        output_path: Optional[str]
    ) -> np.ndarray:
        """
        Convert 2D condensed array to square format using chunked processing.

        Parameters:
        -----------
        condensed_array : np.ndarray
            Condensed format array (NxP)
        residue_pairs : np.ndarray
            Residue pair indices
        n_residues : int
            Number of residues
        chunk_size : int, optional
            Chunk size for processing
        output_path : str, optional
            Path for memory-mapped output

        Returns:
        --------
        np.ndarray
            Square format array (NxMxM)
        """
        n_frames = condensed_array.shape[0]

        if output_path is not None:
            square_array = np.memmap(
                output_path,
                dtype=condensed_array.dtype,
                mode="w+",
                shape=(n_frames, n_residues, n_residues),
            )
        else:
            square_array = np.zeros(
                (n_frames, n_residues, n_residues), dtype=condensed_array.dtype
            )

        for i in range(0, n_frames, chunk_size):
            end_idx = min(i + chunk_size, n_frames)
            chunk = condensed_array[i:end_idx]
            square_chunk = md.geometry.squareform(chunk, residue_pairs)
            square_array[i:end_idx] = square_chunk

        return square_array
