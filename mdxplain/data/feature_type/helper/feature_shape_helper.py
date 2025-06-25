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
Feature array format conversion utilities.

Provides helper functions for converting between square and condensed feature 
array formats. All methods are static and support memory-mapped arrays for 
efficient processing of large datasets.
"""

import mdtraj as md
import numpy as np


class FeatureShapeHelper:
    """
    Static utility class for feature array format conversion and shape operations.

    Handles conversion between square (NxMxM) and condensed (NxP) formats for
    distance matrices and contact maps with memory-mapped array support.
    """

    @staticmethod
    def is_memmap(array):
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
        """
        return hasattr(array, "filename") and array.filename is not None

    @staticmethod
    def squareform_to_condensed(square_array, k=0, output_path=None, chunk_size=None):
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
        """
        i_indices, j_indices = np.triu_indices(square_array.shape[-1], k=k)

        if len(square_array.shape) == 3:
            n_frames = square_array.shape[0]
            n_contacts = len(i_indices)

            if output_path is not None:
                result = np.memmap(
                    output_path,
                    dtype=square_array.dtype,
                    mode="w+",
                    shape=(n_frames, n_contacts),
                )
            else:
                result = np.zeros((n_frames, n_contacts),
                                  dtype=square_array.dtype)

            if FeatureShapeHelper.is_memmap(square_array) and chunk_size is not None:
                for i in range(0, n_frames, chunk_size):
                    end_idx = min(i + chunk_size, n_frames)
                    chunk = square_array[i:end_idx]
                    result[i:end_idx] = chunk[:, i_indices, j_indices]
            else:
                result[:] = square_array[:, i_indices, j_indices]

            return result
        else:
            return square_array[i_indices, j_indices]

    @staticmethod
    def condensed_to_squareform(
        condensed_array, residue_pairs, n_residues, chunk_size=None, output_path=None
    ):
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
        """
        if len(condensed_array.shape) == 2:
            n_frames = condensed_array.shape[0]

            # For 2D arrays, we need to process chunk by chunk if using memmap
            if FeatureShapeHelper.is_memmap(condensed_array) and chunk_size is not None:

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

                # Process in chunks to avoid memory issues
                for i in range(0, n_frames, chunk_size):
                    end_idx = min(i + chunk_size, n_frames)
                    chunk = condensed_array[i:end_idx]
                    square_chunk = md.geometry.squareform(chunk, residue_pairs)
                    square_array[i:end_idx] = square_chunk

            else:
                # Process all at once for non-memmap arrays
                square_array = md.geometry.squareform(
                    condensed_array, residue_pairs)

        elif len(condensed_array.shape) == 1:
            # For 1D arrays, direct conversion
            square_array = md.geometry.squareform(
                condensed_array.reshape(1, -1), residue_pairs
            )[0]
        else:
            raise ValueError("condensed_array must be 1D or 2D")

        return square_array
