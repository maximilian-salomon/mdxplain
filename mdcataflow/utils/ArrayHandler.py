# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# ArrayConverter - Array Format Conversion Utilities
#
# Utility class for converting between square and condensed array formats.
# All methods are static and can be used without instantiation.
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

import numpy as np

class ArrayHandler:
    """
    Utility class for converting between square and condensed array formats.
    All methods are static and can be used without instantiation.
    """
    
    @staticmethod
    def is_memmap(array):
        """Check if array is a memory-mapped array."""
        return hasattr(array, 'filename') and array.filename is not None
    
    @staticmethod
    def squareform_to_condensed(square_array, k=0, output_path=None, chunk_size=None):
        """Convert square format (NxMxM) to condensed format (NxP)."""
        i_indices, j_indices = np.triu_indices(square_array.shape[-1], k=k)
        
        if len(square_array.shape) == 3:
            n_frames = square_array.shape[0]
            n_contacts = len(i_indices)
            
            if output_path is not None:
                result = np.memmap(output_path, dtype=square_array.dtype, mode='w+', 
                                 shape=(n_frames, n_contacts))
            else:
                result = np.zeros((n_frames, n_contacts), dtype=square_array.dtype)
            
            if ArrayHandler.is_memmap(square_array) and chunk_size is not None:
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
    def _apply_symmetry_to_chunk(square_array, start_idx, end_idx, k):
        """Apply symmetry to a chunk of square arrays."""
        chunk = square_array[start_idx:end_idx]
        if k == 0:
            chunk_T = np.transpose(chunk, (0, 2, 1))
            diag_mask = np.eye(chunk.shape[1], dtype=bool)
            square_array[start_idx:end_idx] = chunk + chunk_T - chunk[:, diag_mask, :][:, :, np.newaxis] * diag_mask
        else:
            square_array[start_idx:end_idx] = chunk + np.transpose(chunk, (0, 2, 1))

    @staticmethod
    def _process_condensed_chunk(condensed_array, square_array, start_idx, end_idx, n_residues, k):
        """Process a chunk of condensed array to square format."""
        i_indices, j_indices = np.triu_indices(n_residues, k=k)
        chunk = condensed_array[start_idx:end_idx]
        square_array[start_idx:end_idx, i_indices, j_indices] = chunk
        ArrayHandler._apply_symmetry_to_chunk(square_array, start_idx, end_idx, k)

    @staticmethod
    def condensed_to_squareform(condensed_array, n_residues, k=0, chunk_size=None, output_path=None):
        """Convert condensed format (NxP) to square format (NxMxM)."""
        if len(condensed_array.shape) == 2:
            n_frames = condensed_array.shape[0]
            
            if output_path is not None:
                square_array = np.memmap(output_path, dtype=condensed_array.dtype, mode='w+',
                                       shape=(n_frames, n_residues, n_residues))
            else:
                square_array = np.zeros((n_frames, n_residues, n_residues), dtype=condensed_array.dtype)
            
            if ArrayHandler.is_memmap(condensed_array) and chunk_size is not None:
                for i in range(0, n_frames, chunk_size):
                    end_idx = min(i + chunk_size, n_frames)
                    ArrayHandler._process_condensed_chunk(condensed_array, square_array, i, end_idx, n_residues, k)
            else:
                i_indices, j_indices = np.triu_indices(n_residues, k=k)
                square_array[:, i_indices, j_indices] = condensed_array
                ArrayHandler._apply_symmetry_to_chunk(square_array, 0, n_frames, k)
                
        elif len(condensed_array.shape) == 1:
            square_array = np.zeros((n_residues, n_residues), dtype=condensed_array.dtype)
            i_indices, j_indices = np.triu_indices(n_residues, k=k)
            square_array[i_indices, j_indices] = condensed_array
            if k == 0:
                square_array = square_array + square_array.T - np.diag(np.diag(square_array))
            else:
                square_array = square_array + square_array.T
        else:
            raise ValueError("condensed_array must be 1D or 2D")
        return square_array 