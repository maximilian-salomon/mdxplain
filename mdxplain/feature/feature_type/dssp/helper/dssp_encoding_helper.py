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
DSSP encoding helper utilities for molecular dynamics trajectory analysis.

Helper functions for encoding DSSP secondary structure assignments into
different formats (character, integer, one-hot) with memory-efficient
chunk-wise processing for large datasets.
"""

import numpy as np
from .....utils import DataUtils


class DSSPEncodingHelper:
    """
    Helper class for DSSP encoding operations.
    
    Provides static methods for converting DSSP assignments between different
    encoding formats with support for memory mapping and chunk-wise processing
    for large trajectory datasets.
    
    Examples
    --------
    >>> # Character encoding with space conversion
    >>> char_data = DSSPEncodingHelper.encode_char_chunked(
    ...     dssp_data, False, 1000, "./cache"
    ... )
    
    >>> # Integer encoding for classification
    >>> int_data = DSSPEncodingHelper.encode_integer_chunked(
    ...     dssp_data, classes, 1000, "./cache"
    ... )
    """

    @staticmethod
    def encode_char_chunked(dssp_data: np.ndarray, chunk_size: int, cache_path: str) -> np.ndarray:
        """
        Encode DSSP to character format with chunk-wise processing.
        
        Parameters
        ----------
        dssp_data : numpy.ndarray
            Input DSSP data with shape (n_frames, n_residues)
        chunk_size : int
            Number of frames to process per chunk
        cache_path : str
            Directory path for cache files
            
        Returns
        -------
        numpy.ndarray
            Character-encoded DSSP data
            
        Notes
        -----
        Uses memory mapping for efficient processing of large datasets.
        Space conversion happens centrally in dssp_calculator before this method
        is called, so dssp_data is already cleaned.
        """
        cache_file = DataUtils.get_cache_file_path(
            f'dssp_char_{id(dssp_data)}.dat', cache_path
        )
        encoded = np.memmap(
            cache_file, dtype='U1', mode='w+', shape=dssp_data.shape
        )

        for i in range(0, dssp_data.shape[0], chunk_size):
            end = min(i + chunk_size, dssp_data.shape[0])
            chunk = dssp_data[i:end]
            encoded[i:end] = chunk.astype('U1')
        
        return encoded

    @staticmethod
    def encode_integer(dssp_data: np.ndarray, classes: list) -> np.ndarray:
        """
        Encode DSSP to integer format in-memory.
        
        Parameters
        ----------
        dssp_data : numpy.ndarray
            Input DSSP data with shape (n_frames, n_residues)
        classes : list
            List of class labels for integer mapping

        Returns
        -------
        numpy.ndarray
            Integer-encoded DSSP data
            
        Notes
        -----
        For small datasets that fit in memory. Uses vectorized operations
        for efficient conversion.
        """
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        encoded = np.zeros(dssp_data.shape, dtype=np.int8)
        for class_char, idx in class_to_idx.items():
            encoded[dssp_data == class_char] = idx
        return encoded

    @staticmethod
    def encode_integer_chunked(dssp_data: np.ndarray, classes: list, chunk_size: int, cache_path: str) -> np.ndarray:
        """
        Encode DSSP to integer format with chunk-wise processing.
        
        Parameters
        ----------
        dssp_data : numpy.ndarray
            Input DSSP data with shape (n_frames, n_residues)
        classes : list
            List of class labels for integer mapping
        chunk_size : int
            Number of frames to process per chunk
        cache_path : str
            Directory path for cache files
            
        Returns
        -------
        numpy.ndarray
            Integer-encoded DSSP data
            
        Notes
        -----
        Uses memory mapping for efficient processing of large datasets.
        Processes data in chunks to avoid memory overflow.
        """
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        cache_file = DataUtils.get_cache_file_path(
            f'dssp_int_{id(dssp_data)}.dat', cache_path
        )
        encoded = np.memmap(
            cache_file, dtype=np.int8, mode='w+', shape=dssp_data.shape
        )
        
        for i in range(0, dssp_data.shape[0], chunk_size):
            end = min(i + chunk_size, dssp_data.shape[0])
            chunk = dssp_data[i:end]
            chunk_encoded = np.zeros(chunk.shape, dtype=np.int8)
            
            for class_char, idx in class_to_idx.items():
                chunk_encoded[chunk == class_char] = idx
            
            encoded[i:end] = chunk_encoded
        
        return encoded

    @staticmethod
    def encode_onehot_chunked(dssp_data: np.ndarray, classes: list, chunk_size: int, 
                             cache_path: str) -> np.ndarray:
        """
        Encode DSSP to one-hot format with intelligent processing.
        
        Parameters
        ----------
        dssp_data : numpy.ndarray
            Input DSSP data with shape (n_frames, n_residues)
        classes : list
            List of class labels for one-hot encoding
        chunk_size : int
            Number of frames to process per chunk
        cache_path : str
            Directory path for cache files
        use_memmap : bool, default=True
            Whether to use memory mapping for output array
            
        Returns
        -------
        numpy.ndarray
            One-hot encoded DSSP data with shape (n_frames, n_residues * n_classes)
            
        Notes
        -----
        Uses chunk-wise processing for large datasets or when use_memmap=True.
        For small datasets with use_memmap=False, uses direct vectorized operations.
        """
        n_frames, n_residues = dssp_data.shape
        n_classes = len(classes)
        shape = (n_frames, n_residues * n_classes)
        
        # Use memmap and chunk-wise processing for large datasets
        cache_file = DataUtils.get_cache_file_path(
            f'dssp_onehot_{id(dssp_data)}.dat', cache_path
        )
        encoded = np.memmap(
            cache_file, dtype=np.float32, mode='w+', shape=shape
        )
        
        # Process in chunks
        for i in range(0, n_frames, chunk_size):
            end = min(i + chunk_size, n_frames)
            for class_idx, class_char in enumerate(classes):
                mask = (dssp_data[i:end] == class_char)
                encoded[i:end][:, class_idx::n_classes] = mask

        return encoded

    @staticmethod
    def encode_onehot_direct(dssp_data: np.ndarray, classes: list) -> np.ndarray:
        """
        Encode dssp_data from mdtraj output to one-hot format directly.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            Input DSSP data with shape (n_frames, n_residues)
        classes : list
            List of class labels
            
        Returns
        -------
        numpy.ndarray
            One-hot encoded DSSP data with shape (n_frames, n_residues * n_classes)
        """
        n_frames, n_residues = dssp_data.shape
        n_classes = len(classes)
        shape = (n_frames, n_residues * n_classes)
        encoded = np.zeros(shape, dtype=np.float32)
        
        for class_idx, class_char in enumerate(classes):
            mask = (dssp_data == class_char)
            encoded[:, class_idx::n_classes] = mask

        return encoded
            