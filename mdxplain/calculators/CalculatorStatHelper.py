# MDCatAFlow - A Molecular Dynamics Catalysis Analysis Workflow Tool
# CalculatorStatHelper - Statistical calculations for feature data
#
# Helper class providing statistical calculations for feature data.
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


class CalculatorStatHelper:
    """
    Helper class providing statistical calculations for feature data.
    All methods are static and can be used without instantiation.
    """
    
    # ===== BASIC STATISTICAL METHODS =====

    @staticmethod
    def compute_differences(array1, array2, chunk_size=None, preprocessing_func=None, **func_kwargs):
        """
        Compute differences between two arrays, optionally with preprocessing.
        Default is mean as preprocessing function.
        
        Parameters:
        -----------
        array1 : numpy.ndarray
            First array
        array2 : numpy.ndarray
            Second array
        chunk_size : int, default=None
            Chunk size for memmap processing
            Goes over pairs (spatial dimension), not over frames (always all frames)
        preprocessing_func : function, default=None
            Function to apply to both arrays before computing differences
            (e.g., compute_contact_frequency for contact data)
        **func_kwargs : dict
            Additional keyword arguments for preprocessing_func
            
        Returns:
        --------
        numpy.ndarray
            Differences between processed arrays
        """
        if preprocessing_func is None:
            preprocessing_func = lambda arr, **kw: CalculatorStatHelper.compute_func_per_pair(arr, np.mean, **kw)

        # Apply preprocessing function to both arrays
        processed1 = preprocessing_func(array1, chunk_size=chunk_size, **func_kwargs)
        processed2 = preprocessing_func(array2, chunk_size=chunk_size, **func_kwargs)
        return processed1 - processed2
    
    @staticmethod
    def compute_func_per_pair(array, func, chunk_size=None, **func_kwargs):
        """
        Universal method to compute any function per pair across all frames.
        Chunks over pairs (spatial dimension), not over frames.
        
        Parameters:
        -----------
        array : numpy.ndarray
            Input array (NxMxM for square form or NxP for condensed form)
        func : function
            Numpy function to apply (np.mean, np.std, np.min, np.max, etc.)
        chunk_size : int, default=None
            Chunk size for pair processing (number of pairs per chunk)
        **func_kwargs : dict
            Additional keyword arguments for the function
            
        Returns:
        --------
        numpy.ndarray
            Function result per pair
        """
        if chunk_size is None:
            # No chunking - process all pairs at once
            return func(array, axis=0, **func_kwargs)
        
        # Determine spatial dimensions
        if len(array.shape) == 3:
            # Square format NxMxM
            n_pairs = array.shape[1] * array.shape[2]
            spatial_shape = (array.shape[1], array.shape[2])
            # Flatten spatial dimensions for chunking
            flat_array = array.reshape(array.shape[0], -1)
        else:
            # Condensed format NxP
            n_pairs = array.shape[1]
            spatial_shape = (array.shape[1],)
            flat_array = array
        
        # Process in chunks
        result_chunks = []
        for i in range(0, n_pairs, chunk_size):
            end_idx = min(i + chunk_size, n_pairs)
            chunk_result = func(flat_array[:, i:end_idx], axis=0, **func_kwargs)
            result_chunks.append(chunk_result)
        
        # Concatenate results and reshape back to original spatial dimensions
        result = np.concatenate(result_chunks)
        return result.reshape(spatial_shape)
        
    @staticmethod
    def compute_func_per_frame(array, chunk_size=None, func=None):
        """
        Compute values per frame (sum/mean over spatial dimensions).
        
        Parameters:
        -----------
        array : numpy.ndarray
            Input array
        chunk_size : int, default=None
            Chunk size for memmap processing
            Goes over frames, not over pairs (Always all pairs)
        func : function, default=None
            Function to apply (np.mean, np.sum, etc.). Defaults to np.mean
            
        Returns:
        --------
        numpy.ndarray
            Values per frame
        """
        if func is None:
            func = np.mean
            
        if chunk_size is None:
            # No chunking - process all frames at once
            if len(array.shape) == 3:
                return func(array.reshape(array.shape[0], -1), axis=1)
            else:
                return func(array, axis=1)
        
        # Process in chunks
        result_chunks = []
        for i in range(0, array.shape[0], chunk_size):
            end_idx = min(i + chunk_size, array.shape[0])
            if len(array.shape) == 3:
                chunk_result = func(array[i:end_idx].reshape(end_idx - i, -1), axis=1)
            else:
                chunk_result = func(array[i:end_idx], axis=1)
            result_chunks.append(chunk_result)
        
        return np.concatenate(result_chunks)

    @staticmethod
    def compute_func_per_residue(array, func, chunk_size=None, **func_kwargs):
        """
        Compute values per residue (only for square format arrays).
        
        Parameters:
        -----------
        array : numpy.ndarray
            Input array (must be NxMxM format)
        func : function
            Numpy function to apply
        chunk_size : int, default=None
            Chunk size for memmap processing
        **func_kwargs : dict
            Additional keyword arguments for the function
            
        Returns:
        --------
        numpy.ndarray
            Values per residue
        """
        if len(array.shape) != 3:
            raise ValueError("compute_func_per_residue requires square format arrays (NxMxM)")
        
        if chunk_size is None:
            return func(array, axis=(0, 2), **func_kwargs)
        
        result_chunks = []
        for i in range(0, array.shape[0], chunk_size):
            end_idx = min(i + chunk_size, array.shape[0])
            chunk_result = func(array[i:end_idx], axis=(0, 2), **func_kwargs)
            if len(chunk_result.shape) == 1:
                result_chunks.append(chunk_result)
            else:
                result_chunks.append(chunk_result)
        
        if len(result_chunks[0].shape) == 1:
            return np.mean(result_chunks, axis=0)
        else:
            return np.concatenate(result_chunks, axis=0)

    @staticmethod
    def compute_transitions_within_lagtime(array, threshold=1.0, lag_time=1, chunk_size=None):
        """
        Compute transitions within a lag time for each pair.
        
        Parameters:
        -----------
        array : numpy.ndarray
            Input array
        threshold : float, default=1.0
            Threshold for transition detection
        lag_time : int, default=1
            Number of frames to look ahead
        chunk_size : int, default=None
            Chunk size for processing
            
        Returns:
        --------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(array, threshold, lag_time, chunk_size, mode='lagtime')

    @staticmethod
    def compute_transitions_within_window(array, threshold=1.0, window_size=10, chunk_size=None):
        """
        Compute transitions within a sliding window for each pair.
        
        Parameters:
        -----------
        array : numpy.ndarray
            Input array
        threshold : float, default=1.0
            Threshold for transition detection
        window_size : int, default=10
            Size of the sliding window
        chunk_size : int, default=None
            Chunk size for processing
            
        Returns:
        --------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(array, threshold, window_size, chunk_size, mode='window')

    @staticmethod
    def _compute_transitions_unified(array, threshold, window_size, chunk_size, mode='lagtime'):
        """Unified method for computing transitions."""
        if len(array.shape) == 3:
            output_shape = (array.shape[1], array.shape[2])
            flat_array = array.reshape(array.shape[0], -1)
        else:
            output_shape = (array.shape[1],)
            flat_array = array
        
        result = np.zeros(output_shape)
        
        if chunk_size is not None:
            CalculatorStatHelper._compute_transitions_chunks(flat_array, threshold, window_size, chunk_size, mode, result)
        else:
            CalculatorStatHelper._compute_transitions_direct(flat_array, threshold, window_size, mode, result)
        
        return result

    @staticmethod
    def _compute_transitions_chunks(array, threshold, window_size, chunk_size, mode, result):
        """Compute transitions using chunking."""
        flat_result = result.flatten()
        for i in range(0, array.shape[1], chunk_size):
            end_idx = min(i + chunk_size, array.shape[1])
            chunk = array[:, i:end_idx]
            
            for j in range(chunk.shape[1]):
                if mode == 'lagtime':
                    diff = np.abs(chunk[:-window_size, j] - chunk[window_size:, j])
                    flat_result[i + j] = np.sum(diff >= threshold)
                else:  # window mode - corrected implementation
                    transitions = 0
                    for k in range(len(chunk) - window_size + 1):
                        window_data = chunk[k:k+window_size, j]
                        # Check min/max difference within window
                        window_min = np.min(window_data)
                        window_max = np.max(window_data)
                        if (window_max - window_min) >= threshold:
                            transitions += 1
                    flat_result[i + j] = transitions              

    @staticmethod
    def _compute_transitions_direct(array, threshold, window_size, mode, result):
        """Compute transitions directly without chunking."""
        flat_result = result.flatten()
        for j in range(array.shape[1]):
            if mode == 'lagtime':
                diff = np.abs(array[:-window_size, j] - array[window_size:, j])
                flat_result[j] = np.sum(diff >= threshold)
            else:  # window mode - corrected implementation
                transitions = 0
                for k in range(len(array) - window_size + 1):
                    window_data = array[k:k+window_size, j]
                    # Check min/max difference within window
                    window_min = np.min(window_data)
                    window_max = np.max(window_data)
                    if (window_max - window_min) >= threshold:
                        transitions += 1
                flat_result[j] = transitions

    @staticmethod
    def compute_stability(array, threshold=2.0, window_size=1, chunk_size=None, mode='lagtime'):
        """
        Compute stability (inverse of transitions) for each pair.
        
        Parameters:
        -----------
        array : numpy.ndarray
            Input array
        threshold : float, default=2.0
            Threshold for stability detection
        window_size : int, default=1
            Window size for stability calculation
        chunk_size : int, default=None
            Chunk size for processing
        mode : str, default='lagtime'
            Mode for stability calculation ('lagtime' or 'window')
            
        Returns:
        --------
        numpy.ndarray
            Stability values per pair
        """
        transitions = CalculatorStatHelper._compute_transitions_unified(array, threshold, window_size, chunk_size, mode)
        max_possible_transitions = array.shape[0] - window_size if mode == 'lagtime' else array.shape[0] - window_size + 1
        return 1.0 - (transitions / max_possible_transitions)
