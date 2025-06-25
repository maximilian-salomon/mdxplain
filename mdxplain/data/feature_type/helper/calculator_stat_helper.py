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
Statistical calculations for molecular dynamics feature data.

Provides statistical calculations for feature data with support for memory-mapped
arrays and chunked processing. All methods are static and can be used without 
instantiation across different calculators.
"""

import numpy as np


class CalculatorStatHelper:
    """
    Static utility class for statistical calculations on molecular dynamics feature data.

    Provides efficient statistical computations (mean, std, transitions, etc.) with
    support for memory-mapped arrays and chunked processing for large datasets.
    All methods are static for easy use across different calculators.
    """

    # ===== BASIC STATISTICAL METHODS =====

    @staticmethod
    def compute_differences(
        array1, array2, chunk_size=None, preprocessing_func=None, **func_kwargs
    ):
        """
        Compute differences between two feature arrays with optional preprocessing.

        Parameters:
        -----------
        array1 : np.ndarray
            First feature array
        array2 : np.ndarray
            Second feature array
        chunk_size : int, optional
            Chunk size for memory-mapped processing (over pairs, not frames)
        preprocessing_func : callable, optional
            Function to apply before computing differences (default: mean per pair)
        **func_kwargs : dict
            Additional arguments for preprocessing function

        Returns:
        --------
        np.ndarray
            Element-wise differences between preprocessed arrays
        """
        if preprocessing_func is None:

            def preprocessing_func(arr, **kw):
                """Apply default preprocessing using mean per pair."""
                return CalculatorStatHelper.compute_func_per_pair(arr, np.mean, **kw)

        # Apply preprocessing function to both arrays
        processed1 = preprocessing_func(
            array1, chunk_size=chunk_size, **func_kwargs)
        processed2 = preprocessing_func(
            array2, chunk_size=chunk_size, **func_kwargs)
        return processed1 - processed2

    @staticmethod
    def compute_func_per_pair(array, func, chunk_size=None, **func_kwargs):
        """
        Apply statistical function per feature pair across all frames.

        Parameters:
        -----------
        array : np.ndarray
            Feature array (NxMxM square or NxP condensed format)
        func : callable
            NumPy function to apply (np.mean, np.std, np.min, np.max, etc.)
        chunk_size : int, optional
            Number of pairs to process per chunk
        **func_kwargs : dict
            Additional arguments for the function

        Returns:
        --------
        np.ndarray
            Statistical values per pair (preserves spatial dimensions)
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
            chunk_result = func(
                flat_array[:, i:end_idx], axis=0, **func_kwargs)
            result_chunks.append(chunk_result)

        # Concatenate results and reshape back to original spatial dimensions
        result = np.concatenate(result_chunks)
        return result.reshape(spatial_shape)

    @staticmethod
    def compute_func_per_frame(array, chunk_size=None, func=None):
        """
        Apply statistical function per frame across all pairs.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to process
        chunk_size : int, optional
            Number of frames to process per chunk
        func : callable, optional
            Function to apply (default: np.mean)

        Returns:
        --------
        np.ndarray
            Statistical values per frame
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
                chunk_result = func(
                    array[i:end_idx].reshape(end_idx - i, -1), axis=1)
            else:
                chunk_result = func(array[i:end_idx], axis=1)
            result_chunks.append(chunk_result)

        return np.concatenate(result_chunks)

    @staticmethod
    def compute_func_per_residue(array, func, chunk_size=None, **func_kwargs):
        """
        Apply statistical function per residue (square format only).

        Parameters:
        -----------
        array : np.ndarray
            Square format array (NxMxM)
        func : callable
            NumPy function to apply
        chunk_size : int, optional
            Chunk size for processing
        **func_kwargs : dict
            Additional arguments for the function

        Returns:
        --------
        np.ndarray
            Statistical values per residue
        """
        if len(array.shape) != 3:
            raise ValueError(
                "compute_func_per_residue requires square format arrays (NxMxM)"
            )

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
    def compute_transitions_within_lagtime(
        array, threshold=1.0, lag_time=1, chunk_size=None
    ):
        """
        Count transitions using lag time analysis.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to analyze
        threshold : float, default=1.0
            Threshold for detecting transitions
        lag_time : int, default=1
            Number of frames to look ahead
        chunk_size : int, optional
            Chunk size for processing

        Returns:
        --------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(
            array, threshold, lag_time, chunk_size, mode="lagtime"
        )

    @staticmethod
    def compute_transitions_within_window(
        array, threshold=1.0, window_size=10, chunk_size=None
    ):
        """
        Count transitions using sliding window analysis.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to analyze
        threshold : float, default=1.0
            Threshold for detecting transitions
        window_size : int, default=10
            Size of sliding window
        chunk_size : int, optional
            Chunk size for processing

        Returns:
        --------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(
            array, threshold, window_size, chunk_size, mode="window"
        )

    @staticmethod
    def _compute_transitions_unified(
        array, threshold, window_size, chunk_size, mode="lagtime"
    ):
        """
        Compute transitions using unified internal method.

        Parameters:
        -----------
        array : np.ndarray
            Feature array
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        chunk_size : int or None
            Chunk size for processing
        mode : str
            Computation mode ('lagtime' or 'window')

        Returns:
        --------
        numpyp.ndarray
            Transition counts per pair
        """
        if len(array.shape) == 3:
            output_shape = (array.shape[1], array.shape[2])
            flat_array = array.reshape(array.shape[0], -1)
        else:
            output_shape = (array.shape[1],)
            flat_array = array

        result = np.zeros(output_shape)

        if chunk_size is not None:
            CalculatorStatHelper._compute_transitions_chunks(
                flat_array, threshold, window_size, chunk_size, mode, result
            )
        else:
            CalculatorStatHelper._compute_transitions_direct(
                flat_array, threshold, window_size, mode, result
            )

        return result

    @staticmethod
    def _compute_transitions_chunks(
        array, threshold, window_size, chunk_size, mode, result
    ):
        """
        Compute transitions with chunked processing.

        Parameters:
        -----------
        array : np.ndarray
            Flattened feature array
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        chunk_size : int
            Number of pairs per chunk
        mode : str
            Computation mode
        result : np.ndarray
            Output array to fill

        Returns:
        --------
        None
            Modifies result array in-place
        """
        flat_result = result.flatten()
        for i in range(0, array.shape[1], chunk_size):
            end_idx = min(i + chunk_size, array.shape[1])
            chunk = array[:, i: end_idx]

            for j in range(chunk.shape[1]):
                if mode == "lagtime":
                    diff = np.abs(chunk[:-window_size, j] -
                                  chunk[window_size:, j])
                    flat_result[i + j] = np.sum(diff >= threshold)
                else:  # window mode - corrected implementation
                    transitions = 0
                    for k in range(len(chunk) - window_size + 1):
                        window_data = chunk[k: k + window_size, j]
                        # Check min/max difference within window
                        window_min = np.min(window_data)
                        window_max = np.max(window_data)
                        if (window_max - window_min) >= threshold:
                            transitions += 1
                    flat_result[i + j] = transitions

    @staticmethod
    def _compute_transitions_direct(array, threshold, window_size, mode, result):
        """
        Compute transitions without chunking.

        Parameters:
        -----------
        array : np.ndarray
            Flattened feature array
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        mode : str
            Computation mode
        result : np.ndarray
            Output array to fill

        Returns:
        --------
        None
            Modifies result array in-place
        """
        flat_result = result.flatten()
        for j in range(array.shape[1]):
            if mode == "lagtime":
                diff = np.abs(array[:-window_size, j] - array[window_size:, j])
                flat_result[j] = np.sum(diff >= threshold)
            else:  # window mode - corrected implementation
                transitions = 0
                for k in range(len(array) - window_size + 1):
                    window_data = array[k: k + window_size, j]
                    # Check min/max difference within window
                    window_min = np.min(window_data)
                    window_max = np.max(window_data)
                    if (window_max - window_min) >= threshold:
                        transitions += 1
                flat_result[j] = transitions

    @staticmethod
    def compute_stability(
        array, threshold=2.0, window_size=1, chunk_size=None, mode="lagtime"
    ):
        """
        Calculate stability (inverse of transition rate) per pair.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to analyze
        threshold : float, default=2.0
            Threshold for stability detection
        window_size : int, default=1
            Window size for calculation
        chunk_size : int, optional
            Chunk size for processing
        mode : str, default='lagtime'
            Calculation mode ('lagtime' or 'window')

        Returns:
        --------
        numpy.ndarray
            Stability values per pair (0=unstable, 1=stable)
        """
        transitions = CalculatorStatHelper._compute_transitions_unified(
            array, threshold, window_size, chunk_size, mode
        )
        max_possible_transitions = (
            array.shape[0] - window_size
            if mode == "lagtime"
            else array.shape[0] - window_size + 1
        )
        return 1.0 - (transitions / max_possible_transitions)
