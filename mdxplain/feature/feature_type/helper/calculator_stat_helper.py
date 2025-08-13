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

from .feature_shape_helper import FeatureShapeHelper


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
        array1,
        array2,
        chunk_size=10000,
        use_memmap=False,
        preprocessing_func=None,
        **func_kwargs
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
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        preprocessing_func : callable, optional
            Function to apply before computing differences (default: mean per pair)
        **func_kwargs : dict
            Additional arguments for preprocessing function

        Returns:
        --------
        np.ndarray
            Element-wise differences between preprocessed arrays

        Examples:
        ---------
        >>> array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> CalculatorStatHelper.compute_differences(array1, array2)
        array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        """
        if preprocessing_func is None:

            def preprocessing_func(arr, **kw):
                """Apply default preprocessing using mean per feature."""
                return CalculatorStatHelper.compute_func_per_feature(
                    arr, np.mean, use_memmap=use_memmap, **kw
                )

        # Apply preprocessing function to both arrays
        processed1 = preprocessing_func(array1, chunk_size=chunk_size, **func_kwargs)
        processed2 = preprocessing_func(array2, chunk_size=chunk_size, **func_kwargs)
        return processed1 - processed2

    @staticmethod
    def compute_func_per_feature(
        array, func, chunk_size=10000, use_memmap=False, **func_kwargs
    ):
        """
        Apply statistical function per feature across all frames (2D format).

        Parameters:
        -----------
        array : np.ndarray
            Feature array (NxMxM square or NxP condensed format)
        func : callable
            NumPy function to apply (np.mean, np.std, np.min, np.max, etc.)
        chunk_size : int, optional
            Number of features to process per chunk
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        **func_kwargs : dict
            Additional arguments for the function

        Returns:
        --------
        np.ndarray
            Statistical values per feature (preserves spatial dimensions)

        Examples:
        ---------
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> CalculatorStatHelper.compute_func_per_feature(array, np.mean)
        array([3.0, 5.0, 7.0])
        """
        # Intelligent chunking decision: use chunking if use_memmap=True OR input data is memmap
        should_use_chunking = use_memmap or FeatureShapeHelper.is_memmap(array)

        if not should_use_chunking:
            # No chunking - process all features at once
            return func(array, axis=0, **func_kwargs)

        # Determine spatial dimensions
        if len(array.shape) == 3:
            # Square format NxMxM
            n_features = array.shape[1] * array.shape[2]
            spatial_shape = (array.shape[1], array.shape[2])
            # Flatten spatial dimensions for chunking
            flat_array = array.reshape(array.shape[0], -1)
        else:
            # Condensed format NxP
            n_features = array.shape[1]
            spatial_shape = (array.shape[1],)
            flat_array = array

        # Process in chunks
        result_chunks = []
        for i in range(0, n_features, chunk_size):
            end_idx = min(i + chunk_size, n_features)
            chunk_result = func(flat_array[:, i:end_idx], axis=0, **func_kwargs)
            result_chunks.append(chunk_result)

        # Concatenate results and reshape back to original spatial dimensions
        result = np.concatenate(result_chunks)
        return result.reshape(spatial_shape)

    @staticmethod
    def compute_func_per_frame(array, chunk_size=10000, use_memmap=False, func=None):
        """
        Apply statistical function per frame across all pairs.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to process
        chunk_size : int, optional
            Number of frames to process per chunk
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        func : callable, optional
            Function to apply (default: np.mean)

        Returns:
        --------
        np.ndarray
            Statistical values per frame

        Examples:
        ---------
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> CalculatorStatHelper.compute_func_per_frame(array, np.mean)
        """
        if func is None:
            func = np.mean

        # Intelligent chunking decision: use chunking if use_memmap=True OR input data is memmap
        should_use_chunking = use_memmap or FeatureShapeHelper.is_memmap(array)

        if not should_use_chunking:
            return CalculatorStatHelper._compute_frames_direct(array, func)
        else:
            return CalculatorStatHelper._compute_frames_chunked(array, func, chunk_size)

    @staticmethod
    def _compute_frames_direct(array, func):
        """
        Compute function per frame without chunking.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to process
        func : callable, optional
            Function to apply (default: np.mean)

        Returns:
        --------
        np.ndarray
            Statistical values per frame
        """
        if len(array.shape) == 3:
            return func(array.reshape(array.shape[0], -1), axis=1)
        else:
            return func(array, axis=1)

    @staticmethod
    def _compute_frames_chunked(array, func, chunk_size):
        """
        Compute function per frame with chunking.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to process
        func : callable, optional
            Function to apply (default: np.mean)
        chunk_size : int, optional
            Number of frames to process per chunk

        Returns:
        --------
        np.ndarray
            Statistical values per frame
        """
        result_chunks = []
        for i in range(0, array.shape[0], chunk_size):
            end_idx = min(i + chunk_size, array.shape[0])
            chunk_result = CalculatorStatHelper._process_frame_chunk(
                array, func, i, end_idx
            )
            result_chunks.append(chunk_result)
        return np.concatenate(result_chunks)

    @staticmethod
    def _process_frame_chunk(array, func, start_idx, end_idx):
        """
        Process a single frame chunk.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to process
        func : callable, optional
            Function to apply (default: np.mean)
        start_idx : int
            Start index of the chunk
        end_idx : int
            End index of the chunk

        Returns:
        --------
        np.ndarray
            Statistical values per frame
        """
        if len(array.shape) == 3:
            return func(
                array[start_idx:end_idx].reshape(end_idx - start_idx, -1), axis=1
            )
        else:
            return func(array[start_idx:end_idx], axis=1)

    @staticmethod
    def _convert_2d_to_3d(array, chunk_size=10000):
        """
        Convert 2D condensed array to 3D squareform array.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to process
        chunk_size : int, optional
            Chunk size for processing

        Returns:
        --------
        np.ndarray
            Squareform array (NxMxM)
        """
        n_pairs = array.shape[1]
        n_residues = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)

        pairs = []
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                pairs.append([i, j])

        return FeatureShapeHelper.condensed_to_squareform(
            array, pairs, n_residues, chunk_size=chunk_size
        )

    @staticmethod
    def compute_func_per_column(
        array, func, chunk_size=10000, use_memmap=False, **func_kwargs
    ):
        """
        Apply statistical function per column (3D format required, auto-converts 2D).

        Parameters:
        -----------
        array : np.ndarray
            Feature array - if 2D (condensed), automatically converts to 3D (squareform)
        func : callable
            NumPy function to apply
        chunk_size : int, optional
            Chunk size for processing
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        **func_kwargs : dict
            Additional arguments for the function

        Returns:
        --------
        np.ndarray
            Statistical values per column
        """
        # Convert 2D to 3D if needed
        if len(array.shape) == 2:
            array = CalculatorStatHelper._convert_2d_to_3d(array, chunk_size)

        if len(array.shape) != 3:
            raise ValueError(
                "compute_func_per_column requires 3D arrays or convertible 2D arrays"
            )

        # Intelligent chunking decision: use chunking if use_memmap=True OR input data is memmap
        should_use_chunking = use_memmap or FeatureShapeHelper.is_memmap(array)

        if not should_use_chunking:
            return func(array, axis=(0, 2), **func_kwargs)

        return CalculatorStatHelper._compute_columns_chunked(
            array, func, chunk_size, **func_kwargs
        )

    @staticmethod
    def _compute_columns_chunked(array, func, chunk_size, **func_kwargs):
        """
        Compute function per column with chunking.

        Parameters:
        -----------
        array : np.ndarray
            Feature array to process
        func : callable, optional
            Function to apply (default: np.mean)
        chunk_size : int, optional
            Number of columns to process per chunk
        **func_kwargs : dict
            Additional arguments for the function

        Returns:
        --------
        np.ndarray
            Statistical values per column
        """
        result_chunks = []
        for i in range(0, array.shape[0], chunk_size):
            end_idx = min(i + chunk_size, array.shape[0])
            chunk_result = func(array[i:end_idx], axis=(0, 2), **func_kwargs)
            result_chunks.append(chunk_result)

        return CalculatorStatHelper._combine_chunk_results(result_chunks)

    @staticmethod
    def _combine_chunk_results(result_chunks):
        """
        Combine results from chunked processing.

        Parameters:
        -----------
        result_chunks : list
            List of chunk results

        Returns:
        --------
        np.ndarray
            Combined results
        """
        if len(result_chunks[0].shape) == 1:
            return np.mean(result_chunks, axis=0)
        else:
            return np.concatenate(result_chunks, axis=0)

    @staticmethod
    def compute_transitions_within_lagtime(
        array, threshold=1.0, lag_time=1, chunk_size=10000, use_memmap=False
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
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)

        Returns:
        --------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(
            array, threshold, lag_time, chunk_size, use_memmap, mode="lagtime"
        )

    @staticmethod
    def compute_transitions_within_window(
        array, threshold=1.0, window_size=10, chunk_size=10000, use_memmap=False
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
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)

        Returns:
        --------
        numpy.ndarray
            Transition counts per pair
        """
        return CalculatorStatHelper._compute_transitions_unified(
            array, threshold, window_size, chunk_size, use_memmap, mode="window"
        )

    @staticmethod
    def _compute_transitions_unified(
        array, threshold, window_size, chunk_size, use_memmap=False, mode="lagtime"
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
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
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

        # Intelligent chunking decision: use chunking if use_memmap=True OR input data is memmap
        should_use_chunking = use_memmap or FeatureShapeHelper.is_memmap(array)

        if should_use_chunking:
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
            chunk = array[:, i:end_idx]
            CalculatorStatHelper._process_chunk_transitions(
                chunk, threshold, window_size, mode, flat_result, i
            )

    @staticmethod
    def _process_chunk_transitions(
        chunk, threshold, window_size, mode, flat_result, start_idx
    ):
        """
        Process transitions for a single chunk.

        Parameters:
        -----------
        chunk : np.ndarray
            Chunk of feature array
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size
        mode : str
            Computation mode
        flat_result : np.ndarray
            Flattened result array
        start_idx : int
            Start index of the chunk

        Returns:
        --------
        None
            Modifies flat_result array in-place
        """
        for j in range(chunk.shape[1]):
            if mode == "lagtime":
                flat_result[start_idx + j] = (
                    CalculatorStatHelper._compute_lagtime_transitions(
                        chunk[:, j], threshold, window_size
                    )
                )
            else:
                flat_result[start_idx + j] = (
                    CalculatorStatHelper._compute_window_transitions(
                        chunk[:, j], threshold, window_size
                    )
                )

    @staticmethod
    def _compute_lagtime_transitions(data_column, threshold, window_size):
        """
        Compute lagtime transitions for a single data column.

        Parameters:
        -----------
        data_column : np.ndarray
            Data column to process
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size

        Returns:
        --------
        int
            Number of transitions
        """
        diff = np.abs(data_column[:-window_size] - data_column[window_size:])
        return np.sum(diff >= threshold)

    @staticmethod
    def _compute_window_transitions(data_column, threshold, window_size):
        """
        Compute window transitions for a single data column.

        Parameters:
        -----------
        data_column : np.ndarray
            Data column to process
        threshold : float
            Transition threshold
        window_size : int
            Window or lag size

        Returns:
        --------
        int
            Number of transitions
        """
        transitions = 0
        for k in range(len(data_column) - window_size + 1):
            window_data = data_column[k : k + window_size]
            window_min = np.min(window_data)
            window_max = np.max(window_data)
            if (window_max - window_min) >= threshold:
                transitions += 1
        return transitions

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
                    window_data = array[k : k + window_size, j]
                    # Check min/max difference within window
                    window_min = np.min(window_data)
                    window_max = np.max(window_data)
                    if (window_max - window_min) >= threshold:
                        transitions += 1
                flat_result[j] = transitions

    @staticmethod
    def compute_stability(
        array,
        threshold=2.0,
        window_size=1,
        chunk_size=10000,
        use_memmap=False,
        mode="lagtime",
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
        use_memmap : bool, default=False
            Whether output is for memory-mapped arrays (enables intelligent chunking)
        mode : str, default='lagtime'
            Calculation mode ('lagtime' or 'window')

        Returns:
        --------
        numpy.ndarray
            Stability values per pair (0=unstable, 1=stable)
        """
        transitions = CalculatorStatHelper._compute_transitions_unified(
            array, threshold, window_size, chunk_size, use_memmap, mode
        )
        max_possible_transitions = (
            array.shape[0] - window_size
            if mode == "lagtime"
            else array.shape[0] - window_size + 1
        )
        return 1.0 - (transitions / max_possible_transitions)
