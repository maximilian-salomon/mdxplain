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
Statistical analysis methods for distance calculations.

Provides comprehensive statistical analysis capabilities for distance data
including variability analysis, transition detection, and comparative studies
with support for memory-mapped arrays.
"""

import numpy as np

from ..helper.calculator_stat_helper import CalculatorStatHelper


class DistanceCalculatorAnalysis:
    """
    Analysis methods for distance calculation statistics and metrics.

    Provides statistical analysis capabilities for distance data including
    variability analysis, transition detection, and comparative studies
    with memory-mapped array support.
    """

    def __init__(self, chunk_size=None):
        """
        Initialize distance analysis with chunking configuration.

        Parameters:
        -----------
        chunk_size : int, optional
            Number of frames to process per chunk for memory-mapped arrays

        Examples:
        ---------
        >>> # Default chunking
        >>> analysis = DistanceCalculatorAnalysis()

        >>> # Custom chunk size
        >>> analysis = DistanceCalculatorAnalysis(chunk_size=1000)
        """
        self.chunk_size = chunk_size

    # === PAIR-BASED STATISTICS ===
    def compute_mean(self, distances):
        """
        Compute mean distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Mean distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, np.mean, self.chunk_size
        )

    def compute_std(self, distances):
        """
        Compute standard deviation of distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Standard deviation for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, np.std, self.chunk_size
        )

    def compute_min(self, distances):
        """
        Compute minimum distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Minimum distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, np.min, self.chunk_size
        )

    def compute_max(self, distances):
        """
        Compute maximum distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Maximum distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, np.max, self.chunk_size
        )

    def compute_median(self, distances):
        """
        Compute median distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Median distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, np.median, self.chunk_size
        )

    def compute_variance(self, distances):
        """
        Compute variance of distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Variance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, np.var, self.chunk_size
        )

    def compute_range(self, distances):
        """
        Compute range (peak-to-peak) of distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Range (max - min) for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, np.ptp, self.chunk_size
        )

    def compute_q25(self, distances):
        """
        Compute 25th percentile of distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            25th percentile for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, lambda x, axis: np.percentile(
                x, 25, axis=axis), self.chunk_size
        )

    def compute_q75(self, distances):
        """
        Compute 75th percentile of distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            75th percentile for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances, lambda x, axis: np.percentile(
                x, 75, axis=axis), self.chunk_size
        )

    def compute_iqr(self, distances):
        """
        Compute interquartile range of distances per pair.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Interquartile range (Q75 - Q25) for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances,
            lambda x, axis: np.percentile(x, 75, axis=axis)
            - np.percentile(x, 25, axis=axis),
            self.chunk_size,
        )

    def compute_mad(self, distances):
        """
        Compute median absolute deviation for each distance pair.

        MAD provides robust measure of variability less sensitive to outliers
        than standard deviation. Calculated as median of absolute deviations
        from the median.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            MAD values per distance pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_pair(
            distances,
            lambda x, axis: np.median(
                np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis
            ),
            self.chunk_size,
        )

    # === FRAME-BASED STATISTICS ===
    def distances_per_frame_mean(self, distances):
        """
        Compute mean distances per frame.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Mean distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, np.mean
        )

    def distances_per_frame_std(self, distances):
        """
        Compute standard deviation of distances per frame.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Standard deviation across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, np.std
        )

    def distances_per_frame_min(self, distances):
        """
        Compute minimum distances per frame.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Minimum distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, np.min
        )

    def distances_per_frame_max(self, distances):
        """
        Compute maximum distances per frame.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Maximum distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, np.max
        )

    def distances_per_frame_median(self, distances):
        """
        Compute median distances per frame.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Median distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, np.median
        )

    def distances_per_frame_range(self, distances):
        """
        Compute range of distances per frame.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Range (max - min) across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, np.ptp
        )

    def distances_per_frame_sum(self, distances):
        """
        Compute sum of distances per frame.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns:
        --------
        np.ndarray
            Sum of distances across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, np.sum
        )

    # === RESIDUE-BASED STATISTICS (only for square format) ===
    def compute_per_residue_mean(self, distances):
        """
        Compute mean distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Mean distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.mean, self.chunk_size
        )

    def compute_per_residue_std(self, distances):
        """
        Compute standard deviation of distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Standard deviation for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.std, self.chunk_size
        )

    def compute_per_residue_min(self, distances):
        """
        Compute minimum distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Minimum distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.min, self.chunk_size
        )

    def compute_per_residue_max(self, distances):
        """
        Compute maximum distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Maximum distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.max, self.chunk_size
        )

    def compute_per_residue_median(self, distances):
        """
        Compute median distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Median distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.median, self.chunk_size
        )

    def compute_per_residue_sum(self, distances):
        """
        Compute sum of distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Sum of distances for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.sum, self.chunk_size
        )

    def compute_per_residue_variance(self, distances):
        """
        Compute variance of distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Variance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.var, self.chunk_size
        )

    def compute_per_residue_range(self, distances):
        """
        Compute range of distances per residue. Requires squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in square format with shape (n_frames, n_residues, n_residues)

        Returns:
        --------
        np.ndarray
            Range (max - min) for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_residue(
            distances, np.ptp, self.chunk_size
        )

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, distances, threshold=2.0, lag_time=10):
        """
        Compute transitions within lag time.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)
        threshold : float, default=2.0
            Distance threshold for transition detection in Angstroms
        lag_time : int, default=10
            Number of frames to look ahead for transitions

        Returns:
        --------
        np.ndarray
            Transition counts for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            distances, threshold, lag_time, self.chunk_size
        )

    def compute_transitions_window(self, distances, threshold=2.0, window_size=10):
        """
        Compute transitions within window.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)
        threshold : float, default=2.0
            Distance threshold for transition detection in Angstroms
        window_size : int, default=10
            Size of sliding window for transition analysis

        Returns:
        --------
        np.ndarray
            Transition counts for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_transitions_within_window(
            distances, threshold, window_size, self.chunk_size
        )

    def compute_stability(
        self, distances, threshold=2.0, window_size=10, mode="window"
    ):
        """
        Compute stability analysis.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)
        threshold : float, default=2.0
            Distance threshold for stability detection in Angstroms
        window_size : int, default=10
            Size of analysis window
        mode : str, default="window"
            Analysis mode: "window" or "lagtime"

        Returns:
        --------
        np.ndarray
            Stability scores for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_stability(
            distances, threshold, window_size, self.chunk_size, mode
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, distances1, distances2, preprocessing_func=None):
        """
        Compute differences between two distance datasets.

        Parameters:
        -----------
        distances1 : np.ndarray or np.memmap
            First distance array with shape (n_frames, n_pairs)
        distances2 : np.ndarray or np.memmap
            Second distance array with shape (n_frames, n_pairs)
        preprocessing_func : callable, optional
            Function to apply to each dataset before comparison

        Returns:
        --------
        np.ndarray
            Difference values with shape (n_frames, n_pairs)
        """
        return CalculatorStatHelper.compute_differences(
            distances1, distances2, self.chunk_size, preprocessing_func
        )
