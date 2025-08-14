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

    # Methods that require full data instead of reduced data
    REQUIRES_FULL_DATA = {
        "compute_per_residue_mean",
        "compute_per_residue_std",
        "compute_per_residue_min",
        "compute_per_residue_max",
        "compute_per_residue_median",
        "compute_per_residue_sum",
        "compute_per_residue_variance",
        "compute_per_residue_range",
    }

    def __init__(self, use_memmap=False, chunk_size=10000):
        """
        Initialize distance analysis with chunking configuration.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, default=10000
            Number of frames to process per chunk for memory-mapped arrays

        Examples:
        ---------
        >>> # Default chunking
        >>> analysis = DistanceCalculatorAnalysis()

        >>> # Custom chunk size
        >>> analysis = DistanceCalculatorAnalysis(chunk_size=1000)
        """
        self.use_memmap = use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.mean, self.chunk_size, self.use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.std, self.chunk_size, self.use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.min, self.chunk_size, self.use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.max, self.chunk_size, self.use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.median, self.chunk_size, self.use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.var, self.chunk_size, self.use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.ptp, self.chunk_size, self.use_memmap
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances,
            lambda x, axis: np.percentile(x, 25, axis=axis),
            self.chunk_size,
            self.use_memmap,
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances,
            lambda x, axis: np.percentile(x, 75, axis=axis),
            self.chunk_size,
            self.use_memmap,
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances,
            lambda x, axis: (
                np.percentile(x, 75, axis=axis) - np.percentile(x, 25, axis=axis)
            ),
            self.chunk_size,
            self.use_memmap,
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
        return CalculatorStatHelper.compute_func_per_feature(
            distances,
            lambda x, axis: np.median(
                np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis
            ),
            self.chunk_size,
            self.use_memmap,
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
            distances, self.chunk_size, self.use_memmap, np.mean
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
            distances, self.chunk_size, self.use_memmap, np.std
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
            distances, self.chunk_size, self.use_memmap, np.min
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
            distances, self.chunk_size, self.use_memmap, np.max
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
            distances, self.chunk_size, self.use_memmap, np.median
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
            distances, self.chunk_size, self.use_memmap, np.ptp
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
            distances, self.chunk_size, self.use_memmap, np.sum
        )

    # === PER-COLUMN ANALYSIS (auto-converts 2D to 3D) ===
    def compute_per_residue_mean(self, distances):
        """
        Compute mean distance for each residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Mean distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.mean, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_std(self, distances):
        """
        Compute standard deviation of distances per residue.

        Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Standard deviation for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.std, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_min(self, distances):
        """
        Compute minimum distances per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Minimum distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.min, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_max(self, distances):
        """
        Compute maximum distances per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Maximum distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.max, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_median(self, distances):
        """
        Compute median distances per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Median distance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.median, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_sum(self, distances):
        """
        Compute sum of distances per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Sum of distances for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.sum, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_variance(self, distances):
        """
        Compute variance of distances per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Variance for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.var, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_range(self, distances):
        """
        Compute range of distances per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs) -
            automatically converted to squareform

        Returns:
        --------
        np.ndarray
            Range (max - min) for each residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_func_per_column(
            distances, np.ptp, self.chunk_size, self.use_memmap
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
            distances, threshold, lag_time, self.chunk_size, self.use_memmap
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
            distances, threshold, window_size, self.chunk_size, self.use_memmap
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
            distances, threshold, window_size, self.chunk_size, self.use_memmap, mode
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
            distances1, distances2, self.chunk_size, self.use_memmap, preprocessing_func
        )
