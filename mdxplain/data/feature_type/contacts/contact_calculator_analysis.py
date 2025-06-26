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
Statistical analysis for contact calculations.

Analysis methods for contact calculations with statistical computations
and support for memory-mapped arrays and contact pattern analysis.
"""

import numpy as np

from ..helper.calculator_stat_helper import CalculatorStatHelper


class ContactCalculatorAnalysis:
    """
    Analysis methods for contact calculation statistics and metrics.

    Provides statistical analysis capabilities for contact data including
    frequency calculations, frame-based statistics, residue-based analysis,
    and transition analysis with memory-mapped array support.
    """

    # Methods that require full data instead of reduced data
    REQUIRES_FULL_DATA = {
        "compute_per_residue_mean", "compute_per_residue_std", "compute_per_residue_sum"
    }

    def __init__(self, chunk_size=None):
        """
        Initialize contact analysis with chunking configuration.

        Parameters:
        -----------
        chunk_size : int, optional
            Number of frames to process per chunk for memory-mapped arrays

        Examples:
        ---------
        >>> # Default chunking
        >>> analysis = ContactCalculatorAnalysis()

        >>> # Custom chunk size for large datasets
        >>> analysis = ContactCalculatorAnalysis(chunk_size=1000)
        """
        self.chunk_size = chunk_size

    # === PAIR-BASED STATISTICS ===
    def compute_frequency(self, contacts):
        """
        Compute contact frequency (fraction of frames in contact) per pair.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Binary contact array (0/1 values)

        Returns:
        --------
        numpy.ndarray
            Contact frequencies per pair (0.0 to 1.0)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            contacts, np.mean, self.chunk_size
        )

    # === FRAME-BASED STATISTICS ===
    def contacts_per_frame_abs(self, contacts):
        """
        Compute absolute number of contacts per frame.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Binary contact array

        Returns:
        --------
        numpy.ndarray
            Total contact count per frame
        """
        return CalculatorStatHelper.compute_func_per_frame(
            contacts, self.chunk_size, np.sum
        )

    def contacts_per_frame_percentage(self, contacts):
        """
        Compute percentage of contacts per frame.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Binary contact array

        Returns:
        --------
        numpy.ndarray
            Fraction of pairs in contact per frame (0.0 to 1.0)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            contacts, self.chunk_size, np.mean
        )

    # === PER-COLUMN ANALYSIS (auto-converts 2D to 3D) ===
    def compute_per_residue_mean(self, contacts):
        """
        Compute mean contact frequency per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Contact array in condensed format (n_frames, n_pairs) - automatically converted to squareform

        Returns:
        --------
        numpy.ndarray
            Mean contact frequency per residue
        """
        return CalculatorStatHelper.compute_func_per_column(
            contacts, np.mean, self.chunk_size
        )

    def compute_per_residue_std(self, contacts):
        """
        Compute standard deviation of contacts per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Contact array in condensed format (n_frames, n_pairs) - automatically converted to squareform

        Returns:
        --------
        numpy.ndarray
            Standard deviation of contacts per residue
        """
        return CalculatorStatHelper.compute_func_per_column(
            contacts, np.std, self.chunk_size
        )

    def compute_per_residue_sum(self, contacts):
        """
        Compute total contact count per residue. Auto-converts condensed to squareform.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Contact array in condensed format (n_frames, n_pairs) - automatically converted to squareform

        Returns:
        --------
        numpy.ndarray
            Total contact count per residue
        """
        return CalculatorStatHelper.compute_func_per_column(
            contacts, np.sum, self.chunk_size
        )

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, contacts, threshold=1, lag_time=1):
        """
        Compute contact transitions using lag time analysis.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Binary contact array
        threshold : int, default=1
            Threshold for detecting transitions (contact changes)
        lag_time : int, default=1
            Number of frames to look ahead for transitions

        Returns:
        --------
        numpy.ndarray
            Number of transitions per contact pair
        """
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            contacts, threshold, lag_time, self.chunk_size
        )

    def compute_transitions_window(self, contacts, threshold=1, window_size=10):
        """
        Compute contact transitions using sliding window analysis.

        Parameters:
        -----------
        contacts : numpy.ndarray
            Binary contact array
        threshold : int, default=1
            Threshold for detecting transitions (contact changes)
        window_size : int, default=10
            Size of sliding window for transition detection

        Returns:
        --------
        numpy.ndarray
            Number of transitions per contact pair
        """
        return CalculatorStatHelper.compute_transitions_within_window(
            contacts, threshold, window_size, self.chunk_size
        )

    def compute_stability(self, contacts, threshold=1, window_size=1):
        """
        Compute contact stability (inverse of transition rate).

        Parameters:
        -----------
        contacts : numpy.ndarray
            Binary contact array
        threshold : int, default=1
            Threshold for stability detection
        window_size : int, default=1
            Window size for stability calculation

        Returns:
        --------
        numpy.ndarray
            Stability values per contact pair (0=unstable, 1=stable)
        """
        return CalculatorStatHelper.compute_stability(
            contacts, threshold, window_size, self.chunk_size
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, contacts1, contacts2, preprocessing_func=None):
        """
        Compute differences between two contact datasets.

        Parameters:
        -----------
        contacts1 : numpy.ndarray
            First contact array for comparison
        contacts2 : numpy.ndarray
            Second contact array for comparison
        preprocessing_func : callable, optional
            Function to apply before computing differences (default: frequency)

        Returns:
        --------
        numpy.ndarray
            Element-wise differences between preprocessed contact arrays
        """
        return CalculatorStatHelper.compute_differences(
            contacts1, contacts2, self.chunk_size, preprocessing_func
        )
