# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# contact_calculator_analysis - Analysis methods for contact calculations
#
# Analysis methods for contact calculations with statistical computations.
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

from ..helper.calculator_stat_helper import CalculatorStatHelper


class ContactCalculatorAnalysis:
    """Analysis methods for contact calculations."""

    def __init__(self, chunk_size=None):
        """
        Initialize contact calculator analysis.

        Args:
            chunk_size: Size of chunks for processing large datasets
        """
        self.chunk_size = chunk_size

    # === PAIR-BASED STATISTICS ===
    def compute_frequency(self, contacts):
        """Compute contact frequency per pair."""
        return CalculatorStatHelper.compute_func_per_pair(contacts, np.mean, self.chunk_size)

    # === FRAME-BASED STATISTICS ===
    def contacts_per_frame_abs(self, contacts):
        """Compute absolute number of contacts per frame."""
        return CalculatorStatHelper.compute_func_per_frame(contacts, self.chunk_size, np.sum)

    def contacts_per_frame_percentage(self, contacts):
        """Compute percentage of contacts per frame."""
        return CalculatorStatHelper.compute_func_per_frame(contacts, self.chunk_size, np.mean)

    # === RESIDUE-BASED STATISTICS (only for square format) ===
    def compute_per_residue_mean(self, contacts):
        """Compute mean contacts per residue."""
        return CalculatorStatHelper.compute_func_per_residue(contacts, np.mean, self.chunk_size)

    def compute_per_residue_std(self, contacts):
        """Compute standard deviation of contacts per residue."""
        return CalculatorStatHelper.compute_func_per_residue(contacts, np.std, self.chunk_size)

    def compute_per_residue_sum(self, contacts):
        """Compute sum of contacts per residue."""
        return CalculatorStatHelper.compute_func_per_residue(contacts, np.sum, self.chunk_size)

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, contacts, threshold=1, lag_time=1):
        """Compute transitions within lag time."""
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            contacts, threshold, lag_time, self.chunk_size
        )

    def compute_transitions_window(self, contacts, threshold=1, window_size=10):
        """Compute transitions within window."""
        return CalculatorStatHelper.compute_transitions_within_window(
            contacts, threshold, window_size, self.chunk_size
        )

    def compute_stability(self, contacts, threshold=1, window_size=1):
        """Compute stability analysis."""
        return CalculatorStatHelper.compute_stability(
            contacts, threshold, window_size, self.chunk_size
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, contacts1, contacts2, preprocessing_func=None):
        """Compute differences between two contact datasets."""
        return CalculatorStatHelper.compute_differences(
            contacts1, contacts2, self.chunk_size, preprocessing_func
        )
