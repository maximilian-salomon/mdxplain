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
Statistical analysis for contact calculations.

Analysis methods for contact calculations with statistical computations
and support for memory-mapped arrays and contact pattern analysis.
"""

from typing import Callable, List, Optional, Tuple
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
        "compute_per_residue_mean",
        "compute_per_residue_std",
        "compute_per_residue_sum",
    }

    def __init__(self, use_memmap: bool = False, chunk_size: int = 2000) -> None:
        """
        Initialize contact analysis with chunking configuration.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, default=2000
            Number of frames to process per chunk for memory-mapped arrays

        Examples
        --------
        >>> # Default chunking
        >>> analysis = ContactCalculatorAnalysis()

        >>> # Custom chunk size for large datasets
        >>> analysis = ContactCalculatorAnalysis(chunk_size=1000)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

    # === PAIR-BASED STATISTICS ===
    def compute_frequency(self, contacts: np.ndarray) -> np.ndarray:
        """
        Compute contact frequency (fraction of frames in contact) per pair.

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array (0/1 values)

        Returns
        -------
        numpy.ndarray
            Contact frequencies per pair (0.0 to 1.0)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            contacts, np.mean, self.chunk_size, self.use_memmap
        )

    # === FRAME-BASED STATISTICS ===
    def contacts_per_frame_abs(self, contacts: np.ndarray) -> np.ndarray:
        """
        Compute absolute number of contacts per frame.

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array

        Returns
        -------
        numpy.ndarray
            Total contact count per frame
        """
        return CalculatorStatHelper.compute_func_per_frame(
            contacts, self.chunk_size, self.use_memmap, np.sum
        )

    def contacts_per_frame_percentage(self, contacts: np.ndarray) -> np.ndarray:
        """
        Compute percentage of contacts per frame.

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array

        Returns
        -------
        numpy.ndarray
            Fraction of pairs in contact per frame (0.0 to 1.0)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            contacts, self.chunk_size, self.use_memmap, np.mean
        )

    # === PER-RESIDUE ANALYSIS (reduces over each residue's real partners) ===
    def _per_residue_metric(
        self,
        contacts: np.ndarray,
        metric: str,
        pairs: Optional[List[Tuple[int, int]]] = None,
        n_residues: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reduce condensed contacts to one value per residue over its partners.

        The residue pair of each condensed column comes from ``pairs`` (the
        service passes the real pairs from the feature metadata); absent them the
        columns are assumed to be a full upper triangle.

        Parameters
        ----------
        contacts : numpy.ndarray
            Contact array in condensed format (n_frames, n_pairs)
        metric : str
            Per-residue metric name
        pairs : list of tuple, optional
            Residue index pair for each condensed column, in column order
        n_residues : int, optional
            Number of residues; inferred with pairs when omitted

        Returns
        -------
        numpy.ndarray
            Metric value per residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_per_residue_reduction(
            contacts, pairs, n_residues, metric, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_mean(
        self, contacts: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the mean contact frequency of each residue with its partners.

        Parameters
        ----------
        contacts : numpy.ndarray
            Contact array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        numpy.ndarray
            Mean contact frequency per residue
        """
        return self._per_residue_metric(contacts, "mean", pairs, n_residues)

    def compute_per_residue_std(
        self, contacts: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the standard deviation of each residue's partner contacts.

        Parameters
        ----------
        contacts : numpy.ndarray
            Contact array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        numpy.ndarray
            Standard deviation of contacts per residue
        """
        return self._per_residue_metric(contacts, "std", pairs, n_residues)

    def compute_per_residue_sum(
        self, contacts: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the total contact count of each residue with its partners.

        Parameters
        ----------
        contacts : numpy.ndarray
            Contact array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        numpy.ndarray
            Total contact count per residue
        """
        return self._per_residue_metric(contacts, "sum", pairs, n_residues)

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, contacts: np.ndarray, threshold: int = 1, lag_time: int = 1) -> np.ndarray:
        """
        Compute contact transitions using lag time analysis.

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array
        threshold : int, default=1
            Threshold for detecting transitions (contact changes)
        lag_time : int, default=1
            Number of frames to look ahead for transitions

        Returns
        -------
        numpy.ndarray
            Number of transitions per contact pair
        """
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            contacts, threshold, lag_time, self.chunk_size, self.use_memmap
        )

    def compute_transitions_window(self, contacts: np.ndarray, threshold: int = 1, window_size: int = 10) -> np.ndarray:
        """
        Compute contact transitions using sliding window analysis.

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array
        threshold : int, default=1
            Threshold for detecting transitions (contact changes)
        window_size : int, default=10
            Size of sliding window for transition detection

        Returns
        -------
        numpy.ndarray
            Number of transitions per contact pair
        """
        return CalculatorStatHelper.compute_transitions_within_window(
            contacts, threshold, window_size, self.chunk_size, self.use_memmap
        )

    def compute_stability(self, contacts: np.ndarray, threshold: int = 1, window_size: int = 1) -> np.ndarray:
        """
        Compute contact stability (inverse of transition rate).

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array
        threshold : int, default=1
            Threshold for stability detection
        window_size : int, default=1
            Window size for stability calculation

        Returns
        -------
        numpy.ndarray
            Stability values per contact pair (0=unstable, 1=stable)
        """
        return CalculatorStatHelper.compute_stability(
            contacts, threshold, window_size, self.chunk_size, self.use_memmap
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, contacts1: np.ndarray, contacts2: np.ndarray, preprocessing_func: Optional[Callable] = None) -> np.ndarray:
        """
        Compute differences between two contact datasets.

        Parameters
        ----------
        contacts1 : numpy.ndarray
            First contact array for comparison
        contacts2 : numpy.ndarray
            Second contact array for comparison
        preprocessing_func : callable, optional
            Function to apply before computing differences (default: frequency)

        Returns
        -------
        numpy.ndarray
            Element-wise differences between preprocessed contact arrays
        """
        return CalculatorStatHelper.compute_differences(
            contacts1, contacts2, self.chunk_size, self.use_memmap, preprocessing_func
        )

    def compute_pooled_metric_values(
        self,
        segments: List[np.ndarray],
        metric: str,
        transition_threshold: float = 2.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute pooled metric values across segments.

        Parameters
        ----------
        segments : list
            List of contact arrays
        metric : str
            Metric name
        transition_threshold : float, default=2.0
            Threshold for detecting transitions
        window_size : int, default=10
            Window size for transition analysis
        transition_mode : str, default='window'
            Transition mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for transition analysis

        Returns
        -------
        numpy.ndarray
            Pooled metric values per contact pair
        """
        if not segments:
            return np.array([])
        if metric == "transitions":
            window = lag_time if transition_mode == "lagtime" else window_size
            transitions, _ = CalculatorStatHelper.compute_pooled_transitions(
                segments,
                transition_threshold,
                window,
                self.chunk_size,
                self.use_memmap,
                mode=transition_mode,
            )
            return transitions
        if metric == "stability":
            window = lag_time if transition_mode == "lagtime" else window_size
            return CalculatorStatHelper.compute_pooled_stability(
                segments,
                transition_threshold,
                window,
                self.chunk_size,
                self.use_memmap,
                mode=transition_mode,
            )
        if metric == "frequency":
            # Contact frequency is the mean of a 0/1 column, so it streams.
            return CalculatorStatHelper.compute_pooled_reduction_per_feature(
                segments, "mean", self.chunk_size, self.use_memmap
            )
        return CalculatorStatHelper.compute_pooled_func_per_feature(
            segments,
            lambda block: self._metric_from_pooled(block, metric),
            self.chunk_size,
            self.use_memmap,
        )

    def _metric_from_pooled(self, pooled: np.ndarray, metric: str) -> np.ndarray:
        """
        Compute metric values on pooled data.

        Parameters
        ----------
        pooled : np.ndarray
            Pooled contact array
        metric : str
            Metric name

        Returns
        -------
        numpy.ndarray
            Metric values per contact pair
        """
        if metric == "frequency":
            return self.compute_frequency(pooled)
        raise ValueError(f"Unknown metric: {metric}. Supported: ['frequency', 'stability', 'transitions']")
