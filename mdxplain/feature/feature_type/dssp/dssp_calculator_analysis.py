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
DSSP calculator analysis for molecular dynamics trajectory analysis.

Analysis utilities for DSSP data including secondary structure dynamics,
stability analysis, and transition frequency calculations.
"""

from typing import List, Tuple

import numpy as np
from ....utils.memmap_utils import MemmapUtils
from ....utils.path_utils import PathUtils
from ....utils.resource_utils import ResourceUtils


class DSSPCalculatorAnalysis:
    """
    Analysis utilities for DSSP secondary structure data from MD trajectories.

    Provides statistical analysis methods for DSSP data including
    secondary structure stability, transition analysis, and
    structural dynamics patterns.

    Examples
    --------
    >>> analysis = DSSPCalculatorAnalysis()
    >>> stability = analysis.compute_class_stability(dssp_data)
    >>> transitions = analysis.compute_transition_frequency(dssp_data)
    """

    def __init__(self, full_classes, simplified_classes, use_memmap: bool = False, chunk_size: int = 2000, cache_path: str = "./cache") -> None:
        """
        Initialize DSSP analysis with configuration parameters.

        Parameters
        ----------
        full_classes : list
            List of full DSSP class labels (9 classes)
        simplified_classes : list
            List of simplified DSSP class labels (4 classes)
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, optional
            Number of frames to process per chunk
        cache_path : str, default="./cache"
            Directory path for storing cache files

        Returns
        -------
        None

        Examples
        --------
        >>> # Basic initialization
        >>> analysis = DSSPCalculatorAnalysis()

        >>> # With memory mapping
        >>> analysis = DSSPCalculatorAnalysis(use_memmap=True, chunk_size=1000)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.cache_path = cache_path
        self.full_classes = full_classes
        self.simplified_classes = simplified_classes

    def _detect_encoding(self, dssp_data: np.ndarray) -> tuple:
        """
        Detect DSSP encoding type.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP data array

        Returns
        -------
        tuple
            ('onehot', n_residues, n_classes) or ('standard', n_residues, None)

        Notes
        -----
        Detects encoding by analyzing array shape and content.
        One-hot is detected when features are divisible by class count.
        """
        n_features = dssp_data.shape[1]

        # Check for one-hot by feature count (only for numeric data)
        if dssp_data.dtype.kind in 'fc':  # float or complex
            for n_classes in [4, 9]:  # simplified or full
                if n_features % n_classes == 0:
                    n_residues = n_features // n_classes
                    # Verify it's actually one-hot (sum per block should be 1)
                    sample = dssp_data[0].reshape(n_residues, n_classes)
                    if np.allclose(sample.sum(axis=1), 1.0, atol=1e-5):
                        return 'onehot', n_residues, n_classes
        
        # Standard encoding (char or int)
        return 'standard', n_features, None

    def _onehot_to_indices(self, dssp_data: np.ndarray, n_residues: int, n_classes: int) -> np.ndarray:
        """
        Convert one-hot encoded data to class indices.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            One-hot encoded DSSP data with shape (n_frames, n_residues * n_classes)
        n_residues : int
            Number of residues
        n_classes : int
            Number of classes

        Returns
        -------
        numpy.ndarray
            Class indices with shape (n_frames, n_residues)

        Notes
        -----
        Memmap-conform: Uses chunk-wise processing for memmap arrays.
        """
        n_frames = dssp_data.shape[0]
        
        if self.use_memmap:
            # Create memmap array for indices using DataUtils
            cache_file = PathUtils.get_cache_file_path(f'dssp_indices_{id(dssp_data)}.npy', self.cache_path)
            indices = MemmapUtils.create_memmap(
                path=cache_file,
                dtype=np.int8,
                mode="w+",
                shape=(n_frames, n_residues),
            )
            
            # Chunk-wise processing
            ResourceUtils.tune_memmap(indices, "sequential")
            if MemmapUtils.is_memmap_view(dssp_data):
                ResourceUtils.tune_memmap(dssp_data, "sequential")
            for i in range(0, n_frames, self.chunk_size):
                end = min(i + self.chunk_size, n_frames)
                chunk_reshaped = dssp_data[i:end].reshape(-1, n_residues, n_classes)
                indices[i:end] = np.argmax(chunk_reshaped, axis=2)
                MemmapUtils.evict_memory_range(indices, i, end)
            
            ResourceUtils.tune_memmap(indices, "random")
            if MemmapUtils.is_memmap_view(dssp_data):
                ResourceUtils.tune_memmap(dssp_data, "random")
            return indices
        else:
            # Non-memmap: direct processing
            reshaped = dssp_data.reshape(n_frames, n_residues, n_classes)
            return np.argmax(reshaped, axis=2)

    def compute_transitions_lagtime(self, dssp_data: np.ndarray, lag_time: int = 10) -> np.ndarray:
        """
        Compute transitions with lag time for all DSSP encodings.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP data array (all encodings supported)
        threshold : float, optional
            Ignored for DSSP (always uses != comparison)
        lag_time : int, default=10
            Number of frames to look ahead for transitions

        Returns
        -------
        numpy.ndarray
            Transition counts for each residue

        Notes
        -----
        Works with all DSSP encodings (char, int, one-hot).
        Clean pattern: detect → convert → process.
        """
        # Step 1: Detect encoding
        encoding_type, n_residues, n_classes = self._detect_encoding(dssp_data)
        
        # Step 2: Convert if one-hot
        if encoding_type == 'onehot':
            data = self._onehot_to_indices(dssp_data, n_residues, n_classes)
            n_residues = data.shape[1]  # Update after conversion
        else:
            data = dssp_data
        
        # Step 3: Unified logic for ALL encodings
        n_frames = data.shape[0]
        if lag_time >= n_frames:
            return np.zeros(n_residues, dtype=np.float32)
        
        if self.use_memmap:
            transitions = np.zeros(n_residues, dtype=np.float32)
            if MemmapUtils.is_memmap_view(data):
                ResourceUtils.tune_memmap(data, "sequential")
            for i in range(0, n_frames - lag_time, self.chunk_size):
                end = min(i + self.chunk_size, n_frames - lag_time)
                transitions += (data[i:end] != data[i+lag_time:end+lag_time]).sum(axis=0).astype(np.float32)
            if MemmapUtils.is_memmap_view(data):
                ResourceUtils.tune_memmap(data, "random")
            return transitions
        else:
            return (data[:-lag_time] != data[lag_time:]).sum(axis=0).astype(np.float32)

    def compute_transitions_window(self, dssp_data: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        Compute transitions within sliding window for all DSSP encodings.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP data array (all encodings supported)
        threshold : float, optional
            Ignored for DSSP (always uses != comparison)
        window_size : int, default=10
            Size of sliding window for transition analysis

        Returns
        -------
        numpy.ndarray
            Transition counts for each residue

        Notes
        -----
        Works with all DSSP encodings (char, int, one-hot).
        Clean pattern: detect → convert → process.
        """
        # Step 1: Detect encoding
        encoding_type, n_residues, n_classes = self._detect_encoding(dssp_data)
        
        # Step 2: Convert if one-hot
        if encoding_type == 'onehot':
            data = self._onehot_to_indices(dssp_data, n_residues, n_classes)
            n_residues = data.shape[1]  # Update after conversion
        else:
            data = dssp_data
        
        # Step 3: Unified logic for ALL encodings - sliding window approach
        n_frames = data.shape[0]
        transitions = np.zeros(n_residues, dtype=np.float32)
        
        # For each sliding window, count if ANY transition occurs
        if self.use_memmap:
            # Chunk-wise processing for memmap
            if MemmapUtils.is_memmap_view(data):
                ResourceUtils.tune_memmap(data, "sequential")
            for start in range(0, n_frames - window_size + 1, self.chunk_size):
                end = min(start + self.chunk_size, n_frames - window_size + 1)
                
                for i in range(start, end):
                    window_data = data[i:i + window_size]
                    # Check if any transitions occur within this window
                    diffs = (window_data[1:] != window_data[:-1])
                    window_transitions = diffs.any(axis=0)
                    transitions += window_transitions.astype(np.float32)
            if MemmapUtils.is_memmap_view(data):
                ResourceUtils.tune_memmap(data, "random")
        else:
            # In-memory processing
            for i in range(n_frames - window_size + 1):
                window_data = data[i:i + window_size]
                # Check if any transitions occur within this window
                diffs = (window_data[1:] != window_data[:-1])
                window_transitions = diffs.any(axis=0)
                transitions += window_transitions.astype(np.float32)
        
        return transitions

    def compute_class_frequencies(self, dssp_data: np.ndarray, simplified: bool = True) -> tuple:
        """
        Compute class frequencies for all DSSP encodings.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP data array (all encodings supported)
        simplified : bool, default=True
            Whether to use simplified DSSP classes (4) or full classes (9)
            Ignored for one-hot encoding (auto-detected)

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Tuple containing (frequencies, class_values) where:
            
            - frequencies: shape (n_residues, n_classes) with class frequencies
            - class_values: array of class values

        Notes
        -----
        Works with all DSSP encodings (char, int, one-hot).
        For one-hot: directly averages the encoded values.
        For standard: counts occurrences of each class.
        """
        encoding_type, n_residues, n_classes = self._detect_encoding(dssp_data)
        if encoding_type == 'onehot':
            return self._onehot_class_frequencies(dssp_data, n_residues, n_classes)
        return self._standard_class_frequencies(dssp_data, n_residues, simplified)

    def _frame_blocks(self, data: np.ndarray):
        """
        Yield (start, end) frame ranges, tuning memmap readahead around them.

        The tuning hint prefetches pages for the sequential frame sweep and is
        restored afterwards, so the same block loop serves both in-memory and
        memory-mapped arrays. A single block covers the whole array when the
        chunk size is at least the frame count.

        Parameters
        ----------
        data : numpy.ndarray
            Array whose first axis is frames

        Yields
        ------
        tuple
            (start, end) frame index bounds of the next block

        Examples
        --------
        >>> for start, end in analysis._frame_blocks(dssp_data):
        ...     block = dssp_data[start:end]
        """
        tuned = self.use_memmap and MemmapUtils.is_memmap_view(data)
        if tuned:
            ResourceUtils.tune_memmap(data, "sequential")
        try:
            for start in range(0, data.shape[0], self.chunk_size):
                yield start, min(start + self.chunk_size, data.shape[0])
        finally:
            if tuned:
                ResourceUtils.tune_memmap(data, "random")

    def _onehot_class_frequencies(
        self, dssp_data: np.ndarray, n_residues: int, n_classes: int
    ) -> tuple:
        """
        Average one-hot indicator columns over frames, one frame block at a time.

        Each column is already a 0/1 indicator, so its mean over frames is the
        class frequency. Frames are summed block-wise in float64 and divided at
        the end, so peak memory is one block by the encoded width regardless of
        frame count.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            One-hot DSSP array with shape (n_frames, n_residues * n_classes)
        n_residues : int
            Number of residues
        n_classes : int
            Number of one-hot classes per residue

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (frequencies, class_values); frequencies has shape
            (n_residues, n_classes)

        Examples
        --------
        >>> freqs, classes = analysis._onehot_class_frequencies(data, 20, 4)
        """
        totals = np.zeros(dssp_data.shape[1], dtype=np.float64)
        for start, end in self._frame_blocks(dssp_data):
            totals += dssp_data[start:end].sum(axis=0)
        frequencies = (totals / dssp_data.shape[0]).reshape(n_residues, n_classes)
        classes = self.simplified_classes if n_classes == 4 else self.full_classes
        return frequencies.astype(np.float32), np.array(classes)

    def _standard_class_frequencies(
        self, dssp_data: np.ndarray, n_residues: int, simplified: bool
    ) -> tuple:
        """
        Count each class over frames for char or integer encodings.

        The frequency of a class at a residue is the fraction of frames whose
        value equals that class. Counts accumulate block-wise so peak memory is
        one frame block by the residue count.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP array with shape (n_frames, n_residues)
        n_residues : int
            Number of residues
        simplified : bool
            Whether to use the four simplified classes or the full class set

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (frequencies, class_values); frequencies has shape
            (n_residues, n_classes)

        Examples
        --------
        >>> freqs, classes = analysis._standard_class_frequencies(data, 20, True)
        """
        class_values = np.array(
            self.simplified_classes if simplified else self.full_classes
        )
        frequencies = np.zeros((n_residues, len(class_values)), dtype=np.float32)
        self._accumulate_class_counts(dssp_data, frequencies, class_values)
        frequencies /= dssp_data.shape[0]
        return frequencies, class_values

    def _accumulate_class_counts(
        self,
        dssp_data: np.ndarray,
        frequencies: np.ndarray,
        class_values: np.ndarray,
    ) -> None:
        """
        Add per-class frame counts into frequencies, one frame block at a time.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP array with shape (n_frames, n_residues)
        frequencies : numpy.ndarray
            Running counts with shape (n_residues, n_classes), updated in place
        class_values : numpy.ndarray
            The class values to count

        Returns
        -------
        None

        Examples
        --------
        >>> analysis._accumulate_class_counts(data, frequencies, class_values)
        """
        for start, end in self._frame_blocks(dssp_data):
            self._count_block(dssp_data[start:end], frequencies, class_values)

    @staticmethod
    def _count_block(
        block: np.ndarray,
        frequencies: np.ndarray,
        class_values: np.ndarray,
    ) -> None:
        """
        Add the per-class frame counts of one block into frequencies.

        Parameters
        ----------
        block : numpy.ndarray
            Frame block with shape (block_frames, n_residues)
        frequencies : numpy.ndarray
            Running counts with shape (n_residues, n_classes), updated in place
        class_values : numpy.ndarray
            The class values to count

        Returns
        -------
        None

        Examples
        --------
        >>> DSSPCalculatorAnalysis._count_block(block, frequencies, class_values)
        """
        for class_idx, class_value in enumerate(class_values):
            frequencies[:, class_idx] += (
                (block == class_value).sum(axis=0).astype(np.float32)
            )

    def compute_transition_frequency(self, dssp_data: np.ndarray) -> np.ndarray:
        """
        Compute transition frequency for each residue from DSSP data.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP array with shape (n_frames, n_residues)
            Works with any encoding (integer, character, etc.)

        Returns
        -------
        numpy.ndarray
            Transition frequencies for each residue (number of transitions / frame)

        Notes
        -----
        Works with any DSSP encoding (integer, character, onehot, etc.).
        Counts the number of secondary structure transitions per residue using != comparison.
        """
        n_frames, n_residues = dssp_data.shape
        transition_counts = np.zeros(n_residues, dtype=np.float32)

        if self.use_memmap:
            # Chunk-wise processing with overlap handling
            prev_frame = None
            if MemmapUtils.is_memmap_view(dssp_data):
                ResourceUtils.tune_memmap(dssp_data, "sequential")
            
            for i in range(0, n_frames, self.chunk_size):
                end = min(i + self.chunk_size, n_frames)
                chunk = dssp_data[i:end]
                
                # Handle transitions within chunk
                if len(chunk) > 1:
                    chunk_transitions = (chunk[1:] != chunk[:-1]).sum(axis=0).astype(np.float32)
                    transition_counts += chunk_transitions
                
                # Handle transition at chunk boundary
                if prev_frame is not None and i > 0:
                    boundary_transitions = (chunk[0] != prev_frame)
                    transition_counts += boundary_transitions.astype(np.float32)
                
                prev_frame = chunk[-1]
            if MemmapUtils.is_memmap_view(dssp_data):
                ResourceUtils.tune_memmap(dssp_data, "random")
        else:
            # In-memory computation
            transitions = (dssp_data[1:] != dssp_data[:-1]).sum(axis=0).astype(np.float32)
            transition_counts = transitions.astype(np.float32)

        # Normalize by number of possible transitions (n_frames - 1)
        transition_frequencies = transition_counts / max(1, n_frames - 1)

        return transition_frequencies

    def compute_pooled_transition_frequency(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Compute pooled transition frequency across segments.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_residues) DSSP arrays

        Returns
        -------
        numpy.ndarray
            Pooled transition frequency per residue
        """
        if not segments:
            return np.array([])

        weighted_sum = None
        total_possible = 0
        for segment in segments:
            n_frames = segment.shape[0]
            possible = n_frames - 1
            if possible <= 0:
                continue
            frequency = self.compute_transition_frequency(segment)
            weighted = frequency * possible
            total_possible += possible
            if weighted_sum is None:
                weighted_sum = weighted
            else:
                weighted_sum += weighted

        if weighted_sum is None:
            return np.zeros(segments[0].shape[1], dtype=np.float32)
        if total_possible == 0:
            return np.zeros_like(weighted_sum, dtype=np.float32)
        return weighted_sum / total_possible

    def compute_stability(self, dssp_data: np.ndarray) -> np.ndarray:
        """
        Compute secondary structure stability for each residue.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP array (all encodings supported)

        Returns
        -------
        numpy.ndarray
            Stability scores for each residue (1 - transition_frequency)

        Notes
        -----
        Stability is computed as 1 - transition_frequency.
        High values (near 1) indicate stable secondary structure.
        Low values (near 0) indicate highly dynamic regions.
        """
        frequency = self.compute_transition_frequency(dssp_data)
        return 1.0 - frequency

    def compute_pooled_stability(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Compute pooled stability across segments.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_residues) DSSP arrays

        Returns
        -------
        numpy.ndarray
            Pooled stability per residue
        """
        if not segments:
            return np.array([])

        weighted_sum = None
        total_possible = 0
        for segment in segments:
            n_frames = segment.shape[0]
            possible = n_frames - 1
            if possible <= 0:
                continue
            stability = self.compute_stability(segment)
            weighted = stability * possible
            total_possible += possible
            if weighted_sum is None:
                weighted_sum = weighted
            else:
                weighted_sum += weighted

        if weighted_sum is None:
            return np.ones(segments[0].shape[1], dtype=np.float32)
        if total_possible == 0:
            return np.ones_like(weighted_sum, dtype=np.float32)
        return weighted_sum / total_possible

    def compute_pooled_transitions(
        self,
        segments: List[np.ndarray],
        transition_mode: str = "window",
        window_size: int = 10,
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute pooled transition counts across segments.

        Parameters
        ----------
        segments : list
            List of (n_frames, n_residues) DSSP arrays
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime'
        window_size : int, default=10
            Window size for transitions metric
        lag_time : int, default=1
            Lag time for transitions metric

        Returns
        -------
        numpy.ndarray
            Pooled transition counts per residue
        """
        if not segments:
            return np.array([])

        total_transitions = None
        for segment in segments:
            if transition_mode == "lagtime":
                transitions = self.compute_transitions_lagtime(segment, lag_time=lag_time)
            else:
                transitions = self.compute_transitions_window(segment, window_size=window_size)
            if total_transitions is None:
                total_transitions = transitions.astype(np.float32)
            else:
                total_transitions += transitions.astype(np.float32)

        if total_transitions is None:
            return np.zeros(segments[0].shape[1], dtype=np.float32)
        return total_transitions

    def compute_pooled_class_frequencies(
        self,
        segments: List[np.ndarray],
        simplified: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pooled class frequencies across segments.

        Parameters
        ----------
        segments : list
            List of DSSP arrays
        simplified : bool, default=True
            Whether to use simplified DSSP classes (ignored for one-hot)

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (frequencies, class_values)
        """
        if not segments:
            return np.array([]), np.array([])

        total_frames = 0
        total_freq = None
        class_values = None

        for segment in segments:
            n_frames = segment.shape[0]
            if n_frames <= 0:
                continue
            freqs, class_values = self.compute_class_frequencies(
                segment, simplified=simplified
            )
            if total_freq is None:
                total_freq = freqs * n_frames
            else:
                total_freq += freqs * n_frames
            total_frames += n_frames

        if total_freq is None:
            freqs, class_values = self.compute_class_frequencies(
                segments[0], simplified=simplified
            )
            return freqs, class_values

        if total_frames == 0:
            return total_freq, class_values
        return total_freq / total_frames, class_values

    def compute_pooled_metric_values(
        self,
        segments: List[np.ndarray],
        metric: str,
        transition_mode: str = "window",
        window_size: int = 10,
        lag_time: int = 1,
        simplified: bool = True,
    ) -> np.ndarray:
        """
        Compute pooled metric values across segments.

        Parameters
        ----------
        segments : list
            List of DSSP arrays
        metric : str
            Metric name
        transition_mode : str, default='window'
            Mode for transitions metric: 'window' or 'lagtime'
        window_size : int, default=10
            Window size for transitions metric
        lag_time : int, default=1
            Lag time for transitions metric
        simplified : bool, default=True
            Whether to use simplified DSSP classes (ignored for one-hot)

        Returns
        -------
        numpy.ndarray
            Pooled metric values per residue
        """
        if metric == "transition_frequency":
            return self.compute_pooled_transition_frequency(segments)
        if metric == "stability":
            return self.compute_pooled_stability(segments)
        if metric == "transitions":
            return self.compute_pooled_transitions(
                segments,
                transition_mode=transition_mode,
                window_size=window_size,
                lag_time=lag_time,
            )
        if metric == "class_frequencies":
            frequencies, _ = self.compute_pooled_class_frequencies(
                segments, simplified=simplified
            )
            return np.max(frequencies, axis=1)
        raise ValueError(f"Pooled reduction is not supported for metric '{metric}'.")

    def compute_differences(self, dssp_data: np.ndarray, frame_1: int = 0, frame_2: int = -1) -> np.ndarray:
        """
        Compute differences between two frames for all DSSP encodings.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP array (all encodings supported)
        frame_1 : int, default=0
            First frame index
        frame_2 : int, default=-1
            Second frame index (-1 for last frame)

        Returns
        -------
        numpy.ndarray
            Differences between frames (1.0 where different, 0.0 where same)

        Notes
        -----
        Works with all DSSP encodings (char, int, one-hot).
        Clean pattern: detect → convert → process.
        """
        if frame_2 == -1:
            frame_2 = dssp_data.shape[0] - 1
        
        # Step 1: Detect encoding
        encoding_type, n_residues, n_classes = self._detect_encoding(dssp_data)
        
        # Step 2: Convert if one-hot
        if encoding_type == 'onehot':
            data = self._onehot_to_indices(dssp_data, n_residues, n_classes)
        else:
            data = dssp_data
        
        # Step 3: Unified logic for ALL encodings
        return (data[frame_1] != data[frame_2]).astype(np.float32)

    def compute_dominant_class(self, dssp_data: np.ndarray) -> np.ndarray:
        """
        Compute the dominant (most frequent) secondary structure class for each residue.

        Parameters
        ----------
        dssp_data : numpy.ndarray
            DSSP array with shape (n_frames, n_residues)
            Works with any encoding (integer, character, etc.)

        Returns
        -------
        numpy.ndarray
            Dominant class value for each residue (actual values, not indices)

        Notes
        -----
        Returns the class value that appears most frequently for each residue.
        Works with any DSSP encoding (integer, character, onehot, etc.).
        """
        frequencies, unique_values = self.compute_class_frequencies(dssp_data)
        dominant_indices = np.argmax(frequencies, axis=1)
        dominant_classes = unique_values[dominant_indices]
        
        return dominant_classes
