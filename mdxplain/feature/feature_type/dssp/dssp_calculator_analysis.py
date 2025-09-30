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

import numpy as np
from ....utils.data_utils import DataUtils


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
            cache_file = DataUtils.get_cache_file_path(f'dssp_indices_{id(dssp_data)}.npy', self.cache_path)
            indices = np.memmap(cache_file, dtype=np.int8, mode='w+', 
                               shape=(n_frames, n_residues))
            
            # Chunk-wise processing
            for i in range(0, n_frames, self.chunk_size):
                end = min(i + self.chunk_size, n_frames)
                chunk_reshaped = dssp_data[i:end].reshape(-1, n_residues, n_classes)
                indices[i:end] = np.argmax(chunk_reshaped, axis=2)
            
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
            for i in range(0, n_frames - lag_time, self.chunk_size):
                end = min(i + self.chunk_size, n_frames - lag_time)
                transitions += (data[i:end] != data[i+lag_time:end+lag_time]).sum(axis=0).astype(np.float32)
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
            for start in range(0, n_frames - window_size + 1, self.chunk_size):
                end = min(start + self.chunk_size, n_frames - window_size + 1)
                
                for i in range(start, end):
                    window_data = data[i:i + window_size]
                    # Check if any transitions occur within this window
                    diffs = (window_data[1:] != window_data[:-1])
                    window_transitions = diffs.any(axis=0)
                    transitions += window_transitions.astype(np.float32)
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
            # For one-hot, reshape and average
            n_frames = dssp_data.shape[0]
            reshaped = dssp_data.reshape(n_frames, n_residues, n_classes)
            frequencies = reshaped.mean(axis=0)  # Average over frames
            
            # Get class names based on detected classes
            if n_classes == 4:
                class_values = np.array(self.simplified_classes)
            else:
                class_values = np.array(self.full_classes)
            
            return frequencies, class_values
        
        else:  # standard encoding
            n_frames = dssp_data.shape[0]
            class_values = np.array(self.simplified_classes if simplified else self.full_classes)
            n_classes = len(class_values)
            frequencies = np.zeros((n_residues, n_classes), dtype=np.float32)
            
            if self.use_memmap:
                for i in range(0, n_frames, self.chunk_size):
                    end = min(i + self.chunk_size, n_frames)
                    chunk = dssp_data[i:end]
                    for class_idx, class_value in enumerate(class_values):
                        frequencies[:, class_idx] += (chunk == class_value).sum(axis=0).astype(np.float32)
            else:
                for class_idx, class_value in enumerate(class_values):
                    frequencies[:, class_idx] = (dssp_data == class_value).sum(axis=0).astype(np.float32)
            
            frequencies /= n_frames
            return frequencies, class_values

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
        else:
            # In-memory computation
            transitions = (dssp_data[1:] != dssp_data[:-1]).sum(axis=0).astype(np.float32)
            transition_counts = transitions.astype(np.float32)

        # Normalize by number of possible transitions (n_frames - 1)
        transition_frequencies = transition_counts / max(1, n_frames - 1)

        return transition_frequencies

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
