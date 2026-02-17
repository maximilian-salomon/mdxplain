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
SASA calculator analysis for molecular dynamics trajectory analysis.

Analysis utilities for SASA data including surface area dynamics,
burial analysis, and solvent exposure variability calculations.
"""

from typing import List

import numpy as np

from ..helper.calculator_stat_helper import CalculatorStatHelper


DEFAULT_TRANSITION_THRESHOLD_A2 = 50.0
DEFAULT_BURIAL_THRESHOLD_A2 = 10.0
DEFAULT_EXPOSURE_THRESHOLD_A2 = 100.0


class SASACalculatorAnalysis:
    """
    Analysis utilities for SASA data from MD trajectories.

    Provides statistical analysis methods for SASA data including
    burial/exposure analysis, surface area dynamics, and solvent
    accessibility patterns with complete per-feature and per-frame metrics.

    Examples
    --------
    >>> analysis = SASACalculatorAnalysis()
    >>> mean_sasa = analysis.compute_mean(sasa_data)
    >>> burial_fraction = analysis.compute_burial_fraction(sasa_data, threshold=10.0)
    """

    def __init__(self, use_memmap: bool = False, chunk_size: int = 2000) -> None:
        """
        Initialize SASA analysis with configuration parameters.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, optional
            Number of frames to process per chunk

        Returns
        -------
        None

        Examples
        --------
        >>> # Basic initialization
        >>> analysis = SASACalculatorAnalysis()

        >>> # With memory mapping
        >>> analysis = SASACalculatorAnalysis(use_memmap=True, chunk_size=1000)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

    # ===== PER-FEATURE METHODS (per residue/atom) =====

    def compute_mean(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute mean SASA for each residue/atom.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Mean SASA for each residue/atom

        Examples
        --------
        >>> mean_sasa = analysis.compute_mean(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            sasa_data, np.mean, 
            chunk_size=self.chunk_size, 
            use_memmap=self.use_memmap
        )

    def compute_std(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation for each residue/atom SASA.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Standard deviation for each residue/atom

        Examples
        --------
        >>> std_sasa = analysis.compute_std(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            sasa_data, np.std,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_variance(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute variance for each residue/atom SASA.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Variance for each residue/atom

        Examples
        --------
        >>> var_sasa = analysis.compute_variance(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            sasa_data, np.var,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_min(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute minimum SASA for each residue/atom.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Minimum SASA for each residue/atom

        Examples
        --------
        >>> min_sasa = analysis.compute_min(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            sasa_data, np.min,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_max(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute maximum SASA for each residue/atom.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Maximum SASA for each residue/atom

        Examples
        --------
        >>> max_sasa = analysis.compute_max(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            sasa_data, np.max,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_mad(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute median absolute deviation for each residue/atom SASA.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Median absolute deviation for each residue/atom

        Examples
        --------
        >>> mad_sasa = analysis.compute_mad(sasa_data)
        """
        def mad_func(data, axis=0):
            median = np.median(data, axis=axis, keepdims=True)
            return np.median(np.abs(data - median), axis=axis)
        
        return CalculatorStatHelper.compute_func_per_feature(
            sasa_data, mad_func,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_range(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute range (max - min) for each residue/atom SASA.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Range for each residue/atom

        Examples
        --------
        >>> range_sasa = analysis.compute_range(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            sasa_data, np.ptp,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    # ===== PER-FRAME METHODS (per time step) =====

    def compute_mean_per_frame(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute mean SASA per frame across all residues/atoms.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Mean SASA per frame

        Examples
        --------
        >>> mean_per_frame = analysis.compute_mean_per_frame(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            sasa_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.mean
        )

    def compute_std_per_frame(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation per frame across all residues/atoms.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Standard deviation per frame

        Examples
        --------
        >>> std_per_frame = analysis.compute_std_per_frame(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            sasa_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.std
        )

    def compute_variance_per_frame(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute variance per frame across all residues/atoms.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Variance per frame

        Examples
        --------
        >>> var_per_frame = analysis.compute_variance_per_frame(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            sasa_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.var
        )

    def compute_min_per_frame(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute minimum SASA per frame across all residues/atoms.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Minimum SASA per frame

        Examples
        --------
        >>> min_per_frame = analysis.compute_min_per_frame(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            sasa_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.min
        )

    def compute_max_per_frame(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute maximum SASA per frame across all residues/atoms.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Maximum SASA per frame

        Examples
        --------
        >>> max_per_frame = analysis.compute_max_per_frame(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            sasa_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.max
        )

    def compute_mad_per_frame(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute median absolute deviation per frame across all residues/atoms.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Median absolute deviation per frame

        Examples
        --------
        >>> mad_per_frame = analysis.compute_mad_per_frame(sasa_data)
        """
        def mad_per_frame(data, axis=1):
            median = np.median(data, axis=axis, keepdims=True)
            return np.median(np.abs(data - median), axis=axis)
        
        return CalculatorStatHelper.compute_func_per_frame(
            sasa_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=mad_per_frame
        )

    def compute_range_per_frame(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute range (max - min) per frame across all residues/atoms.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Range per frame

        Examples
        --------
        >>> range_per_frame = analysis.compute_range_per_frame(sasa_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            sasa_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.ptp
        )

    # ===== DIFFERENCES/COMPARISON METHODS =====

    def compute_differences(self, sasa_data: np.ndarray, frame_1: int = 0, frame_2: int = -1) -> np.ndarray:
        """
        Compute differences between two frames.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)
        frame_1 : int, default=0
            First frame index
        frame_2 : int, default=-1
            Second frame index (-1 for last frame)

        Returns
        -------
        numpy.ndarray
            SASA differences between frames

        Examples
        --------
        >>> differences = analysis.compute_differences(sasa_data, 0, -1)
        """
        return CalculatorStatHelper.compute_differences(
            sasa_data[frame_1], sasa_data[frame_2],
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_differences_mean(self, sasa_data_1: np.ndarray, sasa_data_2: np.ndarray) -> np.ndarray:
        """
        Compute differences between mean SASA of two datasets.

        Parameters
        ----------
        sasa_data_1 : numpy.ndarray
            First SASA dataset
        sasa_data_2 : numpy.ndarray
            Second SASA dataset

        Returns
        -------
        numpy.ndarray
            Mean SASA differences between datasets

        Examples
        --------
        >>> diff_means = analysis.compute_differences_mean(sasa_1, sasa_2)
        """
        return CalculatorStatHelper.compute_differences(
            sasa_data_1, sasa_data_2,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    # ===== TRANSITIONS METHODS =====

    def compute_transitions_lagtime(
        self,
        sasa_data: np.ndarray,
        threshold: float = DEFAULT_TRANSITION_THRESHOLD_A2,
        lag_time: int = 1
    ) -> np.ndarray:
        """
        Compute transitions with lag time for each residue/atom.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)
        threshold : float, default=50.0
            Threshold for detecting transitions (in Å²)
        lag_time : int, default=1
            Number of frames to look ahead

        Returns
        -------
        numpy.ndarray
            Transition counts per residue/atom

        Examples
        --------
        >>> transitions = analysis.compute_transitions_lagtime(sasa_data, 50.0, 10)
        """
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            sasa_data,
            threshold=threshold,
            lag_time=lag_time,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_transitions_window(
        self,
        sasa_data: np.ndarray,
        threshold: float = DEFAULT_TRANSITION_THRESHOLD_A2,
        window_size: int = 10
    ) -> np.ndarray:
        """
        Compute transitions within sliding window for each residue/atom.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)
        threshold : float, default=50.0
            Threshold for detecting transitions (in Å²)
        window_size : int, default=10
            Size of sliding window

        Returns
        -------
        numpy.ndarray
            Transition counts per residue/atom

        Examples
        --------
        >>> transitions = analysis.compute_transitions_window(sasa_data, 50.0, 10)
        """
        return CalculatorStatHelper.compute_transitions_within_window(
            sasa_data,
            threshold=threshold,
            window_size=window_size,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_stability(
        self,
        sasa_data: np.ndarray,
        threshold: float = DEFAULT_TRANSITION_THRESHOLD_A2,
        window_size: int = 10,
        mode: str = "lagtime"
    ) -> np.ndarray:
        """
        Compute stability (inverse of transition rate) for each residue/atom.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)
        threshold : float, default=50.0
            Threshold for stability detection (in Å²)
        window_size : int, default=10
            Window size for calculation
        mode : str, default='lagtime'
            Calculation mode ('lagtime' or 'window')

        Returns
        -------
        numpy.ndarray
            Stability values per residue/atom (0=unstable, 1=stable)

        Examples
        --------
        >>> stability = analysis.compute_stability(sasa_data, 50.0, 10, 'window')
        """
        return CalculatorStatHelper.compute_stability(
            sasa_data,
            threshold=threshold,
            window_size=window_size,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            mode=mode
        )

    # ===== SASA-SPECIFIC METHODS =====

    def compute_burial_fraction(
        self,
        sasa_data: np.ndarray,
        threshold: float = DEFAULT_BURIAL_THRESHOLD_A2
    ) -> np.ndarray:
        """
        Compute fraction of time each residue/atom is buried below threshold.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)
        threshold : float, default=10.0
            SASA threshold in Å² below which residue/atom is considered buried

        Returns
        -------
        numpy.ndarray
            Burial fraction for each residue/atom (0-1)

        Notes
        -----
        A burial fraction of 1.0 means the residue/atom is always buried,
        while 0.0 means it is never buried below the threshold.

        Examples
        --------
        >>> burial = analysis.compute_burial_fraction(sasa_data, 10.0)
        """
        if self.use_memmap:
            # Chunk-wise processing for memory efficiency
            burial_fractions = np.zeros(sasa_data.shape[1], dtype=np.float32)
            n_frames = sasa_data.shape[0]
            
            for i in range(0, n_frames, self.chunk_size):
                end = min(i + self.chunk_size, n_frames)
                chunk = sasa_data[i:end]
                
                # Count buried frames in chunk
                buried_chunk = (chunk < threshold).sum(axis=0)
                burial_fractions += buried_chunk
                
            # Normalize by total number of frames
            burial_fractions /= n_frames
        else:
            # In-memory computation
            buried_frames = (sasa_data < threshold).sum(axis=0)
            burial_fractions = buried_frames / sasa_data.shape[0]

        return burial_fractions

    def compute_cv(self, sasa_data: np.ndarray) -> np.ndarray:
        """
        Compute coefficient of variation for each residue/atom SASA.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)

        Returns
        -------
        numpy.ndarray
            Coefficient of variation for each residue/atom

        Notes
        -----
        CV = standard_deviation / mean
        Higher CV indicates more variable SASA over time.

        Examples
        --------
        >>> cv_sasa = analysis.compute_cv(sasa_data)
        """
        mean_vals = self.compute_mean(sasa_data)
        std_vals = self.compute_std(sasa_data)
        return std_vals / (mean_vals + 1e-10)

    def compute_exposure_fraction(
        self,
        sasa_data: np.ndarray,
        threshold: float = DEFAULT_EXPOSURE_THRESHOLD_A2
    ) -> np.ndarray:
        """
        Compute fraction of time each residue/atom is exposed above threshold.

        Parameters
        ----------
        sasa_data : numpy.ndarray
            SASA array with shape (n_frames, n_residues/atoms)
        threshold : float, default=100.0
            SASA threshold in Å² above which residue/atom is considered exposed

        Returns
        -------
        numpy.ndarray
            Exposure fraction for each residue/atom (0-1)

        Notes
        -----
        A exposure fraction of 1.0 means the residue/atom is always exposed,
        while 0.0 means it is never exposed above the threshold.

        Examples
        --------
        >>> exposure = analysis.compute_exposure_fraction(sasa_data, 100.0)
        """
        if self.use_memmap:
            # Chunk-wise processing for memory efficiency
            exposure_fractions = np.zeros(sasa_data.shape[1], dtype=np.float32)
            n_frames = sasa_data.shape[0]
            
            for i in range(0, n_frames, self.chunk_size):
                end = min(i + self.chunk_size, n_frames)
                chunk = sasa_data[i:end]
                
                # Count exposed frames in chunk
                exposed_chunk = (chunk > threshold).sum(axis=0)
                exposure_fractions += exposed_chunk
                
            # Normalize by total number of frames
            exposure_fractions /= n_frames
        else:
            # In-memory computation
            exposed_frames = (sasa_data > threshold).sum(axis=0)
            exposure_fractions = exposed_frames / sasa_data.shape[0]

        return exposure_fractions

    def compute_pooled_metric_values(
        self,
        segments: List[np.ndarray],
        metric: str,
        transition_threshold: float = DEFAULT_TRANSITION_THRESHOLD_A2,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
        threshold_min: float = None,
        threshold_max: float = None,
    ) -> np.ndarray:
        """
        Compute pooled metric values across segments.

        Parameters
        ----------
        segments : list
            List of SASA arrays
        metric : str
            Metric name
        transition_threshold : float, default=50.0
            Threshold for detecting transitions
        window_size : int, default=10
            Window size for transition analysis
        transition_mode : str, default='window'
            Transition mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for transition analysis
        threshold_min : float, optional
            Minimum threshold (used for burial_fraction)
        threshold_max : float, optional
            Maximum threshold (used for exposure_fraction)

        Returns
        -------
        numpy.ndarray
            Pooled metric values per SASA feature
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
        pooled = np.concatenate(segments, axis=0)
        return self._metric_from_pooled(pooled, metric, threshold_min, threshold_max)

    def _metric_from_pooled(
        self,
        pooled: np.ndarray,
        metric: str,
        threshold_min: float = None,
        threshold_max: float = None,
    ) -> np.ndarray:
        """
        Compute metric values on pooled data.

        Parameters
        ----------
        pooled : np.ndarray
            Pooled SASA array
        metric : str
            Metric name
        threshold_min : float, optional
            Minimum threshold (used for burial_fraction)
        threshold_max : float, optional
            Maximum threshold (used for exposure_fraction)

        Returns
        -------
        numpy.ndarray
            Metric values per SASA feature
        """
        metrics = {
            "std": self.compute_std,
            "variance": self.compute_variance,
            "min": self.compute_min,
            "max": self.compute_max,
            "mad": self.compute_mad,
            "mean": self.compute_mean,
            "cv": self.compute_cv,
            "range": self.compute_range,
            "dynamic_range": self.compute_range,
        }
        if metric in metrics:
            return metrics[metric](pooled)
        if metric == "burial_fraction":
            cutoff = threshold_min if threshold_min is not None else DEFAULT_BURIAL_THRESHOLD_A2
            return self.compute_burial_fraction(pooled, cutoff)
        if metric == "exposure_fraction":
            cutoff = threshold_max if threshold_max is not None else DEFAULT_EXPOSURE_THRESHOLD_A2
            return self.compute_exposure_fraction(pooled, cutoff)
        raise ValueError(
            f"Unknown metric: {metric}. Supported: {list(metrics.keys()) + ['transitions', 'stability', 'burial_fraction', 'exposure_fraction']}"
        )
