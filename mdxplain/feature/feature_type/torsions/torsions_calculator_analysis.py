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
Torsion angles calculator analysis for molecular dynamics trajectory analysis.

Analysis utilities for torsion angle data including conformational dynamics,
angular distributions, and circular statistics with complete per-feature and per-frame metrics.
"""

from typing import List, Optional, Tuple

import numpy as np

from ..helper.calculator_stat_helper import CalculatorStatHelper


class TorsionsCalculatorAnalysis:
    """
    Analysis utilities for torsion angle data from MD trajectories.

    Provides statistical analysis methods for torsion angle data including
    conformational dynamics, circular statistics, and angular distributions
    with complete per-feature and per-frame metrics.

    Examples
    --------
    >>> analysis = TorsionsCalculatorAnalysis()
    >>> circular_mean = analysis.compute_circular_mean(torsion_data)
    >>> transitions = analysis.compute_transitions_lagtime(torsion_data, threshold=30.0)
    """

    def __init__(self, use_memmap: bool = False, chunk_size: int = 2000) -> None:
        """
        Initialize torsion analysis with configuration parameters.

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
        >>> analysis = TorsionsCalculatorAnalysis()

        >>> # With memory mapping
        >>> analysis = TorsionsCalculatorAnalysis(use_memmap=True, chunk_size=1000)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

    # ===== PER-FEATURE METHODS (per angle) =====

    def _circular_means(self, torsion_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean sine and mean cosine of each torsion angle.

        Both circular mean and circular variance are derived from this pair, so
        it is computed in one place. The trigonometry runs inside each chunk:
        transforming the whole array up front would allocate a full copy of the
        data before any chunking could take effect, which for a large trajectory
        is the entire dataset.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Mean sine and mean cosine for each torsion angle
        """
        sin_mean = CalculatorStatHelper.compute_reduction_per_feature(
            torsion_data, "mean",
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            transform=lambda block: np.sin(np.radians(block)),
        )
        cos_mean = CalculatorStatHelper.compute_reduction_per_feature(
            torsion_data, "mean",
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            transform=lambda block: np.cos(np.radians(block)),
        )
        return sin_mean, cos_mean

    def compute_mean(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute circular mean for each torsion angle.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Circular mean angle for each torsion in degrees

        Examples
        --------
        >>> mean_angles = analysis.compute_mean(torsion_data)
        """
        sin_mean, cos_mean = self._circular_means(torsion_data)
        return np.degrees(np.arctan2(sin_mean, cos_mean))

    def compute_std(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute circular standard deviation for each torsion angle.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Circular standard deviation for each torsion angle in degrees

        Examples
        --------
        >>> std_angles = analysis.compute_std(torsion_data)
        """
        return self._circular_std_from(self.compute_variance(torsion_data))

    @staticmethod
    def _circular_variance_from(
        sin_mean: np.ndarray, cos_mean: np.ndarray
    ) -> np.ndarray:
        """
        Derive the circular variance from a mean sine and cosine.

        The mean resultant length is the magnitude of an average of unit vectors
        and therefore cannot exceed 1. Rounding can push it a few ulp past 1,
        which would make the variance negative and turn the square root in
        _circular_std_from into NaN — a constant angle is exactly the case that
        lands on that boundary. Clipping restores the mathematical range instead
        of masking it.

        Parameters
        ----------
        sin_mean : numpy.ndarray
            Mean sine per torsion angle
        cos_mean : numpy.ndarray
            Mean cosine per torsion angle

        Returns
        -------
        numpy.ndarray
            Circular variance per torsion angle, between 0 and 1
        """
        mean_resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
        return np.clip(1.0 - mean_resultant_length, 0.0, 1.0)

    @staticmethod
    def _circular_std_from(circular_var: np.ndarray) -> np.ndarray:
        """
        Derive the circular standard deviation from the circular variance.

        Parameters
        ----------
        circular_var : numpy.ndarray
            Circular variance per torsion angle, between 0 and 1

        Returns
        -------
        numpy.ndarray
            Circular standard deviation per torsion angle in degrees
        """
        return np.degrees(np.sqrt(-2 * np.log(1 - circular_var)))

    def compute_variance(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute circular variance for each torsion angle.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Circular variance for each torsion angle (0-1 scale)

        Examples
        --------
        >>> var_angles = analysis.compute_variance(torsion_data)
        """
        return self._circular_variance_from(*self._circular_means(torsion_data))

    def compute_min(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute minimum angle for each torsion.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles)

        Returns
        -------
        numpy.ndarray
            Minimum angle for each torsion

        Examples
        --------
        >>> min_angles = analysis.compute_min(torsion_data)
        """
        return CalculatorStatHelper.compute_reduction_per_feature(
            torsion_data, "min",
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_max(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute maximum angle for each torsion.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles)

        Returns
        -------
        numpy.ndarray
            Maximum angle for each torsion

        Examples
        --------
        >>> max_angles = analysis.compute_max(torsion_data)
        """
        return CalculatorStatHelper.compute_reduction_per_feature(
            torsion_data, "max",
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_mad(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute median absolute deviation for each torsion angle.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles)

        Returns
        -------
        numpy.ndarray
            Median absolute deviation for each torsion angle

        Examples
        --------
        >>> mad_angles = analysis.compute_mad(torsion_data)
        """
        def mad_func(data, axis=0):
            median = np.median(data, axis=axis, keepdims=True)
            return np.median(np.abs(data - median), axis=axis)

        return CalculatorStatHelper.compute_func_per_feature(
            torsion_data, mad_func,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_range(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute angular range for each torsion considering periodicity.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Angular range for each torsion (0-180 degrees)

        Examples
        --------
        >>> range_angles = analysis.compute_range(torsion_data)

        Notes
        -----
        Uses circular statistics to handle periodicity (-180° to 180°).
        Range is computed as the minimum angular distance that contains all data points.
        """
        simple_range = self.compute_max(torsion_data) - self.compute_min(torsion_data)

        # For torsion angles (-180° to 180°), if range > 180°,
        # the actual circular range is smaller going the other way
        return np.where(simple_range > 180.0, 360.0 - simple_range, simple_range)

    # ===== PER-FRAME METHODS (per time step) =====

    def compute_mean_per_frame(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute circular mean angle per frame across all torsions.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Circular mean angle per frame in degrees

        Examples
        --------
        >>> mean_per_frame = analysis.compute_mean_per_frame(torsion_data)
        """
        def circular_mean_frame(data, axis=1):
            sin_data = np.sin(np.radians(data))
            cos_data = np.cos(np.radians(data))
            sin_mean = np.mean(sin_data, axis=axis)
            cos_mean = np.mean(cos_data, axis=axis)
            return np.degrees(np.arctan2(sin_mean, cos_mean))
        
        return CalculatorStatHelper.compute_func_per_frame(
            torsion_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=circular_mean_frame
        )

    def compute_std_per_frame(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute circular standard deviation per frame across all torsions.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Circular standard deviation per frame in degrees

        Examples
        --------
        >>> std_per_frame = analysis.compute_std_per_frame(torsion_data)
        """
        def circular_std_frame(data, axis=1):
            sin_data = np.sin(np.radians(data))
            cos_data = np.cos(np.radians(data))
            sin_mean = np.mean(sin_data, axis=axis)
            cos_mean = np.mean(cos_data, axis=axis)
            
            mean_resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
            circular_var = 1.0 - mean_resultant_length
            return np.degrees(np.sqrt(-2 * np.log(1 - circular_var)))
        
        return CalculatorStatHelper.compute_func_per_frame(
            torsion_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=circular_std_frame
        )

    def compute_variance_per_frame(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute circular variance per frame across all torsions.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Circular variance per frame (0-1 scale)

        Examples
        --------
        >>> var_per_frame = analysis.compute_variance_per_frame(torsion_data)
        """
        def circular_var_frame(data, axis=1):
            sin_data = np.sin(np.radians(data))
            cos_data = np.cos(np.radians(data))
            sin_mean = np.mean(sin_data, axis=axis)
            cos_mean = np.mean(cos_data, axis=axis)
            
            mean_resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
            return 1.0 - mean_resultant_length
        
        return CalculatorStatHelper.compute_func_per_frame(
            torsion_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=circular_var_frame
        )

    def compute_min_per_frame(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute minimum angle per frame across all torsions.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles)

        Returns
        -------
        numpy.ndarray
            Minimum angle per frame

        Examples
        --------
        >>> min_per_frame = analysis.compute_min_per_frame(torsion_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            torsion_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.min
        )

    def compute_max_per_frame(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute maximum angle per frame across all torsions.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles)

        Returns
        -------
        numpy.ndarray
            Maximum angle per frame

        Examples
        --------
        >>> max_per_frame = analysis.compute_max_per_frame(torsion_data)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            torsion_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=np.max
        )

    def compute_mad_per_frame(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute median absolute deviation per frame across all torsions.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles)

        Returns
        -------
        numpy.ndarray
            Median absolute deviation per frame

        Examples
        --------
        >>> mad_per_frame = analysis.compute_mad_per_frame(torsion_data)
        """
        def mad_per_frame(data, axis=1):
            median = np.median(data, axis=axis, keepdims=True)
            return np.median(np.abs(data - median), axis=axis)
        
        return CalculatorStatHelper.compute_func_per_frame(
            torsion_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=mad_per_frame
        )

    def compute_range_per_frame(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute angular range per frame across all torsions with periodicity.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Angular range per frame

        Examples
        --------
        >>> range_per_frame = analysis.compute_range_per_frame(torsion_data)

        Notes
        -----
        Uses circular statistics for proper angular range calculation.
        """
        def circular_range_frame(angles, axis=1):
            max_angles = np.max(angles, axis=axis)
            min_angles = np.min(angles, axis=axis)
            
            # Simple range calculation
            simple_range = max_angles - min_angles
            
            # For torsion angles (-180° to 180°), if range > 180°, 
            # the actual circular range is smaller going the other way
            corrected_range = np.where(simple_range > 180.0, 
                                     360.0 - simple_range, 
                                     simple_range)
            return corrected_range
        
        return CalculatorStatHelper.compute_func_per_frame(
            torsion_data,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            func=circular_range_frame
        )

    # ===== DIFFERENCES/COMPARISON METHODS =====

    def compute_differences(self, torsion_data: np.ndarray, frame_1: int = 0, frame_2: int = -1) -> np.ndarray:
        """
        Compute angle differences between two frames with periodic boundary handling.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles)
        frame_1 : int, default=0
            First frame index
        frame_2 : int, default=-1
            Second frame index (-1 for last frame)

        Returns
        -------
        numpy.ndarray
            Angle differences between frames with proper periodic handling

        Notes
        -----
        Handles periodic boundary conditions for angles (-180 to 180 degrees).

        Examples
        --------
        >>> differences = analysis.compute_differences(torsion_data, 0, -1)
        """
        if frame_2 == -1:
            frame_2 = torsion_data.shape[0] - 1
        
        diff = torsion_data[frame_2] - torsion_data[frame_1]
        
        # Handle periodic boundaries (-180 to 180)
        diff = np.where(diff > 180, diff - 360, diff)
        diff = np.where(diff < -180, diff + 360, diff)
        
        return diff

    def compute_differences_mean(self, torsion_data_1: np.ndarray, torsion_data_2: np.ndarray) -> np.ndarray:
        """
        Compute differences between circular means of two datasets.

        Parameters
        ----------
        torsion_data_1 : numpy.ndarray
            First torsion dataset
        torsion_data_2 : numpy.ndarray
            Second torsion dataset

        Returns
        -------
        numpy.ndarray
            Circular mean angle differences between datasets

        Examples
        --------
        >>> diff_means = analysis.compute_differences_mean(torsion_1, torsion_2)
        """
        def circular_mean_preprocessing(data, **kwargs):
            """Reduce a dataset to its circular mean per angle."""
            # kwargs carry only chunk_size, which already equals self.chunk_size;
            # going through compute_mean also keeps use_memmap, which
            # compute_differences does not forward to a custom preprocessor.
            return self.compute_mean(data)
        
        return CalculatorStatHelper.compute_differences(
            torsion_data_1, torsion_data_2,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
            preprocessing_func=circular_mean_preprocessing
        )

    # ===== TRANSITIONS METHODS =====

    def compute_transitions_lagtime(self, torsion_data: np.ndarray, threshold: float = 30.0, lag_time: int = 1) -> np.ndarray:
        """
        Compute transitions with lag time for each torsion angle with periodic boundaries.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees
        threshold : float, default=30.0
            Threshold for detecting transitions (in degrees)
        lag_time : int, default=1
            Number of frames to look ahead

        Returns
        -------
        numpy.ndarray
            Transition counts per torsion angle

        Examples
        --------
        >>> transitions = analysis.compute_transitions_lagtime(torsion_data, 30.0, 10)
        """
        def angular_difference_check(data, **kwargs):
            n_frames = data.shape[0]
            if lag_time >= n_frames:
                return np.zeros(data.shape[1], dtype=np.float32)
            
            # Compute angular differences with periodic boundaries
            diff = data[lag_time:] - data[:-lag_time]
            # Handle periodic boundaries (-180 to 180)
            diff = np.where(diff > 180, diff - 360, diff)
            diff = np.where(diff < -180, diff + 360, diff)
            
            # Count transitions exceeding threshold
            transitions = (np.abs(diff) > threshold).sum(axis=0)
            return transitions.astype(np.float32)
        
        return CalculatorStatHelper.compute_func_per_feature(
            torsion_data, angular_difference_check,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_transitions_window(self, torsion_data: np.ndarray, threshold: float = 30.0, window_size: int = 10) -> np.ndarray:
        """
        Compute transitions within sliding window for each torsion angle with periodic boundaries.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees
        threshold : float, default=30.0
            Threshold for detecting transitions (in degrees)
        window_size : int, default=10
            Size of sliding window

        Returns
        -------
        numpy.ndarray
            Transition counts per torsion angle

        Examples
        --------
        >>> transitions = analysis.compute_transitions_window(torsion_data, 30.0, 10)
        """
        def angular_window_transitions(data, **kwargs):
            n_frames = data.shape[0]
            transitions = np.zeros(data.shape[1], dtype=np.float32)
            
            for i in range(n_frames - window_size + 1):
                window_data = data[i:i + window_size]
                # Check transitions within window with periodic boundaries
                diff = window_data[1:] - window_data[:-1]
                # Handle periodic boundaries
                diff = np.where(diff > 180, diff - 360, diff)
                diff = np.where(diff < -180, diff + 360, diff)
                
                # Count if any transition occurs within this window
                window_transitions = (np.abs(diff) > threshold).any(axis=0)
                transitions += window_transitions.astype(np.float32)
            
            return transitions
        
        return CalculatorStatHelper.compute_func_per_feature(
            torsion_data, angular_window_transitions,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

    def compute_stability(self, torsion_data: np.ndarray, threshold: float = 30.0, window_size: int = 10, mode: str = "lagtime") -> np.ndarray:
        """
        Compute stability (inverse of transition rate) for each torsion angle with periodic boundaries.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees
        threshold : float, default=30.0
            Threshold for stability detection (in degrees)
        window_size : int, default=10
            Window size for calculation
        mode : str, default='lagtime'
            Calculation mode ('lagtime' or 'window')

        Returns
        -------
        numpy.ndarray
            Stability values per torsion angle (0=unstable, 1=stable)

        Examples
        --------
        >>> stability = analysis.compute_stability(torsion_data, 30.0, 10, 'window')
        """
        if mode == "lagtime":
            transitions = self.compute_transitions_lagtime(torsion_data, threshold, window_size)
        elif mode == "window":
            transitions = self.compute_transitions_window(torsion_data, threshold, window_size)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Use 'lagtime' or 'window'")
        
        # Compute stability as inverse of transition frequency
        max_transitions = torsion_data.shape[0] - 1 if mode == "lagtime" else torsion_data.shape[0] - window_size + 1
        stability = 1.0 - (transitions / max(max_transitions, 1))
        return np.clip(stability, 0.0, 1.0)

    def compute_cv(self, torsion_data: np.ndarray) -> np.ndarray:
        """
        Compute coefficient of variation for each torsion angle.

        Parameters
        ----------
        torsion_data : numpy.ndarray
            Torsion angles array with shape (n_frames, n_angles) in degrees

        Returns
        -------
        numpy.ndarray
            Coefficient of variation for each torsion angle

        Notes
        -----
        CV = circular_standard_deviation / abs(circular_mean)
        Uses circular statistics for proper angular data handling.

        Examples
        --------
        >>> cv_angles = analysis.compute_cv(torsion_data)
        """
        return self._cv_from(
            self.compute_mean(torsion_data), self.compute_std(torsion_data)
        )

    def _cv_from(self, mean_vals: np.ndarray, std_vals: np.ndarray) -> np.ndarray:
        """
        Combine a circular mean and circular deviation into a coefficient of variation.

        Kept separate so the pooled path derives CV from pooled inputs through
        the same expression. A circular mean is signed, so it is taken as an
        absolute value before dividing.

        Parameters
        ----------
        mean_vals : numpy.ndarray
            Circular mean per torsion angle in degrees
        std_vals : numpy.ndarray
            Circular standard deviation per torsion angle in degrees

        Returns
        -------
        numpy.ndarray
            CV values per torsion angle
        """
        return std_vals / (np.abs(mean_vals) + 1e-10)

    def compute_pooled_metric_values(
        self,
        segments: List[np.ndarray],
        metric: str,
        transition_threshold: float = 30.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute pooled metric values across segments.

        Parameters
        ----------
        segments : list
            List of torsion arrays
        metric : str
            Metric name
        transition_threshold : float, default=30.0
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
            Pooled metric values per torsion angle
        """
        if not segments:
            return np.array([])
        if metric in ("transitions", "stability"):
            window = lag_time if transition_mode == "lagtime" else window_size
            return self._pooled_transitions_or_stability(
                segments, metric, transition_threshold, window, transition_mode
            )
        return self._pooled_metric_values(segments, metric)

    def _pooled_transitions_or_stability(
        self,
        segments: List[np.ndarray],
        metric: str,
        threshold: float,
        window: int,
        mode: str,
    ) -> np.ndarray:
        """
        Return pooled transition counts or the stability derived from them.

        Parameters
        ----------
        segments : list
            List of torsion arrays to pool along the frame axis
        metric : str
            Either 'transitions' or 'stability'
        threshold : float
            Transition threshold in degrees
        window : int
            Lag time (lagtime mode) or window size (window mode)
        mode : str
            Either 'lagtime' or 'window'

        Returns
        -------
        numpy.ndarray
            Transition counts, or stability per angle between 0 and 1
        """
        transitions, total_possible = self._pooled_transition_counts(
            segments, threshold, window, mode
        )
        if metric == "transitions":
            return transitions
        if total_possible == 0:
            return np.ones_like(transitions, dtype=float)
        return 1.0 - (transitions / total_possible)

    def _pooled_transition_counts(
        self, segments: List[np.ndarray], threshold: float, window: int, mode: str
    ) -> Tuple[np.ndarray, int]:
        """
        Sum angular transition counts across segments, boundary-safe.

        Each segment is counted on its own so a jump between one segment's last
        frame and the next segment's first frame is never mistaken for a
        transition. The per-segment count uses the periodicity-aware angular
        methods, so a step across the +/-180 degrees wrap counts as the small
        move it is, matching the non-pooled path.

        Parameters
        ----------
        segments : list
            List of torsion arrays to pool along the frame axis
        threshold : float
            Transition threshold in degrees
        window : int
            Lag time (lagtime mode) or window size (window mode)
        mode : str
            Either 'lagtime' or 'window'

        Returns
        -------
        Tuple[numpy.ndarray, int]
            Total transition counts per angle and the total possible transitions
        """
        total = None
        total_possible = 0
        for segment in segments:
            counts, max_possible = self._segment_transition_counts(
                segment, threshold, window, mode
            )
            if counts is None:
                continue
            total_possible += max_possible
            total = counts if total is None else total + counts
        if total is None:
            total = np.zeros(segments[0].shape[1], dtype=float)
        return total, total_possible

    def _segment_transition_counts(
        self, segment: np.ndarray, threshold: float, window: int, mode: str
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Count angular transitions for a single segment.

        Parameters
        ----------
        segment : numpy.ndarray
            One torsion array with shape (n_frames, n_angles)
        threshold : float
            Transition threshold in degrees
        window : int
            Lag time (lagtime mode) or window size (window mode)
        mode : str
            Either 'lagtime' or 'window'

        Returns
        -------
        Tuple[Optional[numpy.ndarray], int]
            Counts per angle and the possible transitions, or (None, 0) if the
            segment is too short to contain a transition
        """
        n_frames = segment.shape[0]
        max_possible = (
            n_frames - window if mode == "lagtime" else n_frames - window + 1
        )
        if max_possible <= 0:
            return None, 0
        if mode == "lagtime":
            return self.compute_transitions_lagtime(segment, threshold, window), max_possible
        return self.compute_transitions_window(segment, threshold, window), max_possible

    def _pooled_circular_means(
        self, segments: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the pooled mean sine and mean cosine of each torsion angle.

        Parameters
        ----------
        segments : list
            List of torsion arrays to pool along the frame axis

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Pooled mean sine and mean cosine for each torsion angle
        """
        sin_mean = CalculatorStatHelper.compute_pooled_reduction_per_feature(
            segments, "mean", self.chunk_size, self.use_memmap,
            transform=lambda block: np.sin(np.radians(block)),
        )
        cos_mean = CalculatorStatHelper.compute_pooled_reduction_per_feature(
            segments, "mean", self.chunk_size, self.use_memmap,
            transform=lambda block: np.cos(np.radians(block)),
        )
        return sin_mean, cos_mean

    def _pooled_metric_values(self, segments: List[np.ndarray], metric: str) -> np.ndarray:
        """
        Compute a pooled metric without materialising the pooled array.

        The circular statistics ride on pooled mean sine and cosine, which
        accumulate over frame blocks. min and max stream directly, range is
        derived from them, and mad needs a whole column at once so it pools one
        feature block at a time.

        Parameters
        ----------
        segments : list
            List of torsion arrays to pool along the frame axis
        metric : str
            Metric name

        Returns
        -------
        numpy.ndarray
            Pooled metric values per torsion angle
        """
        if metric in ("mean", "variance", "std", "cv"):
            return self._pooled_circular_metric(segments, metric)
        if metric in ("min", "max"):
            return CalculatorStatHelper.compute_pooled_reduction_per_feature(
                segments, metric, self.chunk_size, self.use_memmap
            )
        if metric == "range":
            simple_range = self._pooled_metric_values(
                segments, "max"
            ) - self._pooled_metric_values(segments, "min")
            return np.where(simple_range > 180.0, 360.0 - simple_range, simple_range)
        return CalculatorStatHelper.compute_pooled_func_per_feature(
            segments,
            lambda block: self._metric_from_pooled(block, metric),
            self.chunk_size,
            self.use_memmap,
        )

    def _pooled_circular_metric(
        self, segments: List[np.ndarray], metric: str
    ) -> np.ndarray:
        """
        Derive a pooled circular statistic from the pooled sine and cosine means.

        Parameters
        ----------
        segments : list
            List of torsion arrays to pool along the frame axis
        metric : str
            One of 'mean', 'variance', 'std', 'cv'

        Returns
        -------
        numpy.ndarray
            Pooled circular statistic per torsion angle
        """
        sin_mean, cos_mean = self._pooled_circular_means(segments)
        circular_mean = np.degrees(np.arctan2(sin_mean, cos_mean))
        if metric == "mean":
            return circular_mean
        circular_var = self._circular_variance_from(sin_mean, cos_mean)
        if metric == "variance":
            return circular_var
        circular_std = self._circular_std_from(circular_var)
        if metric == "std":
            return circular_std
        return self._cv_from(circular_mean, circular_std)

    def _metric_from_pooled(self, pooled: np.ndarray, metric: str) -> np.ndarray:
        """
        Compute metric values on pooled data.

        Parameters
        ----------
        pooled : np.ndarray
            Pooled torsion array
        metric : str
            Metric name

        Returns
        -------
        numpy.ndarray
            Metric values per torsion angle
        """
        metrics = {
            "std": self.compute_std,
            "variance": self.compute_variance,
            "mad": self.compute_mad,
            "mean": self.compute_mean,
            "cv": self.compute_cv,
            "range": self.compute_range,
            "min": self.compute_min,
            "max": self.compute_max,
        }
        if metric in metrics:
            return metrics[metric](pooled)
        raise ValueError(
            f"Unknown metric: {metric}. Supported: {list(metrics.keys()) + ['transitions', 'stability']}"
        )
