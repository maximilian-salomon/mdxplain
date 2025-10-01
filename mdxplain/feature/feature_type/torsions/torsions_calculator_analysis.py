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
        sin_mean = CalculatorStatHelper.compute_func_per_feature(
            np.sin(np.radians(torsion_data)), np.mean,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )
        cos_mean = CalculatorStatHelper.compute_func_per_feature(
            np.cos(np.radians(torsion_data)), np.mean,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )
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
        circular_var = self.compute_variance(torsion_data)
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
        sin_mean = CalculatorStatHelper.compute_func_per_feature(
            np.sin(np.radians(torsion_data)), np.mean,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )
        cos_mean = CalculatorStatHelper.compute_func_per_feature(
            np.cos(np.radians(torsion_data)), np.mean,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )
        
        mean_resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
        return 1.0 - mean_resultant_length

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
        return CalculatorStatHelper.compute_func_per_feature(
            torsion_data, np.min,
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
        return CalculatorStatHelper.compute_func_per_feature(
            torsion_data, np.max,
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
            return np.median(np.abs(data - np.median(data)), axis=axis)
        
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
        def circular_range(angles, axis=0):
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
        
        return CalculatorStatHelper.compute_func_per_feature(
            torsion_data, circular_range,
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap
        )

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
            sin_mean = CalculatorStatHelper.compute_func_per_feature(
                np.sin(np.radians(data)), np.mean, **kwargs
            )
            cos_mean = CalculatorStatHelper.compute_func_per_feature(
                np.cos(np.radians(data)), np.mean, **kwargs
            )
            return np.degrees(np.arctan2(sin_mean, cos_mean))
        
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
        circular_mean = self.compute_mean(torsion_data)
        circular_std = self.compute_std(torsion_data)
        return circular_std / (np.abs(circular_mean) + 1e-10)
