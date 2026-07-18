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
Statistical analysis for coordinate calculations.

Analysis methods for coordinate calculations with statistical computations
and support for memory-mapped arrays and structural mobility analysis.
"""

from typing import List

import numpy as np

from ..helper.calculator_stat_helper import CalculatorStatHelper

COORDINATES_PER_ATOM = 3


class CoordinatesCalculatorAnalysis:
    """
    Analysis methods for coordinate calculation statistics and metrics.

    Provides statistical analysis capabilities for coordinate data including
    structural variability, mobility analysis, and geometric statistics
    with memory-mapped array support.
    """

    #: Metrics whose pooled value can be accumulated over frame blocks, and the
    #: reduction that backs each. rmsf and cv are derived from these; mad needs a
    #: whole column at once and is pooled one feature block at a time instead.
    POOLED_STREAMING_METRICS = {
        "mean": "mean",
        "std": "std",
        "variance": "var",
        "min": "min",
        "max": "max",
        "range": "ptp",
    }

    def __init__(self, use_memmap: bool = False, chunk_size: int = 2000) -> None:
        """
        Initialize coordinates analysis with chunking configuration.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, default=2000
            Number of frames to process per chunk for memory-mapped arrays

        Examples
        --------
        >>> # Default chunking
        >>> analysis = CoordinatesCalculatorAnalysis()

        >>> # Custom chunk size for large datasets
        >>> analysis = CoordinatesCalculatorAnalysis(chunk_size=1000)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

    # === COORDINATE-BASED STATISTICS ===
    def compute_mean(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute mean coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Mean coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.mean, self.chunk_size, self.use_memmap
        )

    def compute_std(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation of coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Standard deviation for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.std, self.chunk_size, self.use_memmap
        )

    def compute_min(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute minimum coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Minimum coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.min, self.chunk_size, self.use_memmap
        )

    def compute_max(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute maximum coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Maximum coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.max, self.chunk_size, self.use_memmap
        )

    def compute_median(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute median coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Median coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.median, self.chunk_size, self.use_memmap
        )

    def compute_variance(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute variance of coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Variance for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.var, self.chunk_size, self.use_memmap
        )

    def compute_range(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute range (peak-to-peak) of coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Range (max - min) for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.ptp, self.chunk_size, self.use_memmap
        )

    def compute_mad(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute median absolute deviation for each coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            MAD values per coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates,
            lambda x, axis: np.median(
                np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis
            ),
            self.chunk_size,
            self.use_memmap,
        )

    # === FRAME-BASED STATISTICS ===
    def coordinates_per_frame_mean(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute mean coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Mean coordinate across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.mean
        )

    def coordinates_per_frame_std(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation of coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Standard deviation across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.std
        )

    def coordinates_per_frame_min(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute minimum coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Minimum coordinate across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.min
        )

    def coordinates_per_frame_max(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute maximum coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Maximum coordinate across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.max
        )

    def coordinates_per_frame_range(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute range of coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Range (max - min) across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.ptp
        )

    def compute_cv(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute coefficient of variation for each coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            CV values per coordinate with shape (n_coordinates,)
        """
        return self._cv_from(
            self.compute_mean(coordinates), self.compute_std(coordinates)
        )

    def _cv_from(self, mean_vals: np.ndarray, std_vals: np.ndarray) -> np.ndarray:
        """
        Combine a mean and a standard deviation into a coefficient of variation.

        Kept separate so the pooled path derives CV from pooled inputs through
        the same expression. Coordinates are signed, so the mean is taken as an
        absolute value before dividing.

        Parameters
        ----------
        mean_vals : numpy.ndarray
            Mean per coordinate
        std_vals : numpy.ndarray
            Standard deviation per coordinate

        Returns
        -------
        numpy.ndarray
            CV values per coordinate
        """
        return std_vals / (np.abs(mean_vals) + 1e-10)

    def compute_rmsf(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute root mean square fluctuation per atom.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            RMSF values expanded to coordinate format with shape (n_coordinates,)

        Notes
        -----
        The mean of the squared deviation of an atom is the sum of the variances
        of its x, y and z columns, so RMSF is the square root of that sum. Taking
        that route lets the variance stream over frame blocks; forming the
        (n_frames, n_atoms, 3) deviations directly would materialise a full copy
        of the trajectory.
        """
        variance_per_coordinate = CalculatorStatHelper.compute_reduction_per_feature(
            coordinates, "var",
            chunk_size=self.chunk_size,
            use_memmap=self.use_memmap,
        )
        per_atom = np.sqrt(
            variance_per_coordinate.reshape(-1, COORDINATES_PER_ATOM).sum(axis=1)
        )
        # Same RMSF for x, y and z of the same atom
        return np.repeat(per_atom, COORDINATES_PER_ATOM)

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, coordinates: np.ndarray, threshold: float = 1.0, lag_time: int = 10) -> np.ndarray:
        """
        Compute transitions within lag time for coordinates.

        Parameters
        ----------
        coordinates : np.ndarray or np.memmap
            Coordinate array with shape (n_frames, n_coordinates)
        threshold : float, default=1.0
            Position threshold for transition detection in Angstroms
        lag_time : int, default=10
            Number of frames to look ahead for transitions

        Returns
        -------
        np.ndarray
            Transition counts for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            coordinates, threshold, lag_time, self.chunk_size, self.use_memmap
        )

    def compute_transitions_window(self, coordinates: np.ndarray, threshold: float = 1.0, window_size: int = 10) -> np.ndarray:
        """
        Compute transitions within window for coordinates.

        Parameters
        ----------
        coordinates : np.ndarray or np.memmap
            Coordinate array with shape (n_frames, n_coordinates)
        threshold : float, default=1.0
            Position threshold for transition detection in Angstroms
        window_size : int, default=10
            Size of sliding window for transition analysis

        Returns
        -------
        np.ndarray
            Transition counts for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_transitions_within_window(
            coordinates, threshold, window_size, self.chunk_size, self.use_memmap
        )

    def compute_stability(self, coordinates: np.ndarray, threshold: float = 1.0, window_size: int = 10) -> np.ndarray:
        """
        Compute stability analysis for coordinates.

        Parameters
        ----------
        coordinates : np.ndarray or np.memmap
            Coordinate array with shape (n_frames, n_coordinates)
        threshold : float, default=1.0
            Position threshold for stability detection in Angstroms
        window_size : int, default=10
            Size of analysis window

        Returns
        -------
        np.ndarray
            Stability scores for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_stability(
            coordinates, threshold, window_size, self.chunk_size, self.use_memmap
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, coordinates1: np.ndarray, coordinates2: np.ndarray, preprocessing_func = None) -> np.ndarray:
        """
        Compute differences between two coordinate datasets.

        Parameters
        ----------
        coordinates1 : np.ndarray or np.memmap
            First coordinate array with shape (n_frames, n_coordinates)
        coordinates2 : np.ndarray or np.memmap
            Second coordinate array with shape (n_frames, n_coordinates)
        preprocessing_func : callable, optional
            Function to apply to each dataset before comparison

        Returns
        -------
        np.ndarray
            Difference values with shape (n_frames, n_coordinates)
        """
        return CalculatorStatHelper.compute_differences(
            coordinates1, coordinates2, self.chunk_size, self.use_memmap, preprocessing_func
        )

    def compute_pooled_metric_values(
        self,
        segments: List[np.ndarray],
        metric: str,
        transition_threshold: float = 1.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute pooled metric values across segments.

        Parameters
        ----------
        segments : list
            List of coordinate arrays
        metric : str
            Metric name
        transition_threshold : float, default=1.0
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
            Pooled metric values per coordinate
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
        return self._pooled_metric_values(segments, metric)

    def _pooled_metric_values(self, segments: List[np.ndarray], metric: str) -> np.ndarray:
        """
        Compute a pooled metric without materialising the pooled array.

        Streamable metrics accumulate over the frames of every segment. rmsf is
        the square root of the summed x/y/z variances, so it rides on the same
        variance accumulator. Anything that needs a whole column at once (mad)
        pools one feature block at a time.

        Parameters
        ----------
        segments : list
            List of coordinate arrays to pool along the frame axis
        metric : str
            Metric name

        Returns
        -------
        numpy.ndarray
            Pooled metric values per coordinate
        """
        if metric == "cv":
            return self._cv_from(
                self._pooled_metric_values(segments, "mean"),
                self._pooled_metric_values(segments, "std"),
            )
        if metric == "rmsf":
            variance = CalculatorStatHelper.compute_pooled_reduction_per_feature(
                segments, "var", self.chunk_size, self.use_memmap
            )
            per_atom = np.sqrt(
                variance.reshape(-1, COORDINATES_PER_ATOM).sum(axis=1)
            )
            return np.repeat(per_atom, COORDINATES_PER_ATOM)
        reduction = self.POOLED_STREAMING_METRICS.get(metric)
        if reduction is not None:
            return CalculatorStatHelper.compute_pooled_reduction_per_feature(
                segments, reduction, self.chunk_size, self.use_memmap
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
            Pooled coordinate array
        metric : str
            Metric name

        Returns
        -------
        numpy.ndarray
            Metric values per coordinate
        """
        metrics = {
            "std": self.compute_std,
            "variance": self.compute_variance,
            "min": self.compute_min,
            "mad": self.compute_mad,
            "mean": self.compute_mean,
            "max": self.compute_max,
            "cv": self.compute_cv,
            "range": self.compute_range,
            "rmsf": self.compute_rmsf,
        }
        if metric in metrics:
            return metrics[metric](pooled)
        raise ValueError(
            f"Unknown metric: {metric}. Supported: {list(metrics.keys()) + ['transitions', 'stability']}"
        )
