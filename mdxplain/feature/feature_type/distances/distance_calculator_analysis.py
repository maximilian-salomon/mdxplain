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

from typing import Callable, List, Optional, Tuple
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

    #: Metrics whose pooled value can be accumulated over frame blocks, and the
    #: reduction that backs each. Anything absent here (mad) needs a whole column
    #: at once and is pooled one feature block at a time instead.
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
        Initialize distance analysis with chunking configuration.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, default=2000
            Number of frames to process per chunk for memory-mapped arrays

        Examples
        --------
        >>> # Default chunking
        >>> analysis = DistanceCalculatorAnalysis()

        >>> # Custom chunk size
        >>> analysis = DistanceCalculatorAnalysis(chunk_size=1000)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

    # === PAIR-BASED STATISTICS ===
    def compute_mean(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute mean distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Mean distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.mean, self.chunk_size, self.use_memmap
        )

    def compute_std(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation of distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Standard deviation for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.std, self.chunk_size, self.use_memmap
        )

    def compute_min(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute minimum distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Minimum distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.min, self.chunk_size, self.use_memmap
        )

    def compute_max(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute maximum distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Maximum distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.max, self.chunk_size, self.use_memmap
        )

    def compute_median(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute median distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Median distance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.median, self.chunk_size, self.use_memmap
        )

    def compute_variance(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute variance of distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Variance for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.var, self.chunk_size, self.use_memmap
        )

    def compute_range(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute range (peak-to-peak) of distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Range (max - min) for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances, np.ptp, self.chunk_size, self.use_memmap
        )

    def compute_q25(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute 25th percentile of distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            25th percentile for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances,
            lambda x, axis: np.percentile(x, 25, axis=axis),
            self.chunk_size,
            self.use_memmap,
        )

    def compute_q75(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute 75th percentile of distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            75th percentile for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            distances,
            lambda x, axis: np.percentile(x, 75, axis=axis),
            self.chunk_size,
            self.use_memmap,
        )

    def compute_iqr(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute interquartile range of distances per pair.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
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

    def compute_mad(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute median absolute deviation for each distance pair.

        MAD provides robust measure of variability less sensitive to outliers
        than standard deviation. Calculated as median of absolute deviations
        from the median.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
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

    def compute_cv(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute coefficient of variation for each distance pair.

        The coefficient of variation (CV) is the ratio of standard deviation
        to the mean, providing a normalized measure of variability that allows
        comparison of variability across different scales.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            CV values per distance pair with shape (n_pairs,)

        Examples
        --------
        >>> # Compute CV for all distance pairs
        >>> cv_values = analysis.compute_cv(distance_data)
        >>> highly_variable = cv_values > 0.5  # Pairs with high variability
        """
        return self._cv_from(self.compute_mean(distances), self.compute_std(distances))

    def _cv_from(self, mean_vals: np.ndarray, std_vals: np.ndarray) -> np.ndarray:
        """
        Combine a mean and a standard deviation into a coefficient of variation.

        Kept separate so the pooled path derives CV from pooled inputs through
        the same expression. Distances are non-negative, so the mean is used as-is.

        Parameters
        ----------
        mean_vals : numpy.ndarray
            Mean per distance pair
        std_vals : numpy.ndarray
            Standard deviation per distance pair

        Returns
        -------
        numpy.ndarray
            CV values per distance pair
        """
        return std_vals / (mean_vals + 1e-10)

    # === FRAME-BASED STATISTICS ===
    def distances_per_frame_mean(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute mean distances per frame.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Mean distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, self.use_memmap, np.mean
        )

    def distances_per_frame_std(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation of distances per frame.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Standard deviation across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, self.use_memmap, np.std
        )

    def distances_per_frame_min(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute minimum distances per frame.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Minimum distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, self.use_memmap, np.min
        )

    def distances_per_frame_max(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute maximum distances per frame.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Maximum distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, self.use_memmap, np.max
        )

    def distances_per_frame_median(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute median distances per frame.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Median distance across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, self.use_memmap, np.median
        )

    def distances_per_frame_range(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute range of distances per frame.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Range (max - min) across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, self.use_memmap, np.ptp
        )

    def distances_per_frame_sum(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute sum of distances per frame.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)

        Returns
        -------
        np.ndarray
            Sum of distances across all pairs for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            distances, self.chunk_size, self.use_memmap, np.sum
        )

    # === PER-RESIDUE ANALYSIS (reduces over each residue's real partners) ===
    def _per_residue_metric(
        self,
        distances: np.ndarray,
        metric: str,
        pairs: Optional[List[Tuple[int, int]]] = None,
        n_residues: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reduce condensed distances to one value per residue over its partners.

        The residue pair of each condensed column comes from ``pairs``; the
        service passes the real pairs from the feature metadata. When they are
        absent the columns are assumed to be a full upper triangle, which is the
        only inference possible from the column count alone.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        metric : str
            Per-residue metric name
        pairs : list of tuple, optional
            Residue index pair for each condensed column, in column order
        n_residues : int, optional
            Number of residues; inferred with pairs when omitted

        Returns
        -------
        np.ndarray
            Metric value per residue with shape (n_residues,)
        """
        return CalculatorStatHelper.compute_per_residue_reduction(
            distances, pairs, n_residues, metric, self.chunk_size, self.use_memmap
        )

    def compute_per_residue_mean(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the mean distance from each residue to its partners.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Mean distance for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "mean", pairs, n_residues)

    def compute_per_residue_std(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the standard deviation of each residue's partner distances.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Standard deviation for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "std", pairs, n_residues)

    def compute_per_residue_min(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute each residue's closest partner distance.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Minimum distance for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "min", pairs, n_residues)

    def compute_per_residue_max(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute each residue's farthest partner distance.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Maximum distance for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "max", pairs, n_residues)

    def compute_per_residue_median(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the median of each residue's partner distances.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Median distance for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "median", pairs, n_residues)

    def compute_per_residue_sum(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the summed distance from each residue to its partners.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Sum of distances for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "sum", pairs, n_residues)

    def compute_per_residue_variance(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the variance of each residue's partner distances.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Variance for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "variance", pairs, n_residues)

    def compute_per_residue_range(
        self, distances: np.ndarray, pairs=None, n_residues=None
    ) -> np.ndarray:
        """
        Compute the range of each residue's partner distances.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array in condensed format (n_frames, n_pairs)
        pairs : list of tuple, optional
            Residue index pair for each condensed column
        n_residues : int, optional
            Number of residues

        Returns
        -------
        np.ndarray
            Range (max - min) for each residue with shape (n_residues,)
        """
        return self._per_residue_metric(distances, "range", pairs, n_residues)

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, distances: np.ndarray, threshold: float = 2.0, lag_time: int = 10) -> np.ndarray:
        """
        Compute transitions within lag time.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)
        threshold : float, default=2.0
            Distance threshold for transition detection in Angstroms
        lag_time : int, default=10
            Number of frames to look ahead for transitions

        Returns
        -------
        np.ndarray
            Transition counts for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            distances, threshold, lag_time, self.chunk_size, self.use_memmap
        )

    def compute_transitions_window(self, distances: np.ndarray, threshold: float = 2.0, window_size: int = 10) -> np.ndarray:
        """
        Compute transitions within window.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)
        threshold : float, default=2.0
            Distance threshold for transition detection in Angstroms
        window_size : int, default=10
            Size of sliding window for transition analysis

        Returns
        -------
        np.ndarray
            Transition counts for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_transitions_within_window(
            distances, threshold, window_size, self.chunk_size, self.use_memmap
        )

    def compute_stability(
        self, distances: np.ndarray, threshold: float = 2.0, window_size: int = 10, mode: str = "window"
    ) -> np.ndarray:
        """
        Compute stability analysis.

        Parameters
        ----------
        distances : np.ndarray or np.memmap
            Distance array with shape (n_frames, n_pairs)
        threshold : float, default=2.0
            Distance threshold for stability detection in Angstroms
        window_size : int, default=10
            Size of analysis window
        mode : str, default="window"
            Analysis mode: "window" or "lagtime"

        Returns
        -------
        np.ndarray
            Stability scores for each pair with shape (n_pairs,)
        """
        return CalculatorStatHelper.compute_stability(
            distances, threshold, window_size, self.chunk_size, self.use_memmap, mode
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, distances1: np.ndarray, distances2: np.ndarray, preprocessing_func: Optional[Callable] = None) -> np.ndarray:
        """
        Compute differences between two distance datasets.

        Parameters
        ----------
        distances1 : np.ndarray or np.memmap
            First distance array with shape (n_frames, n_pairs)
        distances2 : np.ndarray or np.memmap
            Second distance array with shape (n_frames, n_pairs)
        preprocessing_func : callable, optional
            Function to apply to each dataset before comparison

        Returns
        -------
        np.ndarray
            Difference values with shape (n_frames, n_pairs)
        """
        return CalculatorStatHelper.compute_differences(
            distances1, distances2, self.chunk_size, self.use_memmap, preprocessing_func
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
            List of distance arrays
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
            Pooled metric values per distance pair
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

        Streamable metrics accumulate over the frames of every segment. Anything
        that needs a whole column at once (mad) pools one feature block at a time.

        Parameters
        ----------
        segments : list
            List of distance arrays to pool along the frame axis
        metric : str
            Metric name

        Returns
        -------
        numpy.ndarray
            Pooled metric values per distance pair
        """
        if metric == "cv":
            return self._cv_from(
                self._pooled_metric_values(segments, "mean"),
                self._pooled_metric_values(segments, "std"),
            )
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
            Pooled distance array
        metric : str
            Metric name

        Returns
        -------
        numpy.ndarray
            Metric values per distance pair
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
        }
        if metric in metrics:
            return metrics[metric](pooled)
        raise ValueError(f"Unknown metric: {metric}. Supported: {list(metrics.keys()) + ['transitions', 'stability']}")
