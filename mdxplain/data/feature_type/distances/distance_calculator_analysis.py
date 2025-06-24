import numpy as np

from ..helper.calculator_stat_helper import CalculatorStatHelper


class DistanceCalculatorAnalysis:
    """Analysis methods for distance calculations."""

    def __init__(self, chunk_size=None):
        """
        Initialize distance calculator analysis.

        Args:
            chunk_size: Size of chunks for processing large datasets
        """
        self.chunk_size = chunk_size

    # === PAIR-BASED STATISTICS ===
    def compute_mean(self, distances):
        """Compute mean distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(distances, np.mean, self.chunk_size)

    def compute_std(self, distances):
        """Compute standard deviation of distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(distances, np.std, self.chunk_size)

    def compute_min(self, distances):
        """Compute minimum distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(distances, np.min, self.chunk_size)

    def compute_max(self, distances):
        """Compute maximum distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(distances, np.max, self.chunk_size)

    def compute_median(self, distances):
        """Compute median distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(distances, np.median, self.chunk_size)

    def compute_variance(self, distances):
        """Compute variance of distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(distances, np.var, self.chunk_size)

    def compute_range(self, distances):
        """Compute range (peak-to-peak) of distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(distances, np.ptp, self.chunk_size)

    def compute_q25(self, distances):
        """Compute 25th percentile of distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(
            distances, lambda x, axis: np.percentile(x, 25, axis=axis), self.chunk_size
        )

    def compute_q75(self, distances):
        """Compute 75th percentile of distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(
            distances, lambda x, axis: np.percentile(x, 75, axis=axis), self.chunk_size
        )

    def compute_iqr(self, distances):
        """Compute interquartile range of distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(
            distances,
            lambda x, axis: np.percentile(x, 75, axis=axis) - np.percentile(x, 25, axis=axis),
            self.chunk_size,
        )

    def compute_mad(self, distances):
        """Compute median absolute deviation of distances per pair."""
        return CalculatorStatHelper.compute_func_per_pair(
            distances,
            lambda x, axis: np.median(
                np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis
            ),
            self.chunk_size,
        )

    # === FRAME-BASED STATISTICS ===
    def distances_per_frame_mean(self, distances):
        """Compute mean distances per frame."""
        return CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.mean)

    def distances_per_frame_std(self, distances):
        """Compute standard deviation of distances per frame."""
        return CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.std)

    def distances_per_frame_min(self, distances):
        """Compute minimum distances per frame."""
        return CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.min)

    def distances_per_frame_max(self, distances):
        """Compute maximum distances per frame."""
        return CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.max)

    def distances_per_frame_median(self, distances):
        """Compute median distances per frame."""
        return CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.median)

    def distances_per_frame_range(self, distances):
        """Compute range of distances per frame."""
        return CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.ptp)

    def distances_per_frame_sum(self, distances):
        """Compute sum of distances per frame."""
        return CalculatorStatHelper.compute_func_per_frame(distances, self.chunk_size, np.sum)

    # === RESIDUE-BASED STATISTICS (only for square format) ===
    def compute_per_residue_mean(self, distances):
        """Compute mean distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.mean, self.chunk_size)

    def compute_per_residue_std(self, distances):
        """Compute standard deviation of distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.std, self.chunk_size)

    def compute_per_residue_min(self, distances):
        """Compute minimum distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.min, self.chunk_size)

    def compute_per_residue_max(self, distances):
        """Compute maximum distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.max, self.chunk_size)

    def compute_per_residue_median(self, distances):
        """Compute median distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.median, self.chunk_size)

    def compute_per_residue_sum(self, distances):
        """Compute sum of distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.sum, self.chunk_size)

    def compute_per_residue_variance(self, distances):
        """Compute variance of distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.var, self.chunk_size)

    def compute_per_residue_range(self, distances):
        """Compute range of distances per residue."""
        return CalculatorStatHelper.compute_func_per_residue(distances, np.ptp, self.chunk_size)

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, distances, threshold=2.0, lag_time=10):
        """Compute transitions within lag time."""
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            distances, threshold, lag_time, self.chunk_size
        )

    def compute_transitions_window(self, distances, threshold=2.0, window_size=10):
        """Compute transitions within window."""
        return CalculatorStatHelper.compute_transitions_within_window(
            distances, threshold, window_size, self.chunk_size
        )

    def compute_stability(self, distances, threshold=2.0, window_size=10, mode="window"):
        """Compute stability analysis."""
        return CalculatorStatHelper.compute_stability(
            distances, threshold, window_size, self.chunk_size, mode
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, distances1, distances2, preprocessing_func=None):
        """Compute differences between two distance datasets."""
        return CalculatorStatHelper.compute_differences(
            distances1, distances2, self.chunk_size, preprocessing_func
        )
