# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Opus 4.8).
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
Chunk-invariance and correctness tests for the feature analysis classes.

chunk_size and use_memmap are memory knobs, so every analysis metric must return
the same values regardless of how the data is chunked. These tests also pin the
factual definition of the per-residue metrics, which must reduce over a residue's
actual partners and not over the squareform diagonal.
"""

import numpy as np
import pytest

from mdxplain.feature.feature_type.contacts.contact_calculator_analysis import ContactCalculatorAnalysis
from mdxplain.feature.feature_type.coordinates.coordinates_calculator_analysis import CoordinatesCalculatorAnalysis
from mdxplain.feature.feature_type.distances.distance_calculator_analysis import DistanceCalculatorAnalysis
from mdxplain.feature.feature_type.helper.calculator_stat_helper import CalculatorStatHelper
from mdxplain.feature.feature_type.sasa.sasa_calculator_analysis import SASACalculatorAnalysis
from mdxplain.feature.feature_type.torsions.torsions_calculator_analysis import TorsionsCalculatorAnalysis

RTOL = 1e-6
ATOL = 1e-8

CHUNK_SIZES = [1, 2, 7, 10_000]


def _reference_mad(array: np.ndarray) -> np.ndarray:
    """
    Compute the per-column median absolute deviation.

    Parameters
    ----------
    array : numpy.ndarray
        Input array with shape (n_frames, n_features)

    Returns
    -------
    numpy.ndarray
        MAD per feature
    """
    median = np.median(array, axis=0, keepdims=True)
    return np.median(np.abs(array - median), axis=0)


@pytest.fixture
def angles() -> np.ndarray:
    """
    Build a deterministic torsion angle array in degrees.

    Returns
    -------
    numpy.ndarray
        Angles with shape (40, 6)
    """
    rng = np.random.default_rng(20260718)
    return rng.uniform(-180.0, 180.0, (40, 6)).astype(np.float32)


@pytest.fixture
def positive() -> np.ndarray:
    """
    Build a deterministic positive-valued array.

    Returns
    -------
    numpy.ndarray
        Values with shape (40, 6)
    """
    rng = np.random.default_rng(20260719)
    return rng.uniform(1.0, 20.0, (40, 6)).astype(np.float32)


class TestMadIsChunkInvariant:
    """
    MAD must be a per-column median absolute deviation for every feature type.

    torsions once centred on a median taken over the whole chunk (no axis
    argument), so its result changed with chunk_size and matched no definition.
    """

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_torsions_mad(self, angles, chunk_size):
        """torsions MAD must equal the per-column reference."""
        analysis = TorsionsCalculatorAnalysis(use_memmap=True, chunk_size=chunk_size)
        np.testing.assert_allclose(
            analysis.compute_mad(angles), _reference_mad(angles), rtol=RTOL, atol=ATOL
        )

    def test_mad_is_per_column_when_a_block_spans_features(self):
        """
        A block holding several feature columns still takes a per-column median.

        This is the exact condition the axis-less inner median corrupted. The
        chunk_size sweep only reaches it for one size and one data shape, so the
        block width is pinned above one here to exercise the multi-column path
        directly, regardless of which chunk_sizes are swept.
        """
        rng = np.random.default_rng(20260725)
        angles = rng.uniform(-180.0, 180.0, (8, 5)).astype(np.float32)
        block_width = CalculatorStatHelper.resolve_output_block_size(8, 8, 5)
        assert block_width > 1, "test premise: the block must span several columns"
        analysis = TorsionsCalculatorAnalysis(use_memmap=True, chunk_size=8)
        np.testing.assert_allclose(
            analysis.compute_mad(angles), _reference_mad(angles), rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_distances_mad(self, positive, chunk_size):
        """Control: distances MAD is already correct and chunk-invariant."""
        analysis = DistanceCalculatorAnalysis(use_memmap=True, chunk_size=chunk_size)
        np.testing.assert_allclose(
            analysis.compute_mad(positive), _reference_mad(positive), rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_sasa_mad(self, positive, chunk_size):
        """Control: sasa MAD is already correct and chunk-invariant."""
        analysis = SASACalculatorAnalysis(use_memmap=True, chunk_size=chunk_size)
        np.testing.assert_allclose(
            analysis.compute_mad(positive), _reference_mad(positive), rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_coordinates_mad(self, positive, chunk_size):
        """Control: coordinates MAD is already correct and chunk-invariant."""
        analysis = CoordinatesCalculatorAnalysis(use_memmap=True, chunk_size=chunk_size)
        np.testing.assert_allclose(
            analysis.compute_mad(positive), _reference_mad(positive), rtol=RTOL, atol=ATOL
        )


class TestTorsionsConstantAngles:
    """
    A rigid torsion must report zero spread, not NaN.

    The mean resultant length is an average of unit vectors and cannot exceed 1,
    but rounding can push it a few ulp past it. The circular variance then goes
    negative and the square root behind the circular deviation returns NaN, which
    silently fails every threshold comparison and drops the feature.
    """

    ANGLES = np.column_stack(
        [np.full(10, -60.0), np.full(10, -45.0), np.full(10, 0.0), np.full(10, 180.0)]
    ).astype(np.float32)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_variance_is_zero_and_finite(self, chunk_size):
        """Circular variance of a constant angle is exactly zero."""
        analysis = TorsionsCalculatorAnalysis(use_memmap=True, chunk_size=chunk_size)
        variance = analysis.compute_variance(self.ANGLES)
        assert np.all(np.isfinite(variance))
        np.testing.assert_allclose(variance, 0.0, atol=1e-12)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_std_is_zero_and_finite(self, chunk_size):
        """Circular deviation of a constant angle is exactly zero, never NaN."""
        analysis = TorsionsCalculatorAnalysis(use_memmap=True, chunk_size=chunk_size)
        std = analysis.compute_std(self.ANGLES)
        assert np.all(np.isfinite(std)), f"NaN in circular std: {std}"
        np.testing.assert_allclose(std, 0.0, atol=1e-6)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_pooled_std_is_zero_and_finite(self, chunk_size):
        """The pooled path must clamp the same way the direct path does."""
        analysis = TorsionsCalculatorAnalysis(use_memmap=True, chunk_size=chunk_size)
        std = analysis.compute_pooled_metric_values([self.ANGLES, self.ANGLES], "std")
        assert np.all(np.isfinite(std)), f"NaN in pooled circular std: {std}"
        np.testing.assert_allclose(std, 0.0, atol=1e-6)


class TestPerResidueExcludesDiagonal:
    """
    Per-residue metrics must reduce over a residue's real partners.

    The squareform diagonal holds the self-distance 0. Reducing with
    axis=(0, 2) includes it, which makes per_residue_min identically zero and
    biases mean/std/var/range.

    Layout: 4 residues, full upper triangle, pair order
    (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
    """

    CONDENSED = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)

    def test_min_ignores_self_distance(self):
        """per_residue_min must be the closest partner, never the diagonal zero."""
        analysis = DistanceCalculatorAnalysis(use_memmap=False, chunk_size=2000)
        actual = analysis.compute_per_residue_min(self.CONDENSED)
        np.testing.assert_allclose(actual, [1.0, 1.0, 2.0, 3.0], rtol=RTOL, atol=ATOL)

    def test_mean_ignores_self_distance(self):
        """per_residue_mean must average over partners only."""
        analysis = DistanceCalculatorAnalysis(use_memmap=False, chunk_size=2000)
        actual = analysis.compute_per_residue_mean(self.CONDENSED)
        expected = [6.0 / 3.0, 10.0 / 3.0, 12.0 / 3.0, 14.0 / 3.0]
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    def test_sum_over_partners(self):
        """per_residue_sum is unaffected by the diagonal but pinned for completeness."""
        analysis = DistanceCalculatorAnalysis(use_memmap=False, chunk_size=2000)
        actual = analysis.compute_per_residue_sum(self.CONDENSED)
        np.testing.assert_allclose(actual, [6.0, 10.0, 12.0, 14.0], rtol=RTOL, atol=ATOL)


class TestPooledMetricsAreChunkInvariant:
    """Pooled metrics must not depend on chunk_size or use_memmap."""

    SEGMENT_SIZES = (13, 5, 9)

    def _segments(self, seed: int, low: float, high: float) -> list:
        """
        Build deterministic segments.

        Parameters
        ----------
        seed : int
            Random seed
        low : float
            Lower bound
        high : float
            Upper bound

        Returns
        -------
        list
            List of (n_frames, 6) arrays
        """
        rng = np.random.default_rng(seed)
        return [rng.uniform(low, high, (n, 6)).astype(np.float32) for n in self.SEGMENT_SIZES]

    @pytest.mark.parametrize(
        "metric", ["mean", "std", "variance", "min", "max", "cv", "range", "mad"]
    )
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_distances_pooled(self, metric, chunk_size):
        """Pooled distance metrics are identical for every chunking."""
        segments = self._segments(20260720, 1.0, 20.0)
        reference = DistanceCalculatorAnalysis(
            use_memmap=False, chunk_size=10_000
        ).compute_pooled_metric_values(segments, metric)
        actual = DistanceCalculatorAnalysis(
            use_memmap=True, chunk_size=chunk_size
        ).compute_pooled_metric_values(segments, metric)
        np.testing.assert_allclose(actual, reference, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_contacts_pooled_frequency(self, chunk_size):
        """Pooled contact frequency is identical for every chunking."""
        rng = np.random.default_rng(20260721)
        segments = [
            (rng.random((n, 6)) > 0.5).astype(np.float32) for n in self.SEGMENT_SIZES
        ]
        reference = ContactCalculatorAnalysis(
            use_memmap=False, chunk_size=10_000
        ).compute_pooled_metric_values(segments, "frequency")
        actual = ContactCalculatorAnalysis(
            use_memmap=True, chunk_size=chunk_size
        ).compute_pooled_metric_values(segments, "frequency")
        np.testing.assert_allclose(actual, reference, rtol=RTOL, atol=ATOL)


class TestTorsionsPooledTransitionsRespectPeriodicity:
    """
    Pooled torsion transitions must use the same angular metric as the
    non-pooled path, which wraps at +/-180 degrees.
    """

    def test_wrap_is_not_counted_as_a_jump(self):
        """A -179 -> +179 step is a 2 degree move, not a 358 degree one."""
        segment = np.array(
            [[-179.0], [179.0], [-179.0], [179.0], [-179.0], [179.0]], dtype=np.float32
        )
        analysis = TorsionsCalculatorAnalysis(use_memmap=False, chunk_size=2000)
        pooled = analysis.compute_pooled_metric_values(
            [segment], "transitions", transition_threshold=30.0,
            transition_mode="lagtime", lag_time=1,
        )
        assert pooled[0] == 0.0
