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
Unit tests for CalculatorStatHelper.

Guards the central invariant of the chunking layer: chunk_size and use_memmap
are memory/IO knobs only, never numerical parameters. A reduction must return
the same values for every chunk_size and for both memmap settings, and those
values must equal the plain NumPy reduction over the whole array.
"""

from unittest.mock import patch

import numpy as np
import pytest

from mdxplain.feature.feature_type.helper.calculator_stat_helper import CalculatorStatHelper

# Deliberately tight: the acceptance criterion is ~3 decimals, but a loose
# threshold would let real defects through.
RTOL = 1e-6
ATOL = 1e-8

# 1 and 2 force many chunks, 7 forces a short final chunk (13 % 7 != 0),
# a huge value forces the single-chunk path.
CHUNK_SIZES = [1, 2, 7, 10_000]

PER_FEATURE_REDUCTIONS = [
    ("mean", np.mean),
    ("std", np.std),
    ("var", np.var),
    ("min", np.min),
    ("max", np.max),
    ("sum", np.sum),
    ("ptp", np.ptp),
    ("median", np.median),
]


def _mad(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute median absolute deviation per feature.

    Parameters
    ----------
    array : numpy.ndarray
        Input array
    axis : int, default=0
        Reduction axis

    Returns
    -------
    numpy.ndarray
        MAD per feature
    """
    median = np.median(array, axis=axis, keepdims=True)
    return np.median(np.abs(array - median), axis=axis)


@pytest.fixture
def sample_2d() -> np.ndarray:
    """
    Build a deterministic (n_frames, n_features) array.

    Returns
    -------
    numpy.ndarray
        Sample array with 13 frames and 5 features
    """
    rng = np.random.default_rng(20260715)
    return rng.uniform(1.0, 20.0, (13, 5)).astype(np.float32)


@pytest.fixture
def angles_2d() -> np.ndarray:
    """
    Build a deterministic torsion angle array in degrees.

    Returns
    -------
    numpy.ndarray
        Angles with shape (13, 5)
    """
    rng = np.random.default_rng(20260723)
    return rng.uniform(-180.0, 180.0, (13, 5)).astype(np.float32)


@pytest.fixture
def sample_3d() -> np.ndarray:
    """
    Build a deterministic (n_frames, n_residues, n_residues) array.

    Returns
    -------
    numpy.ndarray
        Sample array with 13 frames and a 4x4 spatial block
    """
    rng = np.random.default_rng(20260716)
    return rng.uniform(1.0, 20.0, (13, 4, 4)).astype(np.float32)


def _as_memmap(array: np.ndarray, path) -> np.ndarray:
    """
    Write an array to disk and return it as a read-back memmap.

    Parameters
    ----------
    array : numpy.ndarray
        Array to persist
    path : pathlib.Path
        Destination file path

    Returns
    -------
    numpy.ndarray
        Memmap view of the same data
    """
    memmap = np.memmap(path, dtype=array.dtype, mode="w+", shape=array.shape)
    memmap[:] = array
    memmap.flush()
    return np.memmap(path, dtype=array.dtype, mode="r", shape=array.shape)


class TestResolveOutputBlockSize:
    """
    resolve_output_block_size converts a frames budget into an output-axis width.

    chunk_size counts frames, so a reduction that must hold every frame of a
    column shrinks its output axis instead, keeping the same memory budget.
    """

    def test_budget_identity(self):
        """block x n_rows must stay within the chunk_size x n_units budget."""
        chunk_size, n_rows, n_units = 2000, 1_086_688, 46_971
        block = CalculatorStatHelper.resolve_output_block_size(chunk_size, n_rows, n_units)
        assert block * n_rows <= chunk_size * n_units
        # ...and use the budget rather than collapsing to a single unit.
        assert (block + 1) * n_rows > chunk_size * n_units

    def test_real_world_shape(self):
        """The reported dataset lands at 86 columns, not the raw chunk_size."""
        assert CalculatorStatHelper.resolve_output_block_size(2000, 1_086_688, 46_971) == 86

    def test_more_features_than_frames_uses_all_units(self):
        """Wide, short data needs no shrinking; one block is the whole axis."""
        assert CalculatorStatHelper.resolve_output_block_size(2000, 10, 50) == 50

    def test_never_below_one(self):
        """An extreme frame count still yields a usable block."""
        assert CalculatorStatHelper.resolve_output_block_size(1, 10**9, 4) == 1

    def test_never_above_n_units(self):
        """The block can never exceed the output axis."""
        assert CalculatorStatHelper.resolve_output_block_size(10**9, 10, 7) == 7

    @pytest.mark.parametrize("chunk_size", [None, 0])
    def test_falsy_chunk_size_means_single_block(self, chunk_size):
        """Falsy chunk_size keeps today's unchunked behaviour: one block."""
        assert CalculatorStatHelper.resolve_output_block_size(chunk_size, 1000, 42) == 42

    def test_zero_units(self):
        """Degenerate input must not produce a zero or negative step."""
        assert CalculatorStatHelper.resolve_output_block_size(2000, 1000, 0) == 1

    def test_zero_rows(self):
        """Degenerate input must not divide by zero."""
        assert CalculatorStatHelper.resolve_output_block_size(2000, 0, 42) == 42


class TestComputeReductionPerFeature:
    """
    Streaming reductions must match NumPy and never depend on the chunking.

    There is no separate unchunked formula, so a single block is just the
    n_chunks == 1 case of the same code.
    """

    STREAMABLE = [
        ("sum", np.sum),
        ("mean", np.mean),
        ("min", np.min),
        ("max", np.max),
        ("ptp", np.ptp),
        ("var", np.var),
        ("std", np.std),
    ]

    @pytest.mark.parametrize("name,func", STREAMABLE)
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_matches_numpy(self, sample_2d, name, func, chunk_size, use_memmap):
        """Every combination must equal the plain NumPy reduction."""
        expected = func(sample_2d, axis=0)
        actual = CalculatorStatHelper.compute_reduction_per_feature(
            sample_2d, name, chunk_size, use_memmap
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("name,func", STREAMABLE)
    def test_memmap_input_matches_ram_input(self, sample_2d, tmp_path, name, func):
        """A memmap must give the same answer as the identical RAM array."""
        memmap = _as_memmap(sample_2d, tmp_path / f"stream_{name}.dat")
        expected = func(sample_2d, axis=0)
        actual = CalculatorStatHelper.compute_reduction_per_feature(memmap, name, 7, False)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_3d_input_preserves_spatial_shape(self, sample_3d, chunk_size):
        """A (n_frames, M, M) array reduces to (M, M)."""
        expected = np.mean(sample_3d, axis=0)
        actual = CalculatorStatHelper.compute_reduction_per_feature(
            sample_3d, "mean", chunk_size, True
        )
        assert actual.shape == (4, 4)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_transform_runs_inside_the_block(self, angles_2d, chunk_size):
        """
        The transform must be applied per block, not to the whole array.

        This is what keeps the torsion sin/cos work from materialising a full
        copy before any chunking happens.
        """
        expected = np.mean(np.sin(np.radians(angles_2d)), axis=0)
        actual = CalculatorStatHelper.compute_reduction_per_feature(
            angles_2d, "mean", chunk_size, True,
            transform=lambda block: np.sin(np.radians(block)),
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    def test_variance_survives_a_large_offset(self):
        """
        Chan's combination must not lose the spread under a huge mean.

        A one-pass E[x^2] - E[x]^2 collapses here; molecular distances look
        exactly like this (5 nm mean, 0.01 nm spread).
        """
        rng = np.random.default_rng(20260722)
        values = (1e6 + rng.normal(0.0, 0.01, (500, 3))).astype(np.float64)
        expected = np.var(values, axis=0)
        actual = CalculatorStatHelper.compute_reduction_per_feature(
            values, "var", 7, True
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_unknown_reduction_raises(self, sample_2d):
        """A non-streamable reduction must be rejected, not silently wrong."""
        with pytest.raises(ValueError, match="Unknown reduction"):
            CalculatorStatHelper.compute_reduction_per_feature(sample_2d, "median")

    def test_every_advertised_reduction_has_an_accumulator(self, sample_2d):
        """
        The whitelist must not advertise anything the dispatcher cannot serve.

        Both are derived from one registry, so this can only break if someone
        adds a family without an accumulator.
        """
        for reduction in CalculatorStatHelper.STREAMING_REDUCTIONS:
            result = CalculatorStatHelper.compute_reduction_per_feature(
                sample_2d, reduction, 7, True
            )
            assert result.shape == (sample_2d.shape[1],), reduction

    def test_abandoned_generator_reverts_the_access_hint(self, sample_2d, tmp_path):
        """
        A consumer that stops early must revert the memmap's access hint.

        The generator sets a 'sequential' madvise hint up front and reverts it to
        'random' in a finally. Asserting the actual tune_memmap sequence is what
        makes this a regression test: without the finally, an early close would
        skip the revert and leave only ['sequential'].
        """
        memmap = _as_memmap(sample_2d, tmp_path / "abandoned.dat")
        target = (
            "mdxplain.feature.feature_type.helper.calculator_stat_helper"
            ".ResourceUtils.tune_memmap"
        )
        with patch(target) as tune:
            blocks = CalculatorStatHelper._iter_frame_blocks(memmap, 2, True, None)
            next(blocks)  # sets the 'sequential' hint, does not yet revert it
            assert [call.args[1] for call in tune.call_args_list] == ["sequential"]
            blocks.close()  # must run the finally and revert to 'random'
            assert [call.args[1] for call in tune.call_args_list] == [
                "sequential",
                "random",
            ]


class TestComputeFuncPerFeatureInvariance:
    """compute_func_per_feature must not depend on chunk_size or use_memmap."""

    @pytest.mark.parametrize("name,func", PER_FEATURE_REDUCTIONS)
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_matches_numpy_reduction(self, sample_2d, name, func, chunk_size, use_memmap):
        """Every combination must equal func(array, axis=0)."""
        expected = func(sample_2d, axis=0)
        actual = CalculatorStatHelper.compute_func_per_feature(
            sample_2d, func, chunk_size, use_memmap
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_mad_matches_reference(self, sample_2d, chunk_size, use_memmap):
        """MAD is the reduction that regressed in torsions; pin it here too."""
        expected = _mad(sample_2d, axis=0)
        actual = CalculatorStatHelper.compute_func_per_feature(
            sample_2d, _mad, chunk_size, use_memmap
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("name,func", PER_FEATURE_REDUCTIONS)
    def test_memmap_input_matches_ram_input(self, sample_2d, tmp_path, name, func):
        """A memmap-backed array must give the same answer as the RAM array."""
        memmap = _as_memmap(sample_2d, tmp_path / f"per_feature_{name}.dat")
        expected = CalculatorStatHelper.compute_func_per_feature(sample_2d, func, 7, False)
        actual = CalculatorStatHelper.compute_func_per_feature(memmap, func, 7, False)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


class TestComputeFuncPerFrameInvariance:
    """compute_func_per_frame must not depend on chunk_size or use_memmap."""

    @pytest.mark.parametrize("name,func", [("mean", np.mean), ("std", np.std), ("sum", np.sum)])
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_matches_numpy_reduction(self, sample_2d, name, func, chunk_size, use_memmap):
        """Every combination must equal func(array, axis=1)."""
        expected = func(sample_2d, axis=1)
        actual = CalculatorStatHelper.compute_func_per_frame(
            sample_2d, chunk_size, use_memmap, func
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


class TestPerResidueReduction:
    """
    Per-residue reduction pools each residue's real partner columns.

    The value must equal the plain reduction over the pooled partner values
    (no squareform, no self-distance diagonal) and must not depend on chunking.
    """

    # 4 residues, full upper triangle: columns 0..5 are the pairs below.
    PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    COLUMNS_BY_RESIDUE = {0: [0, 1, 2], 1: [0, 3, 4], 2: [1, 3, 5], 3: [2, 4, 5]}

    def _pooled_reference(self, data, reduction):
        """Reduce each residue over the pooled values of its partner columns."""
        return np.array([
            reduction(data[:, cols].ravel())
            for cols in self.COLUMNS_BY_RESIDUE.values()
        ])

    @pytest.mark.parametrize(
        "metric,reduction",
        [
            ("mean", np.mean),
            ("std", np.std),
            ("variance", np.var),
            ("min", np.min),
            ("max", np.max),
            ("sum", np.sum),
            ("range", np.ptp),
            ("median", np.median),
        ],
    )
    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_matches_pooled_reference(self, metric, reduction, chunk_size, use_memmap):
        """Every combination equals the pooled reduction over partner columns."""
        rng = np.random.default_rng(20260724)
        data = rng.uniform(1.0, 20.0, (23, 6)).astype(np.float32)
        expected = self._pooled_reference(data, reduction)
        actual = CalculatorStatHelper.compute_per_residue_reduction(
            data, self.PAIRS, 4, metric, chunk_size, use_memmap
        )
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    def test_residue_without_partners_is_zero(self):
        """A residue that appears in no retained pair reduces to zero, not NaN."""
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        # residue 3 participates in no pair
        actual = CalculatorStatHelper.compute_per_residue_reduction(
            data, [(0, 1), (0, 2)], 4, "mean"
        )
        assert actual.shape == (4,)
        assert actual[3] == 0.0

    def test_unknown_metric_raises(self):
        """A metric with no per-residue combiner is rejected."""
        data = np.array([[1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown per-residue metric"):
            CalculatorStatHelper.compute_per_residue_reduction(data, [(0, 1)], 2, "cv")


class TestTransitionsInvariance:
    """Transition counts must not depend on chunk_size or use_memmap."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_lagtime_invariant(self, sample_2d, chunk_size, use_memmap):
        """Lagtime transitions are identical for every chunking."""
        reference = CalculatorStatHelper.compute_transitions_within_lagtime(
            sample_2d, threshold=2.0, lag_time=2, chunk_size=10_000, use_memmap=False
        )
        actual = CalculatorStatHelper.compute_transitions_within_lagtime(
            sample_2d, threshold=2.0, lag_time=2, chunk_size=chunk_size, use_memmap=use_memmap
        )
        np.testing.assert_allclose(actual, reference, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_window_invariant(self, sample_2d, chunk_size, use_memmap):
        """Window transitions are identical for every chunking."""
        reference = CalculatorStatHelper.compute_transitions_within_window(
            sample_2d, threshold=2.0, window_size=3, chunk_size=10_000, use_memmap=False
        )
        actual = CalculatorStatHelper.compute_transitions_within_window(
            sample_2d, threshold=2.0, window_size=3, chunk_size=chunk_size, use_memmap=use_memmap
        )
        np.testing.assert_allclose(actual, reference, rtol=RTOL, atol=ATOL)


class TestPooledTransitionsInvariance:
    """Pooled transitions must not depend on chunk_size and must be boundary-safe."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_pooled_transitions_invariant(self, chunk_size):
        """Pooling is over segments; chunking must not change the counts."""
        rng = np.random.default_rng(20260717)
        segments = [rng.uniform(1.0, 20.0, (n, 5)).astype(np.float32) for n in (13, 5, 9)]
        reference, ref_total = CalculatorStatHelper.compute_pooled_transitions(
            segments, 2.0, 2, 10_000, False, mode="lagtime"
        )
        actual, total = CalculatorStatHelper.compute_pooled_transitions(
            segments, 2.0, 2, chunk_size, True, mode="lagtime"
        )
        assert total == ref_total
        np.testing.assert_allclose(actual, reference, rtol=RTOL, atol=ATOL)

    def test_pooled_transitions_do_not_cross_segment_boundaries(self):
        """A jump between two segments must not be counted as a transition."""
        low = np.full((6, 1), 1.0, dtype=np.float32)
        high = np.full((6, 1), 100.0, dtype=np.float32)
        transitions, _ = CalculatorStatHelper.compute_pooled_transitions(
            [low, high], threshold=2.0, window_size=1, chunk_size=10, mode="lagtime"
        )
        assert transitions[0] == 0.0
