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

"""Unit tests for AutomaticParameterHelper gamma="scale" variance.

The KernelPCA gamma="scale" heuristic divides by the variance of the data. That
variance must be a single, well-defined quantity — the population variance over
every element, matching ``np.var(data)`` — regardless of whether it is computed
directly or streamed in row-chunks for a memory-mapped input. These tests pin
that quantity down and guard the two ways it used to go wrong:

* the streamed branch returning the mean of the per-feature variances, a
  different number that shifted the kernel whenever ``use_memmap`` was toggled;
* chunk boundaries perturbing the result away from ``np.var``.
"""

import numpy as np
import pytest

from mdxplain.decomposition.decomposition_type.helper.automatic_parameter_helper import (
    AutomaticParameterHelper,
)

CHUNK_SIZES = [1, 2, 7, 1000, 100000]


def _heterogeneous_scale_data():
    """Build a matrix whose global variance differs from the per-feature mean.

    Feature 0 sits near 0.4 and feature 1 near 5.0, each with a spread two
    orders of magnitude smaller than the gap between their means. The variance
    of all elements is therefore dominated by that between-feature gap and is
    far larger than the mean of the two per-feature variances. This separation
    is what lets the tests tell the correct global variance apart from the
    quantity the old chunked branch returned; without it both numbers would
    coincide and an assertion could not distinguish them.

    Returns
    -------
    numpy.ndarray
        A (501, 2) float64 matrix with strongly heterogeneous feature scales.
    """
    rng = np.random.RandomState(0)
    n_frames = 501
    feature_a = 0.4 + 0.01 * rng.randn(n_frames)
    feature_b = 5.0 + 0.02 * rng.randn(n_frames)
    return np.column_stack([feature_a, feature_b]).astype(np.float64)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_total_variance_matches_np_var(chunk_size):
    """Streamed total variance reproduces np.var for any chunk size.

    ``compute_total_variance`` walks the matrix in row-chunks and accumulates a
    two-pass population variance. Its result must equal ``np.var(data)`` over
    every element no matter how the rows are partitioned, including a chunk size
    of 1 (every row its own block) and one larger than the frame count (a single
    block). The parametrised sizes also cover an uneven final chunk, the case a
    naive per-chunk average would get wrong.
    """
    data = _heterogeneous_scale_data()
    streamed = AutomaticParameterHelper.compute_total_variance(data, chunk_size)
    assert streamed == pytest.approx(float(np.var(data)), rel=1e-6)


def test_total_variance_is_not_the_per_feature_mean():
    """Result is the global variance, not the mean of per-feature variances.

    This is the direct regression guard for the gamma="scale" fix. The previous
    chunked branch centred each feature on its own mean and averaged the
    resulting per-feature variances, dropping the between-feature term. For data
    whose features live at different scales that term dominates, so the two
    quantities are far apart. The test first asserts they genuinely differ (so a
    passing result is meaningful), then pins ``compute_total_variance`` to the
    global variance and confirms it is not the old per-feature mean.
    """
    data = _heterogeneous_scale_data()
    global_var = float(np.var(data))
    per_feature_mean = float(np.mean(np.var(data, axis=0)))
    # The two quantities must genuinely differ, or the test proves nothing.
    assert not np.isclose(global_var, per_feature_mean, rtol=0.1)
    streamed = AutomaticParameterHelper.compute_total_variance(data, 7)
    assert streamed == pytest.approx(global_var, rel=1e-6)
    assert not np.isclose(streamed, per_feature_mean, rtol=0.1)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_gamma_scale_is_memmap_invariant(chunk_size):
    """gamma="scale" is identical with and without the streamed path.

    ``calculate_gamma_scale`` picks the streamed variance when ``use_memmap`` is
    set and ``data.var()`` otherwise. Because both compute the same global
    variance, the returned gamma must be the same value for either flag and for
    every chunk size; otherwise toggling a pure memory knob would change the
    fitted kernel. The test also pins gamma to its closed form
    ``1 / (n_features * np.var(data))`` so a regression in either branch is
    caught, not just a disagreement between them.
    """
    data = _heterogeneous_scale_data()
    streamed = AutomaticParameterHelper.calculate_gamma_scale(
        data, use_memmap=True, chunk_size=chunk_size
    )
    direct = AutomaticParameterHelper.calculate_gamma_scale(
        data, use_memmap=False
    )
    expected = 1.0 / (data.shape[1] * float(np.var(data)))
    assert streamed == pytest.approx(direct, rel=1e-6)
    assert streamed == pytest.approx(expected, rel=1e-6)


def test_gamma_scale_invariant_over_memmap_input(tmp_path):
    """A real memmap input yields the same gamma as its in-memory copy.

    The other tests exercise the streamed branch with an in-RAM array behind the
    ``use_memmap`` flag. This one writes the same data to an on-disk
    ``numpy.memmap`` and feeds it through the streamed path, so the row-chunked
    reads run against an actual memory-mapped buffer. The resulting gamma must
    match the one computed directly from the in-memory array, confirming the
    memory strategy leaves the parameter untouched.
    """
    data = _heterogeneous_scale_data()
    path = tmp_path / "gamma.dat"
    memmap = np.memmap(path, dtype=np.float64, mode="w+", shape=data.shape)
    memmap[:] = data[:]
    memmap.flush()
    gamma_memmap = AutomaticParameterHelper.calculate_gamma_scale(
        memmap, use_memmap=True, chunk_size=7
    )
    gamma_array = AutomaticParameterHelper.calculate_gamma_scale(
        data, use_memmap=False
    )
    assert gamma_memmap == pytest.approx(gamma_array, rel=1e-6)
