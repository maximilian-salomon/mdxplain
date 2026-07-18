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

"""Unit tests for RepresentativeFinderHelper per-feature variance scaling.

Representative selection normalises each feature's margin by that feature's
standard deviation. The scale must be the feature's true spread, computed the
same way whether or not the data is streamed in chunks. These tests cover both
the numeric core (``_streamed_feature_variances``) and the scale mapping
(``_compute_feature_scales``), and specifically guard the failure that motivated
the two-pass rewrite: a one-pass ``E[x^2] - E[x]^2`` cancels for a molecular
distance (mean ~5, spread ~0.01), gets clamped to zero, and silently collapses
the scale to the 1.0 fallback, which then changes which frame is chosen as the
representative.
"""

import numpy as np
import pytest

from mdxplain.feature_importance.helper.representative_finder_helper import (
    RepresentativeFinderHelper,
)

CHUNK_SIZES = [1, 3, 16, 500, 100000]


def _mixed_scale_data():
    """Build float32 features spanning easy and cancellation-prone regimes.

    Feature 0 mimics a molecular distance: a mean of about 5 with a spread of
    about 0.01, three orders of magnitude smaller, which is exactly where a
    one-pass ``E[x^2] - E[x]^2`` loses all significant digits. Feature 1 is
    zero-centred with unit-scale spread and feature 2 is offset to about 100
    with a moderate spread, so the suite also covers ordinary, well-conditioned
    columns. The data is float32 to match the real feature matrices and to make
    a cancelling one-pass formula fail loudly rather than marginally.

    Returns
    -------
    numpy.ndarray
        A (2000, 3) float32 matrix with one cancellation-prone column.
    """
    rng = np.random.RandomState(1)
    n_frames = 2000
    feature_distance = 5.0 + 0.01 * rng.randn(n_frames)
    feature_centered = 0.0 + 2.0 * rng.randn(n_frames)
    feature_offset = 100.0 + 0.5 * rng.randn(n_frames)
    return np.column_stack(
        [feature_distance, feature_centered, feature_offset]
    ).astype(np.float32)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_streamed_variance_matches_np_var(chunk_size):
    """Streamed per-feature variance reproduces np.var for any chunk size.

    ``_streamed_feature_variances`` makes two passes over the rows — one for the
    per-feature mean, one for the squared deviations — accumulating in float64.
    Its output must equal ``np.var(axis=0)`` for each selected column no matter
    how the rows are split, from a chunk size of 1 up to one exceeding the frame
    count (a single block). The comparison is made against a float64 reference so
    the assertion measures the algorithm, not the float32 storage of the input.
    """
    data = _mixed_scale_data()
    indices = [0, 1, 2]
    streamed = RepresentativeFinderHelper._streamed_feature_variances(
        data, indices, chunk_size
    )
    expected = np.var(data[:, indices].astype(np.float64), axis=0)
    assert streamed == pytest.approx(expected, rel=1e-5)


def test_streamed_variance_survives_cancellation():
    """The small-spread, large-mean feature keeps its real, non-zero variance.

    This is the core regression guard. For the distance-like column the true
    variance is about ``1e-4``; a one-pass ``E[x^2] - E[x]^2`` subtracts two
    numbers near 25 and returns noise or a negative value that the old code
    clamped to zero. The test asserts the two-pass result matches the float64
    reference variance and, independently, that it stays well above zero so a
    reintroduced cancellation cannot pass by landing near — but not exactly at —
    the right magnitude.
    """
    data = _mixed_scale_data()
    streamed = RepresentativeFinderHelper._streamed_feature_variances(
        data, [0], 16
    )
    expected = float(np.var(data[:, 0].astype(np.float64)))
    assert streamed[0] == pytest.approx(expected, rel=1e-4)
    # A cancelling one-pass would clamp this toward zero; it must not.
    assert streamed[0] > 1e-6


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_feature_scales_are_chunk_invariant(chunk_size):
    """Chunked scales match the unchunked np.std path for every feature.

    ``_compute_feature_scales`` takes the direct ``np.std`` path when the data
    fits in one chunk and the streamed path otherwise. Both must return the same
    scale per feature, and both must equal ``np.std(axis=0)``. The test compares
    the chunked result (small ``chunk_size``) against the unchunked result
    (``chunk_size=None``) and against a float64 ``np.std`` reference for all
    three columns, confirming ``chunk_size`` only chooses the reduction strategy
    and never the value.
    """
    data = _mixed_scale_data()
    indices = [0, 1, 2]
    chunked = RepresentativeFinderHelper._compute_feature_scales(
        data, indices, chunk_size=chunk_size
    )
    unchunked = RepresentativeFinderHelper._compute_feature_scales(
        data, indices, chunk_size=None
    )
    reference = np.std(data[:, indices].astype(np.float64), axis=0)
    for position, feat_idx in enumerate(indices):
        assert chunked[feat_idx] == pytest.approx(
            unchunked[feat_idx], rel=1e-5
        )
        assert chunked[feat_idx] == pytest.approx(
            float(reference[position]), rel=1e-4
        )


def test_distance_feature_keeps_real_scale_not_fallback():
    """The distance-like feature scales by its real std, not the 1.0 fallback.

    This closes the loop from the numeric core to the observable behaviour.
    ``_compute_feature_scales`` substitutes a scale of 1.0 whenever the computed
    standard deviation drops below the ``1e-6`` guard — the exact path the old
    cancellation took when it clamped the distance variance to zero. Here the
    real standard deviation is about 0.01, far below 1.0, so a fallback of 1.0
    would be a hundred-fold over-estimate that rescales the margin and can select
    a different representative. The test asserts the returned scale is the real
    standard deviation and is not the fallback.
    """
    data = _mixed_scale_data()
    scales = RepresentativeFinderHelper._compute_feature_scales(
        data, [0], chunk_size=16
    )
    real_std = float(np.std(data[:, 0].astype(np.float64)))
    assert real_std < 1.0  # the fallback would be a gross over-estimate here
    assert scales[0] == pytest.approx(real_std, rel=1e-4)
    assert scales[0] != pytest.approx(1.0, rel=1e-2)
