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
Per-type reuse tests for the clustering calculators.

Each algorithm is checked flag-based via ``metadata["reused"]``: a matching
cache is reused, a changed input (same shape) or a changed parameter is
recomputed, and the sidecar is always written under use_memmap.
"""

import os
from unittest import mock

import numpy as np
import pytest

from mdxplain.clustering.cluster_type.dbscan.dbscan_calculator import (
    DBSCANCalculator,
)
from mdxplain.clustering.cluster_type.hdbscan.hdbscan_calculator import (
    HDBSCANCalculator,
)
from mdxplain.clustering.cluster_type.dpa.dpa_calculator import DPACalculator
from mdxplain.utils.memmap_reuse_helper import MemmapReuseHelper
from mdxplain.utils.memmap_utils import MemmapUtils
from mdxplain.utils.path_utils import PathUtils


def _two_clusters(seed):
    """Return a reproducible two-cluster 2D dataset."""
    rng = np.random.RandomState(seed)
    return np.vstack(
        [rng.randn(40, 2) * 0.3, rng.randn(40, 2) * 0.3 + 8.0]
    ).astype(np.float32)


# DPA requires an explicit, full parameter set (normally supplied by the DPA
# cluster type); Z is the defining density threshold recorded in the cache.
_DPA = {
    "Z": 1.0,
    "metric": "euclidean",
    "affinity": "nearest_neighbors",
    "density_algo": "PAk",
    "k_max": 40,
    "D_thr": 23.92812698,
    "dim_algo": "twoNN",
    "blockAn": False,
    "block_ratio": 20,
    "frac": 1.0,
}


SPECS = [
    {
        "name": "dbscan",
        "cls": DBSCANCalculator,
        "dat": "dbscan_standard_labels.dat",
        "kwargs": {"eps": 1.0, "min_samples": 3},
        "changed": {"eps": 100.0, "min_samples": 3},
    },
    {
        "name": "hdbscan",
        "cls": HDBSCANCalculator,
        "dat": "hdbscan_standard_labels.dat",
        "kwargs": {"min_cluster_size": 5},
        "changed": {"min_cluster_size": 15},
    },
    {
        "name": "dpa",
        "cls": DPACalculator,
        "dat": "dpa_standard_labels.dat",
        "kwargs": _DPA,
        "changed": {**_DPA, "Z": 3.0},
    },
]

# All clustering calculators run the actual clustering in _perform_clustering,
# reached only after the reuse check.
_COMPUTE_METHOD = "_perform_clustering"


def _run(spec, cache_dir, data, reuse, kwargs, expect_compute=None):
    """Compute once and return (labels_copy, reused_flag).

    ``expect_compute`` asserts whether the clustering step ran: True requires
    it (a fresh or recomputed run), False forbids it (a reused run must skip
    reclustering), None skips the check.
    """
    calc = spec["cls"](
        cache_path=cache_dir, use_memmap=True, chunk_size=100
    )
    calc.reuse_memmap_cache = reuse
    with mock.patch.object(
        calc, _COMPUTE_METHOD, wraps=getattr(calc, _COMPUTE_METHOD)
    ) as spy:
        labels, metadata = calc.compute(data, **kwargs)
    if expect_compute is True:
        assert spy.call_count >= 1, (
            f"{spec['name']}: expected reclustering, but the cache was reused"
        )
    elif expect_compute is False:
        assert spy.call_count == 0, (
            f"{spec['name']}: reclustered instead of reusing the cache"
        )
    out = np.array(labels)
    MemmapUtils.close_memmaps_for_path(
        PathUtils.get_cache_file_path(spec["dat"], cache_dir)
    )
    return out, metadata["reused"]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_reuse_hit_returns_cached_labels(spec, tmp_path):
    """Same input and params: the second run reuses without reclustering."""
    cache = str(tmp_path)
    data = _two_clusters(0)

    first, reused_first = _run(
        spec, cache, data, False, spec["kwargs"], expect_compute=True
    )
    assert reused_first is False

    second, reused_second = _run(
        spec, cache, data, True, spec["kwargs"], expect_compute=False
    )
    assert reused_second is True
    assert np.array_equal(second, first)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_changed_input_is_recomputed(spec, tmp_path):
    """A different input of the same shape is recomputed, not reused stale."""
    cache = str(tmp_path)
    data_a = _two_clusters(0)
    data_b = _two_clusters(1)

    _run(spec, cache, data_a, False, spec["kwargs"], expect_compute=True)
    _, reused_second = _run(
        spec, cache, data_b, True, spec["kwargs"], expect_compute=True
    )

    assert reused_second is False


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_changed_param_is_recomputed(spec, tmp_path):
    """A changed defining parameter is recomputed, not reused."""
    cache = str(tmp_path)
    data = _two_clusters(0)

    _run(spec, cache, data, False, spec["kwargs"], expect_compute=True)
    _, reused_second = _run(
        spec, cache, data, True, spec["changed"], expect_compute=True
    )

    assert reused_second is False


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_flag_off_never_reuses_but_writes_sidecar(spec, tmp_path):
    """Disabled reuse recomputes but still writes the sidecar for later."""
    cache = str(tmp_path)
    data = _two_clusters(0)

    _run(spec, cache, data, False, spec["kwargs"], expect_compute=True)
    _, reused_second = _run(
        spec, cache, data, False, spec["kwargs"], expect_compute=True
    )

    assert reused_second is False
    sidecar = MemmapReuseHelper.sidecar_path(
        PathUtils.get_cache_file_path(spec["dat"], cache)
    )
    assert os.path.exists(sidecar)
