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
Per-type reuse tests for the decomposition calculators.

Each decomposition type is checked flag-based via ``metadata["reused"]``:
a matching cache is reused, a changed input (same shape) or a changed
parameter is recomputed, and the sidecar is always written under use_memmap.
"""

import os
from unittest import mock

import numpy as np
import pytest

from mdxplain.decomposition.decomposition_type.pca.pca_calculator import (
    PCACalculator,
)
from mdxplain.decomposition.decomposition_type.kernel_pca.kernel_pca_calculator import (  # noqa: E501
    KernelPCACalculator,
)
from mdxplain.decomposition.decomposition_type.contact_kernel_pca.contact_kernel_pca_calculator import (  # noqa: E501
    ContactKernelPCACalculator,
)
from mdxplain.decomposition.decomposition_type.diffusion_maps.diffusion_maps_calculator import (  # noqa: E501
    DiffusionMapsCalculator,
)
from mdxplain.utils.memmap_reuse_helper import MemmapReuseHelper
from mdxplain.utils.memmap_utils import MemmapUtils
from mdxplain.utils.path_utils import PathUtils


def _float_data(seed):
    """Return a reproducible float coordinate matrix (3 atoms per frame)."""
    rng = np.random.RandomState(seed)
    return np.vstack(
        [rng.rand(15, 9), rng.rand(15, 9) + 4.0]
    ).astype(np.float64)


def _binary_data(seed):
    """Return a reproducible binary contact matrix."""
    rng = np.random.RandomState(seed)
    return (rng.rand(30, 9) > 0.5).astype(np.float64)


SPECS = [
    {
        "name": "pca",
        "cls": PCACalculator,
        "data": _float_data,
        "dat": "pca.dat",
        "kwargs": {"n_components": 3},
        "changed": {"n_components": 4},
        "compute_method": "_fit_transform_to_memmap",
    },
    {
        "name": "kernel_pca",
        "cls": KernelPCACalculator,
        "data": _float_data,
        "dat": "kernel_pca_iterative.dat",
        "kwargs": {"n_components": 3, "gamma": 0.5},
        "changed": {"n_components": 3, "gamma": 1.0},
        "compute_method": "_compute_chunk_wise_rbf_kernel",
    },
    {
        "name": "contact_kernel_pca",
        "cls": ContactKernelPCACalculator,
        "data": _binary_data,
        "dat": "contact_kernel_pca_iterative.dat",
        "kwargs": {"n_components": 3, "gamma": 0.5},
        "changed": {"n_components": 3, "gamma": 1.0},
        "compute_method": "_compute_chunk_wise_rbf_kernel",
    },
    {
        "name": "diffusion_maps",
        "cls": DiffusionMapsCalculator,
        "data": _float_data,
        "dat": "diffusion_maps_iterative.dat",
        "kwargs": {"n_components": 2, "epsilon": 1.0},
        "changed": {"n_components": 2, "epsilon": 3.0},
        "compute_method": "_dispatch_method",
    },
]


def _run(spec, cache_dir, data, reuse, kwargs, expect_compute=None):
    """Compute once and return (result_copy, reused_flag).

    ``expect_compute`` asserts whether the actual computation method ran: True
    requires it (a fresh or recomputed run), False forbids it (a reused run
    must skip recomputation), None skips the check.
    """
    calc = spec["cls"](
        use_memmap=True, cache_path=cache_dir, chunk_size=8
    )
    calc.reuse_memmap_cache = reuse
    method = spec["compute_method"]
    with mock.patch.object(
        calc, method, wraps=getattr(calc, method)
    ) as spy:
        result, metadata = calc.compute(data, **kwargs)
    if expect_compute is True:
        assert spy.call_count >= 1, (
            f"{spec['name']}: expected recomputation, but the cache was reused"
        )
    elif expect_compute is False:
        assert spy.call_count == 0, (
            f"{spec['name']}: recomputed instead of reusing the cache"
        )
    out = np.array(result)
    MemmapUtils.close_memmaps_for_path(
        PathUtils.get_cache_file_path(spec["dat"], cache_dir)
    )
    return out, metadata["reused"]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_reuse_hit_returns_cached_result(spec, tmp_path):
    """Same input and params: the second run reuses without recomputing."""
    cache = str(tmp_path)
    data = spec["data"](0)

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
    data_a = spec["data"](0)
    data_b = spec["data"](1)

    first, _ = _run(
        spec, cache, data_a, False, spec["kwargs"], expect_compute=True
    )
    second, reused_second = _run(
        spec, cache, data_b, True, spec["kwargs"], expect_compute=True
    )

    assert reused_second is False
    assert not np.array_equal(second, first)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_changed_param_is_recomputed(spec, tmp_path):
    """A changed defining parameter is recomputed, not reused."""
    cache = str(tmp_path)
    data = spec["data"](0)

    _run(spec, cache, data, False, spec["kwargs"], expect_compute=True)
    _, reused_second = _run(
        spec, cache, data, True, spec["changed"], expect_compute=True
    )

    assert reused_second is False


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_flag_off_never_reuses_but_writes_sidecar(spec, tmp_path):
    """Disabled reuse recomputes but still writes the sidecar for later."""
    cache = str(tmp_path)
    data = spec["data"](0)

    _run(spec, cache, data, False, spec["kwargs"], expect_compute=True)
    _, reused_second = _run(
        spec, cache, data, False, spec["kwargs"], expect_compute=True
    )

    assert reused_second is False
    sidecar = MemmapReuseHelper.sidecar_path(
        PathUtils.get_cache_file_path(spec["dat"], cache)
    )
    assert os.path.exists(sidecar)
