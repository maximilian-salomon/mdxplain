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
Per-type reuse tests for the feature calculators.

Every feature type is checked flag-based via the ``reused`` marker each
calculator reports: a matching cache is reused (and the actual compute step is
never called), a changed input (same shape) or a changed parameter is
recomputed, and the sidecar is always written under use_memmap. Contacts take
a plain distance array; the trajectory-based calculators run on a small real
MDTraj trajectory built from a bundled PDB structure.
"""

import os
from unittest import mock

import mdtraj as md
import numpy as np
import pytest

from mdxplain.feature.feature_type.contacts.contact_calculator import (
    ContactCalculator,
)
from mdxplain.feature.feature_type.coordinates.coordinates_calculator import (
    CoordinatesCalculator,
)
from mdxplain.feature.feature_type.distances.distance_calculator import (
    DistanceCalculator,
)
from mdxplain.feature.feature_type.dssp.dssp_calculator import DSSPCalculator
from mdxplain.feature.feature_type.sasa.sasa_calculator import SASACalculator
from mdxplain.feature.feature_type.torsions.torsions_calculator import (
    TorsionsCalculator,
)
from mdxplain.utils.memmap_reuse_helper import MemmapReuseHelper
from mdxplain.utils.memmap_utils import MemmapUtils

_PDB = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test_ressources",
    "tutorial_test_pipeline",
    "cache",
    "structure_viz",
    "cluster_0_vs_rest.pdb",
)


def _distances(seed):
    """Build a deterministic condensed distance array in Angstrom."""
    rng = np.random.RandomState(seed)
    return (rng.rand(20, 6) * 8.0).astype(np.float32)


@pytest.fixture(scope="module")
def trajectory():
    """Build identical and jittered real trajectories, or None if unavailable.

    Returns (traj_a, traj_b, res_metadata); traj_b jitters the coordinates so
    it has the same shape but different content than traj_a.
    """
    if not os.path.exists(_PDB):
        return None
    base = md.load(_PDB)
    xyz = np.repeat(base.xyz, 6, axis=0)
    traj_a = md.Trajectory(xyz, base.topology)
    rng = np.random.RandomState(0)
    jitter = rng.normal(0.0, 0.05, xyz.shape).astype(xyz.dtype)
    traj_b = md.Trajectory(xyz + jitter, base.topology)
    res_metadata = [
        {
            "index": i,
            "seqid": res.resSeq,
            "full_name": f"{res.name}{res.resSeq}",
        }
        for i, res in enumerate(base.topology.residues)
    ]
    return traj_a, traj_b, res_metadata


SPECS = [
    {
        "name": "contacts",
        "cls": ContactCalculator,
        "dat": "contacts.dat",
        "kwargs": {"cutoff": 4.5},
        "changed": {"cutoff": 6.0},
        "compute_method": "_fill_contacts",
        "needs_traj": False,
        "needs_res": False,
    },
    {
        "name": "coordinates",
        "cls": CoordinatesCalculator,
        "dat": "coordinates.dat",
        "kwargs": {"selection": "name CA"},
        "changed": {"selection": "backbone"},
        "compute_method": "_extract_coordinates",
        "needs_traj": True,
        "needs_res": False,
    },
    {
        "name": "distances",
        "cls": DistanceCalculator,
        "dat": "distances.dat",
        "kwargs": {"excluded_neighbors": 1},
        "changed": {"excluded_neighbors": 3},
        "compute_method": "_process_trajectory",
        "needs_traj": True,
        "needs_res": True,
    },
    {
        "name": "sasa",
        "cls": SASACalculator,
        "dat": "sasa.dat",
        "kwargs": {"mode": "residue", "probe_radius": 0.14},
        "changed": {"mode": "residue", "probe_radius": 0.2},
        "compute_method": "_compute_sasa",
        "needs_traj": True,
        "needs_res": True,
    },
    {
        "name": "dssp",
        "cls": DSSPCalculator,
        "dat": "dssp.dat",
        "kwargs": {"simplified": True, "encoding": "onehot"},
        "changed": {"simplified": False, "encoding": "onehot"},
        "compute_method": "_compute_dssp_assignments",
        "needs_traj": True,
        "needs_res": True,
    },
    {
        "name": "torsions",
        "cls": TorsionsCalculator,
        "dat": "torsions.dat",
        "kwargs": {
            "calculate_phi": True,
            "calculate_psi": True,
            "calculate_omega": False,
            "calculate_chi": False,
        },
        "changed": {
            "calculate_phi": True,
            "calculate_psi": True,
            "calculate_omega": True,
            "calculate_chi": False,
        },
        "compute_method": "_compute_torsion_angles",
        "needs_traj": True,
        "needs_res": True,
    },
]


def _resolve_inputs(spec, trajectory):
    """Return (input_a, input_b, res_metadata) for a spec.

    Trajectory-based specs use the shared real-trajectory fixture (skipping
    when it is unavailable); contacts use two distinct distance arrays.
    """
    if spec["needs_traj"]:
        if trajectory is None:
            pytest.skip("test structure PDB not available")
        return trajectory
    return _distances(0), _distances(1), None


def _run(spec, path, input_data, reuse, kwargs, res_metadata,
         expect_compute=None):
    """Run one calculator; return (result_copy, reused_flag).

    ``expect_compute`` asserts whether the calculator's compute step ran:
    True requires it (a fresh or recomputed run), False forbids it (a reused
    run must skip recomputation), None skips the check.
    """
    calc = spec["cls"](use_memmap=True, cache_path=path, chunk_size=4)
    calc.reuse_memmap_cache = reuse
    call_kwargs = dict(kwargs)
    if spec["needs_res"]:
        call_kwargs["res_metadata"] = res_metadata
    method = spec["compute_method"]
    with mock.patch.object(
        calc, method, wraps=getattr(calc, method)
    ) as spy:
        result, metadata = calc.compute(input_data, **call_kwargs)
    if expect_compute is True:
        assert spy.call_count >= 1, (
            f"{spec['name']}: expected recomputation, but the cache was reused"
        )
    elif expect_compute is False:
        assert spy.call_count == 0, (
            f"{spec['name']}: recomputed instead of reusing the cache"
        )
    out = np.array(result)
    MemmapUtils.close_memmaps_for_path(path)
    return out, metadata["reused"]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_reuse_hit_returns_cached_result(spec, tmp_path, trajectory):
    """Same input and params: the second run reuses without recomputing."""
    input_a, _, res_metadata = _resolve_inputs(spec, trajectory)
    path = str(tmp_path / spec["dat"])

    first, reused_first = _run(
        spec, path, input_a, False, spec["kwargs"], res_metadata,
        expect_compute=True,
    )
    assert reused_first is False

    second, reused_second = _run(
        spec, path, input_a, True, spec["kwargs"], res_metadata,
        expect_compute=False,
    )
    assert reused_second is True
    assert np.array_equal(second, first)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_changed_input_is_recomputed(spec, tmp_path, trajectory):
    """A different input of the same shape is recomputed, not reused stale."""
    input_a, input_b, res_metadata = _resolve_inputs(spec, trajectory)
    path = str(tmp_path / spec["dat"])

    _run(
        spec, path, input_a, False, spec["kwargs"], res_metadata,
        expect_compute=True,
    )
    _, reused_second = _run(
        spec, path, input_b, True, spec["kwargs"], res_metadata,
        expect_compute=True,
    )
    assert reused_second is False


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_changed_param_is_recomputed(spec, tmp_path, trajectory):
    """A changed defining parameter is recomputed, not reused."""
    input_a, _, res_metadata = _resolve_inputs(spec, trajectory)
    path = str(tmp_path / spec["dat"])

    _run(
        spec, path, input_a, False, spec["kwargs"], res_metadata,
        expect_compute=True,
    )
    _, reused_second = _run(
        spec, path, input_a, True, spec["changed"], res_metadata,
        expect_compute=True,
    )
    assert reused_second is False


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["name"])
def test_flag_off_never_reuses_but_writes_sidecar(spec, tmp_path, trajectory):
    """Disabled reuse recomputes but still writes the sidecar for later."""
    input_a, _, res_metadata = _resolve_inputs(spec, trajectory)
    path = str(tmp_path / spec["dat"])

    _run(
        spec, path, input_a, False, spec["kwargs"], res_metadata,
        expect_compute=True,
    )
    _, reused_second = _run(
        spec, path, input_a, False, spec["kwargs"], res_metadata,
        expect_compute=True,
    )
    assert reused_second is False
    assert os.path.exists(MemmapReuseHelper.sidecar_path(path))
