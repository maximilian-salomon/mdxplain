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

"""Unit tests for MemmapReuseHelper."""

import numpy as np

from mdxplain.utils.memmap_reuse_helper import MemmapReuseHelper


def _write_dat(path, shape, dtype, fill):
    """Write raw C-order bytes of a filled array to a .dat file."""
    np.full(shape, fill, dtype=dtype).tofile(str(path))


def test_reuse_roundtrip_returns_matching_memmap(tmp_path):
    """A matching sidecar reopens the memmap with the exact content."""
    path = tmp_path / "contacts.dat"
    shape = (10, 4)
    _write_dat(path, shape, np.float32, 1.5)
    params = {"cutoff": 4.5, "selection": "all"}
    MemmapReuseHelper.write_sidecar(str(path), shape, np.float32, params)

    reused = MemmapReuseHelper.try_reuse(str(path), np.float32, params)
    assert reused is not None
    assert reused.shape == shape
    assert np.array_equal(reused, np.full(shape, 1.5, dtype=np.float32))


def test_reuse_none_when_params_differ(tmp_path):
    """A cache computed with different params is not reused."""
    path = tmp_path / "contacts.dat"
    shape = (10, 4)
    _write_dat(path, shape, np.float32, 1.0)
    MemmapReuseHelper.write_sidecar(
        str(path), shape, np.float32, {"cutoff": 4.5}
    )
    assert (
        MemmapReuseHelper.try_reuse(str(path), np.float32, {"cutoff": 6.0})
        is None
    )


def test_reuse_none_when_dtype_differs(tmp_path):
    """A cache with a different dtype is not reused."""
    path = tmp_path / "c.dat"
    shape = (8, 2)
    _write_dat(path, shape, np.float32, 1.0)
    MemmapReuseHelper.write_sidecar(str(path), shape, np.float32, {"a": 1})
    assert MemmapReuseHelper.try_reuse(str(path), np.float64, {"a": 1}) is None


def test_reuse_none_when_sidecar_missing(tmp_path):
    """Without a sidecar the memmap is not reused."""
    path = tmp_path / "c.dat"
    _write_dat(path, (8, 2), np.float32, 1.0)
    assert MemmapReuseHelper.try_reuse(str(path), np.float32, {"a": 1}) is None


def test_reuse_none_when_memmap_missing(tmp_path):
    """A sidecar without its .dat file does not reuse."""
    path = tmp_path / "c.dat"
    MemmapReuseHelper.write_sidecar(str(path), (8, 2), np.float32, {"a": 1})
    assert MemmapReuseHelper.try_reuse(str(path), np.float32, {"a": 1}) is None


def test_reuse_none_when_file_truncated(tmp_path):
    """A partially written (truncated) file is rejected by the size check."""
    path = tmp_path / "c.dat"
    shape = (10, 4)
    _write_dat(path, shape, np.float32, 1.0)
    params = {"a": 1}
    MemmapReuseHelper.write_sidecar(str(path), shape, np.float32, params)
    with open(str(path), "r+b") as handle:
        handle.truncate(16)
    assert MemmapReuseHelper.try_reuse(str(path), np.float32, params) is None


def test_reuse_with_payload_roundtrip(tmp_path):
    """try_reuse_with_payload restores the memmap and the pickled payload."""
    path = str(tmp_path / "pca.dat")
    shape = (10, 3)
    _write_dat(path, shape, np.float64, 2.0)
    params = {"n_components": 3, "method": "incremental_pca"}
    meta = {"explained_variance_ratio": [0.5, 0.3, 0.2], "method": "x"}
    MemmapReuseHelper.write_sidecar(
        str(path), shape, np.float64, params, payload=meta
    )

    result = MemmapReuseHelper.try_reuse_with_payload(str(path), params)
    assert result is not None
    memmap, payload = result
    assert memmap.shape == shape
    assert memmap.dtype == np.float64
    assert payload == meta


def test_reuse_with_payload_none_on_param_mismatch(tmp_path):
    """Payload reuse is refused when params differ."""
    path = str(tmp_path / "pca.dat")
    _write_dat(path, (10, 3), np.float64, 1.0)
    MemmapReuseHelper.write_sidecar(
        str(path), (10, 3), np.float64, {"n_components": 3}, payload={"a": 1}
    )
    assert (
        MemmapReuseHelper.try_reuse_with_payload(str(path), {"n_components": 5})
        is None
    )


def test_remove_sidecar_also_removes_payload(tmp_path):
    """Removing the sidecar also removes the pickled payload file."""
    path = str(tmp_path / "pca.dat")
    _write_dat(path, (4, 2), np.float64, 1.0)
    MemmapReuseHelper.write_sidecar(
        str(path), (4, 2), np.float64, {"a": 1}, payload={"m": 1}
    )
    assert MemmapReuseHelper.load_payload(str(path)) == {"m": 1}
    MemmapReuseHelper.remove_sidecar(str(path))
    assert MemmapReuseHelper.load_payload(str(path)) is None
    assert MemmapReuseHelper.try_reuse_with_payload(str(path), {"a": 1}) is None


def test_remove_sidecar_disables_reuse_and_is_idempotent(tmp_path):
    """Removing the sidecar disables reuse and tolerates a missing sidecar."""
    path = tmp_path / "c.dat"
    _write_dat(path, (4, 2), np.float32, 1.0)
    MemmapReuseHelper.write_sidecar(str(path), (4, 2), np.float32, {"a": 1})
    assert (
        MemmapReuseHelper.try_reuse(str(path), np.float32, {"a": 1}) is not None
    )
    MemmapReuseHelper.remove_sidecar(str(path))
    assert MemmapReuseHelper.try_reuse(str(path), np.float32, {"a": 1}) is None
    MemmapReuseHelper.remove_sidecar(str(path))


def test_is_reuse_artifact_recognizes_sidecar_and_payload():
    """The sidecar and payload companions are recognized, the .dat is not."""
    assert MemmapReuseHelper.is_reuse_artifact("cache/pca.dat.reuse.json")
    assert MemmapReuseHelper.is_reuse_artifact("cache/pca.dat.reuse.pkl")
    assert not MemmapReuseHelper.is_reuse_artifact("cache/pca.dat")
    assert not MemmapReuseHelper.is_reuse_artifact("cache/model.pkl")
