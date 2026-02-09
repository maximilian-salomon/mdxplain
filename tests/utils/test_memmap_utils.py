# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex GPT-5.
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

"""Unit tests for MemmapUtils."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mdxplain.utils.memmap_utils import MemmapUtils


def test_prepare_memmap_path_write_mode_creates_parent():
    """Write modes should create parent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "nested" / "file.dat"
        prepared = MemmapUtils.prepare_memmap_path(target, mode="w+")
        assert prepared.endswith(os.path.join("nested", "file.dat"))
        assert target.parent.exists()


def test_prepare_memmap_path_read_mode_does_not_create_parent():
    """Read mode should not create parent directory automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "read_only" / "file.dat"
        prepared = MemmapUtils.prepare_memmap_path(target, mode="r")
        assert prepared.endswith(os.path.join("read_only", "file.dat"))
        assert not target.parent.exists()


def test_is_memmap_view_true_for_memmap_and_view():
    """is_memmap_view should detect both memmap and ndarray views on memmap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "x.dat"
        arr = np.memmap(target, dtype=np.float32, mode="w+", shape=(4, 3))
        view = arr[:, :2]
        assert MemmapUtils.is_memmap_view(arr) is True
        assert MemmapUtils.is_memmap_view(view) is True
        MemmapUtils.close_memmap_view(arr)


def test_is_memmap_view_false_for_plain_ndarray():
    """is_memmap_view should return False for normal ndarray."""
    arr = np.zeros((3, 2), dtype=np.float32)
    assert MemmapUtils.is_memmap_view(arr) is False


def test_close_memmap_view_handles_plain_array_noop():
    """close_memmap_view should no-op for non-memmap arrays."""
    arr = np.ones((2, 2), dtype=np.float32)
    MemmapUtils.close_memmap_view(arr)
    np.testing.assert_array_equal(arr, np.ones((2, 2), dtype=np.float32))


def test_close_memmap_view_is_idempotent():
    """close_memmap_view should be safe when called multiple times."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "x.dat"
        arr = np.memmap(target, dtype=np.float32, mode="w+", shape=(3,))
        MemmapUtils.close_memmap_view(arr)
        MemmapUtils.close_memmap_view(arr)
        assert arr._mmap.closed is True


def test_create_memmap_zero_shape_returns_ndarray():
    """Zero-sized shapes should fallback to plain ndarray."""
    arr = MemmapUtils.create_memmap(
        path="unused.dat",
        dtype=np.float32,
        mode="w+",
        shape=(0, 4),
    )
    assert isinstance(arr, np.ndarray)
    assert not isinstance(arr, np.memmap)
    assert arr.shape == (0, 4)


def test_create_memmap_close_existing_true_closes_old_mapping():
    """Creating with close_existing=True should close stale mapping on same path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "same.dat"
        old = np.memmap(target, dtype=np.float32, mode="w+", shape=(2,))
        new = MemmapUtils.create_memmap(
            path=target,
            dtype=np.float32,
            mode="w+",
            shape=(2,),
            close_existing=True,
        )
        assert old._mmap.closed is True
        assert isinstance(new, np.memmap)
        MemmapUtils.close_memmap_view(new)


def test_create_memmap_close_existing_false_keeps_old_mapping_open():
    """close_existing=False should not proactively close previous mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "same.dat"
        old = np.memmap(target, dtype=np.float32, mode="w+", shape=(2,))
        old[:] = [1.0, 2.0]
        new = MemmapUtils.create_memmap(
            path=target,
            dtype=np.float32,
            mode="r+",
            shape=(2,),
            close_existing=False,
        )
        assert old._mmap.closed is False
        assert new._mmap.closed is False
        MemmapUtils.close_memmap_view(new)
        MemmapUtils.close_memmap_view(old)


def test_create_memmap_access_pattern_none():
    """access_pattern=None should skip tuning and still create memmap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "x.dat"
        arr = MemmapUtils.create_memmap(
            path=target,
            dtype=np.float32,
            mode="w+",
            shape=(3,),
            access_pattern=None,
        )
        assert isinstance(arr, np.memmap)
        MemmapUtils.close_memmap_view(arr)


def test_close_memmaps_under_path_scoped_to_directory():
    """Only memmaps under the given directory should be closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        dir_a = root / "a"
        dir_b = root / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        path_a = dir_a / "a.dat"
        path_b = dir_b / "b.dat"

        mem_a = np.memmap(path_a, dtype=np.float32, mode="w+", shape=(4,))
        mem_b = np.memmap(path_b, dtype=np.float32, mode="w+", shape=(4,))
        mem_a[:] = [1, 2, 3, 4]
        mem_b[:] = [5, 6, 7, 8]

        MemmapUtils.close_memmaps_under_path(dir_a)

        assert mem_a._mmap.closed is True
        assert mem_b._mmap.closed is False

        # Unrelated memmap remains usable.
        mem_b[0] = 42
        mem_b.flush()
        assert mem_b[0] == 42

        MemmapUtils.close_memmap_view(mem_b)


def test_close_memmaps_for_path_exact_match_only():
    """Path-based close should only target the exact memmap path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path_a = root / "a.dat"
        path_b = root / "b.dat"

        mem_a = np.memmap(path_a, dtype=np.float32, mode="w+", shape=(2,))
        mem_b = np.memmap(path_b, dtype=np.float32, mode="w+", shape=(2,))

        MemmapUtils.close_memmaps_for_path(path_a)

        assert mem_a._mmap.closed is True
        assert mem_b._mmap.closed is False

        MemmapUtils.close_memmap_view(mem_b)


def test_close_memmaps_for_path_accepts_alias_path_representation():
    """close_memmaps_for_path should match canonicalized alias paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir()
        memmap_path = data_dir / "x.dat"

        mem = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(3,))

        alias_path = data_dir / "." / "x.dat"
        MemmapUtils.close_memmaps_for_path(alias_path)

        assert mem._mmap.closed is True


def test_close_memmaps_under_path_accepts_file_root_and_closes_exact_file():
    """Passing a file path root should close exactly that mapped file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path_a = root / "a.dat"
        path_b = root / "b.dat"
        mem_a = np.memmap(path_a, dtype=np.float32, mode="w+", shape=(2,))
        mem_b = np.memmap(path_b, dtype=np.float32, mode="w+", shape=(2,))

        MemmapUtils.close_memmaps_under_path(path_a)
        assert mem_a._mmap.closed is True
        assert mem_b._mmap.closed is False
        MemmapUtils.close_memmap_view(mem_b)


def test_close_memmaps_for_path_no_match_does_not_close_other():
    """Non-matching target path should not close unrelated memmaps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mapped = root / "mapped.dat"
        unknown = root / "unknown.dat"
        mem = np.memmap(mapped, dtype=np.float32, mode="w+", shape=(2,))

        MemmapUtils.close_memmaps_for_path(unknown)
        assert mem._mmap.closed is False
        MemmapUtils.close_memmap_view(mem)


def test_create_memmap_defaults_close_existing_for_write_mode_w_plus():
    """Write-truncate modes should default to close_existing=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "x.dat"
        old = np.memmap(target, dtype=np.float32, mode="w+", shape=(2,))
        new = MemmapUtils.create_memmap(
            path=target,
            dtype=np.float32,
            mode="w+",
            shape=(2,),
        )
        assert old._mmap.closed is True
        MemmapUtils.close_memmap_view(new)


def test_create_memmap_defaults_do_not_close_existing_for_read_write_mode():
    """r+ mode should default to close_existing=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "x.dat"
        first = np.memmap(target, dtype=np.float32, mode="w+", shape=(3,))
        second = MemmapUtils.create_memmap(
            path=target,
            dtype=np.float32,
            mode="r+",
            shape=(3,),
        )
        assert first._mmap.closed is False
        assert second._mmap.closed is False
        MemmapUtils.close_memmap_view(second)
        MemmapUtils.close_memmap_view(first)


def test_close_memmaps_under_path_does_not_match_prefix_siblings():
    """Path scoping should not close sibling directories with shared prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        dir_a = root / "cache_a"
        dir_ab = root / "cache_ab"
        dir_a.mkdir()
        dir_ab.mkdir()

        mem_a = np.memmap(dir_a / "a.dat", dtype=np.float32, mode="w+", shape=(2,))
        mem_ab = np.memmap(dir_ab / "ab.dat", dtype=np.float32, mode="w+", shape=(2,))

        MemmapUtils.close_memmaps_under_path(dir_a)

        assert mem_a._mmap.closed is True
        assert mem_ab._mmap.closed is False
        MemmapUtils.close_memmap_view(mem_ab)


def test_close_memmaps_under_path_accepts_alias_directory_path():
    """Alias directory paths should canonicalize to the same cleanup root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        subdir = root / "data"
        subdir.mkdir()
        mem = np.memmap(subdir / "x.dat", dtype=np.float32, mode="w+", shape=(2,))

        alias = root / "data" / "."
        MemmapUtils.close_memmaps_under_path(alias)

        assert mem._mmap.closed is True
