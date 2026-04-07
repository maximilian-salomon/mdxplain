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

"""Unit tests for CleanupUtils."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mdxplain.utils.cleanup_utils import CleanupUtils
from mdxplain.utils.memmap_utils import MemmapUtils


def test_remove_file_existing_path_returns_true_and_deletes():
    """remove_file should delete an existing file and return True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "x.txt"
        target.write_text("hello", encoding="utf-8")
        removed = CleanupUtils.remove_file(target)
        assert removed is True
        assert not target.exists()


def test_remove_file_missing_ok_true_returns_false():
    """remove_file should return False for missing file when missing_ok=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "missing.txt"
        removed = CleanupUtils.remove_file(target, missing_ok=True)
        assert removed is False


def test_remove_file_missing_ok_false_raises():
    """remove_file should raise for missing file when missing_ok=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "missing.txt"
        with pytest.raises(FileNotFoundError):
            CleanupUtils.remove_file(target, missing_ok=False)


def test_remove_file_ignore_errors_true_swallows_oserror(monkeypatch):
    """ignore_errors=True should swallow os.remove OSError and return False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "x.txt"
        target.write_text("hello", encoding="utf-8")

        def _raise(_):
            raise OSError("simulated")

        monkeypatch.setattr("mdxplain.utils.cleanup_utils.os.remove", _raise)
        removed = CleanupUtils.remove_file(target, ignore_errors=True)
        assert removed is False
        assert target.exists()


def test_remove_file_closes_memmap_for_same_path():
    """remove_file should close memmap mapped to same file before deleting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "mapped.dat"
        mapped = np.memmap(target, dtype=np.float32, mode="w+", shape=(3,))
        mapped[:] = [1, 2, 3]
        removed = CleanupUtils.remove_file(target)
        assert removed is True
        assert mapped._mmap.closed is True
        assert not target.exists()


def test_remove_tree_closes_only_memmaps_under_target_path():
    """remove_tree should not close memmaps from unrelated directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target_dir = root / "target"
        other_dir = root / "other"
        target_dir.mkdir()
        other_dir.mkdir()

        target_file = target_dir / "target.dat"
        other_file = other_dir / "other.dat"

        target_map = np.memmap(target_file, dtype=np.float32, mode="w+", shape=(4,))
        other_map = np.memmap(other_file, dtype=np.float32, mode="w+", shape=(4,))
        target_map[:] = [1, 2, 3, 4]
        other_map[:] = [5, 6, 7, 8]

        removed = CleanupUtils.remove_tree(target_dir)
        assert removed is True
        assert not target_dir.exists()

        # Memmap under unrelated directory must remain open and usable.
        assert other_map._mmap.closed is False
        other_map[0] = 99
        other_map.flush()
        assert other_map[0] == 99

        MemmapUtils.close_memmap_view(other_map)
        shutil.rmtree(other_dir)


def test_remove_tree_missing_ok_true_returns_false():
    """remove_tree should return False for missing dir when missing_ok=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "missing_dir"
        removed = CleanupUtils.remove_tree(target, missing_ok=True)
        assert removed is False


def test_remove_tree_missing_ok_false_raises():
    """remove_tree should raise for missing dir when missing_ok=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "missing_dir"
        with pytest.raises(FileNotFoundError):
            CleanupUtils.remove_tree(target, missing_ok=False)


def test_remove_path_dispatches_to_file_removal():
    """remove_path should remove files via file branch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "file.txt"
        target.write_text("x", encoding="utf-8")
        removed = CleanupUtils.remove_path(target)
        assert removed is True
        assert not target.exists()


def test_remove_path_dispatches_to_directory_removal():
    """remove_path should remove directories via tree branch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "dir"
        target.mkdir()
        (target / "a.txt").write_text("x", encoding="utf-8")
        removed = CleanupUtils.remove_path(target)
        assert removed is True
        assert not target.exists()


def test_remove_path_missing_ok_false_raises():
    """remove_path should raise for missing path when missing_ok=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "missing_any"
        with pytest.raises(FileNotFoundError):
            CleanupUtils.remove_path(target, missing_ok=False)


def test_remove_tree_ignore_errors_true_still_returns_true():
    """remove_tree should return True after rmtree call with ignore_errors=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "dir"
        target.mkdir()
        (target / "a.txt").write_text("x", encoding="utf-8")
        removed = CleanupUtils.remove_tree(target, ignore_errors=True)
        assert removed is True
        assert not os.path.exists(target)


def test_remove_tree_raises_oserror_when_rmtree_fails(tmp_path, monkeypatch):
    """remove_tree should propagate OSError when ignore_errors=False."""
    target = tmp_path / "dir"
    target.mkdir()

    def _raise(*_, **__):
        raise OSError("simulated rmtree failure")

    monkeypatch.setattr("mdxplain.utils.cleanup_utils.shutil.rmtree", _raise)
    with pytest.raises(OSError, match="simulated rmtree failure"):
        CleanupUtils.remove_tree(target, ignore_errors=False)


def test_remove_path_directory_closes_only_target_tree_memmaps():
    """remove_path on directory should close memmaps only inside that directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target_dir = root / "target"
        sibling_dir = root / "target_sibling"
        target_dir.mkdir()
        sibling_dir.mkdir()

        target_map = np.memmap(
            target_dir / "target.dat", dtype=np.float32, mode="w+", shape=(2,)
        )
        sibling_map = np.memmap(
            sibling_dir / "sibling.dat", dtype=np.float32, mode="w+", shape=(2,)
        )

        removed = CleanupUtils.remove_path(target_dir)
        assert removed is True
        assert target_map._mmap.closed is True
        assert sibling_map._mmap.closed is False
        MemmapUtils.close_memmap_view(sibling_map)
        shutil.rmtree(sibling_dir)


def test_remove_path_file_closes_only_target_file_memmap():
    """remove_path on file should close exact file mapping only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        file_a = root / "a.dat"
        file_b = root / "b.dat"
        map_a = np.memmap(file_a, dtype=np.float32, mode="w+", shape=(2,))
        map_b = np.memmap(file_b, dtype=np.float32, mode="w+", shape=(2,))

        removed = CleanupUtils.remove_path(file_a)
        assert removed is True
        assert map_a._mmap.closed is True
        assert map_b._mmap.closed is False
        MemmapUtils.close_memmap_view(map_b)
