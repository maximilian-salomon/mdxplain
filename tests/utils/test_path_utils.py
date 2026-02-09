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

"""Unit tests for PathUtils."""

import os
import re
import tempfile
from pathlib import Path

import pytest

from mdxplain.utils.path_utils import PathUtils


def test_prepare_file_path_canonicalizes_alias_segments():
    """prepare_file_path should return canonical path for alias segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        nested = root / "a" / "b"
        nested.mkdir(parents=True)

        alias_path = nested / ".." / "b" / "file.dat"
        normalized = PathUtils.prepare_file_path(alias_path, create_parent=True)

        expected = os.path.realpath(os.path.abspath(os.path.normpath(str(alias_path))))
        assert normalized == expected
        assert os.path.isdir(os.path.dirname(normalized))


def test_create_pipeline_cache_dir_deterministic_layout():
    """create_pipeline_cache_dir should honor explicit UUID and timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = os.path.join(tmpdir, "cache")
        scope_uuid = "0123456789abcdef0123456789abcdef"
        scope_ts = "20260208_201530"

        scoped = PathUtils.create_pipeline_cache_dir(
            base_dir,
            pipeline_uuid=scope_uuid,
            pipeline_timestamp=scope_ts,
        )

        expected = os.path.realpath(
            os.path.abspath(
                os.path.join(base_dir, f"cache_{scope_uuid}_{scope_ts}")
            )
        )
        assert scoped == expected
        assert os.path.isdir(scoped)


def test_create_pipeline_cache_dir_generates_uuid_timestamp_when_omitted():
    """Generated scoped cache path should include UUID and timestamp segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = os.path.join(tmpdir, "cache")
        scoped = PathUtils.create_pipeline_cache_dir(base_dir)

        rel = os.path.relpath(scoped, os.path.realpath(os.path.abspath(base_dir)))
        assert (
            re.fullmatch(r"cache_[0-9a-f]{32}_\d{8}_\d{6}", rel) is not None
        )


def test_prepare_file_path_rejects_invalid_pathlike():
    """Invalid path-like values should raise a ValueError."""
    with pytest.raises(ValueError, match="file path must be a valid path-like value"):
        PathUtils.prepare_file_path(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [None, 42, 3.14])
def test_prepare_file_path_rejects_invalid_pathlike_values(bad_value):
    """prepare_file_path should reject non-path-like values."""
    with pytest.raises(ValueError, match="file path must be a valid path-like value"):
        PathUtils.prepare_file_path(bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("blank", ["", " ", "   ", "\t"])
def test_prepare_file_path_rejects_blank_strings(blank):
    """prepare_file_path should reject blank paths."""
    with pytest.raises(ValueError, match="file path cannot be empty"):
        PathUtils.prepare_file_path(blank)


@pytest.mark.parametrize("create_parent", [False, True])
def test_prepare_file_path_create_parent_flag_behavior(create_parent):
    """create_parent controls whether parent folders are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "nested" / "child" / "file.dat"
        normalized = PathUtils.prepare_file_path(
            target,
            create_parent=create_parent,
            purpose="test file path",
        )
        assert normalized.endswith(os.path.join("nested", "child", "file.dat"))
        assert (target.parent.exists()) is create_parent


def test_prepare_file_path_resolve_symlinks_toggle_alias_segments():
    """resolve_symlinks=False keeps abspath/normpath result without realpath."""
    with tempfile.TemporaryDirectory() as tmpdir:
        alias_path = Path(tmpdir) / "a" / ".." / "b" / "c.dat"
        no_resolve = PathUtils.prepare_file_path(
            alias_path,
            create_parent=False,
            resolve_symlinks=False,
        )
        resolved = PathUtils.prepare_file_path(
            alias_path,
            create_parent=False,
            resolve_symlinks=True,
        )
        assert no_resolve == os.path.abspath(os.path.normpath(str(alias_path)))
        assert resolved == os.path.realpath(no_resolve)


def test_prepare_directory_path_create_false_does_not_create():
    """prepare_directory_path(create=False) should not create missing dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_dir = Path(tmpdir) / "never_created"
        normalized = PathUtils.prepare_directory_path(
            missing_dir, create=False, purpose="test directory"
        )
        assert normalized.endswith("never_created")
        assert not missing_dir.exists()


def test_prepare_directory_path_create_true_creates_directory():
    """prepare_directory_path(create=True) should create missing dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_dir = Path(tmpdir) / "created_dir"
        normalized = PathUtils.prepare_directory_path(
            missing_dir, create=True, purpose="test directory"
        )
        assert normalized.endswith("created_dir")
        assert missing_dir.exists()
        assert missing_dir.is_dir()


def test_get_cache_file_path_with_directory_cache_path():
    """Directory cache path should append cache_name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        path = PathUtils.get_cache_file_path("distances.dat", str(cache_dir))
        assert path.endswith(os.path.join("cache", "distances.dat"))
        assert cache_dir.exists()


def test_get_cache_file_path_with_file_cache_path():
    """File cache path should be used directly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "cache" / "custom_output.dat"
        path = PathUtils.get_cache_file_path("ignored.dat", str(target_file))
        assert path.endswith(os.path.join("cache", "custom_output.dat"))
        assert target_file.parent.exists()


def test_get_cache_file_path_falls_back_to_default_cache(monkeypatch):
    """Empty cache_path should fallback to ./cache/<cache_name>."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        monkeypatch.chdir(tmpdir)
        result = PathUtils.get_cache_file_path("fallback.dat", "")
        assert result == os.path.realpath(
            os.path.abspath(os.path.join(tmpdir, "cache", "fallback.dat"))
        )
        assert os.path.isdir(os.path.join(tmpdir, "cache"))
        os.chdir(original_cwd)


def test_create_pipeline_cache_dir_custom_uuid_and_timestamp():
    """Explicit UUID/timestamp should produce deterministic folder name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "cache_root"
        scoped = PathUtils.create_pipeline_cache_dir(
            base_dir,
            pipeline_uuid="0123456789abcdef0123456789abcdef",
            pipeline_timestamp="20260208_235959",
        )
        expected = os.path.realpath(
            os.path.abspath(
                base_dir / "cache_0123456789abcdef0123456789abcdef_20260208_235959"
            )
        )
        assert scoped == expected
        assert os.path.isdir(scoped)


def test_create_pipeline_cache_dir_default_pattern():
    """Default generated scoped name should match cache_<uuid>_<timestamp>."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scoped = PathUtils.create_pipeline_cache_dir(Path(tmpdir) / "cache_root")
        rel_name = os.path.basename(scoped)
        assert re.fullmatch(r"cache_[0-9a-f]{32}_\d{8}_\d{6}", rel_name)


def test_create_pipeline_cache_dir_two_different_timestamps_are_distinct():
    """Different timestamps should create different scoped directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "cache_root"
        uuid_value = "0123456789abcdef0123456789abcdef"
        first = PathUtils.create_pipeline_cache_dir(
            base_dir,
            pipeline_uuid=uuid_value,
            pipeline_timestamp="20260208_120000",
        )
        second = PathUtils.create_pipeline_cache_dir(
            base_dir,
            pipeline_uuid=uuid_value,
            pipeline_timestamp="20260208_120001",
        )
        assert first != second
        assert os.path.isdir(first)
        assert os.path.isdir(second)


def test_create_pipeline_cache_dir_reuses_already_scoped_directory():
    """Already scoped cache directories should be reused without adding nesting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scoped_dir = (
            Path(tmpdir)
            / "cache_root"
            / "cache_0123456789abcdef0123456789abcdef_20260208_120000"
        )
        scoped_dir.mkdir(parents=True, exist_ok=True)

        reused = PathUtils.create_pipeline_cache_dir(scoped_dir)
        expected = os.path.realpath(os.path.abspath(scoped_dir))
        assert reused == expected
        assert os.path.isdir(reused)
        assert os.path.basename(reused) == (
            "cache_0123456789abcdef0123456789abcdef_20260208_120000"
        )


def test_prepare_directory_path_rejects_invalid_pathlike():
    """prepare_directory_path should reject non path-like values."""
    with pytest.raises(
        ValueError, match="directory path must be a valid path-like value"
    ):
        PathUtils.prepare_directory_path(123)  # type: ignore[arg-type]


def test_prepare_directory_path_resolve_symlinks_false_uses_abspath_normpath():
    """resolve_symlinks=False should avoid realpath canonicalization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        alias_path = Path(tmpdir) / "a" / ".." / "b"
        normalized = PathUtils.prepare_directory_path(
            alias_path,
            create=False,
            resolve_symlinks=False,
        )
        assert normalized == os.path.abspath(os.path.normpath(str(alias_path)))


def test_prepare_file_path_rejects_windows_root_without_drive(monkeypatch):
    """Windows-style root-only absolute paths should fail without drive letter."""
    monkeypatch.setattr("mdxplain.utils.path_utils.os.name", "nt", raising=False)
    with pytest.raises(OSError, match="missing drive letter"):
        PathUtils.prepare_file_path("/invalid/root/path.dat")


def test_prepare_directory_path_rejects_windows_root_without_drive(monkeypatch):
    """Directory path validation should reject invalid Windows absolute roots."""
    monkeypatch.setattr("mdxplain.utils.path_utils.os.name", "nt", raising=False)
    with pytest.raises(OSError, match="missing drive letter"):
        PathUtils.prepare_directory_path("/invalid/root/path", create=False)


def test_get_cache_file_path_accepts_pathlike_directory():
    """get_cache_file_path should accept Path-like cache directory values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache_pathlike"
        path = PathUtils.get_cache_file_path("features.dat", cache_dir)
        assert path.endswith(os.path.join("cache_pathlike", "features.dat"))
        assert cache_dir.exists()
