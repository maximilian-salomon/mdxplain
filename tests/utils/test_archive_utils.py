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

"""Tests for ArchiveUtils behavior."""

from pathlib import Path
import os
import tempfile
import tarfile

import pytest

from mdxplain.utils.archive_utils import ArchiveUtils


class _DummyPipelineData:
    """Minimal pipeline-data stub for archive tests."""

    def __init__(self, cache_dir: str, use_memmap: bool = True):
        self.cache_dir = cache_dir
        self.use_memmap = use_memmap

    def save(self, target_path: str) -> None:
        with open(target_path, "wb") as handle:
            handle.write(b"dummy-pipeline-data")


def test_is_essential_file_variants():
    """Essential file rules should follow extension + use_memmap."""
    assert ArchiveUtils.is_essential_file(".pkl", use_memmap=False) is True
    assert ArchiveUtils.is_essential_file(".dat", use_memmap=True) is True
    assert ArchiveUtils.is_essential_file(".dat", use_memmap=False) is False
    assert ArchiveUtils.is_essential_file(".png", use_memmap=True) is False


def test_is_visualization_and_structure_classification():
    """Visualization and structure extension sets should be recognized."""
    assert ArchiveUtils.is_visualization_file(".png") is True
    assert ArchiveUtils.is_visualization_file(".pdf") is True
    assert ArchiveUtils.is_visualization_file(".dat") is False
    assert ArchiveUtils.is_structure_file(".pdb") is True
    assert ArchiveUtils.is_structure_file(".pml") is True
    assert ArchiveUtils.is_structure_file(".dat") is False


def test_should_include_file_rules_for_visualization_and_structure(tmp_path):
    """File inclusion should respect toggle flags."""
    dat_file = tmp_path / "x.dat"
    png_file = tmp_path / "plot.png"
    pdb_file = tmp_path / "model.pdb"
    dat_file.write_text("x", encoding="utf-8")
    png_file.write_text("x", encoding="utf-8")
    pdb_file.write_text("x", encoding="utf-8")

    assert ArchiveUtils.should_include_file(dat_file, True, True, True) is True
    assert ArchiveUtils.should_include_file(dat_file, True, True, False) is False
    assert ArchiveUtils.should_include_file(png_file, True, True, True) is False
    assert ArchiveUtils.should_include_file(png_file, False, True, True) is True
    assert ArchiveUtils.should_include_file(pdb_file, True, True, True) is True
    assert ArchiveUtils.should_include_file(pdb_file, True, False, True) is False


def test_collect_cache_files_memmap_on_includes_dat_and_zarr(tmp_path):
    """Memmap mode should include .dat and zarr directories."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.dat").write_text("data", encoding="utf-8")
    (cache / "plot.png").write_text("plot", encoding="utf-8")
    zarr_dir = cache / "traj.dask.zarr"
    zarr_dir.mkdir()
    (zarr_dir / "metadata").write_text("meta", encoding="utf-8")

    items = ArchiveUtils.collect_cache_files(
        str(cache),
        exclude_visualizations=True,
        include_structure_files=True,
        use_memmap=True,
    )
    archived = {arc.replace("\\", "/") for _, arc in items}
    assert "cache/a.dat" in archived
    assert "cache/traj.dask.zarr" in archived
    assert "cache/plot.png" not in archived


def test_collect_cache_files_memmap_off_excludes_dat_and_zarr(tmp_path):
    """Non-memmap mode should not archive .dat or zarr items."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.dat").write_text("data", encoding="utf-8")
    zarr_dir = cache / "traj.dask.zarr"
    zarr_dir.mkdir()
    (zarr_dir / "metadata").write_text("meta", encoding="utf-8")
    (cache / "model.pdb").write_text("pdb", encoding="utf-8")

    items = ArchiveUtils.collect_cache_files(
        str(cache),
        exclude_visualizations=True,
        include_structure_files=True,
        use_memmap=False,
    )
    archived = {arc.replace("\\", "/") for _, arc in items}
    assert "cache/a.dat" not in archived
    assert "cache/traj.dask.zarr" not in archived
    assert "cache/model.pdb" in archived


def test_collect_cache_files_missing_cache_returns_empty(tmp_path):
    """Missing cache directories should produce empty file list."""
    missing = tmp_path / "does_not_exist"
    items = ArchiveUtils.collect_cache_files(
        str(missing),
        exclude_visualizations=True,
        include_structure_files=True,
        use_memmap=True,
    )
    assert items == []


def test_estimate_xz_memory_per_thread_mib_level_6():
    """Level 6 should reflect raw and safety-scaled per-thread estimates."""
    assert ArchiveUtils._estimate_xz_memory_per_thread_mib(6, safety_factor=1.0) == 94
    assert ArchiveUtils._estimate_xz_memory_per_thread_mib(6) == pytest.approx(141.0)


def test_resolve_xz_threads_auto_reserve_cores(monkeypatch):
    """Automatic thread selection should keep reserve_cores free."""
    monkeypatch.setattr("mdxplain.utils.archive_utils.os.cpu_count", lambda: 12)
    threads = ArchiveUtils._resolve_xz_threads(reserve_cores=2)
    assert threads == 10


def test_resolve_xz_threads_respects_memory_cap_auto(monkeypatch):
    """Memory cap should include current process RSS when auto-selecting threads."""
    monkeypatch.setattr("mdxplain.utils.archive_utils.os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        ArchiveUtils, "_get_current_process_rss_mib", staticmethod(lambda: 1800)
    )
    threads = ArchiveUtils._resolve_xz_threads(
        xz_threads=None,
        reserve_cores=2,
        xz_level=6,
        xz_max_memory_gb=2.0,  # 2048-1800=248 MiB -> floor(248/141)=1
    )
    assert threads == 1


def test_resolve_xz_threads_respects_memory_cap_explicit(monkeypatch):
    """Memory cap should also bound explicitly requested thread count with RSS."""
    monkeypatch.setattr(
        ArchiveUtils, "_get_current_process_rss_mib", staticmethod(lambda: 1800)
    )
    threads = ArchiveUtils._resolve_xz_threads(
        xz_threads=8,
        reserve_cores=2,
        xz_level=6,
        xz_max_memory_gb=2.0,
    )
    assert threads == 1


def test_resolve_xz_threads_warns_when_available_memory_too_small(monkeypatch):
    """Insufficient available memory should warn and force single-thread mode."""
    monkeypatch.setattr(
        ArchiveUtils, "_get_current_process_rss_mib", staticmethod(lambda: 2000)
    )
    with pytest.warns(RuntimeWarning, match="Available archive memory is below one xz thread"):
        threads = ArchiveUtils._resolve_xz_threads(
            xz_threads=8,
            reserve_cores=2,
            xz_level=6,
            xz_max_memory_gb=2.0,  # 2048-2000=48 MiB < 94 MiB per thread
        )
    assert threads == 1


def test_estimate_xz_memory_per_thread_invalid_level_raises():
    """Invalid xz level should raise ValueError."""
    with pytest.raises(ValueError, match="xz_level must be in range 0-9"):
        ArchiveUtils._estimate_xz_memory_per_thread_mib(11)


@pytest.mark.parametrize(
    "threads,reserve,expected",
    [
        (None, 2, 6),  # with patched cpu=8
        (0, 2, 1),
        (-3, 2, 1),
        (4, 2, 4),
    ],
)
def test_resolve_xz_threads_bounds(monkeypatch, threads, reserve, expected):
    """Thread resolving should clamp to sane values."""
    monkeypatch.setattr("mdxplain.utils.archive_utils.os.cpu_count", lambda: 8)
    value = ArchiveUtils._resolve_xz_threads(
        xz_threads=threads,
        reserve_cores=reserve,
        xz_level=6,
        xz_max_memory_gb=None,
    )
    assert value == expected


def test_create_archive_invalid_compression_raises(tmp_path):
    """Unsupported compression should raise ValueError."""
    pipeline_data = _DummyPipelineData(str(tmp_path / "cache"), use_memmap=False)
    with pytest.raises(ValueError, match="Compression must be one of"):
        ArchiveUtils.create_archive(
            pipeline_data,
            str(tmp_path / "archive"),
            compression="zip",
        )


def test_create_archive_xz_invalid_level_raises(tmp_path):
    """Out-of-range xz compression_level should raise ValueError."""
    cache = tmp_path / "cache"
    cache.mkdir()
    pipeline_data = _DummyPipelineData(str(cache), use_memmap=False)
    with pytest.raises(ValueError, match="compression_level for xz must be in range 0-9"):
        ArchiveUtils.create_archive(
            pipeline_data,
            str(tmp_path / "archive"),
            compression="xz",
            compression_level=10,
        )


@pytest.mark.parametrize("compression", ["xz", "bz2", "gz"])
def test_create_archive_writes_expected_extension(tmp_path, compression):
    """create_archive should append extension automatically if missing."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.pdb").write_text("pdb", encoding="utf-8")
    pipeline_data = _DummyPipelineData(str(cache), use_memmap=False)
    out = ArchiveUtils.create_archive(
        pipeline_data,
        str(tmp_path / "analysis"),
        compression=compression,
    )
    assert out.endswith(f".tar.{compression}")
    assert os.path.exists(out)


def test_create_archive_and_extract_roundtrip(tmp_path):
    """Roundtrip create/extract should restore expected archive contents."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "data.dat").write_text("dat", encoding="utf-8")
    (cache / "plot.png").write_text("plot", encoding="utf-8")
    (cache / "model.pdb").write_text("pdb", encoding="utf-8")
    pipeline_data = _DummyPipelineData(str(cache), use_memmap=True)

    archive = ArchiveUtils.create_archive(
        pipeline_data,
        str(tmp_path / "roundtrip"),
        compression="gz",
        exclude_visualizations=True,
        include_structure_files=True,
    )
    assert os.path.exists(archive)

    extract_dir = ArchiveUtils.extract_archive(archive)
    assert (extract_dir / "pipeline.pkl").exists()
    assert (extract_dir / "cache" / "data.dat").exists()
    assert (extract_dir / "cache" / "model.pdb").exists()
    assert not (extract_dir / "cache" / "plot.png").exists()


def test_extract_archive_missing_file_raises(tmp_path):
    """extract_archive should raise FileNotFoundError for missing archive."""
    with pytest.raises(FileNotFoundError, match="Archive not found"):
        ArchiveUtils.extract_archive(str(tmp_path / "missing.tar.xz"))


def test_extract_archive_custom_target_directory(tmp_path):
    """extract_archive should extract into explicitly provided target directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    pipeline_data = _DummyPipelineData(str(cache), use_memmap=False)
    archive = ArchiveUtils.create_archive(
        pipeline_data,
        str(tmp_path / "custom_target_archive"),
        compression="bz2",
    )
    target = tmp_path / "target_dir"
    extract_dir = ArchiveUtils.extract_archive(archive, extract_to=str(target))
    assert extract_dir == target
    assert (target / "pipeline.pkl").exists()


def test_create_archive_contains_pipeline_pkl_entry(tmp_path):
    """Archive should always include pipeline.pkl."""
    cache = tmp_path / "cache"
    cache.mkdir()
    pipeline_data = _DummyPipelineData(str(cache), use_memmap=False)
    archive = ArchiveUtils.create_archive(
        pipeline_data,
        str(tmp_path / "contains_pipeline"),
        compression="gz",
    )
    with tarfile.open(archive, "r:*") as tar:
        names = tar.getnames()
    assert "pipeline.pkl" in names
