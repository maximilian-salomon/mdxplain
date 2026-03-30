# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex.
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

"""Tests for ArchiveFetchHelper behavior."""

from pathlib import Path

import pytest

from mdxplain.utils.archive_fetch_helper import ArchiveFetchHelper
from mdxplain.utils.archive_utils import ArchiveUtils


def test_validate_load_inputs_verify_requires_sha_by_default():
    """Default verification should require SHA input."""
    with pytest.raises(ValueError, match="Archive verification requires sha"):
        ArchiveFetchHelper.validate_load_inputs(
            file_path="https://example.org/archive.tar.zst",
            verify=True,
            sha=None,
            download_url=None,
        )


def test_validate_load_inputs_sha_enables_verification_even_if_false():
    """Providing SHA input should be accepted even with verify=False."""
    ArchiveFetchHelper.validate_load_inputs(
        file_path="analysis.tar.zst",
        verify=False,
        sha="a" * 64,
        download_url=None,
    )


def test_resolve_expected_sha256_accepts_raw_text_file_and_url(tmp_path):
    """SHA256 resolution should support raw strings, local files, and URLs."""
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"archive-payload")
    digest = ArchiveUtils.compute_sha256(str(payload_path))

    sha_file = tmp_path / "payload.bin.sha"
    sha_file.write_text(f"{digest}  payload.bin\n", encoding="utf-8")

    assert ArchiveFetchHelper.resolve_expected_sha256(digest) == digest
    assert ArchiveFetchHelper.resolve_expected_sha256(str(sha_file)) == digest
    assert ArchiveFetchHelper.resolve_expected_sha256(sha_file.resolve().as_uri()) == digest


def test_resolve_archive_path_downloads_file_url_to_target(tmp_path):
    """Remote archive paths should be downloaded to the requested target path."""
    source_archive = tmp_path / "source.tar.zst"
    source_archive.write_bytes(b"archive-content")
    target_archive = tmp_path / "downloaded" / "archive.tar.zst"

    resolved = ArchiveFetchHelper.resolve_archive_path(
        file_path=str(target_archive),
        cache_dir=str(tmp_path / "cache"),
        verify=False,
        sha=None,
        download_url=source_archive.resolve().as_uri(),
        overwrite=False,
    )

    assert resolved == str(target_archive.resolve())
    assert target_archive.read_bytes() == b"archive-content"


def test_resolve_archive_path_reuses_existing_file_with_warning(tmp_path):
    """Existing target files should be reused when overwrite=False."""
    source_archive = tmp_path / "source.tar.zst"
    source_archive.write_bytes(b"fresh-content")
    target_archive = tmp_path / "downloaded.tar.zst"
    target_archive.write_bytes(b"existing-content")

    with pytest.warns(RuntimeWarning, match="target file already exists"):
        resolved = ArchiveFetchHelper.resolve_archive_path(
            file_path=str(target_archive),
            cache_dir=str(tmp_path / "cache"),
            verify=False,
            sha=None,
            download_url=source_archive.resolve().as_uri(),
            overwrite=False,
        )

    assert resolved == str(target_archive.resolve())
    assert target_archive.read_bytes() == b"existing-content"


def test_verify_archive_sha256_raises_on_mismatch(tmp_path):
    """Archive SHA256 verification should fail on mismatched digests."""
    archive_path = tmp_path / "archive.tar.zst"
    archive_path.write_bytes(b"payload")

    with pytest.raises(ValueError, match="verification failed"):
        ArchiveFetchHelper.verify_archive_sha256(
            file_path=str(archive_path),
            sha="0" * 64,
        )


def test_should_verify_archive_uses_sha_even_when_verify_false():
    """An explicit SHA input should force verification."""
    assert ArchiveFetchHelper.should_verify_archive(False, "a" * 64) is True
    assert ArchiveFetchHelper.should_verify_archive(False, None) is False
