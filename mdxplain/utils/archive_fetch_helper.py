# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex GPT 5.4.
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
Helpers for resolving archive sources and optional SHA256 verification.

This module keeps URL/download handling separate from archive extraction so
PipelineManager can expose a compact API while ArchiveUtils remains focused on
local archive files.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from .archive_utils import ArchiveUtils
from .path_utils import PathUtils


class ArchiveFetchHelper:
    """
    Resolve archive sources and verify downloaded archives when requested.

    The helper accepts either local paths or URLs, downloads remote archives
    into a deterministic local file when needed, and optionally validates the
    resulting archive via SHA256.
    """

    @staticmethod
    def is_url(value: str) -> bool:
        """
        Return whether ``value`` looks like a supported URL.

        Parameters
        ----------
        value : str
            Candidate archive or SHA source.

        Returns
        -------
        bool
            True when the value has an HTTP(S) or file URL scheme.
        """
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https", "file"}

    @staticmethod
    def validate_load_inputs(
        file_path: str,
        verify: bool,
        sha: Optional[str],
        download_url: Optional[str],
    ) -> None:
        """
        Validate public ``load_from_archive`` fetch parameters.

        Parameters
        ----------
        file_path : str
            Local archive path, local download target, or remote archive URL.
        verify : bool
            Requested verification mode.
        sha : str or None
            SHA256 input as raw hash, local file path, or URL.
        download_url : str or None
            Optional remote source URL when ``file_path`` should be treated as
            the local download target.

        Returns
        -------
        None
            Raises ``ValueError`` for invalid argument combinations.
        """
        is_remote = ArchiveFetchHelper.is_url(file_path)
        if download_url is not None and not ArchiveFetchHelper.is_url(download_url):
            raise ValueError(
                "download_url must be a remote URL when provided."
            )
        if download_url is not None and is_remote:
            raise ValueError(
                "file_path must be a local target path when download_url is used."
            )
        if ArchiveFetchHelper.should_verify_archive(verify=verify, sha=sha) and sha is None:
            raise ValueError(
                "Archive verification requires sha to be provided as a hash, "
                "file path, or URL. Set verify=False to skip SHA256 validation."
            )

    @staticmethod
    def should_verify_archive(verify: bool, sha: Optional[str]) -> bool:
        """
        Return whether archive SHA256 verification should be performed.

        Parameters
        ----------
        verify : bool
            User-requested verification flag.
        sha : str or None
            Optional SHA256 input. When present, verification is always enabled.

        Returns
        -------
        bool
            True when archive verification should run.
        """
        return bool(verify or sha is not None)

    @staticmethod
    def resolve_archive_path(
        file_path: str,
        cache_dir: str,
        verify: bool,
        sha: Optional[str],
        download_url: Optional[str],
        overwrite: bool,
    ) -> str:
        """
        Resolve a local archive path from a user-provided source.

        Parameters
        ----------
        file_path : str
            Local archive path, local download target, or remote archive URL.
        cache_dir : str
            Cache root used to derive a default download location.
        verify : bool
            Requested verification mode.
        sha : str or None
            SHA256 input as raw hash, file path, or URL.
        download_url : str or None
            Optional remote source URL when ``file_path`` is the desired local
            archive target.
        overwrite : bool
            Whether an existing download target should be replaced.

        Returns
        -------
        str
            Normalized absolute path to the local archive file.
        """
        ArchiveFetchHelper.validate_load_inputs(
            file_path=file_path,
            verify=verify,
            sha=sha,
            download_url=download_url,
        )
        if download_url is not None:
            target_path = PathUtils.prepare_file_path(
                file_path,
                create_parent=True,
                purpose="archive download path",
            )
            source_url = download_url
            return ArchiveFetchHelper._fetch_remote_archive(
                file_url=source_url,
                target_path=target_path,
                overwrite=overwrite,
            )

        if not ArchiveFetchHelper.is_url(file_path):
            return PathUtils.prepare_file_path(
                file_path,
                create_parent=False,
                purpose="archive path",
            )

        target_path = ArchiveFetchHelper._resolve_default_download_path(
            file_url=file_path,
            cache_dir=cache_dir,
        )
        return ArchiveFetchHelper._fetch_remote_archive(
            file_url=file_path,
            target_path=target_path,
            overwrite=overwrite,
        )

    @staticmethod
    def _fetch_remote_archive(
        file_url: str,
        target_path: str,
        overwrite: bool,
    ) -> str:
        """
        Fetch a remote archive into a local target path.

        Parameters
        ----------
        file_url : str
            Remote archive URL.
        target_path : str
            Local archive path to populate.
        overwrite : bool
            Whether an existing local file may be replaced.

        Returns
        -------
        str
            Normalized absolute path to the local archive file.
        """
        if os.path.exists(target_path) and not overwrite:
            ArchiveFetchHelper._warn_reusing_existing_file(target_path)
            return target_path

        ArchiveFetchHelper._download_to_path(
            file_url=file_url,
            target_path=target_path,
        )
        return target_path

    @staticmethod
    def resolve_expected_sha256(sha: str) -> str:
        """
        Resolve a SHA256 value from raw text, a local file, or a URL.

        Parameters
        ----------
        sha : str
            Raw SHA256 hex string, local path to a ``.sha`` file, or URL.

        Returns
        -------
        str
            Normalized lowercase SHA256 hex string.
        """
        if ArchiveUtils.is_sha256_string(sha):
            return sha.strip().lower()
        if ArchiveFetchHelper.is_url(sha):
            content = ArchiveFetchHelper._read_url_text(sha)
            return ArchiveUtils.parse_sha256_text(content)
        content = ArchiveFetchHelper._read_local_text(sha)
        return ArchiveUtils.parse_sha256_text(content)

    @staticmethod
    def verify_archive_sha256(file_path: str, sha: str) -> None:
        """
        Validate an archive file against an expected SHA256 value.

        Parameters
        ----------
        file_path : str
            Local archive file path.
        sha : str
            Raw SHA256 hex string, local path to a ``.sha`` file, or URL.

        Returns
        -------
        None
            Raises ``ValueError`` if verification fails.
        """
        expected_sha = ArchiveFetchHelper.resolve_expected_sha256(sha)
        actual_sha = ArchiveUtils.compute_sha256(file_path)
        if actual_sha != expected_sha:
            raise ValueError(
                "Archive SHA256 verification failed: "
                f"expected '{expected_sha}', got '{actual_sha}'."
            )

    @staticmethod
    def _resolve_default_download_path(
        file_url: str,
        cache_dir: str,
    ) -> str:
        """
        Resolve the local target path for a remote archive download.

        Parameters
        ----------
        file_url : str
            Remote archive URL.
        cache_dir : str
            Cache root used for the default downloads directory.

        Returns
        -------
        str
            Normalized absolute local path for the downloaded archive.
        """
        downloads_dir = PathUtils.prepare_directory_path(
            Path(cache_dir) / "downloads",
            create=True,
            purpose="archive downloads directory",
        )
        filename = ArchiveFetchHelper._filename_from_url(file_url)
        return PathUtils.prepare_file_path(
            Path(downloads_dir) / filename,
            create_parent=True,
            purpose="archive download path",
        )

    @staticmethod
    def _filename_from_url(file_url: str) -> str:
        """
        Derive a stable filename from a remote archive URL.

        Parameters
        ----------
        file_url : str
            Remote archive URL.

        Returns
        -------
        str
            Filename derived from the URL path.
        """
        parsed = urlparse(file_url)
        candidate = Path(parsed.path).name.strip()
        return candidate or "downloaded_archive.tar.zst"

    @staticmethod
    def _warn_reusing_existing_file(target_path: str) -> None:
        """
        Warn that an existing local archive file is being reused.

        Parameters
        ----------
        target_path : str
            Existing local archive path that will be reused.

        Returns
        -------
        None
            Emits a runtime warning and continues.
        """
        warnings.warn(
            "Archive fetch skipped because the target file already exists: "
            f"'{target_path}'. Reusing the existing file. "
            "Set overwrite=True to replace it with a fresh copy.",
            RuntimeWarning,
            stacklevel=3,
        )

    @staticmethod
    def _download_to_path(file_url: str, target_path: str) -> None:
        """
        Download a remote archive into a local file atomically.

        Parameters
        ----------
        file_url : str
            Remote archive URL.
        target_path : str
            Local file path to populate.

        Returns
        -------
        None
            Downloads the archive and replaces the target atomically.
        """
        target_path = PathUtils.prepare_file_path(
            target_path,
            create_parent=True,
            purpose="archive download path",
        )
        target_dir = os.path.dirname(target_path) or "."
        fd, temp_path = tempfile.mkstemp(
            prefix="mdxplain_download_",
            suffix=".part",
            dir=target_dir,
        )
        os.close(fd)
        try:
            ArchiveFetchHelper._stream_url_to_file(file_url=file_url, target_path=temp_path)
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    @staticmethod
    def _stream_url_to_file(file_url: str, target_path: str) -> None:
        """
        Stream a URL response into a local file.

        Parameters
        ----------
        file_url : str
            Remote archive URL.
        target_path : str
            Local path receiving the response bytes.

        Returns
        -------
        None
            Writes the full response body to ``target_path``.
        """
        with urlopen(file_url) as response, open(target_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    @staticmethod
    def _read_url_text(file_url: str) -> str:
        """
        Read a small text payload from a URL.

        Parameters
        ----------
        file_url : str
            URL pointing to text content such as a ``.sha`` file.

        Returns
        -------
        str
            Decoded UTF-8 text content.
        """
        with urlopen(file_url) as response:
            return response.read().decode("utf-8")

    @staticmethod
    def _read_local_text(path: str) -> str:
        """
        Read text content from a local file path.

        Parameters
        ----------
        path : str
            Local text file path.

        Returns
        -------
        str
            Decoded UTF-8 text content.
        """
        normalized = PathUtils.prepare_file_path(
            path,
            create_parent=False,
            purpose="SHA256 file path",
        )
        with open(normalized, "r", encoding="utf-8") as handle:
            return handle.read()
