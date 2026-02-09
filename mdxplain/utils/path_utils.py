# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from GPT-5.3-Codex.
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
Central path normalization and cache path helpers.
"""

from datetime import datetime
from typing import Optional, Union
import os
import re
import uuid


class PathUtils:
    """
    Central utility methods for path handling.

    Provides normalized absolute path handling and cache file path generation.
    """

    _SCOPED_CACHE_DIR_PATTERN = re.compile(
        r"^cache_[0-9a-f]{32}_\d{8}_\d{6}$"
    )

    @staticmethod
    def _coerce_validate_path(
        path: Union[str, os.PathLike],
        purpose: str = "path",
    ) -> str:
        """
        Coerce a path-like object to string and validate platform-specific forms.

        Parameters
        ----------
        path : str or os.PathLike
            Input path value.
        purpose : str, default="path"
            Human-readable purpose for error messages.

        Returns
        -------
        str
            Validated string path (not yet normalized).
        """
        try:
            raw_path = os.fspath(path)
        except TypeError as exc:
            raise ValueError(f"{purpose} must be a valid path-like value") from exc

        if not isinstance(raw_path, str):
            raise ValueError(f"{purpose} must resolve to a string path")

        if not raw_path.strip():
            raise ValueError(f"{purpose} cannot be empty")

        # Reject POSIX-style absolute paths on Windows when no drive/UNC is present.
        if os.name == "nt":
            if raw_path.startswith("\\\\"):
                return raw_path
            drive, _ = os.path.splitdrive(raw_path)
            if (raw_path.startswith("/") or raw_path.startswith("\\")) and not drive:
                raise OSError(
                    f"Cannot use {purpose} '{raw_path}': "
                    "invalid absolute path on Windows (missing drive letter)"
                )

        return raw_path

    @staticmethod
    def prepare_file_path(
        path: Union[str, os.PathLike],
        create_parent: bool = False,
        purpose: str = "file path",
        resolve_symlinks: bool = True,
    ) -> str:
        """
        Normalize a filesystem path and optionally create its parent directory.

        Parameters
        ----------
        path : str or os.PathLike
            Input file path.
        create_parent : bool, default=False
            If True, create the parent directory of the normalized path.
        purpose : str, default="file path"
            Human-readable purpose for error messages.
        resolve_symlinks : bool, default=True
            If True, canonicalize path aliases/symlinks via ``os.path.realpath``.

        Returns
        -------
        str
            Normalized absolute path.
        """
        raw_path = PathUtils._coerce_validate_path(path, purpose=purpose)
        normalized = os.path.abspath(os.path.normpath(raw_path))
        if resolve_symlinks:
            normalized = os.path.realpath(normalized)
        if create_parent:
            parent_dir = os.path.dirname(normalized)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        return normalized

    @staticmethod
    def prepare_directory_path(
        path: Union[str, os.PathLike],
        create: bool = False,
        purpose: str = "directory path",
        resolve_symlinks: bool = True,
    ) -> str:
        """
        Normalize a directory path and optionally create it.

        Parameters
        ----------
        path : str or os.PathLike
            Input directory path.
        create : bool, default=False
            If True, create the directory after normalization.
        purpose : str, default="directory path"
            Human-readable purpose for error messages.
        resolve_symlinks : bool, default=True
            If True, canonicalize path aliases/symlinks via ``os.path.realpath``.

        Returns
        -------
        str
            Normalized absolute directory path.
        """
        if create:
            try:
                normalized = PathUtils.prepare_file_path(
                    path,
                    create_parent=False,
                    purpose=purpose,
                    resolve_symlinks=resolve_symlinks,
                )
                os.makedirs(normalized, exist_ok=True)
            except OSError as exc:
                raise OSError(
                    f"Cannot create {purpose} '{path}': {exc}"
                ) from exc
            return normalized

        return PathUtils.prepare_file_path(
            path,
            create_parent=False,
            purpose=purpose,
            resolve_symlinks=resolve_symlinks,
        )

    @staticmethod
    def get_cache_file_path(cache_name: str, cache_path: str = "./cache") -> str:
        """
        Build a normalized absolute cache file path.

        Parameters
        ----------
        cache_name : str
            Cache file name (e.g. ``"distances.dat"``).
        cache_path : str, default="./cache"
            Cache directory or explicit cache file path.

        Returns
        -------
        str
            Normalized absolute cache file path.
        """
        if cache_path:
            cache_path = PathUtils.prepare_directory_path(
                cache_path,
                create=False,
                purpose="cache directory",
            )
            if cache_path.endswith(".dat") or "." in os.path.basename(cache_path):
                return PathUtils.prepare_file_path(
                    cache_path,
                    create_parent=True,
                    purpose="cache file path",
                )
            return PathUtils.prepare_file_path(
                os.path.join(cache_path, cache_name),
                create_parent=True,
                purpose="cache file path",
            )
        return PathUtils.prepare_file_path(
            os.path.join("./cache", cache_name),
            create_parent=True,
            purpose="cache file path",
        )

    @staticmethod
    def create_pipeline_cache_dir(
        base_cache_dir: Union[str, os.PathLike],
        *,
        pipeline_uuid: Optional[str] = None,
        pipeline_timestamp: Optional[str] = None,
        purpose: str = "cache directory",
    ) -> str:
        """
        Create and return a per-pipeline scoped cache directory.

        The resulting layout is:
        ``<base_cache_dir>/cache_<pipeline_uuid>_<YYYYMMDD_HHMMSS>``.

        Parameters
        ----------
        base_cache_dir : str or os.PathLike
            User-provided cache root directory.
        pipeline_uuid : str, optional
            Stable pipeline instance UUID. Generated when omitted.
        pipeline_timestamp : str, optional
            Stable timestamp tag for the pipeline. Generated when omitted.
        purpose : str, default="cache directory"
            Human-readable purpose for path validation errors.

        Returns
        -------
        str
            Canonical absolute path to the scoped cache directory.
        """
        normalized_base = PathUtils.prepare_directory_path(
            base_cache_dir,
            create=True,
            purpose=purpose,
        )
        # If the caller already provides a scoped directory, reuse it as-is.
        # This is required for single-file reloads where the same cache scope
        # should remain active across save/load cycles.
        base_name = os.path.basename(os.path.normpath(normalized_base))
        if PathUtils._SCOPED_CACHE_DIR_PATTERN.fullmatch(base_name):
            return normalized_base

        scope_uuid = pipeline_uuid or uuid.uuid4().hex
        scope_timestamp = pipeline_timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        scoped_dir = os.path.join(
            normalized_base,
            f"cache_{scope_uuid}_{scope_timestamp}",
        )
        return PathUtils.prepare_directory_path(
            scoped_dir,
            create=True,
            purpose="scoped pipeline cache directory",
        )
