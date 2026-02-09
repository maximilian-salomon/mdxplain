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

"""
Centralized file and directory cleanup helpers with deterministic memmap release.
"""

from __future__ import annotations

from typing import Any, Union
import gc
import os
import shutil

from .memmap_utils import MemmapUtils
from .path_utils import PathUtils


class CleanupUtils:
    """
    Utility methods for safe file and directory deletion.
    """

    @staticmethod
    def close_zarr_store(store: Any) -> None:
        """
        Best-effort close for a zarr group/store handle.

        Parameters
        ----------
        store : Any
            Zarr group-like object (or None).

        Returns
        -------
        None
            Attempts to close both group and backend store handles.
        """
        if store is None:
            return

        close_group_fn = getattr(store, "close", None)
        if callable(close_group_fn):
            try:
                close_group_fn()
            except Exception:
                pass

        backend_store = getattr(store, "store", None)
        close_backend_fn = getattr(backend_store, "close", None)
        if callable(close_backend_fn):
            try:
                close_backend_fn()
            except Exception:
                pass

    @staticmethod
    def remove_file(
        path: Union[str, os.PathLike],
        *,
        missing_ok: bool = True,
        ignore_errors: bool = False,
        purpose: str = "file path",
    ) -> bool:
        """
        Remove a file after closing tracked memmaps for the same path.

        Parameters
        ----------
        path : str or os.PathLike
            File path to remove.
        missing_ok : bool, default=True
            If True, no error is raised when path does not exist.
        ignore_errors : bool, default=False
            If True, OSError from delete operations is suppressed.
        purpose : str, default="file path"
            Human-readable purpose for path validation errors.

        Returns
        -------
        bool
            True if a file was removed, otherwise False.
        """
        normalized_path = PathUtils.prepare_file_path(
            path,
            create_parent=False,
            purpose=purpose,
        )

        if not os.path.exists(normalized_path):
            if missing_ok:
                return False
            raise FileNotFoundError(normalized_path)

        MemmapUtils.close_memmaps_for_path(normalized_path)
        gc.collect()

        if ignore_errors:
            try:
                os.remove(normalized_path)
            except OSError:
                return False
            return True

        os.remove(normalized_path)
        return True

    @staticmethod
    def remove_tree(
        path: Union[str, os.PathLike],
        *,
        missing_ok: bool = True,
        ignore_errors: bool = False,
        purpose: str = "directory path",
    ) -> bool:
        """
        Remove a directory tree after closing tracked memmaps.

        Parameters
        ----------
        path : str or os.PathLike
            Directory path to remove.
        missing_ok : bool, default=True
            If True, no error is raised when path does not exist.
        ignore_errors : bool, default=False
            If True, shutil.rmtree suppresses delete errors.
        purpose : str, default="directory path"
            Human-readable purpose for path validation errors.

        Returns
        -------
        bool
            True if a directory was removed, otherwise False.
        """
        normalized_path = PathUtils.prepare_file_path(
            path,
            create_parent=False,
            purpose=purpose,
        )

        if not os.path.exists(normalized_path):
            if missing_ok:
                return False
            raise FileNotFoundError(normalized_path)

        MemmapUtils.close_memmaps_under_path(normalized_path)
        gc.collect()

        shutil.rmtree(normalized_path, ignore_errors=ignore_errors)
        return True

    @staticmethod
    def remove_path(
        path: Union[str, os.PathLike],
        *,
        missing_ok: bool = True,
        ignore_errors: bool = False,
        purpose: str = "path",
    ) -> bool:
        """
        Remove a path (file or directory) with memmap cleanup.

        Parameters
        ----------
        path : str or os.PathLike
            Path to remove.
        missing_ok : bool, default=True
            If True, no error is raised when path does not exist.
        ignore_errors : bool, default=False
            If True, OSError from delete operations is suppressed.
        purpose : str, default="path"
            Human-readable purpose for path validation errors.

        Returns
        -------
        bool
            True if a path was removed, otherwise False.
        """
        normalized_path = PathUtils.prepare_file_path(
            path,
            create_parent=False,
            purpose=purpose,
        )

        if not os.path.exists(normalized_path):
            if missing_ok:
                return False
            raise FileNotFoundError(normalized_path)

        if os.path.isdir(normalized_path):
            return CleanupUtils.remove_tree(
                normalized_path,
                missing_ok=missing_ok,
                ignore_errors=ignore_errors,
                purpose=purpose,
            )

        return CleanupUtils.remove_file(
            normalized_path,
            missing_ok=missing_ok,
            ignore_errors=ignore_errors,
            purpose=purpose,
        )
