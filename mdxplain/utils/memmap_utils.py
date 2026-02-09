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
Centralized helpers for memmap path handling and lifecycle operations.
"""

from typing import Any, Optional, Tuple, Union
import gc
import os

import numpy as np

from .path_utils import PathUtils
from .resource_utils import ResourceUtils


class MemmapUtils:
    """
    Utility methods for memmap path prep, creation, and cleanup.
    """

    @staticmethod
    def prepare_memmap_path(path: Union[str, os.PathLike], mode: str) -> str:
        """
        Normalize memmap file path and create parent directory for write modes.

        Parameters
        ----------
        path : str or os.PathLike
            Target memmap file path.
        mode : str
            Memmap mode (e.g. ``"w+"``, ``"r+"``, ``"r"``).

        Returns
        -------
        str
            Normalized absolute memmap path.
        """
        create_parent = mode in {"w+", "w+b"}
        return PathUtils.prepare_file_path(
            path,
            create_parent=create_parent,
            purpose="memmap path",
        )

    @staticmethod
    def _canonical_normcase(path: Union[str, os.PathLike]) -> str:
        """
        Canonicalize a path for robust cross-platform path equality checks.
        """
        canonical = PathUtils.prepare_file_path(
            path,
            create_parent=False,
            purpose="memmap path",
            resolve_symlinks=True,
        )
        return os.path.normcase(canonical)

    @staticmethod
    def is_memmap_view(array: Any) -> bool:
        """
        Check whether an array is backed by a numpy memmap (including views).

        Parameters
        ----------
        array : Any
            Array or view to check.

        Returns
        -------
        bool
            True if the array is a memmap or view on a memmap.
        """
        base = array
        seen = set()
        while isinstance(base, np.ndarray) and id(base) not in seen:
            if isinstance(base, np.memmap):
                return True
            seen.add(id(base))
            base = base.base
        return False

    @staticmethod
    def close_memmap_view(array: Any) -> None:
        """
        Close underlying memmap handle(s) for an array or array view.

        Parameters
        ----------
        array : Any
            Array, memmap, or view potentially backed by a memmap.

        Returns
        -------
        None
            Flushes and closes any discovered memmap handles.
        """
        base = array
        seen = set()
        while isinstance(base, np.ndarray) and id(base) not in seen:
            seen.add(id(base))
            if isinstance(base, np.memmap):
                try:
                    base.flush()
                except Exception:
                    pass
                mmap_obj = getattr(base, "_mmap", None)
                if mmap_obj is not None:
                    try:
                        mmap_obj.close()
                    except Exception:
                        pass
            base = base.base

    @staticmethod
    def close_memmaps_for_path(path: Union[str, os.PathLike]) -> None:
        """
        Close all tracked memmaps whose filename matches the given path.

        Parameters
        ----------
        path : str or os.PathLike
            Target memmap file path.

        Returns
        -------
        None
        """
        target = MemmapUtils._canonical_normcase(path)
        for obj in gc.get_objects():
            if not isinstance(obj, np.memmap):
                continue
            filename = getattr(obj, "filename", None)
            if not filename:
                continue
            current = MemmapUtils._canonical_normcase(filename)
            if current == target:
                MemmapUtils.close_memmap_view(obj)

    @staticmethod
    def close_memmaps_under_path(path: Union[str, os.PathLike]) -> None:
        """
        Close tracked memmaps whose filename is inside the given directory path.

        Parameters
        ----------
        path : str or os.PathLike
            Directory path root for memmap cleanup.

        Returns
        -------
        None
        """
        root = MemmapUtils._canonical_normcase(path)

        for obj in gc.get_objects():
            if not isinstance(obj, np.memmap):
                continue
            filename = getattr(obj, "filename", None)
            if not filename:
                continue
            current = MemmapUtils._canonical_normcase(filename)
            try:
                common = os.path.commonpath([current, root])
            except ValueError:
                # Different mount/drive => definitely not under root.
                continue
            if common == root:
                MemmapUtils.close_memmap_view(obj)

    @staticmethod
    def create_memmap(
        path: Union[str, os.PathLike],
        dtype: Union[np.dtype, str, type],
        mode: str,
        shape: Tuple[int, ...],
        *,
        close_existing: Optional[bool] = None,
        access_pattern: Optional[str] = "random",
    ) -> np.ndarray:
        """
        Create a memmap with standardized path prep and optional stale-handle cleanup.

        Parameters
        ----------
        path : str or os.PathLike
            Target memmap file path.
        dtype : numpy dtype, str, or type
            Data type for the array.
        mode : str
            Memmap mode (e.g. ``"w+"``, ``"r+"``, ``"r"``).
        shape : tuple[int, ...]
            Desired output shape.
        close_existing : bool, optional
            Whether to close currently tracked memmaps for this path before create.
            If None, defaults to True only for write-truncate modes (``"w+"``, ``"w+b"``).
        access_pattern : str, optional
            Access hint for ResourceUtils.tune_memmap (e.g. ``"random"``, ``"sequential"``).
            If None, no hint is applied.

        Returns
        -------
        np.ndarray
            Memmap-backed array when possible, otherwise regular ndarray for empty shapes.
        """
        normalized_shape = tuple(int(dim) for dim in shape)
        if any(dim == 0 for dim in normalized_shape):
            return np.zeros(normalized_shape, dtype=dtype)

        normalized_path = MemmapUtils.prepare_memmap_path(path, mode=mode)
        if close_existing is None:
            close_existing = mode in {"w+", "w+b"}
        if close_existing:
            MemmapUtils.close_memmaps_for_path(normalized_path)
            gc.collect()

        parent_dir = os.path.dirname(normalized_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        array = np.memmap(
            normalized_path,
            dtype=dtype,
            mode=mode,
            shape=normalized_shape,
        )
        if access_pattern:
            ResourceUtils.tune_memmap(array, access_pattern)
        return array
