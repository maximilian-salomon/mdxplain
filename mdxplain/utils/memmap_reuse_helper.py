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

"""
Reuse of persistent memmap results from disk via a sidecar marker.

A persistent memmap ``.dat`` file has no header, and a write-truncate
(``w+``) memmap is created at full size and zero-filled, so a run aborted
mid-fill leaves a full-size, partly-zero file that cannot be told apart from
valid data by size or content. This helper writes a small JSON sidecar next
to the ``.dat`` only after the fill loop completes, recording the shape,
dtype, and the parameters that define the result. On a later run the memmap
is reused only when the sidecar exists and its dtype, parameters, and the
on-disk size all match the request -- which rules out both partial files and
results computed with different parameters.
"""

import hashlib
import json
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .memmap_utils import MemmapUtils

_SIDECAR_SUFFIX = ".reuse.json"
_PAYLOAD_SUFFIX = ".reuse.pkl"


class MemmapReuseHelper:
    """
    Write and validate reuse sidecars for persistent memmap results.

    All methods are static and stateless. Parameters recorded in the sidecar
    must be plain JSON-serializable values (str, int, float, bool, None, and
    lists/dicts thereof) so they compare cleanly across runs.
    """

    @staticmethod
    def sidecar_path(memmap_path: str) -> str:
        """
        Return the sidecar metadata path for a memmap file.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.

        Returns
        -------
        str
            Path of the sidecar file next to the memmap.
        """
        return f"{memmap_path}{_SIDECAR_SUFFIX}"

    @staticmethod
    def is_reuse_artifact(path: str) -> bool:
        """
        Check whether a path is a reuse sidecar or payload file.

        The JSON sidecar and the optional pickle payload are companions of a
        persistent memmap ``.dat`` file. They belong with that ``.dat`` -- for
        example when a cache directory is bundled into a shareable archive --
        so a later run can still validate and reuse the cached result.

        Parameters
        ----------
        path : str
            File path to test.

        Returns
        -------
        bool
            True when the path ends with the sidecar or payload suffix.
        """
        return path.endswith((_SIDECAR_SUFFIX, _PAYLOAD_SUFFIX))

    @staticmethod
    def hash_array(array: np.ndarray, chunk_rows: Optional[int] = 2000) -> str:
        """
        Compute a content hash of an array without materializing it fully.

        Reads the array in row-chunks so a memory-mapped input never has to be
        held in RAM at once. Shape and dtype are folded into the digest so two
        arrays with different layout cannot collide. Used to verify that a
        cached result was produced from the same input before reusing it.

        Parameters
        ----------
        array : numpy.ndarray
            Array whose content is hashed (regular or memory-mapped).
        chunk_rows : int, optional
            Number of leading-axis rows read per chunk. A falsy or negative value
            falls back to the default, since callers pass a configured chunk_size
            straight through and it may legitimately be unset.

        Returns
        -------
        str
            Hex digest identifying the array content.
        """
        step = chunk_rows if chunk_rows and chunk_rows > 0 else 2000
        hasher = hashlib.blake2b()
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        total = int(array.shape[0]) if array.ndim else 0
        for start in range(0, total, step):
            chunk = np.ascontiguousarray(array[start : start + step])
            hasher.update(chunk.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def write_sidecar(
        memmap_path: str,
        shape: Sequence[int],
        dtype: Union[np.dtype, str, type],
        params: Dict[str, Any],
        payload: Optional[Any] = None,
    ) -> None:
        """
        Write the reuse sidecar after a memmap has been fully written.

        Call this only once the fill loop has completed, so the sidecar's
        presence marks the memmap as complete.

        Parameters
        ----------
        memmap_path : str
            Path of the fully written memmap ``.dat`` file.
        shape : sequence of int
            Shape of the written array.
        dtype : numpy dtype, str, or type
            Data type of the written array.
        params : dict
            JSON-serializable parameters that define the result.
        payload : Any, optional
            Extra picklable data restored on reuse (for example result
            metadata that cannot be recomputed cheaply). Written to a
            companion ``.reuse.pkl`` file when provided.

        Returns
        -------
        None
            Writes the sidecar file next to the memmap.
        """
        record = {
            "shape": [dim for dim in shape],
            "dtype": str(np.dtype(dtype)),
            "params": params,
        }
        with open(
            MemmapReuseHelper.sidecar_path(memmap_path),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(record, handle, sort_keys=True)
        if payload is not None:
            with open(
                MemmapReuseHelper._payload_path(memmap_path), "wb"
            ) as handle:
                pickle.dump(payload, handle)

    @staticmethod
    def _payload_path(memmap_path: str) -> str:
        """
        Return the payload companion path for a memmap file.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.

        Returns
        -------
        str
            Path of the pickled payload file next to the memmap.
        """
        return f"{memmap_path}{_PAYLOAD_SUFFIX}"

    @staticmethod
    def load_payload(memmap_path: str) -> Optional[Any]:
        """
        Load the pickled reuse payload for a memmap if present.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.

        Returns
        -------
        Any or None
            The unpickled payload, or None when no payload file exists.
        """
        path = MemmapReuseHelper._payload_path(memmap_path)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def remove_sidecar(memmap_path: str) -> None:
        """
        Remove the reuse sidecar for a memmap if it exists.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.

        Returns
        -------
        None
            Removes the sidecar and payload files when present.
        """
        for path in (
            MemmapReuseHelper.sidecar_path(memmap_path),
            MemmapReuseHelper._payload_path(memmap_path),
        ):
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def try_reuse(
        memmap_path: str,
        dtype: Union[np.dtype, str, type],
        params: Dict[str, Any],
        mode: str = "r",
    ) -> Optional[np.ndarray]:
        """
        Reopen a cached memmap when its sidecar matches the request.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.
        dtype : numpy dtype, str, or type
            Expected data type of the result.
        params : dict
            JSON-serializable parameters the cached result must match.
        mode : str, default="r"
            Memmap open mode for the reused array.

        Returns
        -------
        np.ndarray or None
            The reopened memmap when the sidecar exists and dtype, params,
            and on-disk size all match; otherwise None.
        """
        recorded = MemmapReuseHelper._load_valid_sidecar(
            memmap_path, params, dtype
        )
        if recorded is None:
            return None
        return MemmapUtils.create_memmap(
            path=memmap_path,
            dtype=dtype,
            mode=mode,
            shape=tuple(recorded["shape"]),
            close_existing=False,
        )

    @staticmethod
    def try_reuse_with_payload(
        memmap_path: str,
        params: Dict[str, Any],
        mode: str = "r",
    ) -> Optional[Tuple[np.ndarray, Any]]:
        """
        Reopen a cached memmap and its payload when the sidecar matches.

        Unlike ``try_reuse``, the dtype is taken from the sidecar rather than
        supplied, which suits results whose dtype is only known after the
        (skipped) computation.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.
        params : dict
            JSON-serializable parameters the cached result must match.
        mode : str, default="r"
            Memmap open mode for the reused array.

        Returns
        -------
        Tuple[np.ndarray, Any] or None
            The reopened memmap and its stored payload when the sidecar
            matches on params and on-disk size; otherwise None.
        """
        recorded = MemmapReuseHelper._load_valid_sidecar(memmap_path, params)
        if recorded is None:
            return None
        payload = MemmapReuseHelper.load_payload(memmap_path)
        if payload is None:
            return None
        memmap_array = MemmapUtils.create_memmap(
            path=memmap_path,
            dtype=recorded["dtype"],
            mode=mode,
            shape=tuple(recorded["shape"]),
            close_existing=False,
        )
        return memmap_array, payload

    @staticmethod
    def _load_valid_sidecar(
        memmap_path: str,
        params: Dict[str, Any],
        dtype: Optional[Union[np.dtype, str, type]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the sidecar record if it exists and matches the request.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.
        params : dict
            JSON-serializable parameters the cached result must match.
        dtype : numpy dtype, str, or type, optional
            Expected data type. When None, the dtype is read from the sidecar
            and not validated (used when the dtype is only known post-compute).

        Returns
        -------
        Dict[str, Any] or None
            The validated sidecar record, or None on any mismatch.
        """
        recorded = MemmapReuseHelper._read_sidecar(memmap_path)
        if recorded is None:
            return None
        if not MemmapReuseHelper._sidecar_matches(recorded, params, dtype):
            return None
        check_dtype = dtype if dtype is not None else recorded["dtype"]
        if not MemmapReuseHelper._file_size_matches(
            memmap_path, recorded, check_dtype
        ):
            return None
        return recorded

    @staticmethod
    def _read_sidecar(memmap_path: str) -> Optional[Dict[str, Any]]:
        """
        Read the sidecar JSON when both the memmap and sidecar exist.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.

        Returns
        -------
        Dict[str, Any] or None
            The sidecar record, or None when either file is missing.
        """
        sidecar = MemmapReuseHelper.sidecar_path(memmap_path)
        if not os.path.exists(memmap_path):
            return None
        if not os.path.exists(sidecar):
            return None
        with open(sidecar, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _sidecar_matches(
        recorded: Dict[str, Any],
        params: Dict[str, Any],
        dtype: Optional[Union[np.dtype, str, type]] = None,
    ) -> bool:
        """
        Check parameter equality and (optional) dtype against a sidecar.

        Parameters
        ----------
        recorded : dict
            Sidecar record loaded from disk.
        params : dict
            Parameters the cached result must match.
        dtype : numpy dtype, str, or type, optional
            Expected data type; skipped when None.

        Returns
        -------
        bool
            True when params match and, if given, dtype matches.
        """
        if recorded.get("params") != params:
            return False
        if dtype is not None and recorded.get("dtype") != str(np.dtype(dtype)):
            return False
        return True

    @staticmethod
    def _file_size_matches(
        memmap_path: str,
        recorded: Dict[str, Any],
        dtype: Union[np.dtype, str, type],
    ) -> bool:
        """
        Check the on-disk size equals the recorded shape times itemsize.

        Parameters
        ----------
        memmap_path : str
            Path of the memmap ``.dat`` file.
        recorded : dict
            Sidecar payload loaded from disk.
        dtype : numpy dtype, str, or type
            Expected data type of the result.

        Returns
        -------
        bool
            True when the file size matches the recorded shape and dtype.
        """
        shape = recorded.get("shape")
        if not isinstance(shape, list) or not shape:
            return False
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return os.path.getsize(memmap_path) == expected
