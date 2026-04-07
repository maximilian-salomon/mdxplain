# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
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
DaskMDTrajectory Archive Helper - save/load trajectory archives.

Handles creation and extraction of self-contained ``.dask_traj`` archives
that bundle the pickle metadata with the underlying Zarr cache directory.
The resulting archive is fully portable: no external files are required
after loading.

Archive layout (inside the compressed tar)::

    traj.pkl   – pickle of the DaskMDTrajectory object (metadata only)
    zarr/      – the Zarr cache directory with all coordinate data
"""

from __future__ import annotations

import os
import pickle
import tarfile
import tempfile
from typing import TYPE_CHECKING

import zstandard as zstd

from ....utils.path_utils import PathUtils
from .zarr_cache_helper import ZarrCacheHelper

if TYPE_CHECKING:
    from ...entities.dask_md_trajectory import DaskMDTrajectory

# File extension for trajectory archives
ARCHIVE_EXTENSION = ".dask_traj"

# Fixed entry names inside every archive
_PKL_ENTRY = "traj.pkl"
_ZARR_ENTRY = "zarr"


class DaskMDTrajectoryArchiveHelper:
    """
    Save and load DaskMDTrajectory objects as portable ``.dask_traj`` archives.

    An archive is a zstd-compressed tar that contains:

    - ``traj.pkl`` – pickle of the trajectory object (metadata only, no arrays)
    - ``zarr/``    – the Zarr cache directory with all coordinate data

    The archive can be moved or shared freely.  On load the Zarr cache is
    extracted next to the archive file and the trajectory handles are
    reconnected automatically.  Subsequent loads reuse the already-extracted
    cache without re-extracting.

    Methods
    -------
    save(trajectory, filepath)
        Write a ``.dask_traj`` archive for *trajectory*.
    load(filepath)
        Read a ``.dask_traj`` archive and return a ready-to-use trajectory.
    """

    @staticmethod
    def save(trajectory: DaskMDTrajectory, filepath: str) -> None:
        """
        Save a DaskMDTrajectory to a portable self-contained archive.

        Serialises the trajectory metadata as a pickle and bundles it with
        the underlying Zarr cache into a single zstd-compressed tar archive.
        The resulting ``.dask_traj`` file can be transferred to another
        machine or directory without carrying the Zarr cache separately.

        Parameters
        ----------
        trajectory : DaskMDTrajectory
            Trajectory instance to archive.
        filepath : str
            Destination path.  The ``.dask_traj`` extension is appended
            automatically when not already present.

        Returns
        -------
        None

        Examples
        --------
        >>> DaskMDTrajectoryArchiveHelper.save(traj, "output/run1")
        Trajectory saved: output/run1.dask_traj
        """
        filepath = DaskMDTrajectoryArchiveHelper._normalize_path(filepath)
        filepath = PathUtils.prepare_file_path(
            filepath, create_parent=True, purpose="trajectory archive"
        )

        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = os.path.join(tmp, _PKL_ENTRY)
            with open(pkl_path, "wb") as fh:
                pickle.dump(trajectory, fh)

            DaskMDTrajectoryArchiveHelper._write_archive(
                filepath, pkl_path, trajectory.zarr_cache_path
            )

        print(f"Trajectory saved: {filepath}")

    @staticmethod
    def load(filepath: str) -> DaskMDTrajectory:
        """
        Load a DaskMDTrajectory from a ``.dask_traj`` archive.

        On the first call the archive is extracted to a sibling directory
        (``<name>_extracted/``).  Subsequent calls reuse the extracted
        directory without re-extracting, so repeated loads are fast.

        After extraction the trajectory handles (Dask arrays, Zarr store)
        are reconnected to the extracted Zarr cache.

        Parameters
        ----------
        filepath : str
            Path to a ``.dask_traj`` archive created by :py:meth:`save`.

        Returns
        -------
        DaskMDTrajectory
            Fully initialised trajectory with coordinate access ready.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not point to an existing archive.

        Examples
        --------
        >>> traj = DaskMDTrajectoryArchiveHelper.load("output/run1.dask_traj")
        >>> print(traj.n_frames)
        501
        """
        filepath = PathUtils.prepare_file_path(
            filepath, create_parent=False, purpose="trajectory archive"
        )
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trajectory archive not found: {filepath}")

        extract_dir, zarr_path = DaskMDTrajectoryArchiveHelper._extraction_paths(filepath)

        if not os.path.exists(zarr_path):
            DaskMDTrajectoryArchiveHelper._extract_archive(filepath, extract_dir)

        pkl_path = os.path.join(extract_dir, _PKL_ENTRY)
        with open(pkl_path, "rb") as fh:
            instance = pickle.load(fh)

        DaskMDTrajectoryArchiveHelper._reconnect(instance, extract_dir, zarr_path)
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_path(filepath: str) -> str:
        """
        Return *filepath* with the archive extension guaranteed.

        Parameters
        ----------
        filepath : str
            Raw destination path, with or without the extension.

        Returns
        -------
        str
            *filepath* with ``ARCHIVE_EXTENSION`` appended if missing.
        """
        if not filepath.endswith(ARCHIVE_EXTENSION):
            return filepath + ARCHIVE_EXTENSION
        return filepath

    @staticmethod
    def _extraction_paths(filepath: str) -> tuple[str, str]:
        """
        Derive the extraction directory and expected Zarr path from *filepath*.

        Parameters
        ----------
        filepath : str
            Absolute path to the ``.dask_traj`` archive.

        Returns
        -------
        tuple[str, str]
            ``(extract_dir, zarr_path)`` where *extract_dir* is the sibling
            directory used for extraction and *zarr_path* is the path of
            the ``zarr/`` subdirectory inside it.
        """
        base = filepath[: -len(ARCHIVE_EXTENSION)]
        extract_dir = base + "_extracted"
        zarr_path = os.path.join(extract_dir, _ZARR_ENTRY)
        return extract_dir, zarr_path

    @staticmethod
    def _write_archive(archive_path: str, pkl_path: str, zarr_cache_path: str) -> None:
        """
        Create a zstd-compressed tar archive containing the pickle and Zarr cache.

        Parameters
        ----------
        archive_path : str
            Destination path for the archive file.
        pkl_path : str
            Path to the temporary pickle file to include as ``traj.pkl``.
        zarr_cache_path : str
            Path to the Zarr cache directory to include as ``zarr/``.
            The directory is skipped silently when it does not exist.

        Returns
        -------
        None
        """
        compressor = zstd.ZstdCompressor(level=6, threads=-1)
        with open(archive_path, "wb") as out_fh:
            with compressor.stream_writer(out_fh) as zstd_fh:
                with tarfile.open(fileobj=zstd_fh, mode="w|") as tar:
                    tar.add(pkl_path, arcname=_PKL_ENTRY)
                    if os.path.exists(zarr_cache_path):
                        tar.add(zarr_cache_path, arcname=_ZARR_ENTRY)

    @staticmethod
    def _extract_archive(archive_path: str, extract_dir: str) -> None:
        """
        Extract a zstd-compressed tar archive to *extract_dir*.

        Parameters
        ----------
        archive_path : str
            Path to the ``.dask_traj`` archive to extract.
        extract_dir : str
            Target directory.  Created if it does not exist.

        Returns
        -------
        None
        """
        os.makedirs(extract_dir, exist_ok=True)
        decompressor = zstd.ZstdDecompressor()
        with open(archive_path, "rb") as in_fh:
            with decompressor.stream_reader(in_fh) as zstd_fh:
                with tarfile.open(fileobj=zstd_fh, mode="r|") as tar:
                    tar.extractall(extract_dir, filter="data")

    @staticmethod
    def _reconnect(
        instance: DaskMDTrajectory, extract_dir: str, zarr_path: str
    ) -> None:
        """
        Patch Zarr paths and reload Dask/Zarr handles after extraction.

        Updates ``zarr_cache_path``, ``_cache_dir``, and ``cache_manager``
        on *instance* to point to the extracted location, then calls
        ``_reload_from_cache()`` to rebuild all Dask arrays and open the
        Zarr store.

        Parameters
        ----------
        instance : DaskMDTrajectory
            Unpickled trajectory instance whose handles must be reconnected.
        extract_dir : str
            Root extraction directory (parent of ``zarr/``).
        zarr_path : str
            Absolute path to the extracted ``zarr/`` directory.

        Returns
        -------
        None
        """
        instance.zarr_cache_path = zarr_path
        instance._cache_dir = extract_dir
        instance.cache_manager = ZarrCacheHelper(
            chunk_size=instance.chunk_size, cache_dir=extract_dir
        )
        instance._reload_from_cache()
