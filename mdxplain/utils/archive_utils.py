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
Archive utilities for pipeline persistence and sharing.

This module provides utilities for creating and extracting compressed
archives containing pipeline data. Supports filtering of visualization
files and structure files for flexible archive creation.
"""

import os
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import psutil
from .path_utils import PathUtils
from .progress_utils import ProgressUtils
from xopen import xopen


class ArchiveUtils:
    """
    Utilities for creating and extracting pipeline archives.

    Provides static methods for compressing pipeline data into portable
    archives and extracting them. Supports selective inclusion of files
    based on type (essential data, visualizations, structure files).

    Examples
    --------
    >>> # Create archive from pipeline data
    >>> archive_path = ArchiveUtils.create_archive(
    ...     pipeline_data, "analysis.tar.xz"
    ... )

    >>> # Extract archive
    >>> extract_dir = ArchiveUtils.extract_archive("analysis.tar.xz")
    """

    @staticmethod
    def is_essential_file(suffix: str, use_memmap: bool) -> bool:
        """
        Check if file is essential for pipeline load.

        Essential files depend on memmap usage. Pickle always essential.
        Memmap files (.dat) only essential if use_memmap=True.

        Parameters
        ----------
        suffix : str
            File extension (lowercase with dot)
        use_memmap : bool
            Whether pipeline uses memory mapping

        Returns
        -------
        bool
            True if file is essential for pipeline loading

        Examples
        --------
        >>> ArchiveUtils.is_essential_file('.dat', use_memmap=True)
        True
        >>> ArchiveUtils.is_essential_file('.dat', use_memmap=False)
        False
        >>> ArchiveUtils.is_essential_file('.pkl', use_memmap=False)
        True
        """
        if suffix == '.pkl':
            return True
        if suffix == '.dat' and use_memmap:
            return True
        return False

    @staticmethod
    def is_zarr_directory(path: Path) -> bool:
        """
        Check if path is a zarr archive directory.

        Zarr archives are directories used for trajectory caching
        with DaskMDTrajectory. Essential for trajectory loading.

        Parameters
        ----------
        path : Path
            Path to check

        Returns
        -------
        bool
            True if path is zarr directory

        Examples
        --------
        >>> path = Path("cache/traj0.dask.zarr")
        >>> ArchiveUtils.is_zarr_directory(path)
        True
        """
        return path.is_dir() and '.zarr' in path.name

    @staticmethod
    def is_visualization_file(suffix: str) -> bool:
        """
        Check if file is visualization output.

        Visualization files are plot outputs that can be regenerated
        and are typically excluded from minimal archives.

        Parameters
        ----------
        suffix : str
            File extension (lowercase with dot)

        Returns
        -------
        bool
            True if file is a visualization output

        Examples
        --------
        >>> ArchiveUtils.is_visualization_file('.png')
        True
        >>> ArchiveUtils.is_visualization_file('.dat')
        False
        """
        return suffix in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']

    @staticmethod
    def is_structure_file(suffix: str) -> bool:
        """
        Check if file is structure output.

        Structure files include PDB coordinates and PyMOL scripts
        generated from feature importance analysis.

        Parameters
        ----------
        suffix : str
            File extension (lowercase with dot)

        Returns
        -------
        bool
            True if file is a structure file

        Examples
        --------
        >>> ArchiveUtils.is_structure_file('.pdb')
        True
        >>> ArchiveUtils.is_structure_file('.dat')
        False
        """
        return suffix in ['.pdb', '.pml']

    @staticmethod
    def should_include_file(
        file_path: Path,
        exclude_visualizations: bool,
        include_structure_files: bool,
        use_memmap: bool
    ) -> bool:
        """
        Determine if file should be included in archive.

        Applies filtering logic based on file type and user preferences.
        Essential files depend on memmap usage.

        Parameters
        ----------
        file_path : Path
            Path to file to check
        exclude_visualizations : bool
            If True, exclude plot outputs (PNG, PDF, etc.)
        include_structure_files : bool
            If True, include PDB/PML structure files
        use_memmap : bool
            Whether pipeline uses memory mapping

        Returns
        -------
        bool
            True if file should be included in archive

        Examples
        --------
        >>> path = Path("cache/features.dat")
        >>> ArchiveUtils.should_include_file(path, True, True, True)
        True
        >>> ArchiveUtils.should_include_file(path, True, True, False)
        False
        >>> path = Path("plots/landscape.png")
        >>> ArchiveUtils.should_include_file(path, True, True, True)
        False
        """
        suffix = file_path.suffix.lower()

        if ArchiveUtils.is_essential_file(suffix, use_memmap):
            return True

        if ArchiveUtils.is_structure_file(suffix):
            return include_structure_files

        if ArchiveUtils.is_visualization_file(suffix):
            return not exclude_visualizations

        return False

    @staticmethod
    def _is_inside_zarr(file_path: Path) -> bool:
        """
        Check if file is inside a zarr directory.

        Parameters
        ----------
        file_path : Path
            File path to check

        Returns
        -------
        bool
            True if file is inside zarr directory
        """
        return any(ArchiveUtils.is_zarr_directory(p) for p in file_path.parents)

    @staticmethod
    def _add_zarr_to_archive(
        zarr_path: Path,
        cache_path: Path,
        items_list: list,
        processed_set: set
    ) -> None:
        """
        Add zarr directory to archive list.

        Parameters
        ----------
        zarr_path : Path
            Path to zarr directory
        cache_path : Path
            Cache directory path
        items_list : list
            List to append items to
        processed_set : set
            Set of processed zarr directories
        """
        if zarr_path not in processed_set:
            relative_path = zarr_path.relative_to(cache_path)
            archive_path = Path("cache") / relative_path
            items_list.append((str(zarr_path), str(archive_path)))
            processed_set.add(zarr_path)

    @staticmethod
    def _process_item_for_archive(
        item_path: Path,
        cache_path: Path,
        items_list: list,
        processed_zarr: set,
        exclude_viz: bool,
        include_struct: bool,
        use_memmap: bool
    ) -> None:
        """
        Process single item for archive inclusion.

        Parameters
        ----------
        item_path : Path
            Path to item
        cache_path : Path
            Cache directory path
        items_list : list
            List to append items to
        processed_zarr : set
            Set of processed zarr directories
        exclude_viz : bool
            Exclude visualizations
        include_struct : bool
            Include structure files
        use_memmap : bool
            Whether pipeline uses memory mapping
        """
        if ArchiveUtils.is_zarr_directory(item_path):
            if use_memmap:
                ArchiveUtils._add_zarr_to_archive(
                    item_path, cache_path, items_list, processed_zarr
                )
        elif item_path.is_file():
            ArchiveUtils._add_file_to_archive(
                item_path, cache_path, items_list,
                exclude_viz, include_struct, use_memmap
            )

    @staticmethod
    def _add_file_to_archive(
        file_path: Path,
        cache_path: Path,
        items_list: list,
        exclude_visualizations: bool,
        include_structure_files: bool,
        use_memmap: bool
    ) -> None:
        """
        Add file to archive list if it matches criteria.

        Parameters
        ----------
        file_path : Path
            Path to file
        cache_path : Path
            Cache directory path
        items_list : list
            List to append items to
        exclude_visualizations : bool
            Exclude visualization files
        include_structure_files : bool
            Include structure files
        use_memmap : bool
            Whether pipeline uses memory mapping
        """
        if ArchiveUtils._is_inside_zarr(file_path):
            return

        if ArchiveUtils.should_include_file(
            file_path, exclude_visualizations, include_structure_files,
            use_memmap
        ):
            relative_path = file_path.relative_to(cache_path)
            archive_path = Path("cache") / relative_path
            items_list.append((str(file_path), str(archive_path)))

    @staticmethod
    def collect_cache_files(
        cache_dir: str,
        exclude_visualizations: bool,
        include_structure_files: bool,
        use_memmap: bool
    ) -> List[Tuple[str, str]]:
        """
        Collect all files and zarr directories from cache for archiving.

        Recursively scans cache directory and collects files and
        zarr directories matching the specified filter criteria.

        Parameters
        ----------
        cache_dir : str
            Path to cache directory
        exclude_visualizations : bool
            If True, exclude plot outputs
        include_structure_files : bool
            If True, include PDB/PML files
        use_memmap : bool
            Whether pipeline uses memory mapping

        Returns
        -------
        List[Tuple[str, str]]
            List of (absolute_path, archive_path) tuples

        Examples
        --------
        >>> files = ArchiveUtils.collect_cache_files(
        ...     "./cache", exclude_visualizations=True,
        ...     include_structure_files=True, use_memmap=True
        ... )
        >>> len(files) > 0
        True

        Notes
        -----
        - Files are filtered by extension
        - Zarr directories only included if use_memmap=True
        - .dat files only included if use_memmap=True
        - Zarr directories are added as directories, not individual files
        """
        cache_path = Path(cache_dir)
        items_to_archive = []
        processed_zarr_dirs = set()

        if not cache_path.exists():
            return items_to_archive

        for item_path in cache_path.rglob('*'):
            ArchiveUtils._process_item_for_archive(
                item_path, cache_path, items_to_archive,
                processed_zarr_dirs, exclude_visualizations,
                include_structure_files, use_memmap
            )

        return items_to_archive

    @staticmethod
    def _resolve_xz_threads(
        xz_threads: Optional[int] = None,
        reserve_cores: int = 2,
        xz_level: int = 6,
        xz_max_memory_gb: Optional[float] = None,
    ) -> int:
        """
        Resolve xz thread count with sensible defaults.

        Parameters
        ----------
        xz_threads : int, optional
            Explicit thread count. If None, derive from CPU count.
        reserve_cores : int, default=2
            Number of CPU cores to keep free when xz_threads is None.
        xz_level : int, default=6
            xz compression level (preset 0-9).
        xz_max_memory_gb : float, optional
            Soft memory cap for xz compression in GiB. The cap is applied by
            reducing thread count based on estimated per-thread memory usage.

        Returns
        -------
        int
            Thread count for xz compression.
        """
        if xz_threads is not None:
            resolved_threads = max(1, int(xz_threads))
        else:
            cpu_count = os.cpu_count() or 1
            resolved_threads = max(1, cpu_count - max(0, int(reserve_cores)))

        if xz_max_memory_gb is None or xz_max_memory_gb <= 0:
            return resolved_threads

        per_thread_mib = ArchiveUtils._estimate_xz_memory_per_thread_mib(xz_level)
        budget_mib = int(xz_max_memory_gb * 1024)

        rss_mib = ArchiveUtils._get_current_process_rss_mib()
        available_mib = budget_mib - rss_mib

        if available_mib < per_thread_mib:
            warnings.warn(
                (
                    "Available archive memory is below one xz thread estimate; "
                    f"budget={xz_max_memory_gb:.3f} GiB ({budget_mib} MiB), "
                    f"rss_mib={rss_mib}, "
                    f"per_thread_estimate={per_thread_mib} MiB. "
                    "Proceeding with threads=1."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return 1

        max_threads_by_memory = max(1, available_mib // per_thread_mib)
        return max(1, min(resolved_threads, max_threads_by_memory))

    @staticmethod
    def _get_current_process_rss_mib() -> int:
        """
        Return current process RSS in MiB.

        Returns
        -------
        int
            Current RSS in MiB.
        """
        rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
        return max(0, rss_bytes // (1024 * 1024))

    @staticmethod
    def _estimate_xz_memory_per_thread_mib(xz_level: int, safety_factor: float = 1.5) -> int:
        """
        Estimate xz compressor memory usage per thread in MiB.

        The values mirror xz preset behavior and are used only to cap
        thread count against a user-defined memory budget.
        Source of the constants:
        `xz -vv -T1 -<level> -c /dev/null >/dev/null` and parsing
        "X MiB of memory is required." from xz output.
        Example measured with XZ Utils 5.4.5.

        Parameters
        ----------
        xz_level : int
            xz compression level (preset 0-9).
        safety_factor : float, optional
            Safety factor to account for memory overhead, by default 1.5.

        Returns
        -------
        int
            Estimated memory usage per thread in MiB.
        """
        # These are per-thread compressor memory requirements (MiB) for
        # xz presets 0-9 from `xz -vv -T1 -<level>`.
        per_thread_mib = {
            0: 3,
            1: 9,
            2: 17,
            3: 32,
            4: 48,
            5: 94,
            6: 94,
            7: 186,
            8: 370,
            9: 674,
        }
        if xz_level not in per_thread_mib:
            raise ValueError("xz_level must be in range 0-9")
        return per_thread_mib[xz_level] * safety_factor

    @staticmethod
    def _add_archive_items(
        tar: tarfile.TarFile,
        temp_pkl: str,
        files_to_archive: List[Tuple[str, str]],
    ) -> None:
        """
        Add pipeline pickle and collected cache files to an open tar stream.

        Parameters
        ----------
        tar : tarfile.TarFile
            Open tar file object.
        temp_pkl : str
            Path to temporary pipeline pickle file.
        files_to_archive : List[Tuple[str, str]]
            Archive file list as (source_path, archive_path).
        """
        tar.add(temp_pkl, arcname="pipeline.pkl")
        for file_path, archive_name in ProgressUtils.iterate(
            files_to_archive,
            desc="Adding files to archive",
            unit="file",
        ):
            tar.add(file_path, arcname=archive_name)

    @staticmethod
    def create_archive(
        pipeline_data,
        archive_path: str,
        compression: str = "xz",
        exclude_visualizations: bool = True,
        include_structure_files: bool = True,
        compression_level: Optional[int] = None,
        xz_threads: Optional[int] = None,
        reserve_cores: int = 2,
        xz_max_memory_gb: Optional[float] = None,
    ) -> str:
        """
        Create compressed archive with pipeline and cache files.

        Creates tar archive containing pipeline pickle and filtered
        cache directory files with maximum compression.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object to save
        archive_path : str
            Path for output archive (extension added if missing)
        compression : str, default="xz"
            Compression method: "xz", "bz2", or "gz"
        exclude_visualizations : bool, default=True
            If True, exclude plot outputs
        include_structure_files : bool, default=True
            If True, include PDB/PML files
        compression_level : int, optional
            Compression level override. For xz this maps to preset (0-9).
        xz_threads : int, optional
            Thread count for xz compression via xopen. If None, uses
            ``max(1, cpu_count - reserve_cores)``.
        reserve_cores : int, default=2
            Number of CPU cores to keep free for automatic xz thread selection.
        xz_max_memory_gb : float, optional
            Soft memory cap for xz compression in GiB. The cap is applied by
            reducing thread count based on estimated memory use per thread.

        Returns
        -------
        str
            Path to created archive file

        Raises
        ------
        ValueError
            If compression method not supported

        Examples
        --------
        >>> archive = ArchiveUtils.create_archive(
        ...     pipeline_data, "analysis.tar.xz"
        ... )
        >>> Path(archive).exists()
        True

        Notes
        -----
        - Uses tempfile for pickle creation
        - Preserves relative paths in archive
        - xz provides best compression ratio
        - xz archives are written with multithreading via xopen
        - With use_memmap=False: Only pickle needed (all data in objects)
        - With use_memmap=True: Pickle + .dat files + zarr directories
        - tar.add() automatically handles both files and directories
        """
        compression_modes = {"xz": "w:xz", "bz2": "w:bz2", "gz": "w:gz"}
        if compression not in compression_modes:
            raise ValueError(
                f"Compression must be one of {list(compression_modes.keys())}"
            )

        archive_full_path = f"{archive_path}"
        if not archive_full_path.endswith(f".tar.{compression}"):
            archive_full_path = f"{archive_path}.tar.{compression}"
        archive_full_path = PathUtils.prepare_file_path(
            archive_full_path,
            create_parent=True,
            purpose="archive output path",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pkl = os.path.join(temp_dir, "pipeline.pkl")
            pipeline_data.save(temp_pkl)

            files_to_archive = ArchiveUtils.collect_cache_files(
                pipeline_data.cache_dir,
                exclude_visualizations,
                include_structure_files,
                pipeline_data.use_memmap
            )

            if compression == "xz":
                xz_level = 6 if compression_level is None else int(compression_level)
                if not 0 <= xz_level <= 9:
                    raise ValueError("compression_level for xz must be in range 0-9")

                threads = ArchiveUtils._resolve_xz_threads(
                    xz_threads=xz_threads,
                    reserve_cores=reserve_cores,
                    xz_level=xz_level,
                    xz_max_memory_gb=xz_max_memory_gb,
                )
                with xopen(
                    archive_full_path,
                    mode="wb",
                    compresslevel=xz_level,
                    threads=threads,
                ) as compressed_file:
                    with tarfile.open(fileobj=compressed_file, mode="w|") as tar:
                        ArchiveUtils._add_archive_items(
                            tar=tar,
                            temp_pkl=temp_pkl,
                            files_to_archive=files_to_archive,
                        )
            else:
                tar_kwargs = {}
                if compression_level is not None:
                    tar_kwargs["compresslevel"] = int(compression_level)

                with tarfile.open(
                    archive_full_path, compression_modes[compression], **tar_kwargs
                ) as tar:
                    ArchiveUtils._add_archive_items(
                        tar=tar,
                        temp_pkl=temp_pkl,
                        files_to_archive=files_to_archive,
                    )

        return archive_full_path

    @staticmethod
    def extract_archive(
        archive_path: str,
        extract_to: str = None
    ) -> Path:
        """
        Extract archive and return extraction directory.

        Extracts compressed tar archive preserving directory structure.
        Creates extraction directory if it does not exist.

        Parameters
        ----------
        archive_path : str
            Path to archive file
        extract_to : str, optional
            Directory to extract to. If None, uses archive parent
            directory with archive stem as subdirectory name.

        Returns
        -------
        Path
            Path to extraction directory

        Raises
        ------
        FileNotFoundError
            If archive does not exist

        Examples
        --------
        >>> extract_dir = ArchiveUtils.extract_archive("analysis.tar.xz")
        >>> (extract_dir / "pipeline.pkl").exists()
        True

        >>> extract_dir = ArchiveUtils.extract_archive(
        ...     "analysis.tar.xz",
        ...     extract_to="./restored"
        ... )

        Notes
        -----
        - Automatically detects compression from file extension
        - Creates parent directories if needed
        - Preserves file permissions and timestamps
        """
        archive_path = Path(
            PathUtils.prepare_file_path(
                archive_path,
                create_parent=False,
                purpose="archive path",
            )
        )
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        if extract_to is None:
            extract_dir = archive_path.parent / archive_path.stem.replace(
                '.tar', ''
            )
        else:
            extract_dir = Path(
                PathUtils.prepare_directory_path(
                    extract_to,
                    create=True,
                    purpose="archive extraction directory",
                )
            )

        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(extract_dir, filter="data")

        return extract_dir
