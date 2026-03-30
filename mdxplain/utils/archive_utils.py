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

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import zstandard as zstd

from .path_utils import PathUtils
from .progress_utils import ProgressUtils


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
    ...     pipeline_data, "analysis.tar.zst"
    ... )

    >>> # Extract archive
    >>> extract_dir = ArchiveUtils.extract_archive("analysis.tar.zst")
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
    def _resolve_zstd_threads(
        zstd_threads: Optional[int] = None,
        reserve_cores: int = 2,
    ) -> int:
        """
        Resolve zstd thread count with deterministic defaults.

        Parameters
        ----------
        zstd_threads : int, optional
            Explicit thread count. If None, derive from CPU count.
        reserve_cores : int, default=2
            Number of CPU cores to keep free when ``zstd_threads`` is None.

        Returns
        -------
        int
            Thread count used for zstd compression.
        """
        if zstd_threads is not None:
            return max(1, int(zstd_threads))
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - max(0, int(reserve_cores)))

    @staticmethod
    def _create_zstd_archive(
        archive_full_path: str,
        temp_pkl: str,
        files_to_archive: List[Tuple[str, str]],
        zstd_level: int,
        zstd_threads: int,
    ) -> None:
        """
        Create zstd-compressed tar archive via zstandard streaming.

        Parameters
        ----------
        archive_full_path : str
            Output archive path (including extension).
        temp_pkl : str
            Temporary path to ``pipeline.pkl``.
        files_to_archive : list of tuple[str, str]
            Archive entries as ``(source_path, archive_path)``.
        zstd_level : int
            zstd compression level (1-19).
        zstd_threads : int
            Number of zstd worker threads.

        Returns
        -------
        None
            Creates archive at ``archive_full_path``.
        """
        with open(archive_full_path, "wb") as output_file:
            compressor = zstd.ZstdCompressor(
                level=zstd_level,
                threads=zstd_threads,
            )
            with compressor.stream_writer(output_file) as zstd_writer:
                with tarfile.open(fileobj=zstd_writer, mode="w|") as tar:
                    ArchiveUtils._add_archive_items(
                        tar=tar,
                        temp_pkl=temp_pkl,
                        files_to_archive=files_to_archive,
                    )

    @staticmethod
    def _normalize_archive_output_path(
        archive_path: str,
        compression: str,
    ) -> str:
        """
        Normalize output archive path to ``.tar.<compression>``.

        Parameters
        ----------
        archive_path : str
            User-provided archive base path or full filename.
        compression : str
            Target compression extension.

        Returns
        -------
        str
            Normalized archive path with exactly one ``.tar.<compression>``.
        """
        archive_base = str(archive_path)
        known_suffixes = (".tar.zst", ".tar.gz", ".tar.bz2", ".tar.xz")
        for suffix in known_suffixes:
            if archive_base.endswith(suffix):
                archive_base = archive_base[: -len(suffix)]
                break
        return f"{archive_base}.tar.{compression}"

    @staticmethod
    def is_sha256_string(value: str) -> bool:
        """
        Check whether ``value`` is a raw SHA256 hex digest.

        Parameters
        ----------
        value : str
            Candidate SHA256 string.

        Returns
        -------
        bool
            True when the value is a 64-character hexadecimal digest.
        """
        candidate = value.strip().lower()
        if len(candidate) != 64:
            return False
        return all(char in "0123456789abcdef" for char in candidate)

    @staticmethod
    def parse_sha256_text(text: str) -> str:
        """
        Parse a SHA256 value from raw text or ``sha256sum``-style content.

        Parameters
        ----------
        text : str
            Raw text containing a SHA256 digest.

        Returns
        -------
        str
            Normalized lowercase SHA256 digest.

        Raises
        ------
        ValueError
            If no valid SHA256 digest can be parsed from the text.
        """
        stripped = text.strip()
        if not stripped:
            raise ValueError("SHA256 input cannot be empty.")
        token = stripped.split()[0]
        if not ArchiveUtils.is_sha256_string(token):
            raise ValueError("Could not parse a valid SHA256 digest.")
        return token.lower()

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """
        Compute the SHA256 digest of a local file.

        Parameters
        ----------
        file_path : str
            Path to the file to hash.

        Returns
        -------
        str
            Lowercase SHA256 digest.
        """
        normalized = PathUtils.prepare_file_path(
            file_path,
            create_parent=False,
            purpose="SHA256 file path",
        )
        digest = hashlib.sha256()
        with open(normalized, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def get_sha256_file_path(archive_path: str) -> str:
        """
        Build the sidecar ``.sha`` path for an archive file.

        Parameters
        ----------
        archive_path : str
            Local archive path.

        Returns
        -------
        str
            Normalized absolute path to the sidecar SHA256 file.
        """
        normalized = PathUtils.prepare_file_path(
            archive_path,
            create_parent=False,
            purpose="archive path",
        )
        return PathUtils.prepare_file_path(
            f"{normalized}.sha",
            create_parent=False,
            purpose="archive SHA256 path",
        )

    @staticmethod
    def write_sha256_file(archive_path: str, sha_file_path: Optional[str] = None) -> str:
        """
        Write a ``.sha`` sidecar file for an archive.

        Parameters
        ----------
        archive_path : str
            Local archive path.
        sha_file_path : str, optional
            Explicit output path for the SHA256 sidecar file.

        Returns
        -------
        str
            Path to the written SHA256 file.
        """
        archive_path = PathUtils.prepare_file_path(
            archive_path,
            create_parent=False,
            purpose="archive path",
        )
        sha_file_path = sha_file_path or ArchiveUtils.get_sha256_file_path(archive_path)
        sha_file_path = PathUtils.prepare_file_path(
            sha_file_path,
            create_parent=True,
            purpose="archive SHA256 path",
        )
        digest = ArchiveUtils.compute_sha256(archive_path)
        filename = os.path.basename(archive_path)
        with open(sha_file_path, "w", encoding="utf-8") as handle:
            handle.write(f"{digest}  {filename}\n")
        return sha_file_path

    @staticmethod
    def _ensure_output_paths_writable(
        archive_path: str,
        sha_path: Optional[str],
        overwrite: bool,
    ) -> None:
        """
        Validate archive output paths against the overwrite policy.

        Parameters
        ----------
        archive_path : str
            Target archive file path.
        sha_path : str or None
            Output path for the SHA256 sidecar file when requested.
        overwrite : bool
            Whether existing files may be replaced.

        Returns
        -------
        None
            Raises ``FileExistsError`` when overwrite is disabled and a target
            path already exists.
        """
        if overwrite:
            return
        if os.path.exists(archive_path):
            raise FileExistsError(f"Archive output already exists: {archive_path}")
        if sha_path is not None:
            if os.path.exists(sha_path):
                raise FileExistsError(f"Archive SHA256 output already exists: {sha_path}")

    @staticmethod
    def resolve_sha_output_path(
        archive_path: str,
        sha: Union[bool, str],
    ) -> Optional[str]:
        """
        Resolve the requested SHA256 output path for an archive.

        Parameters
        ----------
        archive_path : str
            Target archive file path.
        sha : bool or str
            ``False`` disables SHA output, ``True`` uses the default sidecar
            path, and a string is treated as an explicit SHA256 output path.

        Returns
        -------
        str or None
            Normalized SHA256 output path when enabled, otherwise None.
        """
        if sha is False:
            return None
        if sha is True:
            return ArchiveUtils.get_sha256_file_path(archive_path)
        return PathUtils.prepare_file_path(
            sha,
            create_parent=True,
            purpose="archive SHA256 path",
        )

    @staticmethod
    def _replace_file_from_temp(temp_path: str, target_path: str) -> None:
        """
        Replace a target file atomically from a temporary file.

        Parameters
        ----------
        temp_path : str
            Temporary file path containing final content.
        target_path : str
            Destination file path.

        Returns
        -------
        None
            Replaces the target file atomically.
        """
        os.replace(temp_path, target_path)

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
        compression: str = "zst",
        exclude_visualizations: bool = True,
        include_structure_files: bool = True,
        compression_level: Optional[int] = None,
        zstd_threads: Optional[int] = None,
        reserve_cores: int = 2,
        sha: Union[bool, str] = True,
        overwrite: bool = False,
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
        compression : str, default="zst"
            Compression method: "zst", "bz2", or "gz"
        exclude_visualizations : bool, default=True
            If True, exclude plot outputs
        include_structure_files : bool, default=True
            If True, include PDB/PML files
        compression_level : int, optional
            Compression level override. For zst this maps to level (1-19).
        zstd_threads : int, optional
            Thread count for zstd compression. If None, uses
            ``max(1, cpu_count - reserve_cores)``.
        reserve_cores : int, default=2
            Number of CPU cores to keep free for automatic zstd thread selection.
        sha : bool or str, default=True
            If True, write ``<archive>.sha`` next to the created archive.
            When a string is provided, it is used as the explicit SHA256
            output path.
        overwrite : bool, default=False
            If True, replace existing archive outputs. When False, existing
            archive or SHA256 files raise ``FileExistsError``.

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
        ...     pipeline_data, "analysis.tar.zst"
        ... )
        >>> Path(archive).exists()
        True

        Notes
        -----
        - Uses tempfile for pickle creation
        - Preserves relative paths in archive
        - zstd compression uses the ``zstandard`` Python library with streaming I/O
        - With use_memmap=False: Only pickle needed (all data in objects)
        - With use_memmap=True: Pickle + .dat files + zarr directories
        - tar.add() automatically handles both files and directories
        """
        compression_modes = {"bz2": "w:bz2", "gz": "w:gz"}
        if compression not in {"zst", "bz2", "gz"}:
            raise ValueError(
                "Compression must be one of ['zst', 'bz2', 'gz']"
            )

        archive_full_path = ArchiveUtils._normalize_archive_output_path(
            archive_path=archive_path,
            compression=compression,
        )
        archive_full_path = PathUtils.prepare_file_path(
            archive_full_path,
            create_parent=True,
            purpose="archive output path",
        )
        sha_output_path = ArchiveUtils.resolve_sha_output_path(
            archive_path=archive_full_path,
            sha=sha,
        )
        ArchiveUtils._ensure_output_paths_writable(
            archive_path=archive_full_path,
            sha_path=sha_output_path,
            overwrite=overwrite,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pkl = os.path.join(temp_dir, "pipeline.pkl")
            pipeline_data.save(temp_pkl)
            temp_archive_path = os.path.join(
                temp_dir,
                os.path.basename(archive_full_path),
            )

            files_to_archive = ArchiveUtils.collect_cache_files(
                pipeline_data.cache_dir,
                exclude_visualizations,
                include_structure_files,
                pipeline_data.use_memmap
            )

            if compression == "zst":
                zstd_level = 6 if compression_level is None else int(compression_level)
                if not 1 <= zstd_level <= 19:
                    raise ValueError("compression_level for zst must be in range 1-19")
                threads = ArchiveUtils._resolve_zstd_threads(
                    zstd_threads=zstd_threads,
                    reserve_cores=reserve_cores,
                )
                ArchiveUtils._create_zstd_archive(
                    archive_full_path=temp_archive_path,
                    temp_pkl=temp_pkl,
                    files_to_archive=files_to_archive,
                    zstd_level=zstd_level,
                    zstd_threads=threads,
                )
            else:
                tar_kwargs = {}
                if compression_level is not None:
                    tar_kwargs["compresslevel"] = int(compression_level)

                with tarfile.open(
                    temp_archive_path, compression_modes[compression], **tar_kwargs
                ) as tar:
                    ArchiveUtils._add_archive_items(
                        tar=tar,
                        temp_pkl=temp_pkl,
                        files_to_archive=files_to_archive,
                    )
            ArchiveUtils._replace_file_from_temp(
                temp_path=temp_archive_path,
                target_path=archive_full_path,
            )
            if sha_output_path is not None:
                ArchiveUtils.write_sha256_file(
                    archive_path=archive_full_path,
                    sha_file_path=sha_output_path,
                )

        return archive_full_path

    @staticmethod
    def _extract_zst_archive(
        archive_path: Path,
        extract_dir: Path,
    ) -> None:
        """
        Extract ``.tar.zst`` archives via zstandard streaming.

        Parameters
        ----------
        archive_path : Path
            Path to ``.tar.zst`` archive file.
        extract_dir : Path
            Destination directory for extracted files.

        Returns
        -------
        None
            Extracts archive contents into ``extract_dir``.

        """
        with open(archive_path, "rb") as input_file:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(input_file) as zstd_reader:
                with tarfile.open(fileobj=zstd_reader, mode="r|") as tar:
                    tar.extractall(extract_dir, filter="data")

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
        >>> extract_dir = ArchiveUtils.extract_archive("analysis.tar.zst")
        >>> (extract_dir / "pipeline.pkl").exists()
        True

        >>> extract_dir = ArchiveUtils.extract_archive(
        ...     "analysis.tar.zst",
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

        if archive_path.name.endswith(".tar.zst"):
            ArchiveUtils._extract_zst_archive(
                archive_path=archive_path,
                extract_dir=extract_dir,
            )
        else:
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_dir, filter="data")

        return extract_dir
