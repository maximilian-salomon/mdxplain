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
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


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
        cache_parent: Path,
        items_list: list,
        processed_set: set
    ) -> None:
        """
        Add zarr directory to archive list.

        Parameters
        ----------
        zarr_path : Path
            Path to zarr directory
        cache_parent : Path
            Parent of cache directory
        items_list : list
            List to append items to
        processed_set : set
            Set of processed zarr directories
        """
        if zarr_path not in processed_set:
            relative_path = zarr_path.relative_to(cache_parent)
            items_list.append((str(zarr_path), str(relative_path)))
            processed_set.add(zarr_path)

    @staticmethod
    def _process_item_for_archive(
        item_path: Path,
        cache_parent: Path,
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
        cache_parent : Path
            Parent of cache directory
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
                    item_path, cache_parent, items_list, processed_zarr
                )
        elif item_path.is_file():
            ArchiveUtils._add_file_to_archive(
                item_path, cache_parent, items_list,
                exclude_viz, include_struct, use_memmap
            )

    @staticmethod
    def _add_file_to_archive(
        file_path: Path,
        cache_parent: Path,
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
        cache_parent : Path
            Parent of cache directory
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
            relative_path = file_path.relative_to(cache_parent)
            items_list.append((str(file_path), str(relative_path)))

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

        cache_parent = cache_path.parent

        for item_path in cache_path.rglob('*'):
            ArchiveUtils._process_item_for_archive(
                item_path, cache_parent, items_to_archive,
                processed_zarr_dirs, exclude_visualizations,
                include_structure_files, use_memmap
            )

        return items_to_archive

    @staticmethod
    def create_archive(
        pipeline_data,
        archive_path: str,
        compression: str = "xz",
        exclude_visualizations: bool = True,
        include_structure_files: bool = True
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

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pkl = os.path.join(temp_dir, "pipeline.pkl")
            pipeline_data.save(temp_pkl)

            files_to_archive = ArchiveUtils.collect_cache_files(
                pipeline_data.cache_dir,
                exclude_visualizations,
                include_structure_files,
                pipeline_data.use_memmap
            )

            with tarfile.open(
                archive_full_path, compression_modes[compression]
            ) as tar:
                tar.add(temp_pkl, arcname="pipeline.pkl")

                for file_path, archive_name in tqdm(
                    files_to_archive,
                    desc="Adding files to archive",
                    unit="file"
                ):
                    tar.add(file_path, arcname=archive_name)

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
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        if extract_to is None:
            extract_dir = archive_path.parent / archive_path.stem.replace(
                '.tar', ''
            )
        else:
            extract_dir = Path(extract_to)

        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(extract_dir)

        return extract_dir
