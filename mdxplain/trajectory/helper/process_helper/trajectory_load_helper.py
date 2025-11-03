# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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
MD trajectory loading strategies and utilities.

Handles the loading methodology for MD trajectories from files and directories.
This class can be easily replaced or extended for different loading strategies.
Supports nested directory structures and trajectory concatenation.
"""

import os
import warnings
from typing import Any, Dict, List, Tuple, Union, Optional
from pathlib import Path

import mdtraj as md
from tqdm import tqdm

from ...entities.dask_md_trajectory import DaskMDTrajectory


# Supported trajectory formats (require separate topology file)
TRAJECTORY_FORMATS = (
    '.xtc', '.trr',  # GROMACS
    '.dcd',  # CHARMM/NAMD
    '.nc', '.ncdf', '.mdcrd', '.crd',  # AMBER
    '.arc', '.lammpstrj', '.xyz',  # Others
)

# Supported topology/structure formats
TOPOLOGY_FORMATS = (
    '.pdb', '.pdb.gz',  # PDB
    '.prmtop', '.parm7', '.prm7', '.top',  # AMBER
    '.ncrst', '.rst', '.rst7',  # AMBER restart
    '.psf',  # CHARMM/XPLOR
    '.mol2', '.xml', '.gsd', '.stk',  # Others
)

# Self-contained formats (have embedded topology, don't need separate topology file)
SELF_CONTAINED_FORMATS = (
    '.h5', '.hdf5', '.lh5',  # HDF5
    '.pdb', '.pdb.gz',  # PDB (can be both topology and trajectory)
    '.dtr',  # DESRES
)


class TrajectoryLoadHelper:
    """
    Internal utility class for loading MD trajectories from files and directories.

    Supports nested directory structures and trajectory concatenation.
    """

    @staticmethod
    def load_trajectories(
        data_input: Union[List[Any], str], 
        concat: bool = False, 
        stride: int = 1,
        selection: Optional[str] = None,
        use_memmap: bool = False,
        chunk_size: int = 1000,
        cache_dir: str = "./cache"
    ) -> Dict[str, List[Any]]:
        """
        Load trajectories from various input types.

        Parameters
        ----------
        data_input : list or str
            List of trajectory objects or path to directory
        concat : bool, default=False
            Whether to concatenate trajectories per system
        stride : int, default=1
            Frame striding parameter
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys containing
            list of loaded trajectory objects and their corresponding names
            
        Examples
        --------
        >>> # Load from trajectory list
        >>> trajs = [traj1, traj2, traj3]
        >>> result = TrajectoryLoadHelper.load_trajectories(trajs, concat=True)
        >>> print(f"Loaded {len(result['trajectories'])} trajectories")
        >>> 
        >>> # Load from directory
        >>> result = TrajectoryLoadHelper.load_trajectories('../data', use_memmap=True)
        >>> # Load with selection
        >>> result = TrajectoryLoadHelper.load_trajectories('../data', selection='protein')
        """
        if isinstance(data_input, list):
            return TrajectoryLoadHelper._load_from_list(data_input, concat, selection, use_memmap, chunk_size, cache_dir)

        if isinstance(data_input, str) and os.path.exists(data_input):
            return TrajectoryLoadHelper._load_from_directory(data_input, concat, stride, selection, use_memmap, chunk_size, cache_dir)

        warnings.warn(
            f"Invalid data input: {data_input}. Expected list of trajectories or valid path."
        )
        return {"trajectories": [], "names": []}

    @staticmethod
    def _load_from_list(
        trajectory_list: List[Any], 
        concat: bool, 
        selection: Optional[str] = None,
        use_memmap: bool = False,
        chunk_size: int = 1000,
        cache_dir: str = "./cache"
    ) -> Dict[str, List[Any]]:
        """
        Handle trajectory list input with optional concatenation.

        Parameters
        ----------
        trajectory_list : list
            List of MDTraj trajectory objects
        concat : bool
            Whether to concatenate all trajectories into one
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        result = TrajectoryLoadHelper._create_initial_result(trajectory_list)
        result = TrajectoryLoadHelper._apply_selection_if_needed(result, selection)
        return TrajectoryLoadHelper._handle_concatenation_if_needed(result, concat)
    
    @staticmethod
    def _create_initial_result(trajectory_list: List[Any]) -> Dict[str, List[Any]]:
        """
        Create initial result structure from trajectory list.
        
        Parameters
        ----------
        trajectory_list : List[Any]
            List of trajectory objects
            
        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        names = [f"provided_traj_{i}" for i in range(len(trajectory_list))]
        return {"trajectories": trajectory_list, "names": names}
    
    @staticmethod
    def _apply_selection_if_needed(
        result: Dict[str, List[Any]], selection: Optional[str]
    ) -> Dict[str, List[Any]]:
        """
        Apply selection to trajectories if specified.
        
        Parameters
        ----------
        result : dict
            Dictionary with 'trajectories' and 'names' keys
        selection : str, optional
            MDTraj selection string
            
        Returns
        -------
        dict
            Dictionary with selected trajectories
        """
        if selection is not None:
            return TrajectoryLoadHelper._apply_selection_to_result(result, selection)
        return result
    
    @staticmethod
    def _handle_concatenation_if_needed(
        result: Dict[str, List[Any]], concat: bool
    ) -> Dict[str, List[Any]]:
        """
        Handle trajectory concatenation if requested.
        
        Parameters
        ----------
        result : dict
            Dictionary with 'trajectories' and 'names' keys
        concat : bool
            Whether to concatenate trajectories
            
        Returns
        -------
        dict
            Dictionary with potentially concatenated trajectories
        """
        if concat and len(result["trajectories"]) > 1:
            return TrajectoryLoadHelper._concatenate_trajectories(result)
        return result
    
    @staticmethod
    def _concatenate_trajectories(result: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Concatenate multiple trajectories into single trajectory.
        
        Parameters
        ----------
        result : dict
            Dictionary with 'trajectories' and 'names' keys
            
        Returns
        -------
        dict
            Dictionary with single concatenated trajectory
        """
        print(f"Concatenating {len(result['trajectories'])} processed trajectories...")
        concatenated = result["trajectories"][0]
        for traj in tqdm(result["trajectories"][1:], desc="Concatenating"):
            concatenated = concatenated.join(traj)
        print(f"Result: 1 concatenated trajectory with {concatenated.n_frames} frames")
        return {
            "trajectories": [concatenated],
            "names": ["provided_traj_concatenated"],
        }

    @staticmethod
    def _load_from_directory(
        directory_path: str, concat: bool, stride: int, selection: Optional[str] = None,
        use_memmap: bool = False, chunk_size: int = 1000, cache_dir: str = "./cache"
    ) -> Dict[str, List[Any]]:
        """
        Load trajectories from directory (flat or nested structure).

        Parameters
        ----------
        directory_path : str
            Path to directory containing trajectory files
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        # Determine directory structure and collect trajectories
        if TrajectoryLoadHelper._has_subdirectories(directory_path):
            return TrajectoryLoadHelper._load_nested_structure(
                directory_path, concat, stride, selection, use_memmap, chunk_size, cache_dir
            )
        else:
            return TrajectoryLoadHelper._load_flat_structure(
                directory_path, concat, stride, selection, use_memmap, chunk_size, cache_dir
            )

    @staticmethod
    def _has_subdirectories(directory_path: str) -> bool:
        """
        Check if directory contains subdirectories.

        Parameters
        ----------
        directory_path : str
            Path to directory to check

        Returns
        -------
        bool
            True if subdirectories exist, False otherwise
        """
        subdirs = [
            d
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d)) and not d.startswith('.')
        ]
        return len(subdirs) > 0

    @staticmethod
    def _load_nested_structure(
        directory_path: str, 
        concat: bool, 
        stride: int, 
        selection: Optional[str] = None,
        use_memmap: bool = False,
        chunk_size: int = 1000,
        cache_dir: str = "./cache"
    ) -> Dict[str, List[Any]]:
        """
        Load from nested directory structure (multiple systems).

        Also handles files in root directory as separate system.

        Parameters
        ----------
        directory_path : str
            Path to directory containing trajectory files
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        trajectories = []
        names = []
        system_summary = []

        # Get subdirectories
        subdirs = [
            d
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d)) and not d.startswith('.')
        ]

        # Load from subdirectories
        for subdir in tqdm(subdirs, desc="Loading systems"):
            subdir_path = os.path.join(directory_path, subdir)
            result = TrajectoryLoadHelper._load_system_trajectories(
                subdir_path, subdir, concat, stride, selection, use_memmap, chunk_size, cache_dir
            )

            if result["trajectories"]:
                trajectories.extend(result["trajectories"])
                names.extend(result["names"])
                system_summary.append((subdir, len(result["trajectories"])))

        # Load root directory files as separate system if present
        TrajectoryLoadHelper._load_root_system_if_present(
            directory_path, concat, stride, trajectories, names, system_summary, selection,
            use_memmap, chunk_size, cache_dir
        )

        TrajectoryLoadHelper._print_loading_summary(system_summary, concat)
        return {"trajectories": trajectories, "names": names}

    @staticmethod
    def _load_flat_structure(directory_path: str, concat: bool, stride: int, selection: Optional[str] = None, 
                            use_memmap: bool = False, chunk_size: int = 1000, cache_dir: str = "./cache") -> Dict[str, List[Any]]:
        """
        Load from flat directory structure (single system).

        Parameters
        ----------
        directory_path : str
            Path to directory containing trajectory files
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        system_name = os.path.basename(directory_path.rstrip("/\\"))
        result = TrajectoryLoadHelper._load_system_trajectories(
            directory_path, system_name, concat, stride, selection, use_memmap, chunk_size, cache_dir
        )

        if result["trajectories"]:
            system_summary = [(system_name, len(result["trajectories"]))]
            TrajectoryLoadHelper._print_loading_summary(system_summary, concat)

        return result

    @staticmethod
    def _print_loading_summary(system_summary: List[Tuple[str, int]], concat: bool) -> None:
        """
        Print summary of loaded trajectories.

        Parameters
        ----------
        system_summary : list
            List of (system_name, trajectory_count) tuples
        concat : bool
            Whether concatenation was used

        Returns
        -------
        None
            Prints loading summary to console
        """
        total_trajectories = sum(count for _, count in system_summary)

        # Print header
        TrajectoryLoadHelper._print_summary_header(
            len(system_summary), total_trajectories, concat
        )

        # Print system details
        for system, count in system_summary:
            TrajectoryLoadHelper._print_system_info(system, count, concat)

    @staticmethod
    def _print_summary_header(num_systems: int, total_trajectories: int, concat: bool) -> None:
        """
        Print the header line of the loading summary.

        Parameters
        ----------
        num_systems : int
            Number of systems loaded
        total_trajectories : int
            Total number of trajectories
        concat : bool
            Whether concatenation was used

        Returns
        -------
        None
            Prints header line to console
        """
        if concat:
            print(
                f"\nLoaded {num_systems} systems with {total_trajectories} "
                f"concatenated trajectories:"
            )
        else:
            print(
                f"\nLoaded {num_systems} systems with {total_trajectories} total trajectories:"
            )

    @staticmethod
    def _print_system_info(system: str, count: int, concat: bool) -> None:
        """
        Print information for a single system.

        Parameters
        ----------
        system : str
            System name
        count : int
            Number of trajectories for this system
        concat : bool
            Whether concatenation was used

        Returns
        -------
        None
            Prints system information to console
        """
        if concat and count > 1:
            print(f"  {system}: 1 concatenated trajectory ({count} files)")
        else:
            print(f"  {system}: {count} trajectories")

    @staticmethod
    def _load_system_trajectories(subdir_path: str, subdir_name: str, concat: bool, stride: int, selection: Optional[str] = None,
                                 use_memmap: bool = False, chunk_size: int = 1000, cache_dir: str = "./cache") -> Dict[str, List[Any]]:
        """
        Load all trajectories for a single system.

        Parameters
        ----------
        subdir_path : str
            Path to directory containing trajectory files
        subdir_name : str
            Name of the system
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        top_files = TrajectoryLoadHelper._get_topology_files_from_directory(subdir_path)
        traj_files = TrajectoryLoadHelper._get_trajectory_files_from_directory(subdir_path)

        # Check if trajectory files are self-contained (have embedded topology)
        if traj_files:
            first_ext = os.path.splitext(traj_files[0])[1].lower()
            is_self_contained = first_ext in SELF_CONTAINED_FORMATS
        else:
            is_self_contained = False

        # Case 1: Trajectory files present with separate topology
        if traj_files and not is_self_contained:
            if len(top_files) != 1:
                raise ValueError(
                    f"Ambiguous file configuration in {subdir_path}: "
                    f"Found {len(top_files)} topology files and {len(traj_files)} trajectory files. "
                    f"Cannot determine topology-trajectory pairing. "
                    f"Need exactly 1 topology file for multiple trajectory files."
                )
            top_path = os.path.join(subdir_path, top_files[0])
            result = TrajectoryLoadHelper._load_trajectory_files_from_directory(
                traj_files, top_path, subdir_path, subdir_name, stride, use_memmap, chunk_size, cache_dir
            )

        # Case 2: Self-contained trajectory files (H5, PDB, DTR, etc.)
        elif traj_files and is_self_contained:
            result = TrajectoryLoadHelper._load_pdb_files_from_directory(
                traj_files, subdir_path, subdir_name, stride, use_memmap, chunk_size, cache_dir
            )

        # Case 3: Only topology files - load each as separate trajectory
        elif top_files and not traj_files:
            result = TrajectoryLoadHelper._load_pdb_files_from_directory(
                top_files, subdir_path, subdir_name, stride, use_memmap, chunk_size, cache_dir
            )

        # Case 4: No files found
        else:
            warnings.warn(f"No trajectory or topology files found in {subdir_path}")
            return {"trajectories": [], "names": []}

        # Apply selection to each trajectory before concatenation if provided
        if selection is not None:
            result = TrajectoryLoadHelper._apply_selection_to_result(result, selection)
        
        return TrajectoryLoadHelper._handle_traj_concatenation(
            result["trajectories"], result["names"], concat, subdir_name
        )

    @staticmethod
    def _get_topology_files_from_directory(subdir_path: str) -> List[str]:
        """
        Get list of all supported topology/structure files in directory.

        Supports PDB, PSF, PRMTOP, MOL2, XML, and other MDTraj-compatible formats.

        Parameters
        ----------
        subdir_path : str
            Path to directory to search

        Returns
        -------
        list
            List of topology filenames (not full paths)
        """
        return [f for f in os.listdir(subdir_path)
                if f.lower().endswith(TOPOLOGY_FORMATS)]

    @staticmethod
    def _get_trajectory_files_from_directory(subdir_path: str) -> List[str]:
        """
        Get list of all supported trajectory files in directory.

        Supports GROMACS (XTC, TRR), CHARMM/NAMD (DCD), AMBER (NC, MDCRD),
        HDF5 (H5), and other MDTraj-compatible trajectory formats.

        Parameters
        ----------
        subdir_path : str
            Path to directory to search

        Returns
        -------
        list
            List of trajectory filenames (not full paths)
        """
        return [f for f in os.listdir(subdir_path)
                if f.lower().endswith(TRAJECTORY_FORMATS)]

    @staticmethod
    def _load_trajectory_files_from_directory(
        traj_files: List[str], top_path: str, subdir_path: str, subdir_name: str, stride: int,
        use_memmap: bool = False, chunk_size: int = 1000, cache_dir: str = "./cache"
    ) -> Dict[str, List[Any]]:
        """
        Load trajectory files with topology.

        Supports all MDTraj trajectory formats (XTC, DCD, TRR, NC, etc.)
        with separate topology files (PDB, PSF, PRMTOP, etc.).

        Parameters
        ----------
        traj_files : list
            List of trajectory filenames (not full paths)
        top_path : str
            Path to topology file
        subdir_path : str
            Path to directory containing trajectory files
        subdir_name : str
            Name of the system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        system_trajs = []
        names = []

        # Extract topology basename for naming
        top_basename = os.path.splitext(os.path.basename(top_path))[0]

        for traj_file in tqdm(traj_files, desc=f"Loading {subdir_name}", leave=False):
            traj_path = os.path.join(subdir_path, traj_file)
            traj_basename = os.path.splitext(traj_file)[0]

            traj = TrajectoryLoadHelper._load_single_trajectory(
                traj_path, top_path, stride, use_memmap, chunk_size, cache_dir
            )
            system_trajs.append(traj)

            # Generate name: directory_topologyname_trajectoryname
            name = f"{subdir_name}_{top_basename}_{traj_basename}"
            names.append(name)

        return {"trajectories": system_trajs, "names": names}

    @staticmethod
    def _load_pdb_files_from_directory(pdb_files: List[str], subdir_path: str, subdir_name: str, stride: int,
                                      use_memmap: bool = False, chunk_size: int = 1000, cache_dir: str = "./cache") -> Dict[str, List[Any]]:
        """
        Load PDB files as separate trajectories.

        Parameters
        ----------
        pdb_files : list
            List of PDB filenames (not full paths)
        subdir_path : str
            Path to directory containing trajectory files
        subdir_name : str
            Name of the system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        system_trajs = []
        names = []

        for pdb in tqdm(pdb_files, desc=f"Loading {subdir_name}", leave=False):
            pdb_path = os.path.join(subdir_path, pdb)
            pdb_basename = os.path.splitext(pdb)[0]

            traj = TrajectoryLoadHelper._load_single_trajectory(
                pdb_path, None, stride, use_memmap, chunk_size, cache_dir
            )
            system_trajs.append(traj)

            # Generate name: directory_pdbname (no xtc for PDB-only trajectories)
            name = f"{subdir_name}_{pdb_basename}"
            names.append(name)

        return {"trajectories": system_trajs, "names": names}

    @staticmethod
    def _handle_traj_concatenation(system_trajs: List[Any], names: List[str], concat: bool, subdir_name: str) -> Dict[str, List[Any]]:
        """
        Handle trajectory concatenation if requested.

        Parameters
        ----------
        system_trajs : list
            List of loaded MDTraj trajectory objects
        names : list
            List of trajectory names
        concat : bool
            Whether to concatenate trajectories per system
        subdir_name : str
            Name of the system

        Returns
        -------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        if concat and len(system_trajs) > 1:
            print(
                f"  Concatenating {len(system_trajs)} trajectories "
                f"for {subdir_name}..."
            )
            concatenated = system_trajs[0]
            for traj in system_trajs[1:]:
                concatenated = concatenated.join(traj)
            return {
                "trajectories": [concatenated],
                "names": [f"{subdir_name}_concatenated"],
            }
        return {"trajectories": system_trajs, "names": names}

    @staticmethod
    def _load_root_system_if_present(
        directory_path: str, concat: bool, stride: int, trajectories: List[Any], names: List[str], system_summary: List[Tuple[str, int]], selection: Optional[str] = None,
        use_memmap: bool = False, chunk_size: int = 1000, cache_dir: str = "./cache"
    ) -> None:
        """
        Load root directory files as separate system if present.

        Parameters
        ----------
        directory_path : str
            Path to directory containing trajectory files
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        trajectories : list
            List of loaded MDTraj trajectory objects
        names : list
            List of trajectory names
        system_summary : list
            List of (system_name, trajectory_count) tuples
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True)
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory (only used when use_memmap=True)

        Returns
        -------
        None
            Loads root directory files as separate system if present
        """
        root_pdb_files = TrajectoryLoadHelper._get_pdb_files_from_directory(directory_path)
        if root_pdb_files:
            root_system_name = os.path.basename(directory_path.rstrip("/\\"))
            root_result = TrajectoryLoadHelper._load_system_trajectories(
                directory_path, root_system_name, concat, stride, selection, use_memmap, chunk_size, cache_dir
            )

            if root_result["trajectories"]:
                trajectories.extend(root_result["trajectories"])
                names.extend(root_result["names"])
                system_summary.append(
                    (root_system_name, len(root_result["trajectories"]))
                )

    @staticmethod
    def _apply_selection_to_result(result: Dict[str, List[Any]], selection: str) -> Dict[str, List[Any]]:
        """
        Apply atom selection to trajectories in result.

        Parameters
        ----------
        result : dict
            Dictionary with 'trajectories' and 'names' keys
        selection : str
            MDTraj selection string to apply

        Returns
        -------
        dict
            Dictionary with selected trajectories

        Notes
        -----
        Selection is applied to each trajectory individually to ensure
        compatible atom counts for concatenation.
        """
        trajectories = result["trajectories"]
        names = result["names"]
        
        if not trajectories:
            return result

        print(f"Applying selection '{selection}' to {len(trajectories)} trajectories...")
        selected_trajectories = []
        
        for i, traj in enumerate(trajectories):
            atom_indices = traj.topology.select(selection)
            if len(atom_indices) == 0:
                print(f"   Warning: Selection '{selection}' returned 0 atoms for trajectory {i}")
            selected_traj = traj.atom_slice(atom_indices)
            selected_trajectories.append(selected_traj)
            print(f"   Trajectory {i} ({names[i]}): {traj.n_atoms} → {selected_traj.n_atoms} atoms")
        
        print(f"Selection completed. All trajectories now have {selected_trajectories[0].n_atoms if selected_trajectories else 0} atoms.")
        
        return {"trajectories": selected_trajectories, "names": names}
    
    @staticmethod
    def _load_single_trajectory(
        trajectory_file: str, 
        topology_file: Optional[str] = None,
        stride: int = 1,
        use_memmap: bool = False,
        chunk_size: int = 1000,
        cache_dir: str = "./cache"
    ) -> Union[DaskMDTrajectory, md.Trajectory]:
        """
        Load a single trajectory file using either md.load or DaskMDTrajectory.
        
        Parameters
        ----------
        trajectory_file : str
            Path to trajectory file
        topology_file : str, optional
            Path to topology file
        stride : int, default=1
            Frame striding parameter
        use_memmap : bool, default=False
            Whether to use DaskMDTrajectory for memory mapping
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory
        cache_dir : str, default="./cache"
            Cache directory for DaskMDTrajectory
            
        Returns
        -------
        Union[DaskMDTrajectory, md.Trajectory]
            Either md.Trajectory or DaskMDTrajectory object
        """
        if use_memmap:
            # Create cache path
            traj_path = Path(trajectory_file)
            zarr_cache_path = os.path.join(cache_dir, f"{traj_path.stem}.dask.zarr")
            
            # Create DaskMDTrajectory
            dask_traj = DaskMDTrajectory(
                trajectory_file=trajectory_file,
                topology_file=topology_file,
                chunk_size=chunk_size,
                zarr_cache_path=zarr_cache_path
            )
            
            # Apply stride if needed
            if stride > 1:
                frame_indices = list(range(0, dask_traj.n_frames, stride))
                dask_traj = dask_traj[frame_indices]
                
            return dask_traj
        # Use regular MDTraj loading
        return md.load(trajectory_file, top=topology_file, stride=stride)
