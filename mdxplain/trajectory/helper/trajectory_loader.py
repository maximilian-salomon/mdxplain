# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
from typing import Any, Dict, List, Union, Optional

import mdtraj as md
from tqdm import tqdm


class TrajectoryLoader:
    """
    Internal utility class for loading MD trajectories from files and directories.

    Supports nested directory structures and trajectory concatenation.
    """

    @staticmethod
    def load_trajectories(
        data_input: Union[List[Any], str], 
        concat: bool = False, 
        stride: int = 1,
        selection: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Load trajectories from various input types.

        Parameters:
        -----------
        data_input : list or str
            List of trajectory objects or path to directory
        concat : bool, default=False
            Whether to concatenate trajectories per system
        stride : int, default=1
            Frame striding parameter
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation

        Returns:
        --------
        dict
            Dictionary with 'trajectories' and 'names' keys containing
            list of loaded trajectory objects and their corresponding names
        """
        if isinstance(data_input, list):
            return TrajectoryLoader._load_from_list(data_input, concat, selection)

        if isinstance(data_input, str) and os.path.exists(data_input):
            return TrajectoryLoader._load_from_directory(data_input, concat, stride, selection)

        warnings.warn(
            f"Invalid data input: {data_input}. Expected list of trajectories or valid path."
        )
        return {"trajectories": [], "names": []}

    @staticmethod
    def _load_from_list(
        trajectory_list: List[Any], 
        concat: bool, 
        selection: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Handle trajectory list input with optional concatenation.

        Parameters:
        -----------
        trajectory_list : list
            List of MDTraj trajectory objects
        concat : bool
            Whether to concatenate all trajectories into one
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation

        Returns:
        --------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        # Generate names for provided trajectories
        names = [f"provided_traj_{i}" for i in range(len(trajectory_list))]

        # Create initial result structure
        result = {"trajectories": trajectory_list, "names": names}
        
        # Apply selection to each trajectory before concatenation if provided
        if selection is not None:
            result = TrajectoryLoader._apply_selection_to_result(result, selection, concat)

        if concat and len(result["trajectories"]) > 1:
            print(f"Concatenating {len(result['trajectories'])} processed trajectories...")
            concatenated = result["trajectories"][0]
            for traj in tqdm(result["trajectories"][1:], desc="Concatenating"):
                concatenated = concatenated.join(traj)
            print(
                f"Result: 1 concatenated trajectory with {concatenated.n_frames} frames"
            )
            return {
                "trajectories": [concatenated],
                "names": ["provided_traj_concatenated"],
            }
        return result

    @staticmethod
    def _load_from_directory(
        directory_path: str, concat: bool, stride: int, selection: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Load trajectories from directory (flat or nested structure).

        Parameters:
        -----------
        directory_path : str
            Path to directory containing trajectory files
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation

        Returns:
        --------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        # Determine directory structure and collect trajectories
        if TrajectoryLoader._has_subdirectories(directory_path):
            return TrajectoryLoader._load_nested_structure(
                directory_path, concat, stride, selection
            )
        else:
            return TrajectoryLoader._load_flat_structure(
                directory_path, concat, stride, selection
            )

    @staticmethod
    def _has_subdirectories(directory_path: str) -> bool:
        """
        Check if directory contains subdirectories.

        Parameters:
        -----------
        directory_path : str
            Path to directory to check

        Returns:
        --------
        bool
            True if subdirectories exist, False otherwise
        """
        subdirs = [
            d
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d))
        ]
        return len(subdirs) > 0

    @staticmethod
    def _load_nested_structure(
        directory_path: str, 
        concat: bool, 
        stride: int, 
        selection: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Load from nested directory structure (multiple systems).

        Also handles files in root directory as separate system.

        Parameters:
        -----------
        directory_path : str
            Path to directory containing trajectory files
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation

        Returns:
        --------
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
            if os.path.isdir(os.path.join(directory_path, d))
        ]

        # Load from subdirectories
        for subdir in tqdm(subdirs, desc="Loading systems"):
            subdir_path = os.path.join(directory_path, subdir)
            result = TrajectoryLoader._load_system_trajectories(
                subdir_path, subdir, concat, stride, selection
            )

            if result["trajectories"]:
                trajectories.extend(result["trajectories"])
                names.extend(result["names"])
                system_summary.append((subdir, len(result["trajectories"])))

        # Load root directory files as separate system if present
        TrajectoryLoader._load_root_system_if_present(
            directory_path, concat, stride, trajectories, names, system_summary, selection
        )

        TrajectoryLoader._print_loading_summary(system_summary, concat)
        return {"trajectories": trajectories, "names": names}

    @staticmethod
    def _load_flat_structure(directory_path, concat, stride, selection=None):
        """
        Load from flat directory structure (single system).

        Parameters:
        -----------
        directory_path : str
            Path to directory containing trajectory files
        concat : bool
            Whether to concatenate trajectories per system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)
        selection : str, optional
            MDTraj selection string to apply to each trajectory before concatenation

        Returns:
        --------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        system_name = os.path.basename(directory_path.rstrip("/\\"))
        result = TrajectoryLoader._load_system_trajectories(
            directory_path, system_name, concat, stride, selection
        )

        if result["trajectories"]:
            system_summary = [(system_name, len(result["trajectories"]))]
            TrajectoryLoader._print_loading_summary(system_summary, concat)

        return result

    @staticmethod
    def _print_loading_summary(system_summary, concat):
        """
        Print summary of loaded trajectories.

        Parameters:
        -----------
        system_summary : list
            List of (system_name, trajectory_count) tuples
        concat : bool
            Whether concatenation was used

        Returns:
        --------
        None
            Prints loading summary to console
        """
        total_trajectories = sum(count for _, count in system_summary)

        # Print header
        TrajectoryLoader._print_summary_header(
            len(system_summary), total_trajectories, concat
        )

        # Print system details
        for system, count in system_summary:
            TrajectoryLoader._print_system_info(system, count, concat)

    @staticmethod
    def _print_summary_header(num_systems, total_trajectories, concat):
        """
        Print the header line of the loading summary.

        Parameters:
        -----------
        num_systems : int
            Number of systems loaded
        total_trajectories : int
            Total number of trajectories
        concat : bool
            Whether concatenation was used

        Returns:
        --------
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
    def _print_system_info(system, count, concat):
        """
        Print information for a single system.

        Parameters:
        -----------
        system : str
            System name
        count : int
            Number of trajectories for this system
        concat : bool
            Whether concatenation was used

        Returns:
        --------
        None
            Prints system information to console
        """
        if concat and count > 1:
            print(f"  {system}: 1 concatenated trajectory ({count} files)")
        else:
            print(f"  {system}: {count} trajectories")

    @staticmethod
    def _load_system_trajectories(subdir_path, subdir_name, concat, stride, selection=None):
        """
        Load all trajectories for a single system.

        Parameters:
        -----------
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

        Returns:
        --------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        pdb_files = TrajectoryLoader._get_pdb_files_from_directory(subdir_path)
        xtc_files = TrajectoryLoader._get_xtc_files_from_directory(subdir_path)

        # Validate file combination
        if not pdb_files:
            warnings.warn(f"No PDB files found in {subdir_path}")
            return {"trajectories": [], "names": []}

        # Case 1: XTC files present - use single PDB as topology
        if xtc_files:
            if len(pdb_files) != 1:
                raise ValueError(
                    f"Ambiguous file configuration in {subdir_path}: "
                    f"Found {len(pdb_files)} PDB files and {len(xtc_files)} XTC files. "
                    f"Cannot determine topology-trajectory pairing. "
                    f"Use either: 1 PDB + multiple XTCs, or multiple PDBs without XTCs."
                )
            pdb_path = os.path.join(subdir_path, pdb_files[0])
            result = TrajectoryLoader._load_trajectory_files_from_directory(
                xtc_files, pdb_path, subdir_path, subdir_name, stride
            )

        # Case 2: No XTC files - load each PDB as separate trajectory
        else:
            result = TrajectoryLoader._load_pdb_files_from_directory(
                pdb_files, subdir_path, subdir_name, stride
            )

        # Apply selection to each trajectory before concatenation if provided
        if selection is not None:
            result = TrajectoryLoader._apply_selection_to_result(result, selection, concat)
        
        return TrajectoryLoader._handle_traj_concatenation(
            result["trajectories"], result["names"], concat, subdir_name
        )

    @staticmethod
    def _get_pdb_files_from_directory(subdir_path):
        """
        Get list of PDB files in directory.

        Parameters:
        -----------
        subdir_path : str
            Path to directory to search

        Returns:
        --------
        list
            List of PDB filenames (not full paths)
        """
        return [f for f in os.listdir(subdir_path) if f.endswith(".pdb")]

    @staticmethod
    def _get_xtc_files_from_directory(subdir_path):
        """
        Get list of XTC files in directory.

        Parameters:
        -----------
        subdir_path : str
            Path to directory to search

        Returns:
        --------
        list
            List of XTC filenames (not full paths)
        """
        return [f for f in os.listdir(subdir_path) if f.endswith(".xtc")]

    @staticmethod
    def _load_trajectory_files_from_directory(
        xtc_files, pdb_path, subdir_path, subdir_name, stride
    ):
        """
        Load XTC files with PDB topology.

        Parameters:
        -----------
        xtc_files : list
            List of XTC filenames (not full paths)
        pdb_path : str
            Path to PDB file
        subdir_path : str
            Path to directory containing trajectory files
        subdir_name : str
            Name of the system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)

        Returns:
        --------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        system_trajs = []
        names = []

        # Extract PDB basename for naming
        pdb_basename = os.path.splitext(os.path.basename(pdb_path))[0]

        for xtc in tqdm(xtc_files, desc=f"Loading {subdir_name}", leave=False):
            xtc_path = os.path.join(subdir_path, xtc)
            xtc_basename = os.path.splitext(xtc)[0]

            traj = md.load(xtc_path, top=pdb_path, stride=stride)
            system_trajs.append(traj)

            # Generate name: directory_pdbname_xtcname
            name = f"{subdir_name}_{pdb_basename}_{xtc_basename}"
            names.append(name)

        return {"trajectories": system_trajs, "names": names}

    @staticmethod
    def _load_pdb_files_from_directory(pdb_files, subdir_path, subdir_name, stride):
        """
        Load PDB files as separate trajectories.

        Parameters:
        -----------
        pdb_files : list
            List of PDB filenames (not full paths)
        subdir_path : str
            Path to directory containing trajectory files
        subdir_name : str
            Name of the system
        stride : int
            Frame striding (1=all frames, 2=every 2nd frame, etc.)

        Returns:
        --------
        dict
            Dictionary with 'trajectories' and 'names' keys
        """
        system_trajs = []
        names = []

        for pdb in tqdm(pdb_files, desc=f"Loading {subdir_name}", leave=False):
            pdb_path = os.path.join(subdir_path, pdb)
            pdb_basename = os.path.splitext(pdb)[0]

            traj = md.load(pdb_path, stride=stride)
            system_trajs.append(traj)

            # Generate name: directory_pdbname (no xtc for PDB-only trajectories)
            name = f"{subdir_name}_{pdb_basename}"
            names.append(name)

        return {"trajectories": system_trajs, "names": names}

    @staticmethod
    def _handle_traj_concatenation(system_trajs, names, concat, subdir_name):
        """
        Handle trajectory concatenation if requested.

        Parameters:
        -----------
        system_trajs : list
            List of loaded MDTraj trajectory objects
        names : list
            List of trajectory names
        concat : bool
            Whether to concatenate trajectories per system
        subdir_name : str
            Name of the system

        Returns:
        --------
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
        directory_path, concat, stride, trajectories, names, system_summary, selection=None
    ):
        """
        Load root directory files as separate system if present.

        Parameters:
        -----------
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

        Returns:
        --------
        None
            Loads root directory files as separate system if present
        """
        root_pdb_files = TrajectoryLoader._get_pdb_files_from_directory(directory_path)
        if root_pdb_files:
            root_system_name = os.path.basename(directory_path.rstrip("/\\"))
            root_result = TrajectoryLoader._load_system_trajectories(
                directory_path, root_system_name, concat, stride, selection
            )

            if root_result["trajectories"]:
                trajectories.extend(root_result["trajectories"])
                names.extend(root_result["names"])
                system_summary.append(
                    (root_system_name, len(root_result["trajectories"]))
                )

    @staticmethod
    def _apply_selection_to_result(result: Dict[str, List[Any]], selection: str, concat: bool) -> Dict[str, List[Any]]:
        """
        Apply atom selection to trajectories in result.

        Parameters:
        -----------
        result : dict
            Dictionary with 'trajectories' and 'names' keys
        selection : str
            MDTraj selection string to apply
        concat : bool
            Whether concatenation was used (for logging purposes)

        Returns:
        --------
        dict
            Dictionary with selected trajectories

        Notes:
        ------
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
