# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
# trajectory_loader - MD Trajectory Loading Strategies
#
# Handles the loading methodology for MD trajectories.
# This class can be easily replaced or extended for different loading strategies.
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

import os
import warnings

import mdtraj as md
from tqdm import tqdm


class TrajectoryLoader:
    """
    Handles the loading methodology for MD trajectories.
    This class can be easily replaced or extended for different loading strategies.
    """

    @staticmethod
    def load_trajectories(data_input, concat=False, stride=1):
        """
        Load trajectories from various input types.

        Parameters:
        -----------
        data_input : list or str
            List of trajectory objects or path to directory
        concat : bool, default=False
            Whether to concatenate trajectories per system
        stride : int, default=1
            Load every stride-th frame from trajectories

        Returns:
        --------
        list
            List of loaded trajectories
        """
        if isinstance(data_input, list):
            return TrajectoryLoader._load_from_list(data_input, concat, stride)

        if isinstance(data_input, str) and os.path.exists(data_input):
            return TrajectoryLoader._load_from_directory(data_input, concat, stride)

        warnings.warn(
            f"Invalid data input: {data_input}. Expected list of trajectories or valid path."
        )
        return []

    @staticmethod
    def _load_from_list(trajectory_list, concat):
        """Load from provided trajectory list."""
        if concat and len(trajectory_list) > 1:
            print(f"Concatenating {len(trajectory_list)} provided trajectories...")
            concatenated = trajectory_list[0]
            for traj in tqdm(trajectory_list[1:], desc="Concatenating"):
                concatenated = concatenated.join(traj)
            print(f"Result: 1 concatenated trajectory with {concatenated.n_frames} frames")
            return [concatenated]
        return trajectory_list

    @staticmethod
    def _load_from_directory(directory_path, concat, stride):
        """Load trajectories from directory structure."""
        # Determine directory structure and collect trajectories
        if TrajectoryLoader._has_subdirectories(directory_path):
            return TrajectoryLoader._load_nested_structure(directory_path, concat, stride)
        return TrajectoryLoader._load_flat_structure(directory_path, concat, stride)

    @staticmethod
    def _has_subdirectories(directory_path):
        """Check if directory contains subdirectories."""
        subdirs = [
            d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))
        ]
        return len(subdirs) > 0

    @staticmethod
    def _load_nested_structure(directory_path, concat, stride):
        """Load trajectories from nested directory structure."""
        trajectories = []
        system_summary = []
        subdirs = [
            d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))
        ]

        for subdir in tqdm(subdirs, desc="Loading systems"):
            subdir_path = os.path.join(directory_path, subdir)
            system_trajs = TrajectoryLoader._load_system_trajectories(
                subdir_path, subdir, concat, stride
            )

            if system_trajs:
                trajectories.extend(system_trajs)
                system_summary.append((subdir, len(system_trajs)))

        TrajectoryLoader._print_loading_summary(system_summary, concat)
        return trajectories

    @staticmethod
    def _load_flat_structure(directory_path, concat, stride):
        """Load trajectories from flat directory structure."""
        trajectories = []
        system_name = os.path.basename(directory_path)
        system_trajs = TrajectoryLoader._load_system_trajectories(
            directory_path, system_name, concat, stride
        )

        if system_trajs:
            trajectories.extend(system_trajs)
            system_summary = [(system_name, len(system_trajs))]
            TrajectoryLoader._print_loading_summary(system_summary, concat)

        return trajectories

    @staticmethod
    def _print_loading_summary(system_summary, concat):
        """Print summary of loaded trajectories."""
        total_trajectories = sum(count for _, count in system_summary)

        if concat:
            print(
                f"\nLoaded {len(system_summary)} systems with "
                f"{total_trajectories} concatenated trajectories:"
            )
        else:
            print(
                f"\nLoaded {len(system_summary)} systems with "
                f"{total_trajectories} total trajectories:"
            )

        for system, count in system_summary:
            if concat and count > 1:
                print(f"  {system}: 1 concatenated trajectory ({count} files)")
            else:
                print(f"  {system}: {count} trajectories")

    @staticmethod
    def _load_system_trajectories(subdir_path, subdir_name, concat, stride):
        """Load all trajectories for a single system."""
        pdb_path = TrajectoryLoader._validate_and_get_pdb_path_from_directory(subdir_path)
        if pdb_path is None:
            return []

        xtc_files = TrajectoryLoader._get_xtc_files_from_directory(subdir_path)
        if not xtc_files:
            return []

        system_trajs = TrajectoryLoader._load_trajectory_files_from_directory(
            xtc_files, pdb_path, subdir_path, subdir_name, stride)

        return TrajectoryLoader._handle_traj_concatenation(system_trajs, concat, subdir_name)

    @staticmethod
    def _validate_and_get_pdb_path_from_directory(subdir_path):
        """Validate PDB files and return path to the single PDB file."""
        pdb_files = [f for f in os.listdir(subdir_path) if f.endswith(".pdb")]

        if len(pdb_files) > 1:
            raise ValueError(f"More than one PDB file found in {subdir_path}: {pdb_files}")

        if not pdb_files:
            return None

        return os.path.join(subdir_path, pdb_files[0])

    @staticmethod
    def _get_xtc_files_from_directory(subdir_path):
        """Get list of XTC files in directory."""
        return [f for f in os.listdir(subdir_path) if f.endswith(".xtc")]

    @staticmethod
    def _load_trajectory_files_from_directory(
            xtc_files, pdb_path, subdir_path, subdir_name, stride):
        """Load all XTC files with the given PDB topology."""
        system_trajs = []
        for xtc in tqdm(xtc_files, desc=f"Loading {subdir_name}", leave=False):
            xtc_path = os.path.join(subdir_path, xtc)
            traj = md.load(xtc_path, top=pdb_path, stride=stride)
            system_trajs.append(traj)
        return system_trajs

    @staticmethod
    def _handle_traj_concatenation(system_trajs, concat, subdir_name):
        """Handle trajectory concatenation if requested."""
        if concat and len(system_trajs) > 1:
            print(f"  Concatenating {len(system_trajs)} trajectories for {subdir_name}...")
            concatenated = system_trajs[0]
            for traj in system_trajs[1:]:
                concatenated = concatenated.join(traj)
            return [concatenated]
        return system_trajs
