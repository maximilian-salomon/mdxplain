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

"""Trajectory consistency checking utilities."""

from typing import List, Tuple

from ..entities.trajectory_data import TrajectoryData


class TrajectoryConsistencyChecker:
    """Utility class for checking trajectory consistency across multiple trajectories."""

    @staticmethod
    def check_trajectory_consistency(traj_data: TrajectoryData) -> None:
        """
        Check if all trajectories have consistent atom and residue counts.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        None
            Prints warnings if trajectory inconsistencies are found

        Examples:
        ---------
        >>> TrajectoryConsistencyChecker.check_trajectory_consistency(traj_data)
        """
        if not traj_data.trajectories or len(traj_data.trajectories) <= 1:
            return

        any_inconsistency = TrajectoryConsistencyChecker._check_all_inconsistencies(
            traj_data
        )
        if any_inconsistency:
            TrajectoryConsistencyChecker._print_consistency_suggestion()

    @staticmethod
    def _check_all_inconsistencies(traj_data: TrajectoryData) -> bool:
        """
        Check for any type of trajectory inconsistency.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        bool
            True if any inconsistency found, False if all consistent
        """
        residue_inconsistent = TrajectoryConsistencyChecker._find_residue_inconsistency(
            traj_data
        )
        if residue_inconsistent:
            return True

        atom_inconsistent = TrajectoryConsistencyChecker._find_atom_inconsistency(
            traj_data
        )
        return atom_inconsistent

    @staticmethod
    def _print_consistency_suggestion() -> None:
        """
        Print suggestion for handling trajectory inconsistencies.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Prints suggestion message to console
        """
        print(
            "\nSuggestion: Use select_atoms() with a selector that applies "
            "to all trajectories\nto find common ground for analysis, or use "
            "remove_trajectory() to exclude\nincompatible trajectories from "
            "the analysis.\n"
        )

    @staticmethod
    def _find_residue_inconsistency(traj_data: TrajectoryData) -> bool:
        """
        Check for residue count inconsistencies between trajectories.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        bool
            True if residue inconsistency found, False if all consistent
        """
        return TrajectoryConsistencyChecker._find_count_inconsistency(
            traj_data,
            count_attr="n_residues",
            count_type="residue",
            plural_type="residues",
        )

    @staticmethod
    def _find_atom_inconsistency(traj_data: TrajectoryData) -> bool:
        """
        Check for atom count inconsistencies between trajectories.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        bool
            True if atom inconsistency found, False if all consistent
        """
        return TrajectoryConsistencyChecker._find_count_inconsistency(
            traj_data, count_attr="n_atoms", count_type="atom", plural_type="atoms"
        )

    @staticmethod
    def _find_count_inconsistency(
        traj_data: TrajectoryData,
        count_attr: str,
        count_type: str,
        plural_type: str,
    ) -> bool:
        """
        Check for count inconsistencies between trajectories.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        count_attr : str
            Attribute name to check (e.g., 'n_residues', 'n_atoms')
        count_type : str
            Single form of count type (e.g., 'residue', 'atom')
        plural_type : str
            Plural form of count type (e.g., 'residues', 'atoms')

        Returns:
        --------
        bool
            True if inconsistency found, False if all consistent
        """
        if not traj_data.trajectories or len(traj_data.trajectories) <= 1:
            return False

        ref_count = getattr(traj_data.trajectories[0], count_attr)
        inconsistent = TrajectoryConsistencyChecker._collect_inconsistent_trajectories(
            traj_data, count_attr, ref_count
        )

        if inconsistent:
            TrajectoryConsistencyChecker._print_inconsistency_warning(
                traj_data, inconsistent, ref_count, count_type, plural_type
            )
            return True

        return False

    @staticmethod
    def _collect_inconsistent_trajectories(
        traj_data: TrajectoryData, count_attr: str, ref_count: int
    ) -> List[Tuple[int, str, int]]:
        """
        Collect trajectories with inconsistent counts.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        count_attr : str
            Attribute name to check
        ref_count : int
            Reference count to compare against

        Returns:
        --------
        list
            List of (index, name, count) tuples for inconsistent trajectories
        """
        inconsistent = []
        for i, traj in enumerate(traj_data.trajectories):
            count = getattr(traj, count_attr)
            if count != ref_count:
                inconsistent.append((i, traj_data.trajectory_names[i], count))
        return inconsistent

    @staticmethod
    def _print_inconsistency_warning(
        traj_data: TrajectoryData,
        inconsistent: List[Tuple[int, str, int]],
        ref_count: int,
        count_type: str,
        plural_type: str,
    ) -> None:
        """
        Print warning message for count inconsistencies.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        inconsistent : list
            List of (index, name, count) tuples for inconsistent trajectories
        ref_count : int
            Reference count from first trajectory
        count_type : str
            Single form of count type
        plural_type : str
            Plural form of count type

        Returns:
        --------
        None
            Prints warning message to console
        """
        inconsistent_list = "\n".join(
            [
                f"  [{idx}] {name}: {count} {plural_type}"
                for idx, name, count in inconsistent
            ]
        )
        print(
            f"\nWARNING: Inconsistent {count_type} counts detected!\n"
            f"Reference trajectory '{traj_data.trajectory_names[0]}' has "
            f"{ref_count} {plural_type}.\n"
            f"Trajectories with different {count_type} counts:\n"
            f"{inconsistent_list}\n\n"
            f"{count_type.capitalize()}-based feature calculations may fail or produce "
            f"incorrect results."
        )
