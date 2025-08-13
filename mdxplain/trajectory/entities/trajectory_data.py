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
Pure MD trajectory data container.

Container for MD trajectory objects with keyword annotation support.
Does not contain feature computations or analysis data - only trajectory
management and keyword metadata.
"""

from typing import Dict, List, Optional, Union

from ...utils.data_utils import DataUtils


class TrajectoryData:
    """
    Pure trajectory data container with keyword support.

    This class serves as a focused container for molecular dynamics trajectories
    and their associated keyword metadata. It provides trajectory management
    without feature computation or analysis dependencies.

    The class supports keyword annotation for trajectories, enabling advanced
    data selection and filtering capabilities through the DataPicker module.

    Examples:
    ---------
    Basic usage:

    >>> traj_data = TrajectoryData()
    >>> # Trajectories loaded via TrajectoryManager
    >>> print(f"Loaded {len(traj_data.trajectories)} trajectories")

    With keyword annotation:

    >>> traj_data.trajectory_keywords = {
    ...     0: ["system_A", "biased", "high_temp"],
    ...     1: ["system_A", "unbiased", "high_temp"]
    ... }
    >>> keywords = traj_data.get_trajectory_keywords(0)
    >>> print(keywords)  # ["system_A", "biased", "high_temp"]

    Attributes:
    -----------
    trajectories : list
        List of loaded MD trajectory objects (MDTraj)
    trajectory_names : list
        List of trajectory names corresponding to trajectories
    trajectory_keywords : dict
        Dictionary mapping trajectory indices/names to keyword lists
    frame_keyword_mapping : dict
        Dictionary mapping global frame indices to keyword lists
    res_label_data : dict or None
        Residue labeling data from nomenclature systems
    """

    def __init__(self):
        """
        Initialize trajectory data container.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Initializes empty trajectory data container

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> print(len(traj_data.trajectories))  # 0
        """
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_keywords: Dict[Union[int, str], List[str]] = {}
        self.frame_keyword_mapping: Dict[int, List[str]] = {}
        self.res_label_data = None

    def get_trajectory_keywords(
        self, trajectory_id: Union[int, str]
    ) -> Optional[List[str]]:
        """
        Get keywords for a specific trajectory.

        Parameters:
        -----------
        trajectory_id : int or str
            Trajectory index or name

        Returns:
        --------
        list or None
            List of keywords for the trajectory, or None if not found

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_data.trajectory_keywords = {0: ["system_A", "biased"]}
        >>> keywords = traj_data.get_trajectory_keywords(0)
        >>> print(keywords)  # ["system_A", "biased"]
        """
        return self.trajectory_keywords.get(trajectory_id)

    def set_trajectory_keywords(
        self, trajectory_id: Union[int, str], keywords: List[str]
    ) -> None:
        """
        Set keywords for a specific trajectory.

        Parameters:
        -----------
        trajectory_id : int or str
            Trajectory index or name
        keywords : list
            List of keyword strings

        Returns:
        --------
        None
            Sets keywords for the specified trajectory

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_data.set_trajectory_keywords(0, ["system_A", "biased"])
        >>> traj_data.set_trajectory_keywords("traj1", ["system_B", "unbiased"])
        """
        self.trajectory_keywords[trajectory_id] = keywords

    def get_frame_keywords(self, frame_index: int) -> Optional[List[str]]:
        """
        Get keywords for a specific global frame index.

        Parameters:
        -----------
        frame_index : int
            Global frame index across all trajectories

        Returns:
        --------
        list or None
            List of keywords for the frame, or None if not found

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> # After frame mapping is built by TrajectoryManager
        >>> keywords = traj_data.get_frame_keywords(100)
        >>> if keywords:
        ...     print(f"Frame 100 has keywords: {keywords}")
        """
        return self.frame_keyword_mapping.get(frame_index)

    def get_trajectory_names(self) -> List[str]:
        """
        Get list of trajectory names.

        Returns:
        --------
        list
            List of trajectory names

        Examples:
        ---------
        >>> names = traj_data.get_trajectory_names()
        >>> print(names)
        ['system1_prot_traj1', 'system1_prot_traj2']
        """
        return self.trajectory_names

    def print_trajectory_info(self) -> None:
        """
        Print information about loaded trajectories.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Prints trajectory information to console

        Examples:
        ---------
        >>> traj_data.print_trajectory_info()
        Loaded 3 trajectories:
          [0] system1_prot_traj1: 1000 frames, keywords: ['system_A', 'biased']
          [1] system1_prot_traj2: 1500 frames, keywords: ['system_A', 'unbiased']
          [2] system2_prot_traj1: 800 frames, keywords: ['system_B', 'biased']
        """
        if not self.trajectories or not self.trajectory_names:
            print("No trajectories loaded.")
            return

        print(f"Loaded {len(self.trajectories)} trajectories:")
        for i, (traj, name) in enumerate(zip(self.trajectories, self.trajectory_names)):
            keywords = self.get_trajectory_keywords(i) or self.get_trajectory_keywords(
                name
            )
            keyword_str = f", keywords: {keywords}" if keywords else ""
            print(f"  [{i}] {name}: {traj.n_frames} frames{keyword_str}")

    def save(self, save_path: str) -> None:
        """
        Save the TrajectoryData object to disk.

        Parameters:
        -----------
        save_path : str
            Path where to save the TrajectoryData object

        Returns:
        --------
        None
            Saves the TrajectoryData object to the specified path

        Examples:
        ---------
        >>> traj_data.save('trajectory_data.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load a previously saved TrajectoryData object from disk.

        Parameters:
        -----------
        load_path : str
            Path to the saved TrajectoryData file

        Returns:
        --------
        None
            Loads the TrajectoryData object from the specified path

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_data.load('trajectory_data.pkl')
        """
        DataUtils.load_object(self, load_path)

    def reset(self) -> None:
        """
        Reset the trajectory data object to empty state.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Resets all trajectory data and keywords

        Examples:
        ---------
        >>> traj_data.reset()
        >>> print(len(traj_data.trajectories))  # 0
        """
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_keywords = {}
        self.frame_keyword_mapping = {}
        self.res_label_data = None
