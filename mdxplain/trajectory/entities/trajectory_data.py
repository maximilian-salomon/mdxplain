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
Pure MD trajectory data container.

Container for MD trajectory objects with tag annotation support.
Does not contain feature computations or analysis data - only trajectory
management and tag metadata.
"""

from typing import Dict, List, Optional, Union

from ...utils.data_utils import DataUtils
from ..helper.process_helper.selection_resolve_helper import SelectionResolveHelper


class TrajectoryData:
    """
    Pure trajectory data container with tag support.

    This class serves as a focused container for molecular dynamics trajectories
    and their associated tag metadata. It provides trajectory management
    without feature computation or analysis dependencies.

    The class supports tag annotation for trajectories, enabling advanced
    data selection and filtering capabilities through the DataPicker module.

    Examples
    --------
    Basic usage:

    >>> traj_data = TrajectoryData()
    >>> # Trajectories loaded via TrajectoryManager
    >>> print(f"Loaded {len(traj_data.trajectories)} trajectories")

    With tag annotation:

    >>> traj_data.trajectory_tags = {
    ...     0: ["system_A", "biased", "high_temp"],
    ...     1: ["system_A", "unbiased", "high_temp"]
    ... }
    >>> tags = traj_data.get_trajectory_tags(0)
    >>> print(tags)  # ["system_A", "biased", "high_temp"]

    Attributes
    ----------
    trajectories : list
        List of loaded MD trajectory objects (MDTraj)
    trajectory_names : list
        List of trajectory names corresponding to trajectories
    trajectory_tags : dict
        Dictionary mapping trajectory indices/names to tag lists
    res_label_data : dict or None
        Residue labeling data from nomenclature systems
    """

    def __init__(self) -> None:
        """
        Initialize trajectory data container.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Initializes empty trajectory data container

        Examples
        --------
        >>> traj_data = TrajectoryData()
        >>> print(len(traj_data.trajectories))  # 0
        """
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_tags: Dict[Union[int, str], List[str]] = {}
        self.res_label_data: Dict[int, dict] = {}

    def get_trajectory_tags(
        self, trajectory_id: Union[int, str]
    ) -> Optional[List[str]]:
        """
        Get tags for a specific trajectory.

        Parameters
        ----------
        trajectory_id : int or str
            Trajectory index or name

        Returns
        -------
        list or None
            List of tags for the trajectory, or None if not found

        Examples
        --------
        >>> traj_data = TrajectoryData()
        >>> traj_data.trajectory_tags = {0: ["system_A", "biased"]}
        >>> tags = traj_data.get_trajectory_tags(0)
        >>> print(tags)  # ["system_A", "biased"]
        """
        return self.trajectory_tags.get(trajectory_id)

    def set_trajectory_tags(
        self, trajectory_id: Union[int, str], tags: List[str]
    ) -> None:
        """
        Set tags for a specific trajectory.

        Parameters
        ----------
        trajectory_id : int or str
            Trajectory index or name
        tags : list
            List of tag strings

        Returns
        -------
        None
            Sets tags for the specified trajectory

        Examples
        --------
        >>> traj_data = TrajectoryData()
        >>> traj_data.set_trajectory_tags(0, ["system_A", "biased"])
        >>> traj_data.set_trajectory_tags("traj1", ["system_B", "unbiased"])
        """
        self.trajectory_tags[trajectory_id] = tags
    
    @property
    def n_frames_total(self) -> int:
        """
        Get total number of frames across all trajectories.
        
        Returns
        -------
        int
            Total number of frames
        """
        return sum(len(traj) for traj in self.trajectories)

    def get_trajectory_names(self) -> List[str]:
        """
        Get list of trajectory names.

        Returns
        -------
        list
            List of trajectory names

        Examples
        --------
        >>> names = traj_data.get_trajectory_names()
        >>> print(names)
        ['system1_prot_traj1', 'system1_prot_traj2']
        """
        return self.trajectory_names

    def print_info(self) -> None:
        """
        Print information about loaded trajectories.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints trajectory information to console

        Examples
        --------
        >>> traj_data.print_info()
        Loaded 3 trajectories:
          [0] system1_prot_traj1: 1000 frames, tags: ['system_A', 'biased']
          [1] system1_prot_traj2: 1500 frames, tags: ['system_A', 'unbiased']
          [2] system2_prot_traj1: 800 frames, tags: ['system_B', 'biased']
        """
        if self._has_no_trajectories():
            print("No trajectories loaded.")
            return
        
        self._print_trajectory_header()
        self._print_individual_trajectory_info()
    
    def _has_no_trajectories(self) -> bool:
        """
        Check if no trajectories are loaded.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bool
            True if no trajectories are loaded, False otherwise
        """
        return not self.trajectories or not self.trajectory_names
    
    def _print_trajectory_header(self) -> None:
        """
        Print header with trajectory count.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Prints header to console
        """
        print(f"Loaded {len(self.trajectories)} trajectories:")
    
    def _print_individual_trajectory_info(self) -> None:
        """
        Print information for each individual trajectory.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Prints trajectory information to console
        """
        for i, (traj, name) in enumerate(zip(self.trajectories, self.trajectory_names)):
            tags = self._get_trajectory_tags_for_display(i, name)
            tag_str = self._format_tags_string(tags)
            print(f"  [{i}] {name}: {traj.n_frames} frames{tag_str}")
    
    def _get_trajectory_tags_for_display(self, index: int, name: str) -> Optional[List[str]]:
        """
        Get trajectory tags for display, trying index first then name.
        
        Parameters
        ----------
        index : int
            Trajectory index to lookup
        name : str
            Trajectory name as fallback lookup
        
        Returns
        -------
        Optional[List[str]]
            List of tags if found, None if no tags exist for the trajectory
        """
        return self.get_trajectory_tags(index) or self.get_trajectory_tags(name)
    
    def _format_tags_string(self, tags: Optional[List[str]]) -> str:
        """
        Format tags for display string.
        
        Parameters
        ----------
        tags : Optional[List[str]]
            List of tags to format, or None
        
        Returns
        -------
        str
            Formatted string with tags or empty string if no tags
        """
        return f", tags: {tags}" if tags else ""

    def save(self, save_path: str) -> None:
        """
        Save the TrajectoryData object to disk.

        Parameters
        ----------
        save_path : str
            Path where to save the TrajectoryData object

        Returns
        -------
        None
            Saves the TrajectoryData object to the specified path

        Examples
        --------
        >>> traj_data.save('trajectory_data.pkl')
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path: str) -> None:
        """
        Load a previously saved TrajectoryData object from disk.

        Parameters
        ----------
        load_path : str
            Path to the saved TrajectoryData file

        Returns
        -------
        None
            Loads the TrajectoryData object from the specified path

        Examples
        --------
        >>> traj_data = TrajectoryData()
        >>> traj_data.load('trajectory_data.pkl')
        """
        DataUtils.load_object(self, load_path)

    def reset(self) -> None:
        """
        Reset the trajectory data object to empty state.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Resets all trajectory data and tags

        Examples
        --------
        >>> traj_data.reset()
        >>> print(len(traj_data.trajectories))  # 0
        """
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_tags = {}
        self.res_label_data: Dict[int, dict] = {}

    def get_trajectory_indices(
        self,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all"
    ) -> List[int]:
        """
        Get trajectory indices based on selection criteria.

        Central method for trajectory selection supporting indices, names,
        tags, and combinations thereof. Uses SelectionResolveHelper for
        consistent trajectory resolution across all modules.

        Parameters
        ----------
        traj_selection : int, str, list, or "all"
            Selection criteria:
            
            - int: trajectory index
            - str: trajectory name, tag (prefixed with "tag:") or advanced formats:
                
                * Range: "0-3", "id 0-3" → [0, 1, 2, 3]
                * Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
                * Single number: "7", "id 7" → [7]
                * Pattern: "system_*" → fnmatch pattern matching
            - list: mix of indices/names/tags/patterns
            - "all": all trajectories

        Returns
        -------
        List[int]
            List of trajectory indices matching the selection criteria

        Examples
        --------
        >>> # All trajectories
        >>> indices = traj_data.get_trajectory_indices("all")

        >>> # Single trajectory by index
        >>> indices = traj_data.get_trajectory_indices(0)

        >>> # Range format
        >>> indices = traj_data.get_trajectory_indices("0-3")  # [0, 1, 2, 3]

        >>> # Comma list format
        >>> indices = traj_data.get_trajectory_indices("1,2,4,5")  # [1, 2, 4, 5]

        >>> # Pattern matching
        >>> indices = traj_data.get_trajectory_indices("system_*")

        >>> # By tag
        >>> indices = traj_data.get_trajectory_indices("tag:system_A")

        >>> # Mixed selection
        >>> indices = traj_data.get_trajectory_indices([0, "traj1", "tag:biased"])
        """
        return SelectionResolveHelper.get_indices_to_process(self, traj_selection)
