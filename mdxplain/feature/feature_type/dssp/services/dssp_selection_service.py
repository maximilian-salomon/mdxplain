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

"""Service for adding DSSP selections with DSSP-specific reduction methods."""
from __future__ import annotations
from typing import Optional, Union, List, TYPE_CHECKING

from ...interfaces.selection_service_base import SelectionServiceBase

if TYPE_CHECKING:
    from .....feature_selection.manager.feature_selector_manager import FeatureSelectorManager
    from .....pipeline.entities.pipeline_data import PipelineData


class DSSPSelectionService(SelectionServiceBase):
    """
    Service for selecting DSSP features with DSSP-specific reduction methods.

    Knows ALL reduction metrics from DSSPReduceService:
    
    - transitions, transition_frequency, stability, class_frequencies

    This service provides methods to add DSSP (secondary structure) feature
    selections with optional post-selection reduction. Each reduction method
    applies filtering ONLY to the specific selection where it's defined.

    Examples
    --------
    Basic selection without reduction:
    >>> service("test", "res ALA")

    With transitions reduction:
    >>> service.with_transitions_reduction("test", "res ALA", threshold_min=5)

    With stability reduction:
    >>> service.with_stability_reduction("test", "helix_region",
    ...     threshold_min=0.8)
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData):
        """
        Initialize DSSP selection service.

        Creates a service for adding DSSP (secondary structure) feature
        selections with optional post-selection reduction based on
        secondary structure-specific metrics.

        Parameters
        ----------
        manager : FeatureSelectorManager
            Manager instance for executing add operations
        pipeline_data : PipelineData
            Pipeline data container with trajectory and feature data

        Returns
        -------
        None
            Initializes service with manager and pipeline_data references

        Examples
        --------
        >>> from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
        >>> pipeline = PipelineManager()
        >>> service = pipeline.feature_selector.add.dssp
        >>> # Service is now ready to add DSSP selections
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "dssp"

    def __call__(
        self,
        selector_name: str,
        selection: str = "all",
        use_reduced: bool = False,
        common_denominator: bool = True,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        require_all_partners: bool = False,
    ) -> None:
        """
        Add DSSP selection without reduction.

        Adds DSSP (secondary structure) feature selection to the named
        selector without applying any statistical filtering. All DSSP
        features matching the selection criteria will be included.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string (e.g., "res ALA", "resid 123-140", "all")
        use_reduced : bool, default=False
            Whether to use reduced data (True) or original data (False)
        common_denominator : bool, default=True
            Whether to find common features across trajectories in traj_selection
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns
        -------
        None
            Adds DSSP selection to the named selector

        Examples
        --------
        >>> service("analysis", "res ALA")
        >>> service("analysis", "resid 120-140", use_reduced=True)
        """
        self._manager.add_selection(
            self._pipeline_data, selector_name, self._feature_type, selection,
            use_reduced, common_denominator, traj_selection, require_all_partners
        )

    def with_transitions_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False, window_size: int = 10, transition_mode: str = 'direct', lag_time: int = 1) -> None:
        """
        Add DSSP with transitions reduction.

        Filters DSSP features based on the number of secondary structure
        transitions. Higher values indicate more dynamic secondary structure.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        cross_trajectory : bool, default=True
            If True, only keep features that pass threshold in ALL trajectories
        use_reduced : bool, default=False
            Whether to use reduced data
        common_denominator : bool, default=True
            Whether to find common features across trajectories
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process
        require_all_partners : bool, default=False
            For pairwise features, require all partners present
        window_size : int, default=10
            Window size for transition detection
        transition_mode : str, default='direct'
            Mode for transition calculation ('direct' or 'window')
        lag_time : int, default=1
            Lag time for transition detection

        Returns
        -------
        None
            Adds DSSP selection with transitions reduction

        Examples
        --------
        >>> service.with_transitions_reduction("test", "res ALA", threshold_min=5)
        >>> service.with_transitions_reduction("test", "loop_regions",
        ...     threshold_min=10, window_size=20)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        extra_params = {"window_size": window_size, "transition_mode": transition_mode, "lag_time": lag_time}
        self._add_reduction_config(selector_name, "transitions", threshold_min, threshold_max, cross_trajectory, extra_params)

    def with_transition_frequency_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add DSSP with transition frequency reduction.

        Filters DSSP features based on the frequency of secondary structure
        transitions normalized by trajectory length. Higher values indicate
        residues with more frequent secondary structure changes.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum transition frequency threshold
        threshold_max : float, optional
            Maximum transition frequency threshold
        cross_trajectory : bool, default=True
            If True, only keep features that pass threshold in ALL trajectories
        use_reduced : bool, default=False
            Whether to use reduced data
        common_denominator : bool, default=True
            Whether to find common features across trajectories
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process
        require_all_partners : bool, default=False
            For pairwise features, require all partners present

        Returns
        -------
        None
            Adds DSSP selection with transition frequency reduction

        Examples
        --------
        >>> service.with_transition_frequency_reduction("test", "res ALA", threshold_min=0.1)
        >>> service.with_transition_frequency_reduction("test", "dynamic_loops", threshold_min=0.2)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "transition_frequency", threshold_min, threshold_max, cross_trajectory)

    def with_stability_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add DSSP with stability reduction.

        Filters DSSP features based on secondary structure stability
        (1 - transition frequency). Higher values indicate more stable secondary structure.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum stability fraction
        threshold_max : float, optional
            Maximum stability fraction
        cross_trajectory : bool, default=True
            If True, only keep features that pass threshold in ALL trajectories
        use_reduced : bool, default=False
            Whether to use reduced data
        common_denominator : bool, default=True
            Whether to find common features across trajectories
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process
        require_all_partners : bool, default=False
            For pairwise features, require all partners present

        Returns
        -------
        None
            Adds DSSP selection with stability reduction

        Examples
        --------
        >>> service.with_stability_reduction("test", "res ALA", threshold_min=0.7)
        >>> service.with_stability_reduction("test", "stable_helices",
        ...     threshold_min=0.9)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "stability", threshold_min, threshold_max, cross_trajectory)

    def with_class_frequencies_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False, target_classes: Optional[List[str]] = None) -> None:
        """
        Add DSSP with class frequencies reduction.

        Filters DSSP features based on the frequency of specific secondary
        structure classes. Takes the class with the highest frequency. 
        This allows selection of residues that spend certain amounts of 
        time in particular secondary structures.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum class frequency threshold
        threshold_max : float, optional
            Maximum class frequency threshold
        cross_trajectory : bool, default=True
            If True, only keep features that pass threshold in ALL trajectories
        use_reduced : bool, default=False
            Whether to use reduced data
        common_denominator : bool, default=True
            Whether to find common features across trajectories
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process
        require_all_partners : bool, default=False
            For pairwise features, require all partners present
        target_classes : list of str, optional
            List of secondary structure classes to analyze (e.g., ['H', 'E', 'C'])

        Returns
        -------
        None
            Adds DSSP selection with class frequencies reduction

        Examples
        --------
        >>> service.with_class_frequencies_reduction("test", "res ALA",
        ...     threshold_min=0.5, target_classes=['H'])
        >>> service.with_class_frequencies_reduction("test", "mixed_structure",
        ...     threshold_min=0.2, target_classes=['H', 'E', 'C'])
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        extra_params = {"target_classes": target_classes} if target_classes else {}
        self._add_reduction_config(selector_name, "class_frequencies", threshold_min, threshold_max, cross_trajectory, extra_params)

