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
Service for adding contacts selections with contacts-specific reduction methods.

This module provides the ContactsAddService class that offers methods for
adding contact feature selections with optional post-selection reduction
based on contact statistics from the ContactsReduceService.
"""
from __future__ import annotations
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .....feature_selection.managers.feature_selector_manager import FeatureSelectorManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ...interfaces.selection_service_base import SelectionServiceBase


class ContactsSelectionService(SelectionServiceBase):
    """
    Service for selecting contacts features with contacts-specific reduction methods.

    Knows ALL reduction metrics from ContactsReduceService:
    
    - frequency, stability, transitions

    This service provides methods to add contact feature selections with
    optional post-selection reduction. Each reduction method applies filtering
    ONLY to the specific selection where it's defined.

    Examples
    --------
    Basic selection without reduction:
    >>> service("test", "resid 120-140")

    With frequency reduction:
    >>> service.with_frequency_reduction("test", "resid 120-140", threshold_min=0.3)

    With transitions reduction and custom parameters:
    >>> service.with_transitions_reduction("test", "binding_site",
    ...     threshold_min=10, transition_threshold=0.5, window_size=20)
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData):
        """
        Initialize contacts selection service.

        Creates a service for adding contact feature selections with optional
        post-selection reduction based on contact-specific metrics.

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
        >>> from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
        >>> pipeline = PipelineManager()
        >>> service = pipeline.feature_selector.add.contacts
        >>> # Service is now ready to add contact selections
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "contacts"

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
        Add contacts selection without reduction.

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
            Adds contacts selection to the named selector

        Examples
        --------
        >>> service("analysis", "resid 120-140")
        >>> service("analysis", "binding_site", use_reduced=True)
        """
        self._manager.add_selection(
            self._pipeline_data,
            selector_name,
            self._feature_type,
            selection,
            use_reduced,
            common_denominator,
            traj_selection,
            require_all_partners
        )

    def with_frequency_reduction(
        self,
        selector_name: str,
        selection: str = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = True,
        use_reduced: bool = False,
        common_denominator: bool = True,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        require_all_partners: bool = False,
    ) -> None:
        """
        Add contacts with frequency reduction.

        Filters contact features based on contact frequency (fraction of frames
        where contact is formed). Features with frequency outside the specified
        thresholds are removed.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum frequency threshold (0.0-1.0)
        threshold_max : float, optional
            Maximum frequency threshold (0.0-1.0)
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
            Adds contacts selection with frequency reduction

        Examples
        --------
        >>> service.with_frequency_reduction("test", "resid 120-140", threshold_min=0.3)
        >>> service.with_frequency_reduction("test", "stable_contacts", threshold_min=0.8, threshold_max=1.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "frequency", threshold_min, threshold_max, cross_trajectory
        )

    def with_stability_reduction(
        self,
        selector_name: str,
        selection: str = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = True,
        use_reduced: bool = False,
        common_denominator: bool = True,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        require_all_partners: bool = False,
    ) -> None:
        """
        Add contacts with stability reduction.

        Filters contact features based on contact stability (measure of how
        consistently a contact is maintained over time). Features with stability
        outside the specified thresholds are removed.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum stability threshold
        threshold_max : float, optional
            Maximum stability threshold
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
            Adds contacts selection with stability reduction

        Examples
        --------
        >>> service.with_stability_reduction("test", "resid 120-140", threshold_min=0.7)
        >>> service.with_stability_reduction("test", "dynamic_contacts", threshold_max=0.3)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "stability", threshold_min, threshold_max, cross_trajectory
        )

    def with_transitions_reduction(
        self,
        selector_name: str,
        selection: str = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 0.5,
        window_size: int = 10,
        transition_mode: str = 'window',
        lag_time: int = 1,
        cross_trajectory: bool = True,
        use_reduced: bool = False,
        common_denominator: bool = True,
        traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all",
        require_all_partners: bool = False,
    ) -> None:
        """
        Add contacts with transitions reduction.

        Filters contact features based on transition frequency or counts (how
        often contacts form/break). Features with transitions outside the
        specified thresholds are removed.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum transitions threshold
        threshold_max : float, optional
            Maximum transitions threshold
        transition_threshold : float, default=0.5
            Contact value threshold to count as transition (0.0-1.0)
        window_size : int, default=10
            Size of sliding window for transition detection
        transition_mode : str, default='window'
            Mode for transition detection ('window', 'direct', 'cumulative')
        lag_time : int, default=1
            Lag time for transition detection (frames)
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
            Adds contacts selection with transitions reduction

        Examples
        --------
        >>> service.with_transitions_reduction("test", "resid 120-140", threshold_min=5)
        >>> service.with_transitions_reduction("test", "dynamic_binding",
        ...     threshold_min=10, transition_threshold=0.8, window_size=20)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)

        extra_params = {
            "transition_threshold": transition_threshold,
            "window_size": window_size,
            "transition_mode": transition_mode,
            "lag_time": lag_time
        }
        self._add_reduction_config(
            selector_name, "transitions", threshold_min, threshold_max,
            cross_trajectory, extra_params
        )

