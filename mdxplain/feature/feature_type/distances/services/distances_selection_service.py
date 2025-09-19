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
Service for adding distances selections with distances-specific reduction methods.

This module provides the DistancesAddService class that offers methods for
adding distance feature selections with optional post-selection reduction
based on statistical metrics from the DistancesReduceService.
"""
from __future__ import annotations
from typing import Optional, Union, List, TYPE_CHECKING

from ...interfaces.selection_service_base import SelectionServiceBase

if TYPE_CHECKING:
    from .....feature_selection.managers.feature_selector_manager import FeatureSelectorManager
    from .....pipeline.entities.pipeline_data import PipelineData


class DistancesSelectionService(SelectionServiceBase):
    """
    Service for selecting distances features with distances-specific reduction methods.

    Knows ALL reduction metrics from DistancesReduceService:
    - cv, std, variance, range, transitions, min, mad, mean, max

    This service provides methods to add distance feature selections with
    optional post-selection reduction. Each reduction method applies filtering
    ONLY to the specific selection where it's defined.

    Examples:
    ---------
    Basic selection without reduction:
    >>> service("test", "res ALA")

    With coefficient of variation reduction:
    >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.1)

    With transitions reduction and custom parameters:
    >>> service.with_transitions_reduction("test", "res ALA",
    ...     threshold_min=5, transition_threshold=2.0, window_size=10)
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData):
        """
        Initialize distances selection service.

        Creates a service for adding distance feature selections with optional
        post-selection reduction based on statistical metrics.

        Parameters:
        -----------
        manager : FeatureSelectorManager
            Manager instance for executing add operations
        pipeline_data : PipelineData
            Pipeline data container with trajectory and feature data

        Returns:
        --------
        None
            Initializes service with manager and pipeline_data references

        Examples:
        ---------
        >>> from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
        >>> pipeline = PipelineManager()
        >>> service = pipeline.feature_selector.add.distances
        >>> # Service is now ready to add distance selections
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "distances"

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
        Add distances selection without reduction.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string (e.g., "res ALA", "resid 123-140", "7x50-8x50", "all")
        use_reduced : bool, default=False
            Whether to use reduced data (True) or original data (False)
        common_denominator : bool, default=True
            Whether to find common features across trajectories in traj_selection
        traj_selection : int, str, list, or "all", default="all"
            Selection of trajectories to process
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns:
        --------
        None
            Adds distances selection to the named selector

        Examples:
        ---------
        >>> service("analysis", "res ALA")
        >>> service("analysis", "resid 120-140", use_reduced=True)
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

    def with_cv_reduction(
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
        Add distances with CV (coefficient of variation) reduction.

        Filters distance features based on coefficient of variation (std/mean).
        Features with CV outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum CV threshold (features with CV below this are removed)
        threshold_max : float, optional
            Maximum CV threshold (features with CV above this are removed)
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

        Returns:
        --------
        None
            Adds distances selection with CV reduction

        Examples:
        ---------
        >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.1)
        >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.05, threshold_max=0.8)
        """
        # 1. Add selection
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)

        # 2. Add reduction config to LAST selection
        self._add_reduction_config(
            selector_name, "cv", threshold_min, threshold_max, cross_trajectory
        )

    def with_std_reduction(
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
        Add distances with standard deviation reduction.

        Filters distance features based on standard deviation.
        Features with standard deviation outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum standard deviation threshold
        threshold_max : float, optional
            Maximum standard deviation threshold
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

        Returns:
        --------
        None
            Adds distances selection with standard deviation reduction

        Examples:
        ---------
        >>> service.with_std_reduction("test", "res ALA", threshold_min=0.5)
        >>> service.with_std_reduction("test", "backbone", threshold_max=2.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "std", threshold_min, threshold_max, cross_trajectory
        )

    def with_variance_reduction(
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
        Add distances with variance reduction.

        Filters distance features based on variance.
        Features with variance outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum variance threshold
        threshold_max : float, optional
            Maximum variance threshold
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

        Returns:
        --------
        None
            Adds distances selection with variance reduction

        Examples:
        ---------
        >>> service.with_variance_reduction("test", "res ALA", threshold_min=0.1)
        >>> service.with_variance_reduction("test", "sidechain", threshold_max=5.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "variance", threshold_min, threshold_max, cross_trajectory
        )

    def with_range_reduction(
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
        Add distances with range reduction.

        Filters distance features based on range (max - min).
        Features with range outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum range threshold
        threshold_max : float, optional
            Maximum range threshold
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

        Returns:
        --------
        None
            Adds distances selection with range reduction

        Examples:
        ---------
        >>> service.with_range_reduction("test", "res ALA", threshold_min=1.0)
        >>> service.with_range_reduction("test", "flexible_loops", threshold_max=10.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "range", threshold_min, threshold_max, cross_trajectory
        )

    def with_transitions_reduction(
        self,
        selector_name: str,
        selection: str = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 2.0,
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
        Add distances with transitions reduction.

        Filters distance features based on transition frequency or counts.
        Features with transitions outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum transitions threshold
        threshold_max : float, optional
            Maximum transitions threshold
        transition_threshold : float, default=2.0
            Distance change threshold to count as transition (Angstroms)
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

        Returns:
        --------
        None
            Adds distances selection with transitions reduction

        Examples:
        ---------
        >>> service.with_transitions_reduction("test", "res ALA", threshold_min=5)
        >>> service.with_transitions_reduction("test", "binding_site",
        ...     threshold_min=10, transition_threshold=1.5, window_size=20)
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

    def with_min_reduction(
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
        Add distances with minimum value reduction.

        Filters distance features based on minimum values.
        Features with minimum values outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum value threshold for minimum distances
        threshold_max : float, optional
            Maximum value threshold for minimum distances
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

        Returns:
        --------
        None
            Adds distances selection with minimum value reduction

        Examples:
        ---------
        >>> service.with_min_reduction("test", "res ALA", threshold_min=2.0)
        >>> service.with_min_reduction("test", "close_contacts", threshold_max=5.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "min", threshold_min, threshold_max, cross_trajectory
        )

    def with_mad_reduction(
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
        Add distances with MAD (median absolute deviation) reduction.

        Filters distance features based on median absolute deviation.
        Features with MAD outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum MAD threshold
        threshold_max : float, optional
            Maximum MAD threshold
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

        Returns:
        --------
        None
            Adds distances selection with MAD reduction

        Examples:
        ---------
        >>> service.with_mad_reduction("test", "res ALA", threshold_min=0.3)
        >>> service.with_mad_reduction("test", "dynamic_region", threshold_max=1.5)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "mad", threshold_min, threshold_max, cross_trajectory
        )

    def with_mean_reduction(
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
        Add distances with mean value reduction.

        Filters distance features based on mean values.
        Features with mean values outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum mean value threshold
        threshold_max : float, optional
            Maximum mean value threshold
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

        Returns:
        --------
        None
            Adds distances selection with mean value reduction

        Examples:
        ---------
        >>> service.with_mean_reduction("test", "res ALA", threshold_min=5.0)
        >>> service.with_mean_reduction("test", "long_range", threshold_max=20.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "mean", threshold_min, threshold_max, cross_trajectory
        )

    def with_max_reduction(
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
        Add distances with maximum value reduction.

        Filters distance features based on maximum values.
        Features with maximum values outside the specified thresholds are removed.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum value threshold for maximum distances
        threshold_max : float, optional
            Maximum value threshold for maximum distances
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

        Returns:
        --------
        None
            Adds distances selection with maximum value reduction

        Examples:
        ---------
        >>> service.with_max_reduction("test", "res ALA", threshold_min=10.0)
        >>> service.with_max_reduction("test", "constrained_bonds", threshold_max=15.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "max", threshold_min, threshold_max, cross_trajectory
        )

