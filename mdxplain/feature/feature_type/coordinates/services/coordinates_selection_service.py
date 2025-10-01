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
Service for adding coordinates selections with coordinates-specific reduction methods.

This module provides the CoordinatesAddService class that offers methods for
adding coordinate feature selections with optional post-selection reduction
based on structural flexibility metrics.
"""
from __future__ import annotations
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .....feature_selection.managers.feature_selector_manager import FeatureSelectorManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ...interfaces.selection_service_base import SelectionServiceBase


class CoordinatesSelectionService(SelectionServiceBase):
    """
    Service for selecting coordinates features with coordinates-specific reduction methods.

    Knows ALL reduction metrics from CoordinatesReduceService:
    - std, rmsf, cv, variance, range, mad, mean, min, max

    This service provides methods to add coordinate feature selections with
    optional post-selection reduction based on structural flexibility metrics.
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData):
        """
        Initialize coordinates selection service.

        Creates a service for adding coordinate feature selections with optional
        post-selection reduction based on structural flexibility metrics.

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
        >>> service = pipeline.feature_selector.add.coordinates
        >>> # Service is now ready to add coordinate selections
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "coordinates"

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
        Add coordinates selection without reduction.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
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
            Adds coordinates selection to the named selector
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

    def with_rmsf_reduction(
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
        Add coordinates with RMSF (Root Mean Square Fluctuation) reduction.

        Filters coordinate features based on atomic positional fluctuations.
        Higher RMSF values indicate more mobile atoms/residues.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum RMSF threshold (Ångströms)
        threshold_max : float, optional
            Maximum RMSF threshold (Ångströms)
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
            Adds coordinates selection with RMSF reduction

        Examples
        --------
        >>> service.with_rmsf_reduction("test", "res ALA", threshold_min=2.0)
        >>> service.with_rmsf_reduction("test", "mobile_loops", threshold_min=3.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "rmsf", threshold_min, threshold_max, cross_trajectory
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
        Add coordinates with standard deviation reduction.

        Filters coordinate features based on standard deviation of atomic
        positions. Higher values indicate more mobile atoms/residues.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum standard deviation threshold (Ångströms)
        threshold_max : float, optional
            Maximum standard deviation threshold (Ångströms)
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
            Adds coordinates selection with standard deviation reduction

        Examples
        --------
        >>> service.with_std_reduction("test", "res ALA", threshold_min=1.5)
        >>> service.with_std_reduction("test", "flexible_regions", threshold_min=2.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "std", threshold_min, threshold_max, cross_trajectory
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
        Add coordinates with CV (coefficient of variation) reduction.

        Filters coordinate features based on coefficient of variation (std/mean)
        of atomic positions. Higher CV values indicate more variable atoms/residues
        relative to their mean position.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum CV threshold
        threshold_max : float, optional
            Maximum CV threshold
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
            Adds coordinates selection with CV reduction

        Examples
        --------
        >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.1)
        >>> service.with_cv_reduction("test", "variable_loops", threshold_min=0.2)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "cv", threshold_min, threshold_max, cross_trajectory
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
        Add coordinates with variance reduction.

        Filters coordinate features based on variance of atomic positions.
        Higher variance values indicate more mobile atoms/residues with
        broader positional distributions.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum variance threshold (Ångströms²)
        threshold_max : float, optional
            Maximum variance threshold (Ångströms²)
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
            Adds coordinates selection with variance reduction

        Examples
        --------
        >>> service.with_variance_reduction("test", "res ALA", threshold_min=2.25)
        >>> service.with_variance_reduction("test", "flexible_regions", threshold_min=4.0)
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
        Add coordinates with range reduction.

        Filters coordinate features based on range (max - min) of atomic
        positions. Higher range values indicate atoms with larger amplitude
        of motion.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum range threshold (Ångströms)
        threshold_max : float, optional
            Maximum range threshold (Ångströms)
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
            Adds coordinates selection with range reduction

        Examples
        --------
        >>> service.with_range_reduction("test", "res ALA", threshold_min=3.0)
        >>> service.with_range_reduction("test", "mobile_loops", threshold_min=5.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "range", threshold_min, threshold_max, cross_trajectory
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
        Add coordinates with MAD (median absolute deviation) reduction.

        Filters coordinate features based on median absolute deviation of
        atomic positions. MAD is more robust to outliers than standard
        deviation while still indicating positional variability.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum MAD threshold (Ångströms)
        threshold_max : float, optional
            Maximum MAD threshold (Ångströms)
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
            Adds coordinates selection with MAD reduction

        Examples
        --------
        >>> service.with_mad_reduction("test", "res ALA", threshold_min=1.0)
        >>> service.with_mad_reduction("test", "robust_flexible", threshold_min=1.5)
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
        Add coordinates with mean reduction.

        Filters coordinate features based on mean position values.
        This can be used to select atoms based on their average
        position along specific coordinate axes.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum mean threshold (Ångströms)
        threshold_max : float, optional
            Maximum mean threshold (Ångströms)
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
            Adds coordinates selection with mean reduction

        Examples
        --------
        >>> service.with_mean_reduction("test", "res ALA", threshold_min=10.0)
        >>> service.with_mean_reduction("test", "central_atoms", threshold_min=5.0, threshold_max=15.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "mean", threshold_min, threshold_max, cross_trajectory
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
        Add coordinates with minimum value reduction.

        Filters coordinate features based on minimum position values
        across the trajectory. This can identify atoms that reach
        specific minimum positions during dynamics.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum threshold for minimum values (Ångströms)
        threshold_max : float, optional
            Maximum threshold for minimum values (Ångströms)
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
            Adds coordinates selection with minimum reduction

        Examples
        --------
        >>> service.with_min_reduction("test", "res ALA", threshold_min=0.0)
        >>> service.with_min_reduction("test", "lower_bound", threshold_max=5.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "min", threshold_min, threshold_max, cross_trajectory
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
        Add coordinates with maximum value reduction.

        Filters coordinate features based on maximum position values
        across the trajectory. This can identify atoms that reach
        specific maximum positions during dynamics.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum threshold for maximum values (Ångströms)
        threshold_max : float, optional
            Maximum threshold for maximum values (Ångströms)
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
            Adds coordinates selection with maximum reduction

        Examples
        --------
        >>> service.with_max_reduction("test", "res ALA", threshold_min=20.0)
        >>> service.with_max_reduction("test", "upper_bound", threshold_min=15.0, threshold_max=25.0)
        """
        self(selector_name, selection, use_reduced, common_denominator,
             traj_selection, require_all_partners)
        self._add_reduction_config(
            selector_name, "max", threshold_min, threshold_max, cross_trajectory
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
        Add coordinates with transitions reduction.

        Filters coordinate features based on the number of positional
        transitions. Higher values indicate more dynamic atoms that
        frequently change their position by more than the threshold.

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
        transition_threshold : float, default=2.0
            Position change threshold (Ångströms) for defining transitions
        window_size : int, default=10
            Window size for transition detection
        transition_mode : str, default='window'
            Mode for transition calculation ('window' or 'direct')
        lag_time : int, default=1
            Lag time for transition detection
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
            Adds coordinates selection with transitions reduction

        Examples
        --------
        >>> service.with_transitions_reduction("test", "res ALA", threshold_min=10)
        >>> service.with_transitions_reduction("test", "dynamic_loops",
        ...     threshold_min=20, transition_threshold=3.0, window_size=20)
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

