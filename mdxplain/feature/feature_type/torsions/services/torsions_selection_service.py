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

"""Service for adding torsions selections with torsions-specific reduction methods."""
from __future__ import annotations
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .....feature_selection.manager.feature_selector_manager import FeatureSelectorManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ...interfaces.selection_service_base import SelectionServiceBase


class TorsionsSelectionService(SelectionServiceBase):
    """
    Service for selecting torsions features with torsions-specific reduction methods.

    Knows ALL reduction metrics from TorsionsReduceService:
    
    - transitions, std, mad, mean, range, min, max, cv, variance

    This service provides methods to add torsions (dihedral angles) feature
    selections with optional post-selection reduction. Each reduction method
    applies filtering ONLY to the specific selection where it's defined.

    Examples
    --------
    Basic selection without reduction:
    >>> service("test", "res ALA")

    With transitions reduction:
    >>> service.with_transitions_reduction("test", "res ALA", threshold_min=10)

    With standard deviation reduction:
    >>> service.with_std_reduction("test", "backbone", threshold_min=20.0)
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData):
        """
        Initialize torsions selection service.

        Creates a service for adding torsions (dihedral angles) feature
        selections with optional post-selection reduction based on
        angular motion metrics.

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
        >>> service = pipeline.feature_selector.add.torsions
        >>> # Service is now ready to add torsions selections
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "torsions"

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
        Add torsions selection without reduction.

        Adds torsions (dihedral angles) feature selection to the named
        selector without applying any statistical filtering. All torsions
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
            Adds torsions selection to the named selector

        Examples
        --------
        >>> service("analysis", "res ALA")
        >>> service("analysis", "resid 120-140", use_reduced=True)
        """
        self._manager.add_selection(
            self._pipeline_data, selector_name, self._feature_type, selection,
            use_reduced, common_denominator, traj_selection, require_all_partners
        )

    def with_transitions_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False, transition_threshold: float = 30.0, window_size: int = 10, transition_mode: str = 'window', lag_time: int = 1) -> None:
        """
        Add torsions with transitions reduction.

        Filters torsion features based on the number of angular transitions.
        Higher values indicate more flexible dihedral angles.

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
        transition_threshold : float, default=30.0
            Angular threshold (degrees) for defining transitions
        window_size : int, default=10
            Window size for transition detection
        transition_mode : str, default='window'
            Mode for transition calculation ('window' or 'direct')
        lag_time : int, default=1
            Lag time for transition detection

        Returns
        -------
        None
            Adds torsions selection with transitions reduction

        Examples
        --------
        >>> service.with_transitions_reduction("test", "res ALA", threshold_min=10)
        >>> service.with_transitions_reduction("test", "flexible_backbone",
        ...     threshold_min=20, transition_threshold=45.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        extra_params = {"transition_threshold": transition_threshold, "window_size": window_size, "transition_mode": transition_mode, "lag_time": lag_time}
        self._add_reduction_config(selector_name, "transitions", threshold_min, threshold_max, cross_trajectory, extra_params)

    def with_std_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with standard deviation reduction.

        Filters torsion features based on standard deviation of angular
        motion. Higher values indicate more flexible dihedral angles.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum standard deviation (degrees)
        threshold_max : float, optional
            Maximum standard deviation (degrees)
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
            Adds torsions selection with standard deviation reduction

        Examples
        --------
        >>> service.with_std_reduction("test", "res ALA", threshold_min=20.0)
        >>> service.with_std_reduction("test", "flexible_loops", threshold_min=30.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "std", threshold_min, threshold_max, cross_trajectory)

    def with_mad_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with MAD (median absolute deviation) reduction.

        Filters torsion features based on median absolute deviation of
        angular motion. MAD is more robust to outliers than standard
        deviation while still indicating angular flexibility.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum MAD threshold (degrees)
        threshold_max : float, optional
            Maximum MAD threshold (degrees)
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
            Adds torsions selection with MAD reduction

        Examples
        --------
        >>> service.with_mad_reduction("test", "res ALA", threshold_min=15.0)
        >>> service.with_mad_reduction("test", "robust_flexible", threshold_min=25.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "mad", threshold_min, threshold_max, cross_trajectory)

    def with_mean_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with mean reduction.

        Filters torsion features based on mean angular values.
        This can be used to select dihedral angles based on their
        average values over the trajectory.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum mean threshold (degrees)
        threshold_max : float, optional
            Maximum mean threshold (degrees)
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
            Adds torsions selection with mean reduction

        Examples
        --------
        >>> service.with_mean_reduction("test", "res ALA", threshold_min=-90.0, threshold_max=90.0)
        >>> service.with_mean_reduction("test", "alpha_helical", threshold_min=-70.0, threshold_max=-50.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "mean", threshold_min, threshold_max, cross_trajectory)

    def with_range_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with range reduction.

        Filters torsion features based on angular range (max - min) of
        dihedral motion. Higher range values indicate more flexible
        dihedral angles with larger amplitude of motion.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum range threshold (degrees)
        threshold_max : float, optional
            Maximum range threshold (degrees)
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
            Adds torsions selection with range reduction

        Examples
        --------
        >>> service.with_range_reduction("test", "res ALA", threshold_min=60.0)
        >>> service.with_range_reduction("test", "flexible_sidechains", threshold_min=120.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "range", threshold_min, threshold_max, cross_trajectory)

    def with_min_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with minimum value reduction.

        Filters torsion features based on minimum angular values
        across the trajectory. This can identify dihedral angles
        that reach specific minimum values during dynamics.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum threshold for minimum values (degrees)
        threshold_max : float, optional
            Maximum threshold for minimum values (degrees)
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
            Adds torsions selection with minimum reduction

        Examples
        --------
        >>> service.with_min_reduction("test", "res ALA", threshold_min=-180.0)
        >>> service.with_min_reduction("test", "negative_angles", threshold_max=-90.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "min", threshold_min, threshold_max, cross_trajectory)

    def with_max_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with maximum value reduction.

        Filters torsion features based on maximum angular values
        across the trajectory. This can identify dihedral angles
        that reach specific maximum values during dynamics.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum threshold for maximum values (degrees)
        threshold_max : float, optional
            Maximum threshold for maximum values (degrees)
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
            Adds torsions selection with maximum reduction

        Examples
        --------
        >>> service.with_max_reduction("test", "res ALA", threshold_min=90.0)
        >>> service.with_max_reduction("test", "positive_angles", threshold_min=120.0, threshold_max=180.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "max", threshold_min, threshold_max, cross_trajectory)

    def with_cv_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with CV (coefficient of variation) reduction.

        Filters torsion features based on coefficient of variation (std/mean)
        of angular motion. Higher CV values indicate more variable dihedral
        angles relative to their mean angular position.

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
            Adds torsions selection with CV reduction

        Examples
        --------
        >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.2)
        >>> service.with_cv_reduction("test", "variable_angles", threshold_min=0.3)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "cv", threshold_min, threshold_max, cross_trajectory)

    def with_variance_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add torsions with variance reduction.

        Filters torsion features based on variance of angular motion.
        Higher values indicate more flexible dihedral angles with
        broader angular distributions.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum variance (degrees²)
        threshold_max : float, optional
            Maximum variance (degrees²)
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
            Adds torsions selection with variance reduction

        Examples
        --------
        >>> service.with_variance_reduction("test", "res ALA", threshold_min=400.0)
        >>> service.with_variance_reduction("test", "dynamic_sidechains", threshold_min=600.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "variance", threshold_min, threshold_max, cross_trajectory)

