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

"""Service for adding SASA selections with SASA-specific reduction methods."""
from __future__ import annotations
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .....feature_selection.managers.feature_selector_manager import FeatureSelectorManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ...interfaces.selection_service_base import SelectionServiceBase


class SasaSelectionService(SelectionServiceBase):
    """
    Service for selecting SASA features with SASA-specific reduction methods.

    Knows ALL reduction metrics from SasaReduceService:
    - cv, std, variance, range, mad, mean, min, max, burial_fraction,
      exposure_fraction, transitions

    This service provides methods to add SASA feature selections with
    optional post-selection reduction. Each reduction method applies filtering
    ONLY to the specific selection where it's defined.

    Examples
    --------
    Basic selection without reduction:
    >>> service("test", "res ALA")

    With coefficient of variation reduction:
    >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.1)

    With burial fraction reduction:
    >>> service.with_burial_fraction_reduction("test", "res ALA",
    ...     threshold_min=0.3, burial_threshold=0.2)
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData):
        """
        Initialize SASA selection service.

        Creates a service for adding SASA feature selections with optional
        post-selection reduction based on statistical metrics.

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
        >>> service = pipeline.feature_selector.add.sasa
        >>> # Service is now ready to add SASA selections
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "sasa"

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
        Add SASA selection without reduction.

        Adds SASA feature selection to the named selector without applying
        any statistical filtering. All SASA features matching the selection
        criteria will be included.

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
            Adds SASA selection to the named selector

        Examples
        --------
        >>> service("analysis", "res ALA")
        >>> service("analysis", "resid 120-140", use_reduced=True)
        """
        self._manager.add_selection(
            self._pipeline_data, selector_name, self._feature_type, selection,
            use_reduced, common_denominator, traj_selection, require_all_partners
        )

    def with_cv_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with CV (coefficient of variation) reduction.

        Filters SASA features based on coefficient of variation (std/mean).
        Features with CV outside the specified thresholds are removed.

        Parameters
        ----------
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

        Returns
        -------
        None
            Adds SASA selection with CV reduction

        Examples
        --------
        >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.1)
        >>> service.with_cv_reduction("test", "res ALA", threshold_min=0.05, threshold_max=0.8)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "cv", threshold_min, threshold_max, cross_trajectory)

    def with_range_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with range reduction.

        Filters SASA features based on range (max - min) of surface
        accessibility values. Higher range values indicate residues
        with more dynamic surface exposure.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum range threshold (Å²)
        threshold_max : float, optional
            Maximum range threshold (Å²)
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
            Adds SASA selection with range reduction

        Examples
        --------
        >>> service.with_range_reduction("test", "res ALA", threshold_min=20.0)
        >>> service.with_range_reduction("test", "dynamic_surface", threshold_min=50.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "range", threshold_min, threshold_max, cross_trajectory)

    def with_std_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with standard deviation reduction.

        Filters SASA features based on standard deviation of surface
        accessibility values. Higher values indicate more dynamic
        surface exposure with greater variability.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum standard deviation threshold (Å²)
        threshold_max : float, optional
            Maximum standard deviation threshold (Å²)
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
            Adds SASA selection with standard deviation reduction

        Examples
        --------
        >>> service.with_std_reduction("test", "res ALA", threshold_min=10.0)
        >>> service.with_std_reduction("test", "variable_exposure", threshold_min=15.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "std", threshold_min, threshold_max, cross_trajectory)

    def with_variance_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with variance reduction.

        Filters SASA features based on variance of surface accessibility
        values. Higher variance indicates residues with more dynamic
        surface exposure and broader SASA distributions.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum variance threshold (Ų)
        threshold_max : float, optional
            Maximum variance threshold (Ų)
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
            Adds SASA selection with variance reduction

        Examples
        --------
        >>> service.with_variance_reduction("test", "res ALA", threshold_min=100.0)
        >>> service.with_variance_reduction("test", "high_variance", threshold_min=200.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "variance", threshold_min, threshold_max, cross_trajectory)

    def with_mad_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with MAD (median absolute deviation) reduction.

        Filters SASA features based on median absolute deviation of
        surface accessibility. MAD is more robust to outliers than
        standard deviation while still indicating surface exposure variability.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum MAD threshold (Å²)
        threshold_max : float, optional
            Maximum MAD threshold (Å²)
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
            Adds SASA selection with MAD reduction

        Examples
        --------
        >>> service.with_mad_reduction("test", "res ALA", threshold_min=8.0)
        >>> service.with_mad_reduction("test", "robust_variable", threshold_min=12.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "mad", threshold_min, threshold_max, cross_trajectory)

    def with_mean_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with mean reduction.

        Filters SASA features based on mean surface accessibility values.
        This can be used to select residues based on their average
        surface exposure over the trajectory.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum mean threshold (Å²)
        threshold_max : float, optional
            Maximum mean threshold (Å²)
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
            Adds SASA selection with mean reduction

        Examples
        --------
        >>> service.with_mean_reduction("test", "res ALA", threshold_min=20.0)
        >>> service.with_mean_reduction("test", "exposed_residues", threshold_min=50.0, threshold_max=150.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "mean", threshold_min, threshold_max, cross_trajectory)

    def with_min_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with minimum value reduction.

        Filters SASA features based on minimum surface accessibility
        values across the trajectory. This can identify residues that
        reach specific minimum exposure levels during dynamics.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum threshold for minimum values (Å²)
        threshold_max : float, optional
            Maximum threshold for minimum values (Å²)
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
            Adds SASA selection with minimum reduction

        Examples
        --------
        >>> service.with_min_reduction("test", "res ALA", threshold_min=0.0)
        >>> service.with_min_reduction("test", "sometimes_buried", threshold_max=10.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "min", threshold_min, threshold_max, cross_trajectory)

    def with_max_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with maximum value reduction.

        Filters SASA features based on maximum surface accessibility
        values across the trajectory. This can identify residues that
        reach specific maximum exposure levels during dynamics.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        threshold_min : float, optional
            Minimum threshold for maximum values (Å²)
        threshold_max : float, optional
            Maximum threshold for maximum values (Å²)
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
            Adds SASA selection with maximum reduction

        Examples
        --------
        >>> service.with_max_reduction("test", "res ALA", threshold_min=100.0)
        >>> service.with_max_reduction("test", "highly_exposed", threshold_min=150.0, threshold_max=250.0)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "max", threshold_min, threshold_max, cross_trajectory)

    def with_burial_fraction_reduction(self, selector_name: str, selection: str = "all", burial_threshold: float = 0.2, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with burial fraction reduction.

        Filters SASA features based on the fraction of time residues are
        buried (SASA below burial_threshold). Higher values indicate more
        buried residues.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        burial_threshold : float, default=0.2
            SASA threshold (Å²) below which residue is considered buried.
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
            Adds SASA selection with burial fraction reduction

        Examples
        --------
        >>> service.with_burial_fraction_reduction("test", "res ALA", burial_threshold=0.15)
        >>> service.with_burial_fraction_reduction("test", "core_residues", burial_threshold=0.1)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "burial_fraction", burial_threshold, None, cross_trajectory)

    def with_exposure_fraction_reduction(self, selector_name: str, selection: str = "all", exposure_threshold: float = 0.2, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with exposure fraction reduction.

        Filters SASA features based on the fraction of time residues are
        exposed (SASA above exposure_threshold). Higher values indicate
        more consistently exposed residues.

        Parameters
        ----------
        selector_name : str
            Name of the feature selector configuration
        selection : str, default="all"
            Selection criteria string
        exposure_threshold : float, default=0.2
            SASA threshold (Å²) above which residue is considered exposed.
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
            Adds SASA selection with exposure fraction reduction

        Examples
        --------
        >>> service.with_exposure_fraction_reduction("test", "res ALA", exposure_threshold=0.3)
        >>> service.with_exposure_fraction_reduction("test", "surface_residues", exposure_threshold=0.5)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        self._add_reduction_config(selector_name, "exposure_fraction", None, exposure_threshold, cross_trajectory)

    def with_transitions_reduction(self, selector_name: str, selection: str = "all", threshold_min: Optional[float] = None, threshold_max: Optional[float] = None, transition_threshold: float = 10.0, window_size: int = 10, transition_mode: str = 'window', lag_time: int = 1, cross_trajectory: bool = True, use_reduced: bool = False, common_denominator: bool = True, traj_selection: Union[int, str, List[Union[int, str]], "all"] = "all", require_all_partners: bool = False) -> None:
        """
        Add SASA with transitions reduction.

        Filters SASA features based on the number of transitions between
        exposed and buried states. Higher values indicate more dynamic
        surface accessibility.

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
        transition_threshold : float, default=10.0
            SASA threshold for defining exposed/buried states
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
            Adds SASA selection with transitions reduction

        Examples
        --------
        >>> service.with_transitions_reduction("test", "res ALA", threshold_min=5)
        >>> service.with_transitions_reduction("test", "dynamic_surface",
        ...     threshold_min=10, transition_threshold=15.0, window_size=20)
        """
        self(selector_name, selection, use_reduced, common_denominator, traj_selection, require_all_partners)
        extra_params = {"transition_threshold": transition_threshold, "window_size": window_size, "transition_mode": transition_mode, "lag_time": lag_time}
        self._add_reduction_config(selector_name, "transitions", threshold_min, threshold_max, cross_trajectory, extra_params)

