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

"""Base class for all feature type selection services."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....feature_selection.managers.feature_selector_manager import FeatureSelectorManager
    from ....pipeline.entities.pipeline_data import PipelineData


class SelectionServiceBase(ABC):
    """
    Base class for all feature type selection services.

    Provides common functionality for feature selection services including:
    - Standard initialization with manager and pipeline_data
    - Common reduction configuration handling
    - Abstract __call__ method that must be implemented by subclasses

    Subclasses must implement:
    - __call__: Main selection method for the feature type
    - Set self._feature_type: String identifier for the feature type

    Examples:
    ---------
    >>> class ContactsSelectionService(SelectionServiceBase):
    ...     def __init__(self, manager, pipeline_data):
    ...         super().__init__(manager, pipeline_data)
    ...         self._feature_type = "contacts"
    ...
    ...     def __call__(self, selector_name, selection="all", ...):
    ...         # Implementation specific to contacts
    ...         self._manager.add_selection(...)
    """

    def __init__(self, manager: FeatureSelectorManager, pipeline_data: PipelineData) -> None:
        """
        Initialize base selection service.

        Parameters:
        -----------
        manager : FeatureSelectorManager
            Manager instance for executing add operations
        pipeline_data : PipelineData
            Pipeline data container with trajectory and feature data

        Returns:
        --------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
        self._feature_type = None  # Must be set by subclass

    @abstractmethod
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
        Add feature selection without reduction.

        This abstract method must be implemented by all subclasses to provide
        the main selection functionality for their specific feature type.

        Parameters:
        -----------
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

        Returns:
        --------
        None
            Adds feature selection to the named selector
        """
        pass

    def _add_reduction_config(
        self,
        selector_name: str,
        metric: str,
        threshold_min: Optional[float],
        threshold_max: Optional[float],
        cross_trajectory: bool = True,
        extra_params: Optional[dict] = None
    ) -> None:
        """
        Store reduction config in the LAST selection of this type.

        This method provides common functionality for storing reduction
        configuration parameters in the most recently added selection
        of the feature type. It creates a configuration dictionary with
        all reduction parameters and attaches it to the selection.

        Parameters:
        -----------
        selector_name : str
            Name of the feature selector configuration
        metric : str
            Statistical metric for reduction (e.g., 'std', 'max', 'transitions')
        threshold_min : float, optional
            Minimum threshold value for the metric
        threshold_max : float, optional
            Maximum threshold value for the metric
        cross_trajectory : bool, default=True
            Whether to apply reduction across all trajectories
        extra_params : dict, optional
            Additional parameters specific to the reduction metric

        Returns:
        --------
        None
            Modifies the last selection of this feature type in-place

        Raises:
        -------
        ValueError
            If no selections exist for the feature type in the selector
        """
        if self._feature_type is None:
            raise ValueError("Feature type not set by subclass")

        selector_data = self._pipeline_data.selected_feature_data[selector_name]

        if self._feature_type not in selector_data.selections:
            raise ValueError(f"No {self._feature_type} selections for selector '{selector_name}'")

        last_selection = selector_data.selections[self._feature_type][-1]

        config = {
            "metric": metric,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "cross_trajectory": cross_trajectory
        }

        if extra_params:
            config.update(extra_params)

        last_selection["reduction"] = config
