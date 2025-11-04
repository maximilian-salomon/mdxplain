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

"""Base class for all feature type reduce services."""

from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...manager.feature_manager import FeatureManager
    from ....pipeline.entities.pipeline_data import PipelineData


class ReduceServiceBase(ABC):
    """
    Base class for all feature type reduce services.

    Provides common functionality for feature reduction services including:

    - Standard initialization with manager and pipeline_data
    - Common validation and error handling

    Subclasses must set:
    
    - self._feature_type: String identifier for the feature type

    Examples
    --------
    >>> class ContactsReduceService(ReduceServiceBase):
    ...     def __init__(self, manager, pipeline_data):
    ...         super().__init__(manager, pipeline_data)
    ...         self._feature_type = "contacts"
    ...
    ...     def frequency(self, **kwargs):
    ...         # Contact-specific reduction method
    ...         return self._manager.reduce_data(...)
    """

    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize base reduce service.

        Parameters
        ----------
        manager : FeatureManager
            Feature manager instance for executing reduce operations
        pipeline_data : PipelineData
            Pipeline data container with trajectory and feature data

        Returns
        -------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
        self._feature_type = None  # Must be set by subclass
