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

"""Factory for contact-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....managers.feature_manager import FeatureManager
    from .....pipeline.entities.pipeline_data import PipelineData

from ..contacts import Contacts
from ...interfaces.reduce_service_base import ReduceServiceBase


class ContactsReduceService(ReduceServiceBase):
    """
    Service for contact-specific reduce metrics.
    
    Provides metrics specifically useful for contact features,
    focusing on contact frequency and stability patterns.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize contact reduce factory.
        
        Parameters:
        -----------
        manager : FeatureManager
            Feature manager instance
        pipeline_data : PipelineData
            Pipeline data container
            
        Returns:
        --------
        None
        """
        super().__init__(manager, pipeline_data)
        self._feature_type = "contacts"
    
    def frequency(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce contacts by contact frequency.
        
        Filters contact features based on how frequently they occur
        (fraction of frames where pair is in contact).
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum contact frequency (0.0 to 1.0)
        threshold_max : float, optional
            Maximum contact frequency (0.0 to 1.0)
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline
            
        Examples:
        ---------
        >>> # Keep persistent contacts (>50% of time)
        >>> pipeline.feature.reduce.contacts.frequency(threshold_min=0.5)
        
        >>> # Keep moderately frequent contacts
        >>> pipeline.feature.reduce.contacts.frequency(
        ...     threshold_min=0.2, 
        ...     threshold_max=0.8
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Contacts,
            metric="frequency",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def stability(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce contacts by stability.
        
        Filters contact features based on their stability over time
        (measure of contact persistence patterns).
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum stability threshold
        threshold_max : float, optional
            Maximum stability threshold
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline
            
        Examples:
        ---------
        >>> # Keep stable contacts
        >>> pipeline.feature.reduce.contacts.stability(threshold_min=0.7)
        
        >>> # Remove very unstable contacts
        >>> pipeline.feature.reduce.contacts.stability(threshold_max=0.3)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Contacts,
            metric="stability",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
    ) -> None:
        """
        Reduce contacts by number of formation/breaking transitions.

        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep dynamic contacts with many transitions
        >>> pipeline.feature.reduce.contacts.transitions(threshold_min=10)

        >>> # Keep stable contacts with few transitions
        >>> pipeline.feature.reduce.contacts.transitions(threshold_max=5)

        >>> # Focus on moderately dynamic contacts
        >>> pipeline.feature.reduce.contacts.transitions(
        ...     threshold_min=3,
        ...     threshold_max=15
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Contacts,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
