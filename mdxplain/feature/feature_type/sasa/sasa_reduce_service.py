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

"""Factory for SASA-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...managers.feature_manager import FeatureManager
    from ....pipeline.entities.pipeline_data import PipelineData

from . import SASA


class SASAReduceService:
    """Service for SASA-specific reduce metrics."""
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """Initialize SASA reduce factory."""
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    def cv(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by coefficient of variation.
        
        Filters SASA features based on their relative variability.
        Higher CV indicates more dynamic surface accessibility.

        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum CV threshold
        threshold_max : float, optional
            Maximum CV threshold

        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="cv",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def range(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by range (max - min).

        Filters SASA features based on their dynamic range.
        Higher range indicates more variability in surface accessibility.

        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum range threshold (nm²)
        threshold_max : float, optional
            Maximum range threshold (nm²)

        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="range",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def std(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by standard deviation.
        
        Filters SASA features based on their variability.
        Higher std indicates more dynamic surface accessibility.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum std threshold (nm²)
        threshold_max : float, optional
            Maximum std threshold (nm²)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="std",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def variance(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by variance.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum variance threshold
        threshold_max : float, optional
            Maximum variance threshold
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="variance",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def mad(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by median absolute deviation.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum MAD threshold
        threshold_max : float, optional
            Maximum MAD threshold
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="mad",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def mean(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by mean value.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum mean SASA in nm²
        threshold_max : float, optional
            Maximum mean SASA in nm²
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="mean",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def min(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by minimum value.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum SASA minimum in nm²
        threshold_max : float, optional
            Maximum SASA minimum in nm²
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="min",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def max(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by maximum value.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum SASA maximum in nm²
        threshold_max : float, optional
            Maximum SASA maximum in nm²
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="max",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def burial_fraction(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by burial fraction.
        
        Filters residues based on fraction of time they are buried.
        Uses threshold_max as burial cutoff (SASA below this = buried).
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum burial fraction (0-1)
        threshold_max : float, optional
            SASA cutoff for buried state (nm²)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="burial_fraction",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def exposure_fraction(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce SASA by exposure fraction.
        
        Filters residues based on fraction of time they are exposed.
        Uses threshold_max as exposure cutoff (SASA above this = exposed).
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum exposure fraction (0-1)
        threshold_max : float, optional
            SASA cutoff for exposed state (nm²)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="exposure_fraction",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 0.5,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> None:
        """
        Reduce SASA by transition detection.
        
        Filters SASA features based on number of transitions.
        Detects changes in surface accessibility patterns.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        transition_threshold : float, default=0.5
            SASA threshold for detecting transitions (nm²)
        window_size : int, default=10
            Window size for transition analysis
        transition_mode : str, default="window"
            Transition analysis mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for transition analysis
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            transition_mode=transition_mode,
            lag_time=lag_time,
        )
