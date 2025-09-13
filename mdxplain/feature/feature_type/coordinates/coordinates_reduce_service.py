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

"""Factory for coordinates-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...managers.feature_manager import FeatureManager
    from ....pipeline.entities.pipeline_data import PipelineData

from . import Coordinates


class CoordinatesReduceService:
    """
    Service for coordinates-specific reduce metrics.
    
    Provides methods to reduce coordinate features based on various statistical metrics.
    Each method applies a different reduction criterion, such as standard deviation,
    root mean square fluctuation, coefficient of variation, variance, range, median absolute deviation,
    mean, min, max, and transitions.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize coordinates reduce factory.
        
        Parameters:
        -----------
        manager : FeatureManager
            The feature manager instance
        pipeline_data : PipelineData
            The pipeline data instance

        Returns:
        --------
        None
            Initializes the coordinates reduce service
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    def std(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce coordinates by standard deviation.

        Standard deviation measures the amount of variation or dispersion in a set of values.

        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum standard deviation threshold (Angstrom)
        threshold_max : float, optional
            Maximum standard deviation threshold (Angstrom)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="std",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def rmsf(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce coordinates by root mean square fluctuation.

        RMSF measures average positional deviation from mean position.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum RMSF threshold (Angstrom)
        threshold_max : float, optional
            Maximum RMSF threshold (Angstrom)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="rmsf",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def cv(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce coordinates by coefficient of variation.
        
        Filters coordinate features based on their relative variability.
        CV = std/mean, indicating positional flexibility.
        
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
            Coordinates,
            metric="cv",
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
        Reduce coordinates by variance.
        
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
            Coordinates,
            metric="variance",
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
        Reduce coordinates by range (max - min).
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum range threshold (Angstrom)
        threshold_max : float, optional
            Maximum range threshold (Angstrom)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="range",
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
        Reduce coordinates by median absolute deviation.
        
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
            Coordinates,
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
        Reduce coordinates by mean position.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum mean position threshold (Angstrom)
        threshold_max : float, optional
            Maximum mean position threshold (Angstrom)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
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
        Reduce coordinates by minimum position.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum position minimum threshold (Angstrom)
        threshold_max : float, optional
            Maximum position minimum threshold (Angstrom)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
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
        Reduce coordinates by maximum position.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum position maximum threshold (Angstrom)
        threshold_max : float, optional
            Maximum position maximum threshold (Angstrom)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Coordinates,
            metric="max",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 1.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> None:
        """
        Reduce coordinates by transition detection.
        
        Filters coordinate features based on number of positional transitions.
        Detects significant changes in atomic positions.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        transition_threshold : float, default=1.0
            Position threshold for detecting transitions (Angstrom)
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
            Coordinates,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            transition_mode=transition_mode,
            lag_time=lag_time,
        )
