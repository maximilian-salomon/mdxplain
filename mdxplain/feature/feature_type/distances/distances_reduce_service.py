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

"""Factory for distance-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...managers.feature_manager import FeatureManager
    from ....pipeline.entities.pipeline_data import PipelineData

from . import Distances


class DistancesReduceService:
    """
    Service for distance-specific reduce metrics.
    
    Provides statistical metrics specifically useful for distance features,
    including variability measures and transition detection.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize distance reduce factory.
        
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
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    def cv(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce distances by coefficient of variation (CV = std/mean).
        
        Filters distance features based on their relative variability.
        Higher CV indicates more dynamic/flexible distance pairs.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum CV threshold (features below are removed)
        threshold_max : float, optional
            Maximum CV threshold (features above are removed)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
            
        Examples:
        ---------
        >>> # Keep moderately variable distances
        >>> pipeline.feature.reduce.distances.cv(threshold_min=0.1, threshold_max=0.5)
        
        >>> # Keep highly variable distances only
        >>> pipeline.feature.reduce.distances.cv(threshold_min=0.3)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Distances,
            metric="cv",
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
        Reduce distances by standard deviation.
        
        Filters distance features based on their absolute variability.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum std threshold in Angstroms
        threshold_max : float, optional
            Maximum std threshold in Angstroms
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
            
        Examples:
        ---------
        >>> # Keep distances with significant variation
        >>> pipeline.feature.reduce.distances.std(threshold_min=0.5)
        
        >>> # Remove extremely variable distances
        >>> pipeline.feature.reduce.distances.std(threshold_max=3.0)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Distances,
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
        Reduce distances by variance.
        
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
            Distances,
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
        Reduce distances by range (max - min).
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum range threshold in Angstroms
        threshold_max : float, optional
            Maximum range threshold in Angstroms
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Distances,
            metric="range",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 2.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> None:
        """
        Reduce distances by transition detection.
        
        Identifies distance features that show conformational transitions
        based on their temporal patterns.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions required
        threshold_max : float, optional
            Maximum number of transitions allowed
        transition_threshold : float, default=2.0
            Z-score threshold for transition detection
        window_size : int, default=10
            Size of sliding window for transition detection
        transition_mode : str, default="window"
            Mode for transition detection ('window' or 'lag')
        lag_time : int, default=1
            Lag time for transition detection (if mode='lag')
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
            
        Examples:
        ---------
        >>> # Find distances with clear transitions
        >>> pipeline.feature.reduce.distances.transitions(
        ...     transition_threshold=3.0, 
        ...     window_size=20
        ... )
        
        >>> # Keep features with moderate number of transitions
        >>> pipeline.feature.reduce.distances.transitions(
        ...     threshold_min=2, 
        ...     threshold_max=10
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Distances,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
            window_size=window_size,
            transition_mode=transition_mode,
            lag_time=lag_time,
        )
    
    def min(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce distances by minimum value.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum distance minimum in Angstroms
        threshold_max : float, optional
            Maximum distance minimum in Angstroms
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Distances,
            metric="min",
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
        Reduce distances by median absolute deviation.
        
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
            Distances,
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
        Reduce distances by mean value.
        
        Filters distance features based on their average values over time.
        Useful for identifying consistently close or distant pairs.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum mean distance threshold (features below are removed)
        threshold_max : float, optional
            Maximum mean distance threshold (features above are removed)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
            
        Examples:
        ---------
        >>> # Keep distances with moderate average values
        >>> pipeline.feature.reduce.distances.mean(threshold_min=2.0, threshold_max=8.0)
        
        >>> # Keep only close contacts on average
        >>> pipeline.feature.reduce.distances.mean(threshold_max=5.0)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Distances,
            metric="mean",
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
        Reduce distances by maximum value.
        
        Filters distance features based on their maximum observed values.
        Useful for identifying pairs that never exceed certain distances.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum maximum distance threshold (features below are removed)
        threshold_max : float, optional
            Maximum maximum distance threshold (features above are removed)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
            
        Examples:
        ---------
        >>> # Keep only pairs that stay within 10 Å
        >>> pipeline.feature.reduce.distances.max(threshold_max=10.0)
        
        >>> # Keep pairs that reach at least 15 Å separation
        >>> pipeline.feature.reduce.distances.max(threshold_min=15.0)
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            Distances,
            metric="max",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
