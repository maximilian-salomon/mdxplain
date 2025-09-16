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
        """
        Initialize SASA reduce factory.

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
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep highly dynamic surface accessibility
        >>> pipeline.feature.reduce.sasa.cv(threshold_min=0.3)

        >>> # Remove extremely variable SASA
        >>> pipeline.feature.reduce.sasa.cv(threshold_max=0.8)

        >>> # Focus on moderately dynamic surface exposure
        >>> pipeline.feature.reduce.sasa.cv(
        ...     threshold_min=0.1,
        ...     threshold_max=0.6
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="cv",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def range(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues with large SASA changes
        >>> pipeline.feature.reduce.sasa.range(threshold_min=1.0)

        >>> # Keep relatively stable surface areas
        >>> pipeline.feature.reduce.sasa.range(threshold_max=0.5)

        >>> # Focus on moderate surface accessibility changes
        >>> pipeline.feature.reduce.sasa.range(
        ...     threshold_min=0.3,
        ...     threshold_max=1.5
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="range",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def std(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues with significant SASA fluctuations
        >>> pipeline.feature.reduce.sasa.std(threshold_min=0.4)

        >>> # Remove extremely variable surface areas
        >>> pipeline.feature.reduce.sasa.std(threshold_max=1.2)

        >>> # Focus on moderate surface accessibility dynamics
        >>> pipeline.feature.reduce.sasa.std(
        ...     threshold_min=0.2,
        ...     threshold_max=0.8
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="std",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def variance(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues with high SASA variance
        >>> pipeline.feature.reduce.sasa.variance(threshold_min=0.5)

        >>> # Remove extremely variable surface areas
        >>> pipeline.feature.reduce.sasa.variance(threshold_max=2.0)

        >>> # Focus on moderate surface area variance
        >>> pipeline.feature.reduce.sasa.variance(
        ...     threshold_min=0.2,
        ...     threshold_max=1.5
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="variance",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def mad(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep robustly variable SASA (outlier-resistant)
        >>> pipeline.feature.reduce.sasa.mad(threshold_min=0.3)

        >>> # Remove extreme SASA outliers
        >>> pipeline.feature.reduce.sasa.mad(threshold_max=1.0)

        >>> # MAD-based selection for robust analysis
        >>> pipeline.feature.reduce.sasa.mad(
        ...     threshold_min=0.15,
        ...     threshold_max=0.8
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="mad",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def mean(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep highly exposed residues
        >>> pipeline.feature.reduce.sasa.mean(threshold_min=2.0)

        >>> # Keep buried/partially exposed residues
        >>> pipeline.feature.reduce.sasa.mean(threshold_max=1.5)

        >>> # Focus on moderately exposed surface areas
        >>> pipeline.feature.reduce.sasa.mean(
        ...     threshold_min=0.5,
        ...     threshold_max=3.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="mean",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def min(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues that can become buried
        >>> pipeline.feature.reduce.sasa.min(threshold_max=0.5)

        >>> # Exclude completely buried residues
        >>> pipeline.feature.reduce.sasa.min(threshold_min=0.2)

        >>> # Focus on specific minimum accessibility range
        >>> pipeline.feature.reduce.sasa.min(
        ...     threshold_min=0.1,
        ...     threshold_max=1.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="min",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def max(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues that can become highly exposed
        >>> pipeline.feature.reduce.sasa.max(threshold_min=3.0)

        >>> # Keep relatively constrained surface areas
        >>> pipeline.feature.reduce.sasa.max(threshold_max=2.5)

        >>> # Focus on specific maximum accessibility range
        >>> pipeline.feature.reduce.sasa.max(
        ...     threshold_min=1.5,
        ...     threshold_max=4.0
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="max",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def burial_fraction(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep frequently buried residues
        >>> pipeline.feature.reduce.sasa.burial_fraction(
        ...     threshold_min=0.7,
        ...     threshold_max=0.5  # SASA cutoff for buried state
        ... )

        >>> # Keep occasionally buried residues
        >>> pipeline.feature.reduce.sasa.burial_fraction(
        ...     threshold_min=0.2,
        ...     threshold_max=1.0
        ... )

        >>> # Focus on intermediate burial patterns
        >>> pipeline.feature.reduce.sasa.burial_fraction(
        ...     threshold_min=0.3,
        ...     threshold_max=0.8
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="burial_fraction",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            cross_trajectory=cross_trajectory,
        )
    
    def exposure_fraction(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories
            
        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep frequently exposed residues
        >>> pipeline.feature.reduce.sasa.exposure_fraction(
        ...     threshold_min=0.6,
        ...     threshold_max=2.0  # SASA cutoff for exposed state
        ... )

        >>> # Keep occasionally exposed residues
        >>> pipeline.feature.reduce.sasa.exposure_fraction(
        ...     threshold_min=0.1,
        ...     threshold_max=3.0
        ... )

        >>> # Focus on intermediate exposure patterns
        >>> pipeline.feature.reduce.sasa.exposure_fraction(
        ...     threshold_min=0.3,
        ...     threshold_max=0.9
        ... )
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            SASA,
            metric="exposure_fraction",
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
        transition_threshold: float = 0.5,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
        cross_trajectory: bool = False,
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
        cross_trajectory : bool, default=False
            If True, find common features across all selected trajectories

        Returns:
        --------
        None
            Updates reduced data in pipeline

        Examples:
        ---------
        >>> # Keep residues with many burial/exposure transitions
        >>> pipeline.feature.reduce.sasa.transitions(
        ...     threshold_min=6,
        ...     transition_threshold=0.8
        ... )

        >>> # Keep stable surface accessibility
        >>> pipeline.feature.reduce.sasa.transitions(
        ...     threshold_max=4,
        ...     window_size=15
        ... )

        >>> # Focus on moderate SASA dynamics
        >>> pipeline.feature.reduce.sasa.transitions(
        ...     threshold_min=3,
        ...     threshold_max=10,
        ...     transition_threshold=0.5
        ... )
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
            cross_trajectory=cross_trajectory,
        )
