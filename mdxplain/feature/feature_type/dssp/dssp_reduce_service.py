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

"""Factory for DSSP-specific reduce operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...managers.feature_manager import FeatureManager
    from ....pipeline.entities.pipeline_data import PipelineData

from . import DSSP


class DSSPReduceService:
    """
    Service for DSSP-specific reduce metrics.
    
    Provides metrics specifically useful for DSSP features,
    focusing on secondary structure transitions and stability.
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize DSSP reduce factory.
        
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
    
    def transitions(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        transition_threshold: float = 1.0,
    ) -> None:
        """
        Reduce DSSP by secondary structure transitions.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum number of transitions
        threshold_max : float, optional
            Maximum number of transitions
        transition_threshold : float, default=1.0
            Threshold for transition detection
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="transitions",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            transition_threshold=transition_threshold,
        )
    
    def transition_frequency(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce DSSP by transition frequency.
        
        Filters residues based on frequency of secondary structure transitions.
        Same as transitions but normalized by trajectory length.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum transition frequency
        threshold_max : float, optional
            Maximum transition frequency
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="transition_frequency",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def stability(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce DSSP by structural stability.
        
        Filters residues based on secondary structure stability.
        Stability = 1 - transition_frequency, higher values = more stable.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum stability threshold (0-1)
        threshold_max : float, optional
            Maximum stability threshold (0-1)
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="stability",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    
    def class_frequencies(
        self,
        traj_selection: Union[str, int, List] = "all",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
    ) -> None:
        """
        Reduce DSSP by secondary structure class frequencies.
        
        Filters residues based on frequency distribution of structural classes.
        Analyzes how often each residue adopts different secondary structures.
        
        Parameters:
        -----------
        traj_selection : str, int, list, default="all"
            Which trajectories to analyze for reduction
        threshold_min : float, optional
            Minimum class frequency threshold
        threshold_max : float, optional
            Maximum class frequency threshold
            
        Returns:
        --------
        None
            Updates reduced data in pipeline
        """
        return self._manager.reduce_data(
            self._pipeline_data,
            DSSP,
            metric="class_frequencies",
            traj_selection=traj_selection,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
