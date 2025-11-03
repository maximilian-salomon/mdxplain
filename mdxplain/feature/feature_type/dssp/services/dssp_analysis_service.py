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

"""Service for DSSP-specific analysis operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING
import numpy as np

from ....services.helper.analysis_data_helper import AnalysisDataHelper
from ..dssp_calculator_analysis import DSSPCalculatorAnalysis
from ...interfaces.analysis_service_base import AnalysisServiceBase
from ..dssp_calculator import DSSPCalculator
if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData


class DSSPAnalysisService(AnalysisServiceBase):
    """
    Service for DSSP-specific analysis operations.
    
    Provides a simplified interface for DSSP analysis methods by automatically
    handling data selection and forwarding method calls to the underlying
    calculator_analysis class. All methods from DSSPCalculatorAnalysis
    are automatically available with additional feature_selector and 
    traj_selection parameters.
    """
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """
        Initialize DSSP analysis service.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data containing DSSP feature data

        Returns
        -------
        None

        Examples
        --------
        >>> # Initialize DSSP analysis service
        >>> dssp_service = DSSPAnalysisService(pipeline.data)
        """
        super().__init__(pipeline_data)
        self._feature_type = "dssp"

        self._calculator = DSSPCalculatorAnalysis(
            full_classes=DSSPCalculator().FULL_CLASSES,
            simplified_classes=DSSPCalculator().SIMPLIFIED_CLASSES,
            use_memmap=pipeline_data.use_memmap,
            chunk_size=pipeline_data.chunk_size
        )
    
    # === TRANSITIONS & ANALYSIS ===
    
    def compute_transitions_lagtime(self, lag_time: int = 10, 
                                   feature_selector: Optional[str] = None, 
                                   traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute DSSP transitions using lag time analysis.
        
        Parameters
        ----------
        lag_time : int, default=10
            Number of frames to look ahead for transitions
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per residue
            
        Examples
        --------
        >>> # Find dynamic secondary structure elements
        >>> pipeline.analysis.features.dssp.compute_transitions_lagtime()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_lagtime(data, lag_time)
    
    def compute_transitions_window(self, window_size: int = 10,
                                  feature_selector: Optional[str] = None, 
                                  traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute DSSP transitions within sliding window.
        
        Parameters
        ----------
        window_size : int, default=10
            Size of sliding window for transition detection
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per residue
            
        Examples
        --------
        >>> # Window-based transition detection for secondary structure
        >>> pipeline.analysis.features.dssp.compute_transitions_window(window_size=20)
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_window(data, window_size)
    
    def compute_class_frequencies(self, simplified: bool = True,
                                 feature_selector: Optional[str] = None, 
                                 traj_selection: Optional[Union[str, int, List]] = None) -> tuple:
        """
        Compute frequencies of DSSP classes.
        
        Parameters
        ----------
        simplified : bool, default=True
            Use simplified classes (H, E, C) vs full classes
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        tuple
            (class_names, frequencies) for each DSSP class
            
        Examples
        --------
        >>> # Get secondary structure composition
        >>> classes, freqs = pipeline.analysis.features.dssp.compute_class_frequencies()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_class_frequencies(data, simplified)
    
    def compute_transition_frequency(self, feature_selector: Optional[str] = None, 
                                    traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute transition frequency between DSSP classes.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition frequency matrix between classes
            
        Examples
        --------
        >>> # Get secondary structure transition matrix
        >>> pipeline.analysis.features.dssp.compute_transition_frequency()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transition_frequency(data)
    
    def compute_stability(self, feature_selector: Optional[str] = None, 
                         traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute stability of DSSP assignments per residue.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Stability values per residue (0=unstable, 1=stable)
            
        Examples
        --------
        >>> # Find most stable secondary structure elements
        >>> pipeline.analysis.features.dssp.compute_stability()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_stability(data)
    
    def compute_differences(self, frame_1: int = 0, frame_2: int = -1,
                           feature_selector: Optional[str] = None, 
                           traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute differences in DSSP assignments between frames.
        
        Parameters
        ----------
        frame_1 : int, default=0
            First frame for comparison
        frame_2 : int, default=-1
            Second frame for comparison (-1 = last frame)
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Differences in DSSP assignments
            
        Examples
        --------
        >>> # Compare first and last frame
        >>> pipeline.analysis.features.dssp.compute_differences()
        
        >>> # Compare specific frames
        >>> pipeline.analysis.features.dssp.compute_differences(frame_1=100, frame_2=200)
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_differences(data, frame_1, frame_2)
    
    def compute_dominant_class(self, feature_selector: Optional[str] = None, 
                              traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute dominant DSSP class per residue across trajectory.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Dominant class index per residue
            
        Examples
        --------
        >>> # Get most common secondary structure per residue
        >>> pipeline.analysis.features.dssp.compute_dominant_class()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_dominant_class(data)
    