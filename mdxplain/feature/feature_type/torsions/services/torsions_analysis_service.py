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

"""Factory for torsions-specific analysis operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING
import numpy as np

from ....services.helpers.analysis_data_helper import AnalysisDataHelper
from ..torsions_calculator_analysis import TorsionsCalculatorAnalysis
from ...interfaces.analysis_service_base import AnalysisServiceBase

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData


class TorsionsAnalysisService(AnalysisServiceBase):
    """
    Service for torsions-specific analysis operations.
    
    Provides a simplified interface for torsion analysis methods by automatically
    handling data selection and forwarding method calls to the underlying
    calculator_analysis class. All methods from TorsionsCalculatorAnalysis
    are automatically available with additional feature_selector and 
    traj_selection parameters.
    """
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """
        Initialize torsions analysis service.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data containing feature data and configuration
        
        Returns
        -------
        None
        """
        super().__init__(pipeline_data)
        self._feature_type = "torsions"
        self._calculator = TorsionsCalculatorAnalysis(
            use_memmap=pipeline_data.use_memmap,
            chunk_size=pipeline_data.chunk_size
        )
    
    # === PER-FEATURE METHODS (per angle) ===
    
    def mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute circular mean for each torsion angle.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Circular mean angle for each torsion in degrees
            
        Examples
        --------
        >>> # Compute circular mean for all torsion angles
        >>> pipeline.analysis.features.torsions.mean()
        
        >>> # Compute mean for selected angles
        >>> pipeline.analysis.features.torsions.mean(feature_selector="phi_psi_angles")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mean(data)
    
    def std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute circular standard deviation for each torsion angle.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Circular standard deviation for each torsion angle in degrees
            
        Examples
        --------
        >>> # Compute std for all torsion angles
        >>> pipeline.analysis.features.torsions.std()
        
        >>> # Find most flexible angles (high std)
        >>> pipeline.analysis.features.torsions.std(feature_selector="backbone_angles")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_std(data)
    
    def variance(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute circular variance for each torsion angle.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Circular variance for each torsion angle (0-1 scale)
            
        Examples
        --------
        >>> # Compute variance for all torsion angles
        >>> pipeline.analysis.features.torsions.variance()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_variance(data)
    
    def min(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum angle for each torsion.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Minimum angle for each torsion
            
        Examples
        --------
        >>> # Find minimum angles observed
        >>> pipeline.analysis.features.torsions.min()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_min(data)
    
    def max(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum angle for each torsion.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Maximum angle for each torsion
            
        Examples
        --------
        >>> # Find maximum angles observed
        >>> pipeline.analysis.features.torsions.max()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_max(data)
    
    def mad(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median absolute deviation for each torsion angle.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Median absolute deviation for each torsion angle
            
        Examples
        --------
        >>> # Robust measure of angle variability
        >>> pipeline.analysis.features.torsions.mad()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mad(data)
    
    def range(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute angular range for each torsion considering periodicity.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Angular range for each torsion (0-180 degrees)
            
        Examples
        --------
        >>> # Find angular range with periodic boundary handling
        >>> pipeline.analysis.features.torsions.range()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_range(data)
    
    def cv(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coefficient of variation for each torsion angle.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Coefficient of variation for each torsion angle
            
        Examples
        --------
        >>> # Find most variable torsions (high CV)
        >>> pipeline.analysis.features.torsions.cv()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_cv(data)
    
    # === PER-FRAME METHODS (per time step) ===
    
    def mean_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute circular mean angle per frame across all torsions.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Circular mean angle per frame in degrees
            
        Examples
        --------
        >>> # Overall conformational state per frame
        >>> pipeline.analysis.features.torsions.mean_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mean_per_frame(data)
    
    def std_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute circular standard deviation per frame across all torsions.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Circular standard deviation per frame in degrees
            
        Examples
        --------
        >>> # Conformational disorder per frame
        >>> pipeline.analysis.features.torsions.std_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_std_per_frame(data)
    
    def variance_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute circular variance per frame across all torsions.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Circular variance per frame (0-1 scale)
            
        Examples
        --------
        >>> # Conformational variance per frame
        >>> pipeline.analysis.features.torsions.variance_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_variance_per_frame(data)
    
    def min_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum angle per frame across all torsions.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Minimum angle per frame
            
        Examples
        --------
        >>> # Most negative angle per frame
        >>> pipeline.analysis.features.torsions.min_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_min_per_frame(data)
    
    def max_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum angle per frame across all torsions.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Maximum angle per frame
            
        Examples
        --------
        >>> # Most positive angle per frame
        >>> pipeline.analysis.features.torsions.max_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_max_per_frame(data)
    
    def mad_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median absolute deviation per frame across all torsions.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Median absolute deviation per frame
            
        Examples
        --------
        >>> # Robust measure of conformational spread per frame
        >>> pipeline.analysis.features.torsions.mad_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mad_per_frame(data)
    
    def range_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute angular range per frame across all torsions with periodicity.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Angular range per frame
            
        Examples
        --------
        >>> # Conformational spread per frame
        >>> pipeline.analysis.features.torsions.range_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_range_per_frame(data)
    
    # === TRANSITIONS/DYNAMICS METHODS ===
    
    def transitions_lagtime(self, threshold: float = 30.0, lag_time: int = 1,
                           feature_selector: Optional[str] = None, 
                           traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute transitions with lag time for each torsion angle with periodic boundaries.
        
        Parameters
        ----------
        threshold : float, default=30.0
            Threshold for detecting transitions (in degrees)
        lag_time : int, default=1
            Number of frames to look ahead
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per torsion angle
            
        Examples
        --------
        >>> # Find dynamic torsions with 30° threshold
        >>> pipeline.analysis.features.torsions.transitions_lagtime(threshold=30.0)
        
        >>> # Slower dynamics with longer lag time
        >>> pipeline.analysis.features.torsions.transitions_lagtime(
        ...     threshold=45.0, lag_time=10
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_lagtime(data, threshold, lag_time)
    
    def transitions_window(self, threshold: float = 30.0, window_size: int = 10,
                          feature_selector: Optional[str] = None, 
                          traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute transitions within sliding window for each torsion angle with periodic boundaries.
        
        Parameters
        ----------
        threshold : float, default=30.0
            Threshold for detecting transitions (in degrees)
        window_size : int, default=10
            Size of sliding window
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per torsion angle
            
        Examples
        --------
        >>> # Window-based transition detection
        >>> pipeline.analysis.features.torsions.transitions_window(
        ...     threshold=30.0, window_size=10
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_window(data, threshold, window_size)
    
    def stability(self, threshold: float = 30.0, window_size: int = 10, mode: str = "lagtime",
                 feature_selector: Optional[str] = None, 
                 traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute stability (inverse of transition rate) for each torsion angle with periodic boundaries.
        
        Parameters
        ----------
        threshold : float, default=30.0
            Threshold for stability detection (in degrees)
        window_size : int, default=10
            Window size for calculation
        mode : str, default='lagtime'
            Calculation mode ('lagtime' or 'window')
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Stability values per torsion angle (0=unstable, 1=stable)
            
        Examples
        --------
        >>> # Find most stable torsions
        >>> pipeline.analysis.features.torsions.stability(
        ...     threshold=30.0, window_size=10, mode='window'
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_stability(data, threshold, window_size, mode)
    
    # === COMPARISON METHODS ===
    
    def differences(self, frame_1: int = 0, frame_2: int = -1,
                   feature_selector: Optional[str] = None, 
                   traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute angle differences between two frames with periodic boundary handling.
        
        Parameters
        ----------
        frame_1 : int, default=0
            First frame index
        frame_2 : int, default=-1
            Second frame index (-1 for last frame)
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Angle differences between frames with proper periodic handling
            
        Examples
        --------
        >>> # Compare first and last frames
        >>> pipeline.analysis.features.torsions.differences(frame_1=0, frame_2=-1)
        
        >>> # Compare specific frames
        >>> pipeline.analysis.features.torsions.differences(frame_1=100, frame_2=200)
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_differences(data, frame_1, frame_2)
    
    def differences_mean(self, feature_selector2: Optional[str] = None, 
                        traj_selection2: Optional[Union[str, int, List]] = None,
                        feature_selector: Optional[str] = None, 
                        traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute differences between circular means of two datasets.
        
        Parameters
        ----------
        feature_selector2 : str, optional
            Second feature selector for comparison
        traj_selection2 : str, int, list, optional
            Second trajectory selection for comparison
        feature_selector : str, optional
            First feature selector for column selection
        traj_selection : str, int, list, optional
            First trajectory selection for row selection
            
        Returns
        -------
        np.ndarray
            Circular mean angle differences between datasets
            
        Examples
        --------
        >>> # Compare mean conformations between conditions
        >>> pipeline.analysis.features.torsions.differences_mean(
        ...     traj_selection=[0, 1],      # Native state
        ...     traj_selection2=[2, 3]      # Denatured state
        ... )
        """
        data1 = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        data2 = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector2, traj_selection2
        )
        return self._calculator.compute_differences_mean(data1, data2)
    