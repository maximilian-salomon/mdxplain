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

"""Factory for SASA-specific analysis operations."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING
import numpy as np

from ....services.helpers.analysis_data_helper import AnalysisDataHelper
from ..sasa_calculator_analysis import SASACalculatorAnalysis
from ...interfaces.analysis_service_base import AnalysisServiceBase

if TYPE_CHECKING:
    from .....pipeline.entities.pipeline_data import PipelineData


class SASAAnalysisService(AnalysisServiceBase):
    """
    Service for SASA-specific analysis operations.
    
    Provides a simplified interface for SASA analysis methods by automatically
    handling data selection and forwarding method calls to the underlying
    calculator_analysis class. All methods from SASACalculatorAnalysis
    are automatically available with additional feature_selector and 
    traj_selection parameters.
    """
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """
        Initialize SASA analysis service.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data containing feature data and configuration

        Returns
        -------
        None
        """
        super().__init__(pipeline_data)
        self._feature_type = "sasa"
        self._calculator = SASACalculatorAnalysis(
            use_memmap=pipeline_data.use_memmap,
            chunk_size=pipeline_data.chunk_size
        )
    
    # === PER-FEATURE METHODS (per residue/atom) ===
    
    def mean(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean SASA for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Mean SASA for each residue/atom
            
        Examples
        --------
        >>> # Compute mean SASA for all residues
        >>> pipeline.analysis.features.sasa.mean()
        
        >>> # Compute mean for selected residues
        >>> pipeline.analysis.features.sasa.mean(feature_selector="surface_residues")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mean(data)
    
    def std(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation of SASA for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Standard deviation for each residue/atom
            
        Examples
        --------
        >>> # Find most variable surface areas
        >>> pipeline.analysis.features.sasa.std()
        
        >>> # Variable exposure in selected region
        >>> pipeline.analysis.features.sasa.std(feature_selector="binding_site")
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_std(data)
    
    def variance(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute variance of SASA for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Variance for each residue/atom
            
        Examples
        --------
        >>> # Compute SASA variance
        >>> pipeline.analysis.features.sasa.variance()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_variance(data)
    
    def min(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum SASA for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Minimum SASA for each residue/atom
            
        Examples
        --------
        >>> # Find most buried state for each residue
        >>> pipeline.analysis.features.sasa.min()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_min(data)
    
    def max(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum SASA for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Maximum SASA for each residue/atom
            
        Examples
        --------
        >>> # Find most exposed state for each residue
        >>> pipeline.analysis.features.sasa.max()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_max(data)
    
    def mad(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median absolute deviation of SASA for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Median absolute deviation for each residue/atom
            
        Examples
        --------
        >>> # Robust measure of SASA variability
        >>> pipeline.analysis.features.sasa.mad()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mad(data)
    
    def range(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute SASA range for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Range (max - min) for each residue/atom
            
        Examples
        --------
        >>> # Find residues with largest burial/exposure changes
        >>> pipeline.analysis.features.sasa.range()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_range(data)
    
    def cv(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute coefficient of variation for each residue/atom.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Coefficient of variation for each residue/atom
            
        Examples
        --------
        >>> # Find most variable residues (high CV)
        >>> pipeline.analysis.features.sasa.cv()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_cv(data)
    
    def burial_fraction(self, threshold: float = 0.1, feature_selector: Optional[str] = None, 
                       traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute burial fraction for each residue/atom.
        
        Parameters
        ----------
        threshold : float, default=0.1
            SASA threshold below which residue is considered buried
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Fraction of time each residue is buried (below threshold)
            
        Examples
        --------
        >>> # Find frequently buried residues (< 0.1 nm²)
        >>> pipeline.analysis.features.sasa.burial_fraction(threshold=0.1)
        
        >>> # More restrictive burial criteria
        >>> pipeline.analysis.features.sasa.burial_fraction(threshold=0.05)
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_burial_fraction(data, threshold)
    
    def exposure_fraction(self, threshold: float = 1.0, feature_selector: Optional[str] = None, 
                         traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute exposure fraction for each residue/atom.
        
        Parameters
        ----------
        threshold : float, default=1.0
            SASA threshold above which residue is considered exposed
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Fraction of time each residue is exposed (above threshold)
            
        Examples
        --------
        >>> # Find frequently exposed residues (> 1.0 nm²)
        >>> pipeline.analysis.features.sasa.exposure_fraction(threshold=1.0)
        
        >>> # Higher exposure threshold
        >>> pipeline.analysis.features.sasa.exposure_fraction(threshold=2.0)
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_exposure_fraction(data, threshold)
    
    # === PER-FRAME METHODS (per time step) ===
    
    def mean_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute mean SASA per frame across all residues/atoms.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Mean SASA per frame
            
        Examples
        --------
        >>> # Overall surface exposure per frame
        >>> pipeline.analysis.features.sasa.mean_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mean_per_frame(data)
    
    def std_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute standard deviation per frame across all residues/atoms.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Standard deviation per frame
            
        Examples
        --------
        >>> # SASA heterogeneity per frame
        >>> pipeline.analysis.features.sasa.std_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_std_per_frame(data)
    
    def variance_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute variance per frame across all residues/atoms.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Variance per frame
            
        Examples
        --------
        >>> # SASA variance per frame
        >>> pipeline.analysis.features.sasa.variance_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_variance_per_frame(data)
    
    def min_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute minimum SASA per frame across all residues/atoms.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Minimum SASA per frame
            
        Examples
        --------
        >>> # Most buried residue per frame
        >>> pipeline.analysis.features.sasa.min_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_min_per_frame(data)
    
    def max_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute maximum SASA per frame across all residues/atoms.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Maximum SASA per frame
            
        Examples
        --------
        >>> # Most exposed residue per frame
        >>> pipeline.analysis.features.sasa.max_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_max_per_frame(data)
    
    def mad_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute median absolute deviation per frame across all residues/atoms.
        
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
        >>> # Robust measure of SASA spread per frame
        >>> pipeline.analysis.features.sasa.mad_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_mad_per_frame(data)
    
    def range_per_frame(self, feature_selector: Optional[str] = None, traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute SASA range per frame across all residues/atoms.
        
        Parameters
        ----------
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Range per frame
            
        Examples
        --------
        >>> # Surface exposure heterogeneity per frame
        >>> pipeline.analysis.features.sasa.range_per_frame()
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_range_per_frame(data)
    
    # === TRANSITIONS/DYNAMICS METHODS ===
    
    def transitions_lagtime(self, threshold: float = 0.5, lag_time: int = 1,
                           feature_selector: Optional[str] = None, 
                           traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute SASA transitions with lag time for each residue/atom.
        
        Parameters
        ----------
        threshold : float, default=0.5
            SASA change threshold for detecting transitions
        lag_time : int, default=1
            Number of frames to look ahead
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per residue/atom
            
        Examples
        --------
        >>> # Find dynamic surface areas with 0.5 nm² threshold
        >>> pipeline.analysis.features.sasa.transitions_lagtime(threshold=0.5)
        
        >>> # Slower dynamics with longer lag time
        >>> pipeline.analysis.features.sasa.transitions_lagtime(
        ...     threshold=1.0, lag_time=10
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_lagtime(data, threshold, lag_time)
    
    def transitions_window(self, threshold: float = 0.5, window_size: int = 10,
                          feature_selector: Optional[str] = None, 
                          traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute SASA transitions within sliding window for each residue/atom.
        
        Parameters
        ----------
        threshold : float, default=0.5
            SASA change threshold for detecting transitions
        window_size : int, default=10
            Size of sliding window
        feature_selector : str, optional
            Name of feature selector for column selection
        traj_selection : str, int, list, optional
            Trajectory selection criteria for row selection
            
        Returns
        -------
        np.ndarray
            Transition counts per residue/atom
            
        Examples
        --------
        >>> # Window-based transition detection
        >>> pipeline.analysis.features.sasa.transitions_window(
        ...     threshold=0.5, window_size=10
        ... )
        """
        data = AnalysisDataHelper.get_selected_data(
            self._pipeline_data, self._feature_type,
            feature_selector, traj_selection
        )
        return self._calculator.compute_transitions_window(data, threshold, window_size)
    
    def stability(self, threshold: float = 0.5, window_size: int = 10, mode: str = "lagtime",
                 feature_selector: Optional[str] = None, 
                 traj_selection: Optional[Union[str, int, List]] = None) -> np.ndarray:
        """
        Compute SASA stability (inverse of transition rate) for each residue/atom.
        
        Parameters
        ----------
        threshold : float, default=0.5
            SASA change threshold for stability detection
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
            Stability values per residue/atom (0=unstable, 1=stable)
            
        Examples
        --------
        >>> # Find most stable surface areas
        >>> pipeline.analysis.features.sasa.stability(
        ...     threshold=0.5, window_size=10, mode='window'
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
        Compute SASA differences between two frames.
        
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
            SASA differences between frames
            
        Examples
        --------
        >>> # Compare first and last frames
        >>> pipeline.analysis.features.sasa.differences(frame_1=0, frame_2=-1)
        
        >>> # Compare specific frames
        >>> pipeline.analysis.features.sasa.differences(frame_1=100, frame_2=200)
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
        Compute differences between mean SASA of two datasets.
        
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
            Mean SASA differences between datasets
            
        Examples
        --------
        >>> # Compare surface exposure between conditions
        >>> pipeline.analysis.features.sasa.differences_mean(
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
    