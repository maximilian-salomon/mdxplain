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

"""Base class for all feature type analysis services."""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....pipeline.entities.pipeline_data import PipelineData

from ...services.helpers.analysis_data_helper import AnalysisDataHelper


class AnalysisServiceBase:
    """
    Base class for all feature type analysis services.
    
    Provides common functionality for all feature analysis services including:

    - Automatic method forwarding via __getattr__ for calculator methods
    - Consistent parameter handling (feature_selector, traj_selection)
    - Data selection and validation logic
    
    Subclasses must set:
    
    - self._feature_type: String identifier for the feature type
    - self._calculator: Calculator instance with analysis methods
    
    Examples
    --------
    >>> class DistancesAnalysisService(AnalysisServiceBase):
    ...     def __init__(self, pipeline_data):
    ...         super().__init__(pipeline_data)
    ...         self._feature_type = "distances"
    ...         self._calculator = DistanceCalculatorAnalysis(...)
    ...         
    ...     # Optional explicit methods for VS Code autocompletion
    ...     def mean(self, feature_selector=None, traj_selection=None):
    ...         # Explicit implementation or delegate to __getattr__
    ...         return super().__getattr__('compute_mean')(feature_selector, traj_selection)
    """
    
    def __init__(self, pipeline_data: PipelineData) -> None:
        """
        Initialize base analysis service.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with all necessary data
            
        Returns
        -------
        None
        """
        self._pipeline_data = pipeline_data
        self._feature_type = None  # Must be set by subclass
        self._calculator = None     # Must be set by subclass
    
    def __getattr__(self, name: str):
        """
        Automatically forward method calls to the underlying calculator.
        
        This provides a fallback for any methods not explicitly defined
        in the service, allowing direct access to all calculator methods
        with automatic data selection handling.
        
        The wrapper automatically adds feature_selector and traj_selection
        parameters to all calculator methods, handling the common pattern
        of data selection before method execution.
        
        Parameters
        ----------
        name : str
            Name of the method being accessed
            
        Returns
        -------
        function
            Wrapped method that handles data selection and calls calculator
            
        Raises
        ------
        AttributeError
            If the method doesn't exist in the calculator
        """
        # Check if the method exists in the calculator and is callable
        if hasattr(self._calculator, name) and callable(getattr(self._calculator, name)):
            calculator_method = getattr(self._calculator, name)
            
            def wrapped_method(
                feature_selector: Optional[str] = None,
                traj_selection: Optional[Union[str, int, List]] = None,
                **kwargs
            ):
                """
                Wrapped calculator method with automatic data selection.
                
                Parameters
                ----------
                feature_selector : str, optional
                    Name of feature selector for column selection
                traj_selection : str, int, list, optional
                    Trajectory selection criteria for row selection
                kwargs
                    Additional arguments passed to the calculator method
                    
                Returns
                -------
                Any
                    Result from the calculator method
                """
                # Get selected data using helper
                data = AnalysisDataHelper.get_selected_data(
                    self._pipeline_data,
                    self._feature_type,
                    feature_selector,
                    traj_selection
                )
                # Call original calculator method with selected data
                return calculator_method(data, **kwargs)
            
            # Copy metadata from original method for better debugging
            wrapped_method.__name__ = name
            wrapped_method.__doc__ = calculator_method.__doc__
            
            return wrapped_method
        
        # Method doesn't exist in calculator
        available_methods = [
            method for method in dir(self._calculator) 
            if not method.startswith('_') and callable(getattr(self._calculator, method))
        ]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Available calculator methods: {available_methods}"
        )
