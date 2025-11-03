# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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

"""
Abstract base class for feature calculator implementations.

Defines the common interface that all calculator classes must implement
to ensure consistency across different feature calculators like distances,
contacts, and other molecular dynamics features.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np


class CalculatorBase(ABC):
    """
    Abstract base class for all feature calculator implementations.

    Defines the common interface that all calculator classes must implement
    to ensure consistency across different feature calculators like distances,
    contacts, and other molecular dynamics features.

    All calculators support memory mapping for efficient handling of large
    datasets and provide statistical analysis capabilities.

    Examples
    --------
    >>> class MyCalculator(CalculatorBase):
    ...     def compute(self, input_data, **kwargs):
    ...         # Implementation here
    ...         return data, feature_names
    ...     def compute_dynamic_values(self, input_data, metric, **kwargs):
    ...         # Implementation here
    ...         return {"filtered_data": data, "statistics": stats}
    """

    def __init__(
        self,
        use_memmap: bool = False,
        cache_path: Optional[str] = "./cache",
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize calculator with common configuration parameters.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for efficient handling of large datasets
        cache_path : str, optional
            Directory path for storing cache files when using memory mapping
        chunk_size : int, optional
            Number of frames to process in each chunk (None for automatic sizing)

        Returns
        -------
        None
        """
        self.use_memmap = use_memmap
        self.cache_path = cache_path
        self.chunk_size = chunk_size

        # Analysis object will be set by subclasses
        self.analysis = None

    @abstractmethod
    def compute(self, input_data: Any, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Compute feature data from input data (must be implemented by subclasses).

        Parameters
        ----------
        input_data : Any
            Input data for computation (trajectories for distances, distances for contacts, etc.)
        kwargs : dict
            Additional calculator-specific parameters

        Returns
        -------
        Tuple[np.ndarray, dict]
            Tuple containing (computed_data, feature_metadata) where computed_data
            is the calculated feature matrix and feature_metadata is structured metadata
        """
        pass

    @abstractmethod
    def compute_dynamic_values(
        self,
        input_data: np.ndarray,
        metric: str,
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        feature_metadata: Optional[list] = None,
        output_path: Optional[str] = None,
        transition_threshold: Optional[float] = None,
        window_size: Optional[int] = None,
        transition_mode: Optional[str] = None,
        lag_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute statistical analysis and filter features based on dynamic criteria.

        Parameters
        ----------
        input_data : np.ndarray
            Feature data matrix to analyze (frames x features)
        metric : str
            Statistical metric to use ('std', 'var', 'range', 'transitions', etc.)
        threshold_min : float, optional
            Minimum threshold value for filtering features
        threshold_max : float, optional
            Maximum threshold value for filtering features
        feature_metadata : list, optional
            Feature metadata corresponding to data columns
        output_path : str, optional
            Path for saving memory-mapped filtered results
        transition_threshold : float, optional
            Threshold for detecting state transitions in features
        window_size : int, optional
            Window size for transition analysis
        transition_mode : str, optional
            Mode for transition computation ('window' or 'lagtime')
        lag_time : int, optional
            Lag time for transition rate calculations

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys: 'filtered_data', 'selected_indices',
            'statistics', 'feature_names', and analysis metadata
        """
        pass
