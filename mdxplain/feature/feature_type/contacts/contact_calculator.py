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
Contact calculator for molecular dynamics trajectory analysis.

Utility class for computing contact maps from distance arrays using distance cutoffs.
Supports memory mapping for large datasets and provides statistical analysis capabilities.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm

from ..helper.calculator_compute_helper import CalculatorComputeHelper
from ..interfaces.calculator_base import CalculatorBase
from .contact_calculator_analysis import ContactCalculatorAnalysis


class ContactCalculator(CalculatorBase):
    """
    Calculator for computing binary contact maps from distance arrays.

    Converts distance data into binary contact matrices by applying distance
    cutoffs. Supports both memory-mapped arrays for large datasets and various
    output formats (square/condensed). Includes statistical analysis capabilities
    for contact frequency, stability, and transition analysis.

    Examples
    --------
    >>> # Basic contact calculation
    >>> calculator = ContactCalculator()
    >>> contacts = calculator.compute(distance_data, cutoff=4.0)

    >>> # With memory mapping
    >>> calculator = ContactCalculator(use_memmap=True, cache_path='./cache/')
    >>> contacts = calculator.compute(distance_data, cutoff=3.5)
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize contact calculator with configuration parameters.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Directory path for storing cache files
        chunk_size : int, optional
            Number of frames to process per chunk

        Returns
        -------
        None

        Examples
        --------
        >>> # Basic initialization
        >>> calculator = ContactCalculator()

        >>> # With memory mapping
        >>> calculator = ContactCalculator(use_memmap=True, cache_path='./cache/')
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self.contacts_path = cache_path
        self.chunk_size = chunk_size
        self.use_memmap = use_memmap

        self.analysis = ContactCalculatorAnalysis(
            use_memmap=self.use_memmap, chunk_size=self.chunk_size
        )

    # ===== MAIN COMPUTATION METHOD =====

    def compute(self, input_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute binary contact maps from distance arrays using distance cutoff.

        Parameters
        ----------
        input_data : numpy.ndarray
            Distance array in condensed format (NxP) in Angstrom
        **kwargs : dict
            Additional parameters:
            - cutoff : float, default=4.5 - Distance cutoff for contact determination

        Returns
        -------
        numpy.ndarray
            Binary contact array where True/1 indicates contact (distance <= cutoff)

        Examples
        --------
        >>> # Basic contact calculation
        >>> contacts = calculator.compute(distance_data, cutoff=4.0)

        >>> # With custom cutoff
        >>> contacts = calculator.compute(distance_data, cutoff=3.5)
        """
        # Extract parameters from kwargs
        distances = input_data
        cutoff = kwargs.get("cutoff", 4.5)

        # Create output array with same shape as input (condensed format)
        contacts = CalculatorComputeHelper.create_output_array(
            self.use_memmap, self.contacts_path, distances.shape, dtype=bool
        )

        # Process in chunks
        if self.chunk_size is None:
            self.chunk_size = distances.shape[0]

        for i in tqdm(range(0, distances.shape[0], self.chunk_size), desc="Computing contacts", unit="chunks"):
            end_idx = min(i + self.chunk_size, distances.shape[0])
            contacts[i:end_idx] = distances[i:end_idx] <= cutoff

        return contacts

    def _compute_metric_values(
        self,
        contacts: np.ndarray,
        metric: str,
        threshold: float,
        window_size: int,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> np.ndarray:
        """
        Compute statistical metric values for contact filtering.

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array
        metric : str
            Metric type ('frequency', 'stability', 'transitions')
        threshold : float
            Threshold for transition detection
        window_size : int
            Window size for analysis
        transition_mode : str, default='window'
            Transition analysis mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for analysis

        Returns
        -------
        numpy.ndarray
            Computed metric values per contact pair

        Raises
        ------
        ValueError
            If the metric is not supported
        """
        # Define simple metrics mapping
        simple_metrics = {
            "frequency": self.analysis.compute_frequency,
            "stability": self.analysis.compute_stability,
        }

        if metric in simple_metrics:
            return simple_metrics[metric](contacts)
        if metric == "transitions":
            return self._compute_transitions_metric(
                contacts, threshold, window_size, transition_mode, lag_time
            )

        supported_metrics = ["frequency", "stability", "transitions"]
        raise ValueError(f"Unknown metric: {metric}. Supported: {supported_metrics}")

    def _compute_transitions_metric(
        self, contacts: np.ndarray, threshold: float, window_size: int, transition_mode: str, lag_time: int
    ) -> int:
        """
        Compute transitions metric based on specified mode and parameters.

        Parameters
        ----------
        contacts : numpy.ndarray
            Binary contact array
        threshold : float
            Threshold for transition detection
        window_size : int
            Window size for analysis
        transition_mode : str, default='window'
            Transition analysis mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for analysis

        Returns
        -------
        int
            Number of transitions
        """
        if transition_mode == "window":
            return self.analysis.compute_transitions_window(
                contacts, threshold=threshold, window_size=window_size
            )
        if transition_mode == "lagtime":
            return self.analysis.compute_transitions_lagtime(
                contacts, threshold=threshold, lag_time=lag_time
            )
        raise ValueError(
            f"Unknown transition mode: {transition_mode}. Supported: 'window', 'lagtime'"
        )

    def compute_dynamic_values(
        self,
        input_data: np.ndarray,
        metric: str = "frequency",
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        feature_metadata: Optional[list] = None,
        output_path: Optional[str] = None,
        transition_threshold: float = 2.0,
        window_size: int = 10,
        transition_mode: str = "window",
        lag_time: int = 1,
    ) -> Dict[str, Any]:
        """
        Filter and select dynamic contact pairs based on statistical criteria.

        Parameters
        ----------
        input_data : numpy.ndarray
            Binary contact array to analyze (n_frames x n_pairs)
        metric : str, default='frequency'
            Statistical metric for filtering contacts:
            - 'frequency': Contact frequency (fraction of frames in contact)
            - 'stability': Contact stability (persistence over time)
            - 'transitions': Number of contact formation/breaking events
        threshold_min : float, optional
            Minimum threshold for filtering (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold for filtering (metric_values <= threshold_max)
        transition_threshold : float, default=2.0
            Threshold for detecting contact transitions (only for 'transitions' metric)
        window_size : int, default=10
            Window size for transition analysis
        feature_metadata : list, optional
            Contact pair metadata corresponding to data columns
        output_path : str, optional
            Path for memory-mapped filtered output
        transition_mode : str, default='window'
            Transition analysis mode ('window' or 'lagtime')
        lag_time : int, default=1
            Lag time for transition analysis (when transition_mode='lagtime')

        Returns
        -------
        dict
            Dictionary with keys: 'indices', 'values', 'dynamic_data', 'feature_metadata',
            'metric_used', 'n_dynamic', 'total_pairs', 'threshold_min', 'threshold_max'

        Examples
        --------
        >>> # Select frequently formed contacts (> 50% frequency)
        >>> result = calculator.compute_dynamic_values(
        ...     contact_data, metric='frequency', threshold_min=0.5
        ... )
        >>> frequent_contacts = result['dynamic_data']
        >>> print(f"Found {result['n_dynamic']} frequent contacts")

        >>> # Select moderately stable contacts (20% <= frequency <= 80%)
        >>> result = calculator.compute_dynamic_values(
        ...     contact_data, metric='frequency',
        ...     threshold_min=0.2, threshold_max=0.8
        ... )

        >>> # Select highly dynamic contacts (many transitions)
        >>> result = calculator.compute_dynamic_values(
        ...     contact_data, metric='transitions', threshold_min=10,
        ...     transition_threshold=1, window_size=5
        ... )
        >>> dynamic_contacts = result['dynamic_data']
        """
        # Compute metric values using helper method
        metric_values = self._compute_metric_values(
            input_data,
            metric,
            transition_threshold,
            window_size,
            transition_mode,
            lag_time,
        )

        # Use the common helper
        return CalculatorComputeHelper.compute_dynamic_values(
            data=input_data,
            metric_values=metric_values,
            metric_name=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            feature_metadata=feature_metadata,
            use_memmap=self.use_memmap,
            output_path=output_path,
            chunk_size=self.chunk_size,
        )
