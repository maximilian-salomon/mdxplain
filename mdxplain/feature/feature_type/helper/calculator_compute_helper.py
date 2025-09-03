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
Common helper methods for calculator dynamic value computation.

Provides common functionality for dynamic value computation across different
calculators (ContactCalculator, DistanceCalculator, etc.) with memory-mapped
array support and statistical filtering capabilities.
"""

import warnings
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
from tqdm import tqdm

from .feature_shape_helper import FeatureShapeHelper


class CalculatorComputeHelper:
    """
    Static utility class for dynamic feature filtering and selection operations.

    Provides common functionality for filtering features based on statistical
    criteria with memory-mapped array support. Used by distance and contact
    calculators for dynamic value computation.
    """

    @staticmethod
    def compute_dynamic_values(
        data: np.ndarray,
        metric_values: np.ndarray,
        metric_name: str,
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        feature_metadata: Optional[List[Any]] = None,
        use_memmap: bool = False,
        output_path: Optional[str] = None,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Filter feature data based on statistical metric thresholds.

        Parameters:
        -----------
        data : np.ndarray
            Feature data array to filter
        metric_values : np.ndarray
            Pre-computed statistical values for filtering
        metric_name : str
            Name of the metric for reporting
        threshold_min : float, optional
            Minimum threshold (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold (metric_values <= threshold_max)
        feature_metadata : list, optional
            Feature metadata
        use_memmap : bool, default=False
            Whether to use memory mapping for output
        output_path : str, optional
            Path for memory-mapped output file
        chunk_size : int, default=1000
            Chunk size for processing

        Returns:
        --------
        dict
            Dictionary with keys: 'indices', 'values', 'dynamic_data',
            'feature_names', 'metric_used', 'n_dynamic', 'total_pairs'

        Examples:
        --------
        # High values only (> 0.5)
        result = compute_dynamic_values(data, values, 'cv', threshold_min=0.5)

        # Range filtering (0.2 <= values <= 0.8)
        result = compute_dynamic_values(data, values, 'cv', threshold_min=0.2, threshold_max=0.8)

        # Low values only (< 0.3)
        result = compute_dynamic_values(data, values, 'cv', threshold_max=0.3)
        """
        # Create filter mask
        mask = CalculatorComputeHelper._create_threshold_mask(
            metric_values, threshold_min, threshold_max
        )

        n_selected = np.sum(mask.flatten())
        CalculatorComputeHelper._validate_selection_results(
            n_selected, metric_name, threshold_min, threshold_max
        )

        # Handle feature names and extract data
        feature_info = CalculatorComputeHelper._handle_feature_metadata(
            feature_metadata, mask
        )
        dynamic_data = CalculatorComputeHelper._extract_dynamic_data(
            data, mask, use_memmap, output_path, chunk_size
        )

        return {
            "indices": np.where(mask),
            "values": metric_values[mask],
            "dynamic_data": dynamic_data,
            "feature_names": feature_info,
            "metric_used": metric_name,
            "n_dynamic": n_selected,
            "total_pairs": mask.size,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
        }

    @staticmethod
    def _create_threshold_mask(metric_values: np.ndarray, threshold_min: Optional[float], threshold_max: Optional[float]) -> np.ndarray:
        """
        Create boolean mask based on threshold conditions.

        Parameters:
        -----------
        metric_values : np.ndarray
            Pre-computed statistical values for filtering
        threshold_min : float, optional
            Minimum threshold (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold (metric_values <= threshold_max)

        Returns:
        --------
        np.ndarray
            Boolean mask
        """
        if threshold_min is None and threshold_max is None:
            raise ValueError(
                "At least one of 'threshold_min' or 'threshold_max' must be provided"
            )

        mask = np.ones(metric_values.shape, dtype=bool)

        if threshold_min is not None:
            mask = mask & (metric_values >= threshold_min)

        if threshold_max is not None:
            mask = mask & (metric_values <= threshold_max)

        return mask

    @staticmethod
    def _validate_selection_results(
        n_selected: int, 
        metric_name: str, 
        threshold_min: Optional[float], 
        threshold_max: Optional[float]
    ) -> None:
        """
        Validate that selection found some results.

        Parameters:
        -----------
        n_selected : int
            Number of selected values
        metric_name : str
            Name of the metric for reporting
        threshold_min : float, optional
            Minimum threshold (metric_values >= threshold_min)
        threshold_max : float, optional
            Maximum threshold (metric_values <= threshold_max)

        Returns:
        --------
        None
            Raises warning if no values found within thresholds
        """
        if n_selected == 0:
            threshold_desc = f"min={threshold_min}, max={threshold_max}"
            warnings.warn(
                f"No values found within the specified threshold criteria. "
                f"Metric: {metric_name}, Thresholds: {threshold_desc}"
            )

    @staticmethod
    def _handle_feature_metadata(feature_metadata: Optional[List[Any]], mask: np.ndarray) -> Optional[List[Any]]:
        """
        Extract feature names based on filter mask.

        Parameters:
        -----------
        feature_metadata : list or None
            Original feature metadata
        mask : np.ndarray
            Boolean filter mask

        Returns:
        --------
        np.ndarray or np.ndarray
            Filtered feature names or mask if names invalid
        """
        if feature_metadata is not None:
            if len(feature_metadata) != mask.size:
                warnings.warn(
                    f"feature_names length ({len(feature_metadata)}) doesn't match "
                    f"mask size ({mask.size}). Returning mask instead."
                )
                return mask
            else:
                return np.array(feature_metadata)[mask.flatten()]
        return mask

    @staticmethod
    def _extract_dynamic_data(data: np.ndarray, mask: np.ndarray, use_memmap: bool, output_path: Optional[str], chunk_size: int) -> np.ndarray:
        """
        Extract filtered data using boolean mask.

        Parameters:
        -----------
        data : np.ndarray
            Original feature data
        mask : np.ndarray
            Boolean filter mask
        use_memmap : bool
            Whether to use memory mapping
        output_path : str or None
            Path for memory-mapped output
        chunk_size : int
            Chunk size for processing

        Returns:
        --------
        np.ndarray
            Filtered data array
        """
        if use_memmap:
            if output_path is None:
                raise ValueError("output_path must be provided when use_memmap=True")

            n_selected = np.sum(mask.flatten())
            dynamic_data = np.memmap(
                output_path,
                dtype=data.dtype,
                mode="w+",
                shape=(data.shape[0], n_selected),
            )
            CalculatorComputeHelper._fill_memmap_data(
                data, dynamic_data, mask, chunk_size
            )
        else:
            # Extract based on data format
            if len(data.shape) == 3:  # Square format
                indices = np.where(mask)
                dynamic_data = data[:, indices[0], indices[1]]
            else:  # Condensed format or 2D
                data_flat = data.reshape(data.shape[0], -1)
                dynamic_data = data_flat[:, mask.flatten()]

        return dynamic_data

    @staticmethod
    def _fill_memmap_data(data: np.ndarray, dynamic_data: np.ndarray, mask: np.ndarray, chunk_size: int) -> None:
        """
        Fill memory-mapped array with filtered data.

        Parameters:
        -----------
        data : np.ndarray
            Original feature data
        dynamic_data : np.memmap
            Memory-mapped output array
        mask : np.ndarray
            Boolean filter mask
        chunk_size : int
            Chunk size for processing

        Returns:
        --------
        None
            Fills dynamic_data array in-place
        """
        is_square_format = len(data.shape) == 3
        is_memmap_data = FeatureShapeHelper.is_memmap(data)

        if is_memmap_data:
            CalculatorComputeHelper._fill_memmap_chunked(
                data, dynamic_data, mask, chunk_size, is_square_format
            )
        else:
            CalculatorComputeHelper._fill_regular_chunked(
                data, dynamic_data, mask, chunk_size, is_square_format
            )

    @staticmethod
    def _fill_memmap_chunked(data: np.ndarray, dynamic_data: np.ndarray, mask: np.ndarray, chunk_size: int, is_square_format: bool) -> None:
        """
        Fill memory-mapped data in chunks.

        Parameters:
        -----------
        data : np.ndarray
            Original feature data
        dynamic_data : np.memmap
            Memory-mapped output array
        mask : np.ndarray
            Boolean filter mask
        chunk_size : int
            Chunk size for processing
        is_square_format : bool
            Whether the data is in square format

        Returns:
        --------
        None
            Fills dynamic_data array in-place
        """
        for i in tqdm(range(0, data.shape[0], chunk_size), 
                      desc="Computing dynamic values", unit="chunks"):
            end_idx = min(i + chunk_size, data.shape[0])
            chunk = data[i:end_idx]
            CalculatorComputeHelper._process_chunk(
                chunk, dynamic_data, mask, i, end_idx, is_square_format
            )

    @staticmethod
    def _fill_regular_chunked(data: np.ndarray, dynamic_data: np.ndarray, mask: np.ndarray, chunk_size: int, is_square_format: bool) -> None:
        """
        Fill regular data in chunks.

        Parameters:
        -----------
        data : np.ndarray
            Original feature data
        dynamic_data : np.ndarray
            Memory-mapped output array
        mask : np.ndarray
            Boolean filter mask
        chunk_size : int
            Chunk size for processing
        is_square_format : bool
            Whether the data is in square format

        Returns:
        --------
        None
            Fills dynamic_data array in-place
        """
        if is_square_format:
            indices = np.where(mask)
            for i in tqdm(range(0, data.shape[0], chunk_size),
                          desc="Extracting pair data", unit="chunks"):
                end_idx = min(i + chunk_size, data.shape[0])
                dynamic_data[i:end_idx] = data[i:end_idx, indices[0], indices[1]]
        else:
            data_flat = data.reshape(data.shape[0], -1)
            for i in tqdm(range(0, data.shape[0], chunk_size),
                          desc="Extracting mask data", unit="chunks"):
                end_idx = min(i + chunk_size, data.shape[0])
                dynamic_data[i:end_idx] = data_flat[i:end_idx, mask.flatten()]

    @staticmethod
    def _process_chunk(chunk: np.ndarray, dynamic_data: np.ndarray, mask: np.ndarray, start_idx: int, end_idx: int, is_square_format: bool) -> None:
        """
        Process a single chunk of data.

        Parameters:
        -----------
        chunk : np.ndarray
            Chunk of feature data
        dynamic_data : np.ndarray
            Memory-mapped output array
        mask : np.ndarray
            Boolean filter mask
        start_idx : int
            Start index of the chunk
        end_idx : int
            End index of the chunk
        is_square_format : bool
            Whether the data is in square format

        Returns:
        --------
        None
            Fills dynamic_data array in-place
        """
        if is_square_format:
            indices = np.where(mask)
            dynamic_data[start_idx:end_idx] = chunk[:, indices[0], indices[1]]
        else:
            chunk_flat = chunk.reshape(chunk.shape[0], -1)
            dynamic_data[start_idx:end_idx] = chunk_flat[:, mask.flatten()]

    @staticmethod
    def calculate_output_dimensions(input_shape: Tuple[int, ...], squareform: bool, k: Optional[int]) -> Tuple[int, ...]:
        """
        Calculate output array dimensions after format conversion.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input array
        squareform : bool
            Whether output should be square format
        k : int
            Diagonal offset parameter

        Returns:
        --------
        tuple[tuple, int or None]
            Output shape and number of residues
        """
        is_square_input = len(input_shape) == 3

        if not CalculatorComputeHelper._needs_format_conversion(
            is_square_input, squareform
        ):
            return input_shape, None

        return CalculatorComputeHelper._convert_dimensions(
            input_shape, is_square_input, squareform, k
        )

    @staticmethod
    def _needs_format_conversion(is_square_input: bool, squareform: bool) -> bool:
        """
        Check if format conversion is needed.

        Parameters:
        -----------
        is_square_input : bool
            Whether the input is in square format
        squareform : bool
            Whether output should be square format

        Returns:
        --------
        bool
            True if format conversion is needed
        """
        return (is_square_input and not squareform) or (
            not is_square_input and squareform
        )

    @staticmethod
    def _convert_dimensions(input_shape: Tuple[int, ...], is_square_input: bool, squareform: bool, k: Optional[int]) -> Tuple[int, ...]:
        """
        Convert dimensions based on input and output format.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input array
        is_square_input : bool
            Whether the input is in square format
        squareform : bool
            Whether output should be square format
        k : int
            Diagonal offset parameter

        Returns:
        --------
        tuple[tuple, int or None]
            Output shape and number of residues
        """
        if is_square_input and not squareform:
            return CalculatorComputeHelper._square_to_condensed_dims(input_shape, k)
        else:
            return CalculatorComputeHelper._condensed_to_square_dims(input_shape, k)

    @staticmethod
    def _square_to_condensed_dims(input_shape: Tuple[int, ...], k: Optional[int]) -> Tuple[int, ...]:
        """
        Convert square format dimensions to condensed format.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input array
        k : int
            Diagonal offset parameter

        Returns:
        --------
        tuple[tuple, int]
            Output shape and number of residues
        """
        n_residues = input_shape[1]
        if k == 0:
            n_contacts = n_residues * (n_residues + 1) // 2
        else:
            n_contacts = n_residues * (n_residues - k) // 2 - sum(range(k))
        return (input_shape[0], n_contacts), n_residues

    @staticmethod
    def _condensed_to_square_dims(input_shape: Tuple[int, ...], k: Optional[int]) -> Tuple[int, ...]:
        """
        Convert condensed format dimensions to square format.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input array
        k : int
            Diagonal offset parameter

        Returns:
        --------
        tuple[tuple, int]
            Output shape and number of residues
        """
        n_contacts = input_shape[1]
        n_residues = k + int((-1 + np.sqrt(1 + 8 * n_contacts)) / 2)
        return (input_shape[0], n_residues, n_residues), n_residues

    @staticmethod
    def create_output_array(use_memmap: bool, path: str, output_shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """
        Create output array (regular or memory-mapped).

        Parameters:
        -----------
        use_memmap : bool
            Whether to use memory mapping
        path : str or None
            File path for memory-mapped array
        output_shape : tuple
            Shape of output array
        dtype : np.dtype
            Data type of array

        Returns:
        --------
        numpy.ndarray or numpy.memmap
            Created output array
        """
        if use_memmap:
            return np.memmap(path, dtype=dtype, mode="w+", shape=output_shape)
        else:
            return np.zeros(output_shape, dtype=dtype)
