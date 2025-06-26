# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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

import numpy as np

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
        data,
        metric_values,
        metric_name,
        threshold_min=None,
        threshold_max=None,
        feature_names=None,
        use_memmap=False,
        output_path=None,
        chunk_size=1000,
    ):
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
        feature_names : list, optional
            Feature pair names
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
        # Create mask based on threshold conditions
        mask = np.ones(metric_values.shape, dtype=bool)

        if threshold_min is not None:
            mask = mask & (metric_values >= threshold_min)

        if threshold_max is not None:
            mask = mask & (metric_values <= threshold_max)

        if threshold_min is None and threshold_max is None:
            raise ValueError(
                "At least one of 'threshold_min' or 'threshold_max' must be provided"
            )

        n_selected = np.sum(mask.flatten())

        if n_selected == 0:
            threshold_desc = f"min={threshold_min}, max={threshold_max}"
            warnings.warn(
                f"No values found within the specified threshold criteria. "
                f"Metric: {metric_name}, Thresholds: {threshold_desc}"
            )

        # Handle feature names
        feature_info = CalculatorComputeHelper._handle_feature_names(
            feature_names, mask
        )

        # Extract dynamic data
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
    def _handle_feature_names(feature_names, mask):
        """
        Extract feature names based on filter mask.

        Parameters:
        -----------
        feature_names : list or None
            Original feature names
        mask : np.ndarray
            Boolean filter mask

        Returns:
        --------
        np.ndarray or np.ndarray
            Filtered feature names or mask if names invalid
        """
        if feature_names is not None:
            if len(feature_names) != mask.size:
                warnings.warn(
                    f"feature_names length ({len(feature_names)}) doesn't match "
                    f"mask size ({mask.size}). Returning mask instead."
                )
                return mask
            else:
                return np.array(feature_names)[mask.flatten()]
        return mask

    @staticmethod
    def _extract_dynamic_data(data, mask, use_memmap, output_path, chunk_size):
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
                raise ValueError(
                    "output_path must be provided when use_memmap=True")

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
    def _fill_memmap_data(data, dynamic_data, mask, chunk_size):
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
        if FeatureShapeHelper.is_memmap(data):
            for i in range(0, data.shape[0], chunk_size):
                end_idx = min(i + chunk_size, data.shape[0])
                chunk = data[i:end_idx]
                if len(chunk.shape) == 3:  # Square format
                    indices = np.where(mask)
                    dynamic_data[i:end_idx] = chunk[:, indices[0], indices[1]]
                else:  # Condensed format
                    chunk_flat = chunk.reshape(chunk.shape[0], -1)
                    dynamic_data[i:end_idx] = chunk_flat[:, mask.flatten()]
        else:
            if len(data.shape) == 3:  # Square format
                indices = np.where(mask)
                for i in range(0, data.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, data.shape[0])
                    dynamic_data[i:end_idx] = data[i:end_idx,
                                                   indices[0], indices[1]]
            else:  # Condensed format
                data_flat = data.reshape(data.shape[0], -1)
                for i in range(0, data.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, data.shape[0])
                    dynamic_data[i:end_idx] = data_flat[i:end_idx,
                                                        mask.flatten()]

    @staticmethod
    def calculate_output_dimensions(input_shape, squareform, k):
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
        needs_conversion = (len(input_shape) == 3 and not squareform) or (
            len(input_shape) == 2 and squareform
        )

        if needs_conversion:
            if len(input_shape) == 3 and not squareform:
                # Square to condensed
                n_residues = input_shape[1]
                if k == 0:
                    n_contacts = n_residues * (n_residues + 1) // 2
                else:
                    n_contacts = n_residues * \
                        (n_residues - k) // 2 - sum(range(k))
                output_shape = (input_shape[0], n_contacts)
            else:
                # Condensed to square
                n_contacts = input_shape[1]
                n_residues = k + int((-1 + np.sqrt(1 + 8 * n_contacts)) / 2)
                output_shape = (input_shape[0], n_residues, n_residues)
        else:
            output_shape = input_shape
            n_residues = n_contacts = None

        return output_shape, n_residues

    @staticmethod
    def create_output_array(use_memmap, path, output_shape, dtype):
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
