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

"""
Centroid calculation helper for trajectory analysis.

This module provides utilities for computing centroids (mean frames) and
finding frames closest to centroids. Used for representative frame selection
in clustering and other analyses.
"""

import numpy as np


class CentroidHelper:
    """
    Helper class for centroid calculations.

    Provides method to find frames closest to the centroid (mean)
    using memory-efficient chunked processing for large datasets.

    Examples
    --------
    >>> # Find centroid frame with fast numpy
    >>> best_idx = CentroidHelper.find_centroid(
    ...     selected_data, use_memmap=False
    ... )

    >>> # Find centroid frame with memmap-safe chunked processing
    >>> best_idx = CentroidHelper.find_centroid(
    ...     selected_data, use_memmap=True, chunk_size=1000
    ... )
    """

    @staticmethod
    def find_centroid(
        selected_data: np.ndarray,
        use_memmap: bool = False,
        chunk_size: int = 1000
    ) -> int:
        """
        Find frame closest to centroid (mean).

        Computes the centroid (mean) of all frames and finds the frame
        that minimizes Euclidean distance to it. Uses fast numpy operations
        for small datasets or chunked processing for large memmap datasets.

        Parameters
        ----------
        selected_data : np.ndarray
            Data array with shape (n_frames, n_features)
        use_memmap : bool, default=False
            Whether to use chunked processing for memmap
        chunk_size : int, default=1000
            Number of frames to process per chunk when use_memmap=True

        Returns
        -------
        int
            Local index of centroid frame

        Examples
        --------
        >>> # Fast mode for standard numpy arrays
        >>> data = np.random.rand(1000, 100)
        >>> idx = CentroidHelper.find_centroid(data, use_memmap=False)
        >>> print(idx)  # Index between 0 and 999

        >>> # Memmap mode for large datasets
        >>> idx = CentroidHelper.find_centroid(
        ...     large_memmap_data, use_memmap=True, chunk_size=500
        ... )
        """
        if use_memmap:
            return CentroidHelper._find_centroid_chunked(
                selected_data, chunk_size
            )
        else:
            return CentroidHelper._find_centroid_fast(selected_data)

    @staticmethod
    def _find_centroid_fast(selected_data: np.ndarray) -> int:
        """
        Find centroid using fast numpy operations.

        Parameters
        ----------
        selected_data : np.ndarray
            Data array with shape (n_frames, n_features)

        Returns
        -------
        int
            Local index of centroid frame

        Examples
        --------
        >>> data = np.random.rand(1000, 100)
        >>> idx = CentroidHelper._find_centroid_fast(data)
        >>> print(idx)  # Index between 0 and 999
        """
        centroid = np.mean(selected_data, axis=0)
        distances = np.linalg.norm(selected_data - centroid, axis=1)
        return np.argmin(distances)

    @staticmethod
    def _find_centroid_chunked(
        selected_data: np.ndarray,
        chunk_size: int
    ) -> int:
        """
        Find centroid using chunked processing.

        Parameters
        ----------
        selected_data : np.ndarray
            Data array with shape (n_frames, n_features)
        chunk_size : int
            Number of frames to process per chunk

        Returns
        -------
        int
            Local index of centroid frame

        Examples
        --------
        >>> idx = CentroidHelper._find_centroid_chunked(
        ...     large_memmap_data, chunk_size=500
        ... )
        """
        centroid = CentroidHelper._compute_centroid_chunked(
            selected_data, chunk_size
        )
        return CentroidHelper._find_closest_to_centroid_chunked(
            selected_data, centroid, chunk_size
        )

    @staticmethod
    def _compute_centroid_chunked(
        selected_data: np.ndarray,
        chunk_size: int
    ) -> np.ndarray:
        """
        Compute centroid using chunked processing.

        Parameters
        ----------
        selected_data : np.ndarray
            Data array with shape (n_frames, n_features)
        chunk_size : int
            Number of frames to process per chunk

        Returns
        -------
        np.ndarray
            Centroid vector with shape (n_features,)

        Examples
        --------
        >>> centroid = CentroidHelper._compute_centroid_chunked(
        ...     large_memmap_data, chunk_size=500
        ... )
        >>> print(centroid.shape)  # (n_features,)
        """
        running_sum = None
        total_count = 0
        n_frames = selected_data.shape[0]

        for start_idx in range(0, n_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, n_frames)
            chunk = selected_data[start_idx:end_idx]

            if running_sum is None:
                running_sum = np.sum(chunk, axis=0)
            else:
                running_sum += np.sum(chunk, axis=0)
            total_count += chunk.shape[0]

        return running_sum / total_count

    @staticmethod
    def _find_closest_to_centroid_chunked(
        selected_data: np.ndarray,
        centroid: np.ndarray,
        chunk_size: int
    ) -> int:
        """
        Find frame closest to given centroid using chunked processing.

        Parameters
        ----------
        selected_data : np.ndarray
            Data array with shape (n_frames, n_features)
        centroid : np.ndarray
            Centroid vector with shape (n_features,)
        chunk_size : int
            Number of frames to process per chunk

        Returns
        -------
        int
            Local index of frame closest to centroid

        Examples
        --------
        >>> centroid = np.mean(data, axis=0)
        >>> idx = CentroidHelper._find_closest_to_centroid_chunked(
        ...     large_memmap_data, centroid, chunk_size=500
        ... )
        >>> print(idx)  # Index of closest frame
        """
        best_distance = np.inf
        best_idx = 0
        current_offset = 0
        n_frames = selected_data.shape[0]

        for start_idx in range(0, n_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, n_frames)
            chunk = selected_data[start_idx:end_idx]

            distances = np.linalg.norm(chunk - centroid, axis=1)
            chunk_min_idx = np.argmin(distances)
            chunk_min_dist = distances[chunk_min_idx]

            if chunk_min_dist < best_distance:
                best_distance = chunk_min_dist
                best_idx = current_offset + chunk_min_idx

            current_offset += chunk.shape[0]

        return best_idx
