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
Statistical analysis for coordinate calculations.

Analysis methods for coordinate calculations with statistical computations
and support for memory-mapped arrays and structural mobility analysis.
"""

import numpy as np

from ..helper.calculator_stat_helper import CalculatorStatHelper


class CoordinatesCalculatorAnalysis:
    """
    Analysis methods for coordinate calculation statistics and metrics.

    Provides statistical analysis capabilities for coordinate data including
    structural variability, mobility analysis, and geometric statistics
    with memory-mapped array support.
    """

    def __init__(self, use_memmap: bool = False, chunk_size: int = 2000) -> None:
        """
        Initialize coordinates analysis with chunking configuration.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        chunk_size : int, default=2000
            Number of frames to process per chunk for memory-mapped arrays

        Examples
        --------
        >>> # Default chunking
        >>> analysis = CoordinatesCalculatorAnalysis()

        >>> # Custom chunk size for large datasets
        >>> analysis = CoordinatesCalculatorAnalysis(chunk_size=1000)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size

    # === COORDINATE-BASED STATISTICS ===
    def compute_mean(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute mean coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Mean coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.mean, self.chunk_size, self.use_memmap
        )

    def compute_std(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation of coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Standard deviation for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.std, self.chunk_size, self.use_memmap
        )

    def compute_min(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute minimum coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Minimum coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.min, self.chunk_size, self.use_memmap
        )

    def compute_max(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute maximum coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Maximum coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.max, self.chunk_size, self.use_memmap
        )

    def compute_median(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute median coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Median coordinate for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.median, self.chunk_size, self.use_memmap
        )

    def compute_variance(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute variance of coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Variance for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.var, self.chunk_size, self.use_memmap
        )

    def compute_range(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute range (peak-to-peak) of coordinates per coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Range (max - min) for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates, np.ptp, self.chunk_size, self.use_memmap
        )

    def compute_mad(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute median absolute deviation for each coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            MAD values per coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_func_per_feature(
            coordinates,
            lambda x, axis: np.median(
                np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis
            ),
            self.chunk_size,
            self.use_memmap,
        )

    # === FRAME-BASED STATISTICS ===
    def coordinates_per_frame_mean(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute mean coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Mean coordinate across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.mean
        )

    def coordinates_per_frame_std(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute standard deviation of coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Standard deviation across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.std
        )

    def coordinates_per_frame_min(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute minimum coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Minimum coordinate across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.min
        )

    def coordinates_per_frame_max(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute maximum coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Maximum coordinate across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.max
        )

    def coordinates_per_frame_range(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute range of coordinates per frame.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            Range (max - min) across all coordinates for each frame with shape (n_frames,)
        """
        return CalculatorStatHelper.compute_func_per_frame(
            coordinates, self.chunk_size, self.use_memmap, np.ptp
        )

    def compute_cv(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute coefficient of variation for each coordinate.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            CV values per coordinate with shape (n_coordinates,)
        """
        mean_vals = self.compute_mean(coordinates)
        std_vals = self.compute_std(coordinates)
        return std_vals / (np.abs(mean_vals) + 1e-10)

    def compute_rmsf(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute root mean square fluctuation per atom.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate array with shape (n_frames, n_coordinates)

        Returns
        -------
        numpy.ndarray
            RMSF values expanded to coordinate format with shape (n_coordinates,)
        """
        # Reshape to (n_frames, n_atoms, 3) for proper per-atom calculation
        n_coords = coordinates.shape[1]
        n_atoms = n_coords // 3
        coords_3d = coordinates.reshape(coordinates.shape[0], n_atoms, 3)
        
        # Compute mean position per atom
        mean_positions = np.mean(coords_3d, axis=0)  # (n_atoms, 3)
        
        # Compute RMSF per atom
        deviations = coords_3d - mean_positions  # (n_frames, n_atoms, 3)
        rmsf_per_atom = np.sqrt(np.mean(np.sum(deviations**2, axis=2), axis=0))  # (n_atoms,)
        
        # Expand back to coordinate format (same RMSF for x,y,z of same atom)
        rmsf_expanded = np.repeat(rmsf_per_atom, 3)
        return rmsf_expanded

    # === TRANSITION ANALYSIS ===
    def compute_transitions_lagtime(self, coordinates: np.ndarray, threshold: float = 1.0, lag_time: int = 10) -> np.ndarray:
        """
        Compute transitions within lag time for coordinates.

        Parameters
        ----------
        coordinates : np.ndarray or np.memmap
            Coordinate array with shape (n_frames, n_coordinates)
        threshold : float, default=1.0
            Position threshold for transition detection in Angstroms
        lag_time : int, default=10
            Number of frames to look ahead for transitions

        Returns
        -------
        np.ndarray
            Transition counts for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_transitions_within_lagtime(
            coordinates, threshold, lag_time, self.chunk_size, self.use_memmap
        )

    def compute_transitions_window(self, coordinates: np.ndarray, threshold: float = 1.0, window_size: int = 10) -> np.ndarray:
        """
        Compute transitions within window for coordinates.

        Parameters
        ----------
        coordinates : np.ndarray or np.memmap
            Coordinate array with shape (n_frames, n_coordinates)
        threshold : float, default=1.0
            Position threshold for transition detection in Angstroms
        window_size : int, default=10
            Size of sliding window for transition analysis

        Returns
        -------
        np.ndarray
            Transition counts for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_transitions_within_window(
            coordinates, threshold, window_size, self.chunk_size, self.use_memmap
        )

    def compute_stability(self, coordinates: np.ndarray, threshold: float = 1.0, window_size: int = 10) -> np.ndarray:
        """
        Compute stability analysis for coordinates.

        Parameters
        ----------
        coordinates : np.ndarray or np.memmap
            Coordinate array with shape (n_frames, n_coordinates)
        threshold : float, default=1.0
            Position threshold for stability detection in Angstroms
        window_size : int, default=10
            Size of analysis window

        Returns
        -------
        np.ndarray
            Stability scores for each coordinate with shape (n_coordinates,)
        """
        return CalculatorStatHelper.compute_stability(
            coordinates, threshold, window_size, self.chunk_size, self.use_memmap
        )

    # === COMPARISON METHODS ===
    def compute_differences(self, coordinates1: np.ndarray, coordinates2: np.ndarray, preprocessing_func = None) -> np.ndarray:
        """
        Compute differences between two coordinate datasets.

        Parameters
        ----------
        coordinates1 : np.ndarray or np.memmap
            First coordinate array with shape (n_frames, n_coordinates)
        coordinates2 : np.ndarray or np.memmap
            Second coordinate array with shape (n_frames, n_coordinates)
        preprocessing_func : callable, optional
            Function to apply to each dataset before comparison

        Returns
        -------
        np.ndarray
            Difference values with shape (n_frames, n_coordinates)
        """
        return CalculatorStatHelper.compute_differences(
            coordinates1, coordinates2, self.chunk_size, self.use_memmap, preprocessing_func
        )
