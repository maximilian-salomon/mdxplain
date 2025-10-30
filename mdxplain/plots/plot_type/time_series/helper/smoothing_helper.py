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
Helper for data smoothing in time series plots.

Provides multiple smoothing methods for trajectory feature visualization
including moving average and Savitzky-Golay filtering.
"""

from typing import Optional
import numpy as np
from scipy.signal import savgol_filter


class SmoothingHelper:
    """
    Helper for smoothing time series data.

    Provides moving average and Savitzky-Golay filtering methods
    with parameter validation and automatic window size adjustment.

    Examples
    --------
    >>> y_smooth = SmoothingHelper.apply_smoothing(
    ...     y_data, "savitzky", window=51, polyorder=3
    ... )
    >>> y_smooth = SmoothingHelper.apply_smoothing(
    ...     y_data, "moving_average", window=20
    ... )
    """

    @staticmethod
    def apply_smoothing(
        y_values: np.ndarray,
        method: Optional[str],
        window: int,
        polyorder: int = 3
    ) -> np.ndarray:
        """
        Apply smoothing to time series data.

        Parameters
        ----------
        y_values : np.ndarray
            Input data to smooth
        method : str or None
            Smoothing method ("moving_average", "savitzky", or None)
        window : int
            Window size in frames (must be odd for Savitzky-Golay)
        polyorder : int, default=3
            Polynomial order for Savitzky-Golay filter

        Returns
        -------
        np.ndarray
            Smoothed data (same length as input)

        Raises
        ------
        ValueError
            If parameters are invalid or method is unknown

        Examples
        --------
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> y_smooth = SmoothingHelper.apply_smoothing(
        ...     y, "savitzky", window=3, polyorder=1
        ... )

        Notes
        -----
        Returns original data if method is None.
        Window size is automatically adjusted if too large for data.
        """
        if method is None:
            return y_values

        data_length = len(y_values)
        adjusted_window = SmoothingHelper._adjust_window_size(
            window, data_length, method
        )

        SmoothingHelper.validate_smoothing_params(
            method, adjusted_window, polyorder, data_length
        )

        if method == "moving_average":
            return SmoothingHelper._smooth_moving_average(
                y_values, adjusted_window
            )
        elif method == "savitzky":
            return SmoothingHelper._smooth_savitzky_golay(
                y_values, adjusted_window, polyorder
            )
        else:
            raise ValueError(
                f"Unknown smoothing method: {method}. "
                f"Use 'moving_average' or 'savitzky'."
            )

    @staticmethod
    def _adjust_window_size(
        window: int,
        data_length: int,
        method: str
    ) -> int:
        """
        Adjust window size to fit data length.

        Parameters
        ----------
        window : int
            Requested window size
        data_length : int
            Length of data
        method : str
            Smoothing method

        Returns
        -------
        int
            Adjusted window size

        Examples
        --------
        >>> adjusted = SmoothingHelper._adjust_window_size(100, 50, "savitzky")
        >>> print(adjusted)  # 49 (largest odd number < 50)

        Notes
        -----
        For Savitzky-Golay, ensures window is odd and < data_length.
        For moving average, ensures window <= data_length.
        """
        if window >= data_length:
            return SmoothingHelper._adjust_oversized_window(
                data_length, method
            )

        return SmoothingHelper._ensure_odd_for_savitzky(window, method)

    @staticmethod
    def _adjust_oversized_window(data_length: int, method: str) -> int:
        """
        Adjust window when larger than data.

        Parameters
        ----------
        data_length : int
            Length of data
        method : str
            Smoothing method

        Returns
        -------
        int
            Adjusted window size

        Examples
        --------
        >>> adjusted = SmoothingHelper._adjust_oversized_window(50, "savitzky")

        Notes
        -----
        Returns largest valid window for given data length and method.
        """
        max_window = data_length - 1
        if method == "savitzky":
            return max_window if max_window % 2 == 1 else max_window - 1
        return max_window

    @staticmethod
    def _ensure_odd_for_savitzky(window: int, method: str) -> int:
        """
        Ensure window is odd for Savitzky-Golay.

        Parameters
        ----------
        window : int
            Window size
        method : str
            Smoothing method

        Returns
        -------
        int
            Adjusted window (odd if Savitzky-Golay)

        Examples
        --------
        >>> adjusted = SmoothingHelper._ensure_odd_for_savitzky(50, "savitzky")
        >>> print(adjusted)  # 51

        Notes
        -----
        Only modifies window for Savitzky-Golay method.
        """
        if method == "savitzky" and window % 2 == 0:
            return window + 1
        return window

    @staticmethod
    def validate_smoothing_params(
        method: str,
        window: int,
        polyorder: int,
        data_length: int
    ) -> None:
        """
        Validate smoothing parameters.

        Parameters
        ----------
        method : str
            Smoothing method
        window : int
            Window size
        polyorder : int
            Polynomial order (for Savitzky-Golay)
        data_length : int
            Length of data to smooth

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If parameters are invalid

        Examples
        --------
        >>> SmoothingHelper.validate_smoothing_params(
        ...     "savitzky", 51, 3, 1000
        ... )  # OK

        Notes
        -----
        Checks:
        - Window must be positive
        - Window must be < data_length
        - For Savitzky-Golay: window must be odd
        - For Savitzky-Golay: polyorder must be < window
        """
        SmoothingHelper._validate_window_basic(window, data_length)
        if method == "savitzky":
            SmoothingHelper._validate_savitzky_params(window, polyorder)

    @staticmethod
    def _validate_window_basic(window: int, data_length: int) -> None:
        """
        Validate basic window parameters.

        Parameters
        ----------
        window : int
            Window size
        data_length : int
            Data length

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If window is invalid

        Examples
        --------
        >>> SmoothingHelper._validate_window_basic(51, 1000)

        Notes
        -----
        Checks window is positive and smaller than data length.
        """
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        if window >= data_length:
            raise ValueError(
                f"Window size ({window}) must be smaller than "
                f"data length ({data_length})"
            )

    @staticmethod
    def _validate_savitzky_params(window: int, polyorder: int) -> None:
        """
        Validate Savitzky-Golay specific parameters.

        Parameters
        ----------
        window : int
            Window size
        polyorder : int
            Polynomial order

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If parameters invalid for Savitzky-Golay

        Examples
        --------
        >>> SmoothingHelper._validate_savitzky_params(51, 3)

        Notes
        -----
        Checks window is odd and polyorder < window.
        """
        if window % 2 == 0:
            raise ValueError(
                f"Savitzky-Golay window must be odd, got {window}"
            )
        if polyorder >= window:
            raise ValueError(
                f"Polynomial order ({polyorder}) must be less than "
                f"window size ({window})"
            )

    @staticmethod
    def _smooth_moving_average(
        y_values: np.ndarray,
        window: int
    ) -> np.ndarray:
        """
        Apply moving average smoothing.

        Parameters
        ----------
        y_values : np.ndarray
            Input data
        window : int
            Window size

        Returns
        -------
        np.ndarray
            Smoothed data

        Examples
        --------
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> y_smooth = SmoothingHelper._smooth_moving_average(y, 3)

        Notes
        -----
        Uses numpy.convolve with uniform kernel.
        Mode='same' ensures output has same length as input.
        """
        kernel = np.ones(window) / window
        return np.convolve(y_values, kernel, mode='same')

    @staticmethod
    def _smooth_savitzky_golay(
        y_values: np.ndarray,
        window: int,
        polyorder: int
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay filter smoothing.

        Parameters
        ----------
        y_values : np.ndarray
            Input data
        window : int
            Window size (must be odd)
        polyorder : int
            Polynomial order

        Returns
        -------
        np.ndarray
            Smoothed data

        Examples
        --------
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> y_smooth = SmoothingHelper._smooth_savitzky_golay(y, 3, 1)

        Notes
        -----
        Uses scipy.signal.savgol_filter.
        Preserves peaks and valleys better than moving average.
        """
        return savgol_filter(y_values, window, polyorder)
