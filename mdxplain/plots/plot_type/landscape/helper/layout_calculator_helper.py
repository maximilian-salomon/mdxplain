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
Helper class for layout calculations in landscape plots.

Provides methods for calculating grid layouts, subplot arrangements,
and figure sizes for displaying multiple 2D projections.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List


class LayoutCalculatorHelper:
    """
    Helper class for layout calculations in multi-dimensional plots.

    Provides static methods for calculating optimal grid layouts,
    figure sizes, dimension pairs, and subplot positions.

    Examples
    --------
    >>> # Calculate grid layout
    >>> n_rows, n_cols = LayoutCalculatorHelper.calculate_grid_layout(4)
    >>> print(f"Grid: {n_rows}x{n_cols}")

    >>> # Create dimension pairs
    >>> pairs = LayoutCalculatorHelper.create_dimension_pairs([0, 1, 2, 3])
    >>> print(pairs)  # [(0, 1), (2, 3)]
    """

    @staticmethod
    def calculate_grid_layout(n_plots: int) -> Tuple[int, int]:
        """
        Calculate optimal grid layout for given number of plots.

        Creates as square a grid as possible, preferring
        more columns than rows for better screen aspect ratio.

        Parameters
        ----------
        n_plots : int
            Number of subplots to arrange

        Returns
        -------
        n_rows : int
            Number of rows in grid
        n_cols : int
            Number of columns in grid

        Examples
        --------
        >>> # Single plot
        >>> LayoutCalculatorHelper.calculate_grid_layout(1)
        (1, 1)

        >>> # Two plots
        >>> LayoutCalculatorHelper.calculate_grid_layout(2)
        (1, 2)

        >>> # Four plots (square grid)
        >>> LayoutCalculatorHelper.calculate_grid_layout(4)
        (2, 2)

        >>> # Six plots
        >>> LayoutCalculatorHelper.calculate_grid_layout(6)
        (2, 3)

        >>> # Nine plots (square grid)
        >>> LayoutCalculatorHelper.calculate_grid_layout(9)
        (3, 3)

        Notes
        -----
        Algorithm finds smallest integer n_cols >= sqrt(n_plots),
        then calculates n_rows = ceil(n_plots / n_cols).
        """
        if n_plots <= 0:
            raise ValueError("n_plots must be positive")

        # Square root gives target for square grid
        sqrt_n = np.sqrt(n_plots)

        # Round up to get number of columns
        n_cols = int(np.ceil(sqrt_n))

        # Calculate rows needed
        n_rows = int(np.ceil(n_plots / n_cols))

        return n_rows, n_cols

    @staticmethod
    def calculate_figure_size(
        n_rows: int,
        n_cols: int,
        subplot_size: float = 4.0
    ) -> Tuple[float, float]:
        """
        Calculate figure size based on grid layout.

        Parameters
        ----------
        n_rows : int
            Number of rows in grid
        n_cols : int
            Number of columns in grid
        subplot_size : float, default=4.0
            Size of each subplot in inches

        Returns
        -------
        fig_width : float
            Figure width in inches
        fig_height : float
            Figure height in inches

        Examples
        --------
        >>> # Single plot (4x4 inches)
        >>> LayoutCalculatorHelper.calculate_figure_size(1, 1)
        (4.0, 4.0)

        >>> # 2x2 grid (8x8 inches)
        >>> LayoutCalculatorHelper.calculate_figure_size(2, 2)
        (8.0, 8.0)

        >>> # 2x3 grid with custom subplot size
        >>> LayoutCalculatorHelper.calculate_figure_size(2, 3, subplot_size=5.0)
        (15.0, 10.0)

        Notes
        -----
        Total figure size = subplot_size * (n_cols, n_rows)
        This ensures consistent subplot sizes regardless of grid arrangement.
        """
        fig_width = n_cols * subplot_size
        fig_height = n_rows * subplot_size

        return fig_width, fig_height

    @staticmethod
    def create_dimension_pairs(dimensions: List[int]) -> List[Tuple[int, int]]:
        """
        Create consecutive pairs from dimension list.

        Pairs dimensions sequentially for grid plotting:
        [0, 1, 2, 3] -> [(0, 1), (2, 3)]

        Parameters
        ----------
        dimensions : List[int]
            List of dimension indices (must be even length)

        Returns
        -------
        List[Tuple[int, int]]
            List of dimension pairs

        Raises
        ------
        ValueError
            If dimensions has odd length

        Examples
        --------
        >>> # Two dimensions (one plot)
        >>> LayoutCalculatorHelper.create_dimension_pairs([0, 1])
        [(0, 1)]

        >>> # Four dimensions (two plots)
        >>> LayoutCalculatorHelper.create_dimension_pairs([0, 1, 2, 3])
        [(0, 1), (2, 3)]

        >>> # Six dimensions (three plots)
        >>> LayoutCalculatorHelper.create_dimension_pairs([0, 1, 2, 3, 4, 5])
        [(0, 1), (2, 3), (4, 5)]

        >>> # Odd number raises error
        >>> LayoutCalculatorHelper.create_dimension_pairs([0, 1, 2])
        ValueError: Dimension list must have even length

        Notes
        -----
        This consecutive pairing strategy creates intuitive projections.
        For example, PCA [0,1,2,3] shows (PC1 vs PC2) and (PC3 vs PC4).
        """
        if len(dimensions) % 2 != 0:
            raise ValueError("Dimension list must have even length")

        pairs = []
        for i in range(0, len(dimensions), 2):
            pairs.append((dimensions[i], dimensions[i + 1]))

        return pairs

    @staticmethod
    def get_subplot_position(
        plot_index: int,
        n_rows: int,
        n_cols: int
    ) -> Tuple[int, int]:
        """
        Get subplot position (row, col) for given plot index.

        Parameters
        ----------
        plot_index : int
            Index of plot (0-based)
        n_rows : int
            Number of rows in grid
        n_cols : int
            Number of columns in grid

        Returns
        -------
        row : int
            Row index (0-based)
        col : int
            Column index (0-based)

        Examples
        --------
        >>> # 2x2 grid
        >>> LayoutCalculatorHelper.get_subplot_position(0, 2, 2)
        (0, 0)
        >>> LayoutCalculatorHelper.get_subplot_position(1, 2, 2)
        (0, 1)
        >>> LayoutCalculatorHelper.get_subplot_position(2, 2, 2)
        (1, 0)
        >>> LayoutCalculatorHelper.get_subplot_position(3, 2, 2)
        (1, 1)

        >>> # 2x3 grid
        >>> LayoutCalculatorHelper.get_subplot_position(0, 2, 3)
        (0, 0)
        >>> LayoutCalculatorHelper.get_subplot_position(3, 2, 3)
        (1, 0)

        Notes
        -----
        Uses row-major ordering (fill rows left to right, top to bottom).
        """
        row = plot_index // n_cols
        col = plot_index % n_cols

        return row, col
