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
Helper class for grid layout computation in plots.

Provides intelligent grid layout algorithms for arranging subplots
in a quadratic figure with proportional spacing based on content counts.
"""

from typing import List, Tuple


class GridLayoutHelper:
    """
    Helper class for grid layout computation.

    Provides static methods for computing uniform subplot arrangements
    in matplotlib figures. Creates grids where each item occupies
    equal width (colspan=1), arranged in rows and columns.

    Examples
    --------
    >>> # Compute uniform layout for 10 features with max 4 columns
    >>> layout, n_rows, n_cols = GridLayoutHelper.compute_uniform_grid_layout(10, 4)
    >>> print(f"Grid: {n_rows}x{n_cols}")
    Grid: 3x4
    >>> print(layout[0])  # First item
    (0, 0, 0, 1)  # item_index=0, row=0, col=0, colspan=1
    >>> print(layout[4])  # Start of second row
    (4, 1, 0, 1)  # item_index=4, row=1, col=0, colspan=1
    """

    @staticmethod
    def compute_uniform_grid_layout(
        n_items: int,
        max_cols: int
    ) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
        """
        Compute uniform grid layout with equal-width items.

        Creates grid layout where each item occupies exactly one cell
        (colspan=1), arranged left-to-right, top-to-bottom, with
        maximum of max_cols columns per row.

        Parameters
        ----------
        n_items : int
            Total number of items to arrange
        max_cols : int
            Maximum columns per row

        Returns
        -------
        layout : List[Tuple[int, int, int, int]]
            List of (item_index, row, col, colspan) tuples
        n_rows : int
            Number of grid rows
        n_cols : int
            Number of grid columns (actual columns used, <= max_cols)

        Examples
        --------
        >>> # 10 items with max 4 columns
        >>> layout, n_rows, n_cols = GridLayoutHelper.compute_uniform_grid_layout(10, 4)
        >>> print(f"Grid: {n_rows}x{n_cols}")
        Grid: 3x4
        >>> print(layout[0])  # First item
        (0, 0, 0, 1)  # item_index=0, row=0, col=0, colspan=1
        >>> print(layout[4])  # Fifth item (start of second row)
        (4, 1, 0, 1)  # item_index=4, row=1, col=0, colspan=1

        >>> # 3 items with max 5 columns
        >>> layout, n_rows, n_cols = GridLayoutHelper.compute_uniform_grid_layout(3, 5)
        >>> print(f"Grid: {n_rows}x{n_cols}")
        Grid: 1x3

        >>> # Empty case
        >>> layout, n_rows, n_cols = GridLayoutHelper.compute_uniform_grid_layout(0, 4)
        >>> print(n_rows, n_cols)
        0 1

        Notes
        -----
        Algorithm:
        1. Determine actual columns used: min(n_items, max_cols)
        2. Calculate rows needed: ceil(n_items / n_cols)
        3. Pack items left-to-right, top-to-bottom

        Each item gets:
        - Unique position (row, col)
        - colspan=1 (uniform width)
        - Sequential item_index for tracking

        Used for density and violin plots where each subplot shows
        one feature or one feature+DataSelector combination.
        """
        if n_items == 0:
            return [], 0, 1

        # Actual columns used (don't exceed items count)
        n_cols = min(max_cols, n_items)

        # Calculate rows needed (ceiling division)
        n_rows = (n_items + n_cols - 1) // n_cols

        # Create layout: pack left-to-right, top-to-bottom
        layout = []
        for item_idx in range(n_items):
            row = item_idx // n_cols
            col = item_idx % n_cols
            layout.append((item_idx, row, col, 1))  # colspan=1 for all

        return layout, n_rows, n_cols
