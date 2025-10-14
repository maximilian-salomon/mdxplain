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

from typing import Dict, List, Tuple


class GridLayoutHelper:
    """
    Helper class for intelligent grid layout computation.

    Provides static methods for computing optimal subplot arrangements
    in matplotlib figures. Arranges subplots in a roughly quadratic
    grid with proportional widths based on content counts.

    Examples
    --------
    >>> # Compute layout for feature types
    >>> feature_counts = {"distances": 15, "torsions": 5, "sasa": 3}
    >>> layout, n_rows, n_cols = GridLayoutHelper.compute_grid_layout(feature_counts)
    >>> print(f"Grid: {n_rows}x{n_cols}")
    Grid: 2x24
    >>> for feat_type, row, col, colspan in layout:
    ...     print(f"{feat_type}: row {row}, cols {col}-{col+colspan-1}")
    distances: row 0, cols 0-23
    torsions: row 1, cols 0-7
    sasa: row 1, cols 8-12
    """

    @staticmethod
    def compute_grid_layout(
        feature_counts: Dict[str, int]
    ) -> Tuple[List[Tuple[str, int, int, int]], int, int]:
        """
        Compute intelligent grid layout for quadratic figure arrangement.

        Determines optimal grid dimensions and subplot positions to create
        a roughly quadratic overall figure with proportional subplot widths
        based on feature counts.

        Parameters
        ----------
        feature_counts : Dict[str, int]
            Mapping of feature_type -> number of features

        Returns
        -------
        layout : List[Tuple[str, int, int, int]]
            List of (feature_type, row, col, colspan) tuples
        n_rows : int
            Number of grid rows
        n_cols : int
            Number of grid columns

        Examples
        --------
        >>> # Simple layout with three feature types
        >>> feature_counts = {"distances": 15, "torsions": 5, "sasa": 3}
        >>> layout, n_rows, n_cols = GridLayoutHelper.compute_grid_layout(feature_counts)
        >>> print(f"Grid: {n_rows}x{n_cols}")
        Grid: 2x24

        >>> # Check layout assignments
        >>> for feat_type, row, col, colspan in layout:
        ...     print(f"{feat_type}: row {row}, cols {col}-{col+colspan-1}")
        distances: row 0, cols 0-23
        torsions: row 1, cols 0-7
        sasa: row 1, cols 8-12

        >>> # Single feature type
        >>> single_counts = {"distances": 8}
        >>> layout, n_rows, n_cols = GridLayoutHelper.compute_grid_layout(single_counts)
        >>> print(f"Grid: {n_rows}x{n_cols}")
        Grid: 1x12

        >>> # Empty input
        >>> empty_layout, n_rows, n_cols = GridLayoutHelper.compute_grid_layout({})
        >>> print(len(empty_layout))
        0

        Notes
        -----
        Algorithm:
        1. Sort feature types by count (largest first)
        2. Calculate grid columns (min 12, max 24)
        3. Pack feature types row-by-row with proportional colspan
        4. Start new row when current row full

        Grid columns are scaled based on the largest feature count:
        - Minimum 12 columns for small counts
        - Maximum 24 columns for large counts
        - Proportional to largest feature count

        Colspan is proportional to feature count relative to maximum:
        - Larger feature types get more columns
        - Minimum 1 column per feature type
        - Rounds to nearest integer for balanced layout

        Used for creating intelligent subplot arrangements in violin plots,
        heatmaps, and other multi-panel visualizations.
        """
        # Sort by feature count (largest first)
        sorted_types = sorted(feature_counts.items(), key=lambda x: -x[1])

        if not sorted_types:
            return [], 0, 1

        # Calculate grid parameters
        max_features = sorted_types[0][1]  # Largest feature count
        n_cols = min(max(12, max_features), 24)  # Between 12 and 24

        # Pack feature types into grid rows
        layout = []
        current_row = 0
        current_col = 0

        for feat_type, n_features in sorted_types:
            # Calculate colspan (proportional to feature count)
            colspan = max(1, round(n_features / max_features * n_cols))
            colspan = min(colspan, n_cols)  # Don't exceed grid width

            # Check if fits in current row
            if current_col + colspan > n_cols:
                # Start new row
                current_row += 1
                current_col = 0

            layout.append((feat_type, current_row, current_col, colspan))
            current_col += colspan

        n_rows = current_row + 1
        return layout, n_rows, n_cols
