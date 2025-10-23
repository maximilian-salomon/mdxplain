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
Helper for optimizing cluster label sequences to plotting blocks.

Converts frame-by-frame cluster labels into contiguous blocks for
efficient visualization, reducing the number of matplotlib draw calls
from O(n_frames) to O(n_transitions).
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


class BlockOptimizerHelper:
    """
    Helper class for optimizing cluster label sequences.

    Converts sequential cluster labels into block representations
    (start_frame, end_frame, cluster_id) for efficient plotting.

    Examples
    --------
    >>> # Convert labels to blocks
    >>> labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    >>> blocks = BlockOptimizerHelper.labels_to_blocks(labels)
    >>> blocks
    [(0, 2, 0), (3, 4, 1), (5, 7, 2)]

    >>> # Single cluster
    >>> labels = np.array([5, 5, 5, 5])
    >>> blocks = BlockOptimizerHelper.labels_to_blocks(labels)
    >>> blocks
    [(0, 3, 5)]
    """

    @staticmethod
    def labels_to_blocks(labels: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Convert cluster label array to block representation.

        Groups consecutive identical cluster labels into contiguous blocks
        for efficient plotting. Reduces draw calls from O(n_frames) to
        O(n_transitions).

        Parameters
        ----------
        labels : numpy.ndarray, shape (n_frames,)
            Cluster label for each frame

        Returns
        -------
        List[Tuple[int, int, int]]
            List of (start_frame, end_frame, cluster_id) blocks.
            Indices are inclusive on both ends.

        Examples
        --------
        >>> # Three clusters with transitions
        >>> labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        >>> blocks = BlockOptimizerHelper.labels_to_blocks(labels)
        >>> blocks
        [(0, 2, 0), (3, 4, 1), (5, 7, 2)]

        >>> # With noise (-1)
        >>> labels = np.array([0, 0, -1, -1, 1, 1])
        >>> blocks = BlockOptimizerHelper.labels_to_blocks(labels)
        >>> blocks
        [(0, 1, 0), (2, 3, -1), (4, 5, 1)]

        >>> # Empty array
        >>> labels = np.array([])
        >>> blocks = BlockOptimizerHelper.labels_to_blocks(labels)
        >>> blocks
        []

        Notes
        -----
        Uses vectorized numpy operations for performance:
        - np.diff() to detect label changes
        - np.where() to find transition points
        - O(n) time complexity, minimal memory overhead
        """
        if len(labels) == 0:
            return []

        # Find where cluster changes occur
        changes = np.diff(labels, prepend=labels[0] - 1)
        change_indices = np.where(changes != 0)[0]

        # Create blocks
        blocks = []
        for i in range(len(change_indices)):
            start = change_indices[i]
            end = change_indices[i + 1] - 1 if i + 1 < len(change_indices) else len(labels) - 1
            cluster_id = int(labels[start])
            blocks.append((start, end, cluster_id))

        return blocks
