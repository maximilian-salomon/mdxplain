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

"""Unit tests for SelectionMatrixHelper internals."""

import numpy as np

from mdxplain.pipeline.helper.selection_matrix_helper import SelectionMatrixHelper


class TestSelectionMatrixHelperRuns:
    """Test contiguous run splitting and mapping materialization."""

    def test_build_contiguous_runs_mixed_segments(self):
        """Split source/destination indices into expected contiguous run blocks."""
        src_cols = np.array([1, 2, 3, 7, 8, 10], dtype=np.int32)
        dst_offsets = np.array([0, 1, 2, 3, 4, 9], dtype=np.int32)

        runs = SelectionMatrixHelper._build_contiguous_runs(src_cols, dst_offsets)

        assert runs == [
            (1, 4, 0, 3),   # src 1..3 -> dst 0..2
            (7, 9, 3, 5),   # src 7..8 -> dst 3..4
            (10, 11, 9, 10),  # src 10 -> dst 9
        ]

    def test_materialize_frame_mapping_from_chunks(self):
        """Build final row->(traj, frame) dict from compact chunk arrays."""
        traj_chunks = [
            np.array([0, 0], dtype=np.int32),
            np.array([2], dtype=np.int32),
        ]
        frame_chunks = [
            np.array([5, 9], dtype=np.int32),
            np.array([3], dtype=np.int32),
        ]

        mapping = SelectionMatrixHelper._materialize_frame_mapping(
            traj_chunks,
            frame_chunks,
        )

        assert mapping == {
            0: (0, 5),
            1: (0, 9),
            2: (2, 3),
        }


class TestSelectionMatrixHelperCopy:
    """Test low-level block copy behavior for direct and indexed row access."""

    def test_copy_feature_block_direct_row_slice_with_offset(self):
        """Copy from a direct source row slice using source_row_offset."""
        source_data = np.arange(30 * 8, dtype=np.float32).reshape(30, 8)
        baseline = (np.arange(4 * 6, dtype=np.float32).reshape(4, 6) + 1000.0)
        matrix = baseline.copy()

        frame_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        src_cols = np.array([1, 2, 5], dtype=np.int32)
        dst_offsets = np.array([0, 1, 4], dtype=np.int32)

        SelectionMatrixHelper._copy_feature_block(
            matrix=matrix,
            source_data=source_data,
            metadata={},
            frame_indices=frame_indices,
            direct_row_slice=True,
            source_row_offset=10,
            start_row=0,
            start_col=1,
            src_cols=src_cols,
            dst_offsets=dst_offsets,
        )

        expected = baseline.copy()
        block = source_data[10:14][:, [1, 2, 5]]
        expected[:, 1] = block[:, 0]
        expected[:, 2] = block[:, 1]
        expected[:, 5] = block[:, 2]

        np.testing.assert_array_equal(matrix, expected)

    def test_copy_feature_block_with_explicit_frame_indices(self):
        """Copy using explicit non-monotonic frame indices."""
        source_data = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
        baseline = (np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 2000.0)
        matrix = baseline.copy()

        frame_indices = np.array([6, 1, 7], dtype=np.int32)
        src_cols = np.array([0, 3], dtype=np.int32)
        dst_offsets = np.array([1, 4], dtype=np.int32)

        SelectionMatrixHelper._copy_feature_block(
            matrix=matrix,
            source_data=source_data,
            metadata={},
            frame_indices=frame_indices,
            direct_row_slice=False,
            source_row_offset=0,
            start_row=0,
            start_col=0,
            src_cols=src_cols,
            dst_offsets=dst_offsets,
        )

        expected = baseline.copy()
        expected[:, 1] = source_data[frame_indices, 0]
        expected[:, 4] = source_data[frame_indices, 3]

        np.testing.assert_array_equal(matrix, expected)
