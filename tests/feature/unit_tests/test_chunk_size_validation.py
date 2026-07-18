# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Opus 4.8).
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
Tests for chunk_size and stride validation and for chunk-size floors.

An invalid chunk_size must be rejected where it is configured. Otherwise it
travels deep into a chunk loop and surfaces as range() arg 3 must not be zero,
which says nothing about the actual mistake.
"""

import numpy as np
import pytest

from mdxplain.feature.manager.feature_manager import FeatureManager
from mdxplain.utils.memmap_reuse_helper import MemmapReuseHelper


class TestFeatureManagerValidatesChunkSize:
    """FeatureManager must reject a chunk_size that cannot drive a chunk loop."""

    @pytest.mark.parametrize("chunk_size", [0, -1, -2000])
    def test_rejects_non_positive(self, tmp_path, chunk_size):
        """Zero or negative would reach range(0, n, step) and raise there."""
        with pytest.raises(ValueError, match="Chunk size must be a positive integer"):
            FeatureManager(chunk_size=chunk_size, cache_dir=str(tmp_path))

    @pytest.mark.parametrize("chunk_size", [2.5, "2000", None])
    def test_rejects_non_int(self, tmp_path, chunk_size):
        """A non-int is equally unusable as a step."""
        with pytest.raises(ValueError, match="Chunk size must be a positive integer"):
            FeatureManager(chunk_size=chunk_size, cache_dir=str(tmp_path))

    def test_accepts_positive_int(self, tmp_path):
        """The valid case still constructs."""
        manager = FeatureManager(chunk_size=2000, cache_dir=str(tmp_path))
        assert manager.chunk_size == 2000


class TestHashArrayChunkRows:
    """hash_array is handed a configured chunk_size straight through."""

    @pytest.mark.parametrize("chunk_rows", [None, 0, -5])
    def test_falsy_chunk_rows_fall_back(self, chunk_rows):
        """A falsy value must not raise; callers pass an unset chunk_size."""
        array = np.arange(24, dtype=np.float32).reshape(6, 4)
        assert MemmapReuseHelper.hash_array(array, chunk_rows)

    def test_hash_is_independent_of_chunk_rows(self):
        """The digest describes the content, never the read granularity."""
        array = np.arange(24, dtype=np.float32).reshape(6, 4)
        digests = {MemmapReuseHelper.hash_array(array, rows) for rows in (1, 2, 5, 1000)}
        assert len(digests) == 1

    def test_hash_detects_content_change(self):
        """A different array must give a different digest."""
        first = np.arange(24, dtype=np.float32).reshape(6, 4)
        second = first.copy()
        second[3, 2] = 99.0
        assert MemmapReuseHelper.hash_array(first, 2) != MemmapReuseHelper.hash_array(second, 2)
