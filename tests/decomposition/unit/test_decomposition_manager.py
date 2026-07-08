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

"""Unit tests for DecompositionManager removal and reset behaviour."""

import os

import numpy as np
import pytest

from mdxplain.decomposition.entities.decomposition_data import (
    DecompositionData,
)
from mdxplain.decomposition.manager.decomposition_manager import (
    DecompositionManager,
)
from mdxplain.pipeline.entities.pipeline_data import PipelineData


def _memmap_decomposition(cache_dir, name="source.dat", n_frames=50, n_comp=6):
    """Build a decomposition whose transformed data lives in a memmap file."""
    path = os.path.join(cache_dir, name)
    data = np.memmap(
        path, dtype=np.float32, mode="w+", shape=(n_frames, n_comp)
    )
    rng = np.random.RandomState(0)
    data[:] = rng.rand(n_frames, n_comp).astype(np.float32)
    data.flush()
    decomposition = DecompositionData("kernel_pca")
    decomposition.data = data
    decomposition.metadata = {"n_components": n_comp}
    decomposition.frame_mapping = {i: (0, i) for i in range(n_frames)}
    return decomposition, path


def test_remove_decomposition_deletes_cache_files(tmp_path):
    """remove_decomposition drops the entry and deletes its memmap file."""
    cache = str(tmp_path)
    decomposition, path = _memmap_decomposition(cache)
    pipe = PipelineData()
    pipe.decomposition_data["src"] = decomposition
    manager = DecompositionManager(
        use_memmap=True, chunk_size=64, cache_dir=cache
    )

    assert os.path.exists(path)
    manager.remove_decomposition(pipe, "src")

    assert "src" not in pipe.decomposition_data
    assert not os.path.exists(path)


def test_remove_decomposition_missing_raises(tmp_path):
    """Removing an unknown decomposition raises ValueError."""
    pipe = PipelineData()
    manager = DecompositionManager(
        use_memmap=True, chunk_size=64, cache_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="not found"):
        manager.remove_decomposition(pipe, "missing")


def test_remove_decomposition_leaves_others_untouched(tmp_path):
    """Removing one decomposition keeps the others and their files intact."""
    cache = str(tmp_path)
    first, first_path = _memmap_decomposition(cache, name="a.dat")
    second, second_path = _memmap_decomposition(cache, name="b.dat")
    pipe = PipelineData()
    pipe.decomposition_data["a"] = first
    pipe.decomposition_data["b"] = second
    manager = DecompositionManager(
        use_memmap=True, chunk_size=64, cache_dir=cache
    )

    manager.remove_decomposition(pipe, "a")

    assert "a" not in pipe.decomposition_data
    assert not os.path.exists(first_path)
    assert "b" in pipe.decomposition_data
    assert os.path.exists(second_path)


def test_reset_decompositions_deletes_all_cache_files(tmp_path):
    """reset_decompositions clears entries and deletes their memmap files."""
    cache = str(tmp_path)
    first, first_path = _memmap_decomposition(cache, name="a.dat")
    second, second_path = _memmap_decomposition(cache, name="b.dat")
    pipe = PipelineData()
    pipe.decomposition_data["a"] = first
    pipe.decomposition_data["b"] = second
    manager = DecompositionManager(
        use_memmap=True, chunk_size=64, cache_dir=cache
    )

    manager.reset_decompositions(pipe)

    assert dict(pipe.decomposition_data) == {}
    assert not os.path.exists(first_path)
    assert not os.path.exists(second_path)


def test_reset_decompositions_empty_is_noop(tmp_path):
    """Resetting with no decompositions does nothing and does not error."""
    pipe = PipelineData()
    manager = DecompositionManager(
        use_memmap=True, chunk_size=64, cache_dir=str(tmp_path)
    )
    manager.reset_decompositions(pipe)
    assert dict(pipe.decomposition_data) == {}


def test_reset_decompositions_keeps_in_memory_decompositions(tmp_path):
    """A reset over in-memory (non-memmap) decompositions just clears them."""
    pipe = PipelineData()
    decomposition = DecompositionData("pca")
    decomposition.data = np.zeros((10, 3), dtype=np.float64)
    pipe.decomposition_data["ram"] = decomposition
    manager = DecompositionManager(cache_dir=str(tmp_path))

    manager.reset_decompositions(pipe)
    assert dict(pipe.decomposition_data) == {}
