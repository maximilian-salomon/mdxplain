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

"""Integration tests for DecompositionManager.reduce_components."""

import numpy as np
import pytest

from mdxplain.decomposition.entities.decomposition_data import (
    DecompositionData,
)
from mdxplain.decomposition.manager.decomposition_manager import (
    DecompositionManager,
)
from mdxplain.pipeline.entities.pipeline_data import PipelineData


@pytest.fixture
def source_decomposition():
    """A 200-frame decomposition with 30 variance-ordered components."""
    rng = np.random.RandomState(0)
    src = DecompositionData("kernel_pca")
    src.data = rng.rand(200, 30).astype(np.float64)
    evr = np.sort(rng.rand(30))[::-1]
    evr = evr / evr.sum()
    src.metadata = {
        "n_components": 30,
        "auto_selected": True,
        "explained_variance_ratio": list(evr),
        "explained_variance": list(evr * 42.0),
        "hyperparameters": {"kernel": "rbf"},
    }
    src.frame_mapping = {i: (0, i) for i in range(200)}
    return src


@pytest.fixture
def pipeline(source_decomposition):
    """A PipelineData holding the source under the name 'src'."""
    pipe = PipelineData()
    pipe.decomposition_data["src"] = source_decomposition
    return pipe


@pytest.fixture
def manager():
    """A DecompositionManager instance."""
    return DecompositionManager()


def test_reduce_slices_leading_components(
    manager, pipeline, source_decomposition
):
    """The reduced data equals the leading columns of the source exactly."""
    manager.reduce_components(pipeline, "src", "src_5", 5)
    reduced = pipeline.decomposition_data["src_5"]
    assert np.array_equal(reduced.data, source_decomposition.data[:, :5])


def test_reduce_owns_its_data(manager, pipeline, source_decomposition):
    """The reduced clone holds its own array, not a view on the source."""
    manager.reduce_components(pipeline, "src", "src_5", 5)
    reduced = pipeline.decomposition_data["src_5"]
    assert not np.shares_memory(reduced.data, source_decomposition.data)


def test_reduce_truncates_metadata(manager, pipeline, source_decomposition):
    """Metadata records the new count and truncates the variance arrays."""
    manager.reduce_components(pipeline, "src", "src_5", 5)
    meta = pipeline.decomposition_data["src_5"].metadata
    assert meta["n_components"] == 5
    assert meta["auto_selected"] is False
    assert meta["reduced_from"] == "src"
    expected_ratio = np.asarray(
        source_decomposition.metadata["explained_variance_ratio"]
    )[:5]
    expected_variance = np.asarray(
        source_decomposition.metadata["explained_variance"]
    )[:5]
    assert np.array_equal(meta["explained_variance_ratio"], expected_ratio)
    assert np.array_equal(meta["explained_variance"], expected_variance)
    assert meta["hyperparameters"] == {"kernel": "rbf"}


def test_reduce_shares_frame_mapping(manager, pipeline, source_decomposition):
    """The reduced clone shares the source frame mapping."""
    manager.reduce_components(pipeline, "src", "src_5", 5)
    reduced = pipeline.decomposition_data["src_5"]
    assert reduced.frame_mapping is source_decomposition.frame_mapping


def test_reduce_leaves_source_untouched(
    manager, pipeline, source_decomposition
):
    """The original decomposition keeps its exact data and metadata."""
    original_data = source_decomposition.data.copy()
    original_ratio = list(
        source_decomposition.metadata["explained_variance_ratio"]
    )
    manager.reduce_components(pipeline, "src", "src_5", 5)
    assert np.array_equal(source_decomposition.data, original_data)
    assert (
        source_decomposition.metadata["explained_variance_ratio"]
        == original_ratio
    )
    assert source_decomposition.metadata["n_components"] == 30
    assert source_decomposition.metadata["auto_selected"] is True


def test_reduce_missing_source_raises(manager, pipeline):
    """Reducing an unknown source raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        manager.reduce_components(pipeline, "missing", "x", 5)


def test_reduce_existing_target_raises(manager, pipeline):
    """Reducing onto an existing name without force raises ValueError."""
    with pytest.raises(ValueError, match="already exists"):
        manager.reduce_components(pipeline, "src", "src", 5)


def test_reduce_force_overwrites_target(
    manager, pipeline, source_decomposition
):
    """With force, an existing target is overwritten with the new content."""
    manager.reduce_components(pipeline, "src", "src_5", 5)
    manager.reduce_components(pipeline, "src", "src_5", 3, force=True)
    reduced = pipeline.decomposition_data["src_5"]
    assert np.array_equal(reduced.data, source_decomposition.data[:, :3])


@pytest.mark.parametrize("n_components", [0, 31])
def test_reduce_component_count_out_of_range_raises(
    manager, pipeline, n_components
):
    """A component count outside the valid range raises ValueError."""
    with pytest.raises(ValueError, match="between 1"):
        manager.reduce_components(pipeline, "src", "x", n_components)
