# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex GPT-5.
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

"""Unit tests for SelectionMemmapHelper eviction behavior."""

import tempfile

import numpy as np

from mdxplain.pipeline.helper.selection_memmap_helper import SelectionMemmapHelper


def test_create_memmap_selection_batches_evictions(monkeypatch):
    """Selection writes should evict in buffered ranges instead of every chunk."""
    data = np.arange(40, dtype=np.float32).reshape(10, 4)
    calls = []

    def _evict_stub(_array, start_row, end_row):
        calls.append((int(start_row), int(end_row)))

    monkeypatch.setattr(
        "mdxplain.pipeline.helper.selection_memmap_helper.ResourceUtils.tune_memmap",
        lambda *_args, **_kwargs: {"applied": False, "errors": []},
    )
    monkeypatch.setattr(
        "mdxplain.pipeline.helper.selection_memmap_helper.MemmapUtils.evict_memory_range",
        _evict_stub,
    )
    monkeypatch.setattr(
        "mdxplain.pipeline.helper.selection_memmap_helper.SelectionMemmapHelper._EVICT_EVERY_N_CHUNKS",
        3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = SelectionMemmapHelper.create_memmap_selection(
            data=data,
            indices=[0, 2],
            name="evict_batch",
            data_type="feature",
            feature_type="coords",
            cache_dir=tmpdir,
            chunk_size=2,
        )[0]

        assert calls == [(0, 6), (6, 10)]
        result._mmap.close()


def test_create_memmap_frame_selection_batches_result_evictions(monkeypatch):
    """Frame selection should evict result ranges in fixed chunk batches."""
    result_calls = []

    def _evict_stub(array, start_row, end_row):
        filename = str(getattr(array, "filename", ""))
        if filename.endswith("_frame_selection.dat"):
            result_calls.append((int(start_row), int(end_row)))

    monkeypatch.setattr(
        "mdxplain.pipeline.helper.selection_memmap_helper.ResourceUtils.tune_memmap",
        lambda *_args, **_kwargs: {"applied": False, "errors": []},
    )
    monkeypatch.setattr(
        "mdxplain.pipeline.helper.selection_memmap_helper.MemmapUtils.evict_memory_range",
        _evict_stub,
    )
    monkeypatch.setattr(
        "mdxplain.pipeline.helper.selection_memmap_helper.SelectionMemmapHelper._EVICT_EVERY_N_CHUNKS",
        2,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = f"{tmpdir}/source_frames.dat"
        source = np.memmap(source_path, dtype=np.float32, mode="w+", shape=(12, 3))
        source[:] = np.arange(36, dtype=np.float32).reshape(12, 3)
        frame_indices = [0, 6, 2, 10, 1]

        result = SelectionMemmapHelper.create_memmap_frame_selection(
            data=source,
            frame_indices=frame_indices,
            name="frame_sel",
            cache_dir=tmpdir,
            chunk_size=2,
        )

        assert result_calls == [(0, 4), (4, 5)]

        result._mmap.close()
        source._mmap.close()
