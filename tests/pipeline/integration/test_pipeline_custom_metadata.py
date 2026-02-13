# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Codex.
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

"""Integration tests for pipeline custom metadata APIs."""

import tempfile

import numpy as np
import pytest

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager


class TestPipelineCustomMetadata:
    """Tests for add/get custom metadata behavior and persistence."""

    def test_add_and_get_custom_metadata(self):
        """Stored custom metadata should be retrievable by name."""
        pipeline = PipelineManager(show_progress=False, use_memmap=False)
        payload = {"timings": [0.1, 0.2, 0.3], "label": "benchmark"}

        pipeline.add_custom_metadata("timing_info", payload)
        loaded = pipeline.get_custom_metadata("timing_info")

        assert loaded == payload
        pipeline.close()

    def test_get_custom_metadata_missing_raises(self):
        """Requesting a missing key should raise ValueError."""
        pipeline = PipelineManager(show_progress=False, use_memmap=False)

        with pytest.raises(ValueError, match="not found"):
            pipeline.get_custom_metadata("does_not_exist")

        pipeline.close()

    def test_add_custom_metadata_overwrite_control(self):
        """overwrite flag should control replacement behavior."""
        pipeline = PipelineManager(show_progress=False, use_memmap=False)
        pipeline.add_custom_metadata("shared_key", {"value": 1})

        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_custom_metadata("shared_key", {"value": 2})

        pipeline.add_custom_metadata(
            "shared_key",
            {"value": 2},
            overwrite=True
        )
        assert pipeline.get_custom_metadata("shared_key")["value"] == 2
        pipeline.close()

    def test_add_custom_metadata_warns_when_large(self):
        """Large payloads should trigger warning based on threshold."""
        pipeline = PipelineManager(show_progress=False, use_memmap=False)
        payload = np.zeros((2048,), dtype=np.float64)

        with pytest.warns(RuntimeWarning, match="warning threshold"):
            pipeline.add_custom_metadata(
                name="large_payload",
                value=payload,
                max_size_gb=1e-9
            )

        pipeline.close()

    def test_custom_metadata_persisted_in_single_file_save_load(self):
        """Custom metadata must survive save/load roundtrip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = f"{temp_dir}/pipeline_state.pkl"
            cache_root = f"{temp_dir}/cache"
            load_cache_root = f"{temp_dir}/cache_load"

            pipeline = PipelineManager(
                cache_dir=cache_root,
                show_progress=False,
                use_memmap=False
            )
            payload = {
                "notes": ["alpha", "beta"],
                "score": 42
            }
            pipeline.add_custom_metadata("session_data", payload)
            pipeline.save_to_single_file(save_path)
            pipeline.close()

            loaded_pipeline = PipelineManager.load_from_single_file(
                save_path,
                cache_dir=load_cache_root,
                show_progress=False
            )
            loaded_payload = loaded_pipeline.get_custom_metadata("session_data")

            assert loaded_payload == payload
            loaded_pipeline.close()
