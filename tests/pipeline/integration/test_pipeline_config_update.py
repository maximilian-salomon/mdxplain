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

"""Integration tests for pipeline configuration update functionality."""

import pytest
import tempfile
import os
import re
import numpy as np

from mdxplain.pipeline.manager.pipeline_manager import PipelineManager
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestPipelineConfigUpdate:
    """Test pipeline configuration update functionality."""

    @staticmethod
    def _assert_scoped_cache_path(base_dir: str, scoped_path: str) -> None:
        """
        Assert scoped cache path layout: <base>/cache_<uuid>_<YYYYMMDD_HHMMSS>.
        """
        normalized_base = os.path.abspath(os.path.normpath(base_dir))
        normalized_scoped = os.path.abspath(os.path.normpath(scoped_path))

        assert normalized_scoped.startswith(normalized_base + os.sep)

        rel = os.path.relpath(normalized_scoped, normalized_base)
        assert re.fullmatch(r"cache_[0-9a-f]{32}_\d{8}_\d{6}", rel) is not None

    def test_get_initial_config(self):
        """
        Test that get_config returns initial configuration values.

        Validates that the configuration is properly initialized and
        accessible through the get_config method.
        """
        pipeline = PipelineManager(
            chunk_size=1000,
            cache_dir="./test_cache",
            use_memmap=True
        )

        config = pipeline.get_config()

        assert config["chunk_size"] == 1000
        self._assert_scoped_cache_path("./test_cache", config["cache_dir"])
        assert config["use_memmap"] is True

    def test_update_chunk_size(self, capsys):
        """
        Test updating chunk_size configuration.

        Validates that chunk_size updates are propagated to all
        managers and reflected in get_config output.
        """
        pipeline = PipelineManager(chunk_size=1000)

        # Update chunk size
        pipeline.update_config(chunk_size=5000)

        # Check configuration was updated
        config = pipeline.get_config()
        assert config["chunk_size"] == 5000

        # Check managers were updated
        assert pipeline._data.chunk_size == 5000
        assert pipeline._trajectory_manager.chunk_size == 5000
        assert pipeline._feature_manager.chunk_size == 5000
        assert pipeline._decomposition_manager.chunk_size == 5000
        assert pipeline._feature_importance_manager.chunk_size == 5000

        # Check print output
        captured = capsys.readouterr()
        assert "Configuration updated successfully" in captured.out
        assert "chunk_size: 5000" in captured.out

    def test_update_cache_dir(self, capsys):
        """
        Test updating cache_dir configuration.

        Validates that cache_dir updates create the directory and
        are propagated to all managers.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = PipelineManager(cache_dir="./test_cache")

            new_cache_dir = os.path.join(temp_dir, "new_cache")

            # Update cache directory
            pipeline.update_config(cache_dir=new_cache_dir)

            # Check configuration was updated
            config = pipeline.get_config()
            self._assert_scoped_cache_path(new_cache_dir, config["cache_dir"])
            scoped_cache_dir = config["cache_dir"]

            # Check directory was created
            assert os.path.exists(new_cache_dir)
            assert os.path.exists(scoped_cache_dir)

            # Check managers were updated
            assert pipeline._data.cache_dir == scoped_cache_dir
            assert pipeline._trajectory_manager.cache_dir == scoped_cache_dir
            assert pipeline._feature_manager.cache_dir == scoped_cache_dir
            assert pipeline._decomposition_manager.cache_dir == scoped_cache_dir
            assert pipeline._cluster_manager.cache_dir == scoped_cache_dir
            assert pipeline._feature_importance_manager.cache_dir == scoped_cache_dir

            # Check print output
            captured = capsys.readouterr()
            assert "Configuration updated successfully" in captured.out
            assert f"cache_dir: {scoped_cache_dir}" in captured.out

    def test_update_use_memmap(self, capsys):
        """
        Test updating use_memmap configuration.

        Validates that use_memmap updates are propagated to all
        relevant managers.
        """
        pipeline = PipelineManager(use_memmap=False)

        # Update memmap setting
        pipeline.update_config(use_memmap=True)

        # Check configuration was updated
        config = pipeline.get_config()
        assert config["use_memmap"] is True

        # Check managers were updated
        assert pipeline._data.use_memmap is True
        assert pipeline._trajectory_manager.use_memmap is True
        assert pipeline._feature_manager.use_memmap is True
        assert pipeline._decomposition_manager.use_memmap is True
        assert pipeline._feature_importance_manager.use_memmap is True

        # Check print output
        captured = capsys.readouterr()
        assert "Configuration updated successfully" in captured.out
        assert "use_memmap: True" in captured.out

    def test_update_multiple_config_params(self, capsys):
        """
        Test updating multiple configuration parameters simultaneously.

        Validates that multiple parameter updates work correctly
        in a single call.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = PipelineManager(
                chunk_size=1000,
                cache_dir="./test_cache",
                use_memmap=False
            )

            new_cache_dir = os.path.join(temp_dir, "multi_cache")

            # Update multiple parameters
            pipeline.update_config(
                chunk_size=8000,
                cache_dir=new_cache_dir,
                use_memmap=True
            )

            # Check all configurations were updated
            config = pipeline.get_config()
            assert config["chunk_size"] == 8000
            self._assert_scoped_cache_path(new_cache_dir, config["cache_dir"])
            scoped_cache_dir = config["cache_dir"]
            assert config["use_memmap"] is True

            # Check directory was created
            assert os.path.exists(new_cache_dir)
            assert os.path.exists(scoped_cache_dir)

            # Check print output contains all updates
            captured = capsys.readouterr()
            assert "Configuration updated successfully" in captured.out
            assert "chunk_size: 8000" in captured.out
            assert f"cache_dir: {scoped_cache_dir}" in captured.out
            assert "use_memmap: True" in captured.out

    def test_update_config_validation_errors(self):
        """
        Test that update_config properly validates parameters.

        Validates that invalid parameter values raise appropriate
        ValueError exceptions.
        """
        pipeline = PipelineManager()

        # Test invalid chunk_size
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            pipeline.update_config(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            pipeline.update_config(chunk_size=-100)

        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            pipeline.update_config(chunk_size="invalid")

        # Test invalid cache_dir
        with pytest.raises(
            ValueError, match="cache directory must be a valid path-like value"
        ):
            pipeline.update_config(cache_dir=123)

        # Test invalid use_memmap
        with pytest.raises(ValueError, match="use_memmap must be a boolean"):
            pipeline.update_config(use_memmap="yes")

    def test_update_config_invalid_directory(self):
        """
        Test that update_config handles directory creation failures.

        Validates that OSError is raised when cache directory
        cannot be created.
        """
        pipeline = PipelineManager()

        # Try to create directory in invalid location (assuming /root is not writable)
        invalid_dir = "/root/invalid_cache_dir"

        with pytest.raises(OSError, match="Cannot create cache directory"):
            pipeline.update_config(cache_dir=invalid_dir)

    def test_config_persistence_after_operations(self):
        """
        Test that configuration changes persist across real pipeline operations.

        Validates that updated configuration values remain consistent
        and are not overwritten by actual pipeline operations like
        loading trajectories and computing features.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cache_dir = os.path.join(temp_dir, "old_cache")
            new_cache_dir = os.path.join(temp_dir, "new_cache")

            # Start with old cache directory and enable memmap for file caching
            pipeline = PipelineManager(chunk_size=1000, cache_dir=old_cache_dir, use_memmap=True)
            old_runtime_cache_dir = pipeline.get_config()["cache_dir"]

            # Setup trajectory data
            mock_traj = MockTrajectoryFactory.create_triangle_atoms(n_frames=5, seed=42)
            pipeline._data.trajectory_data.trajectories = [mock_traj]
            pipeline._data.trajectory_data.n_frames = 5
            pipeline._data.trajectory_data.n_atoms = 3
            pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
            pipeline._data.trajectory_data.res_label_data = {
                0: [{"seqid": 0, "full_name": "RES_0"}, {"seqid": 1, "full_name": "RES_1"}, {"seqid": 2, "full_name": "RES_2"}]
            }

            # Update to new cache directory (keeping memmap enabled)
            pipeline.update_config(cache_dir=new_cache_dir)
            new_runtime_cache_dir = pipeline.get_config()["cache_dir"]

            # Perform operation that creates cache files
            pipeline.feature.add.distances(excluded_neighbors=0, force=True)

            # Verify both directories exist
            assert os.path.exists(old_cache_dir), "Old cache base directory should still exist"
            assert os.path.exists(new_cache_dir), "New cache base directory should exist"
            assert os.path.exists(old_runtime_cache_dir), "Old runtime cache dir should still exist"
            assert os.path.exists(new_runtime_cache_dir), "New runtime cache dir should exist"

            # Verify specific cache file exists in new directory only
            expected_cache_file = "distances_test_traj.dat"
            old_cache_file_path = os.path.join(old_runtime_cache_dir, expected_cache_file)
            new_cache_file_path = os.path.join(new_runtime_cache_dir, expected_cache_file)

            # Old cache directory should NOT contain the cache file
            assert not os.path.exists(old_cache_file_path), f"Cache file should NOT exist in old directory: {old_cache_file_path}"

            # New cache directory MUST contain the cache file
            assert os.path.exists(new_cache_file_path), f"Cache file must exist in new directory: {new_cache_file_path}"

            # Verify cache file has expected minimum size (should contain distance data)
            cache_file_size = os.path.getsize(new_cache_file_path)
            assert cache_file_size >= 50, f"Cache file too small: {cache_file_size} bytes (expected >= 50 bytes)"

            # Verify configuration was correctly updated
            config = pipeline.get_config()
            self._assert_scoped_cache_path(new_cache_dir, config["cache_dir"])
            assert config["use_memmap"] is True
            pipeline.close()

    def test_config_update_disables_memmap_no_cache_files(self):
        """
        Test that updating use_memmap=False prevents cache file creation.

        Validates that when use_memmap is disabled via config update,
        no cache files are created during feature operations.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_cache_dir = os.path.join(temp_dir, "cache")

            # Start with memmap enabled
            pipeline = PipelineManager(use_memmap=True, cache_dir=base_cache_dir)
            runtime_cache_dir = pipeline.get_config()["cache_dir"]

            # Setup trajectory data
            mock_traj = MockTrajectoryFactory.create_triangle_atoms(n_frames=5, seed=42)
            pipeline._data.trajectory_data.trajectories = [mock_traj]
            pipeline._data.trajectory_data.n_frames = 5
            pipeline._data.trajectory_data.n_atoms = 3
            pipeline._data.trajectory_data.trajectory_names = ["test_traj"]
            pipeline._data.trajectory_data.res_label_data = {
                0: [{"seqid": 0, "full_name": "RES_0"}, {"seqid": 1, "full_name": "RES_1"}, {"seqid": 2, "full_name": "RES_2"}]
            }

            # Update to disable memmap
            pipeline.update_config(use_memmap=False)

            # Perform operation that would create cache files if memmap was enabled
            pipeline.feature.add.distances(excluded_neighbors=0, force=True)

            # Verify cache directory exists but is empty
            assert os.path.exists(base_cache_dir), "Cache base directory should exist"
            assert os.path.exists(runtime_cache_dir), "Runtime cache directory should exist"

            # Verify specific cache file does NOT exist
            expected_cache_file = "distances_test_traj.dat"
            cache_file_path = os.path.join(runtime_cache_dir, expected_cache_file)
            assert not os.path.exists(cache_file_path), f"Cache file should NOT exist when memmap disabled: {cache_file_path}"

            # Verify no memmap cache files exist (other directories like structure_viz/ are OK)
            cache_files = os.listdir(runtime_cache_dir)
            memmap_files = []
            for f in cache_files:
                fpath = os.path.join(runtime_cache_dir, f)
                if os.path.isfile(fpath):
                    try:
                        arr = np.load(fpath, mmap_mode='r')
                        if isinstance(arr, np.memmap):
                            memmap_files.append(f)
                    except:
                        pass
            assert len(memmap_files) == 0, f"No memmap cache files should exist when memmap disabled, but found: {memmap_files}"

            # Verify configuration was correctly updated
            config = pipeline.get_config()
            assert config["use_memmap"] is False

    def test_partial_config_updates(self):
        """
        Test that partial configuration updates don't affect unchanged parameters.

        Validates that updating only one parameter leaves others unchanged.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            initial_cache_dir = os.path.join(temp_dir, "initial_cache")
            os.makedirs(initial_cache_dir, exist_ok=True)

            pipeline = PipelineManager(
                chunk_size=2000,
                cache_dir=initial_cache_dir,
                use_memmap=True
            )
            initial_runtime_cache_dir = pipeline.get_config()["cache_dir"]

            # Update only chunk_size
            pipeline.update_config(chunk_size=4000)

            # Verify only chunk_size changed
            config = pipeline.get_config()
            assert config["chunk_size"] == 4000
            assert config["cache_dir"] == initial_runtime_cache_dir
            assert config["use_memmap"] is True

            # Update only use_memmap
            pipeline.update_config(use_memmap=False)

            # Verify only use_memmap changed
            config = pipeline.get_config()
            assert config["chunk_size"] == 4000
            assert config["cache_dir"] == initial_runtime_cache_dir
            assert config["use_memmap"] is False

    def test_update_cache_dir_preserves_scope_suffix(self):
        """
        update_config(cache_dir=...) should preserve pipeline scope suffix.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            initial_base = os.path.join(temp_dir, "initial")
            new_base = os.path.join(temp_dir, "new")
            pipeline = PipelineManager(cache_dir=initial_base, use_memmap=True)
            before = os.path.basename(pipeline.get_config()["cache_dir"])
            assert re.fullmatch(r"cache_[0-9a-f]{32}_\d{8}_\d{6}", before)

            pipeline.update_config(cache_dir=new_base)
            after_path = pipeline.get_config()["cache_dir"]
            after = os.path.basename(after_path)

            assert after == before
            assert os.path.dirname(after_path) == os.path.abspath(os.path.normpath(new_base))

    def test_two_pipelines_same_base_have_different_scoped_cache_dirs(self):
        """
        Two pipelines with same base cache should get distinct scoped cache dirs.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base = os.path.join(temp_dir, "shared")
            p1 = PipelineManager(cache_dir=base, use_memmap=True)
            p2 = PipelineManager(cache_dir=base, use_memmap=True)
            try:
                c1 = p1.get_config()["cache_dir"]
                c2 = p2.get_config()["cache_dir"]

                assert c1 != c2
                assert os.path.dirname(c1) == os.path.abspath(os.path.normpath(base))
                assert os.path.dirname(c2) == os.path.abspath(os.path.normpath(base))
            finally:
                p1.close()
                p2.close()

    def test_update_cache_dir_multiple_switches_preserve_scope_and_write_target(self):
        """
        Multiple cache_dir switches should preserve scope suffix and write to latest cache.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_1 = os.path.join(temp_dir, "cache_base_1")
            base_2 = os.path.join(temp_dir, "cache_base_2")
            base_3 = os.path.join(temp_dir, "cache_base_3")

            pipeline = PipelineManager(chunk_size=256, cache_dir=base_1, use_memmap=True)
            try:
                scope_name = os.path.basename(pipeline.get_config()["cache_dir"])

                mock_traj = MockTrajectoryFactory.create_triangle_atoms(n_frames=6, seed=7)
                pipeline._data.trajectory_data.trajectories = [mock_traj]
                pipeline._data.trajectory_data.n_frames = mock_traj.n_frames
                pipeline._data.trajectory_data.n_atoms = mock_traj.n_atoms
                pipeline._data.trajectory_data.trajectory_names = ["switch_traj"]
                pipeline._data.trajectory_data.res_label_data = {
                    0: [
                        {"seqid": 0, "full_name": "RES_0"},
                        {"seqid": 1, "full_name": "RES_1"},
                        {"seqid": 2, "full_name": "RES_2"},
                    ]
                }

                runtime_paths = []
                for cache_base in (base_1, base_2, base_3):
                    pipeline.update_config(cache_dir=cache_base)
                    runtime_cache = pipeline.get_config()["cache_dir"]
                    runtime_paths.append(runtime_cache)

                    assert os.path.basename(runtime_cache) == scope_name
                    assert os.path.dirname(runtime_cache) == os.path.abspath(
                        os.path.normpath(cache_base)
                    )

                    pipeline.feature.add.distances(excluded_neighbors=0, force=True)
                    expected_cache_file = os.path.join(
                        runtime_cache, "distances_switch_traj.dat"
                    )
                    assert os.path.exists(expected_cache_file)

                assert runtime_paths[0] != runtime_paths[1]
                assert runtime_paths[1] != runtime_paths[2]
            finally:
                pipeline.close()
