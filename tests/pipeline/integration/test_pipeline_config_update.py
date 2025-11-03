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
import numpy as np

from mdxplain.pipeline.managers.pipeline_manager import PipelineManager
from tests.fixtures.mock_trajectory_factory import MockTrajectoryFactory


class TestPipelineConfigUpdate:
    """Test pipeline configuration update functionality."""

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
        assert config["cache_dir"] == "./test_cache"
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
            assert config["cache_dir"] == new_cache_dir

            # Check directory was created
            assert os.path.exists(new_cache_dir)

            # Check managers were updated
            assert pipeline._data.cache_dir == new_cache_dir
            assert pipeline._trajectory_manager.cache_dir == new_cache_dir
            assert pipeline._feature_manager.cache_dir == new_cache_dir
            assert pipeline._decomposition_manager.cache_dir == new_cache_dir
            assert pipeline._cluster_manager.cache_dir == new_cache_dir
            assert pipeline._feature_importance_manager.cache_dir == new_cache_dir

            # Check print output
            captured = capsys.readouterr()
            assert "Configuration updated successfully" in captured.out
            assert f"cache_dir: {new_cache_dir}" in captured.out

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
            assert config["cache_dir"] == new_cache_dir
            assert config["use_memmap"] is True

            # Check directory was created
            assert os.path.exists(new_cache_dir)

            # Check print output contains all updates
            captured = capsys.readouterr()
            assert "Configuration updated successfully" in captured.out
            assert "chunk_size: 8000" in captured.out
            assert f"cache_dir: {new_cache_dir}" in captured.out
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
        with pytest.raises(ValueError, match="cache_dir must be a string"):
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

            # Perform operation that creates cache files
            pipeline.feature.add.distances(excluded_neighbors=0, force=True)

            # Verify both directories exist
            assert os.path.exists(old_cache_dir), "Old cache directory should still exist"
            assert os.path.exists(new_cache_dir), "New cache directory should exist"

            # Verify specific cache file exists in new directory only
            expected_cache_file = "distances_test_traj.dat"
            old_cache_file_path = os.path.join(old_cache_dir, expected_cache_file)
            new_cache_file_path = os.path.join(new_cache_dir, expected_cache_file)

            # Old cache directory should NOT contain the cache file
            assert not os.path.exists(old_cache_file_path), f"Cache file should NOT exist in old directory: {old_cache_file_path}"

            # New cache directory MUST contain the cache file
            assert os.path.exists(new_cache_file_path), f"Cache file must exist in new directory: {new_cache_file_path}"

            # Verify cache file has expected minimum size (should contain distance data)
            cache_file_size = os.path.getsize(new_cache_file_path)
            assert cache_file_size >= 50, f"Cache file too small: {cache_file_size} bytes (expected >= 50 bytes)"

            # Verify configuration was correctly updated
            config = pipeline.get_config()
            assert config["cache_dir"] == new_cache_dir
            assert config["use_memmap"] is True

    def test_config_update_disables_memmap_no_cache_files(self):
        """
        Test that updating use_memmap=False prevents cache file creation.

        Validates that when use_memmap is disabled via config update,
        no cache files are created during feature operations.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")

            # Start with memmap enabled
            pipeline = PipelineManager(use_memmap=True, cache_dir=cache_dir)

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
            assert os.path.exists(cache_dir), "Cache directory should exist"

            # Verify specific cache file does NOT exist
            expected_cache_file = "distances_test_traj.dat"
            cache_file_path = os.path.join(cache_dir, expected_cache_file)
            assert not os.path.exists(cache_file_path), f"Cache file should NOT exist when memmap disabled: {cache_file_path}"

            # Verify no memmap cache files exist (other directories like structure_viz/ are OK)
            cache_files = os.listdir(cache_dir)
            memmap_files = []
            for f in cache_files:
                fpath = os.path.join(cache_dir, f)
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

            # Update only chunk_size
            pipeline.update_config(chunk_size=4000)

            # Verify only chunk_size changed
            config = pipeline.get_config()
            assert config["chunk_size"] == 4000
            assert config["cache_dir"] == initial_cache_dir
            assert config["use_memmap"] is True

            # Update only use_memmap
            pipeline.update_config(use_memmap=False)

            # Verify only use_memmap changed
            config = pipeline.get_config()
            assert config["chunk_size"] == 4000
            assert config["cache_dir"] == initial_cache_dir
            assert config["use_memmap"] is False
