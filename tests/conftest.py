# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
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

"""Global test configuration and fixtures."""

import gc
import pytest
import shutil
import time
from pathlib import Path

from mdxplain.utils.memmap_utils import MemmapUtils


def _close_memmaps_under_path(path: Path) -> None:
    """
    Force-close tracked memmaps only under the given path.
    """
    MemmapUtils.close_memmaps_under_path(path)


def _safe_rmtree(path: Path, retries: int = 8, delay_seconds: float = 0.15) -> bool:
    """
    Best-effort directory removal with retries for Windows file locks.
    """
    for attempt in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return True
        except (OSError, PermissionError):
            if attempt == retries - 1:
                return False
            _close_memmaps_under_path(path)
            gc.collect()
            time.sleep(delay_seconds)
    return not path.exists()


def _cleanup_cache_dirs(project_root: Path) -> None:
    """
    Cleanup shared cache folders used by tests.
    """
    cache_dirs = [
        project_root / "cache",
        project_root / "test_cache",
    ]
    failed = []
    for cache_dir in cache_dirs:
        _close_memmaps_under_path(cache_dir)
        if not _safe_rmtree(cache_dir):
            failed.append(str(cache_dir))

    # Most tests assume ./cache exists.
    (project_root / "cache").mkdir(exist_ok=True)

    if failed:
        raise RuntimeError(
            "Failed to cleanup test cache directories (likely open memmap lock): "
            + ", ".join(failed)
        )


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """
    Auto-cleanup fixture that removes test artifacts after each test.
    
    This fixture runs automatically after every test to clean up:
    
    - Cache directories and files
    - Temporary data files (.dat, .npy, .memmap)
    - Pipeline data remnants
    
    Returns:
    --------
    None
        Yields control to test, then performs cleanup
    """
    project_root = Path(__file__).parent.parent

    # Pre-cleanup: avoid stale locks/files from previous failed tests.
    _cleanup_cache_dirs(project_root)

    # Run test
    yield
    
    # Post-cleanup
    _cleanup_cache_dirs(project_root)
