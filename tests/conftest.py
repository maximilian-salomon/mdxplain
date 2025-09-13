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

import pytest
import shutil
from pathlib import Path


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
    # Run test
    yield
    
    # Cleanup after test
    project_root = Path(__file__).parent.parent
    
    # Clean cache directory
    cache_dir = project_root / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(exist_ok=True)  # Recreate empty cache dir
    
    # Remove temporary data files
    temp_patterns = [
        "*.dat",
        "*.npy", 
        "*.memmap"
    ]
    
    for pattern in temp_patterns:
        for temp_file in project_root.rglob(pattern):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
            except (OSError, PermissionError):
                # Some files might be in use, skip them
                pass
    
    # Clean pytest cache in tests directory
    pytest_cache = project_root / "tests" / ".pytest_cache"
    if pytest_cache.exists():
        try:
            shutil.rmtree(pytest_cache)
        except (OSError, PermissionError):
            pass
    
    # Clean __pycache__ directories in tests
    for pycache in (project_root / "tests").rglob("__pycache__"):
        try:
            if pycache.is_dir():
                shutil.rmtree(pycache)
        except (OSError, PermissionError):
            pass