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

"""
Utility functions for saving and loading Python objects with memmap support.

This module provides utility class for saving and loading Python objects
with memmap support. Works with any Python object, not just TrajectoryData.
Preserves memmap properties correctly.
"""

from typing import Any, Union
import os

import numpy as np

from .helper.load_and_save_helper import LoadAndSaveHelper


class DataUtils:
    """
    Utility class for saving and loading Python objects with memory-mapped array support.

    Provides methods to serialize and deserialize Python objects that contain
    memory-mapped numpy arrays while preserving memmap properties and file
    references. Works with any Python object, not just mdxplain classes.

    Examples
    --------
    >>> # Save any object with memmap support
    >>> DataUtils.save_object(my_object, 'data/my_object.pkl')

    >>> # Load into existing object
    >>> new_object = MyClass()
    >>> DataUtils.load_object(new_object, 'data/my_object.pkl')
    """

    @staticmethod
    def save_object(obj: Any, save_path: str) -> None:
        """
        Save any Python object while preserving memory-mapped array properties.

        Parameters
        ----------
        obj : object
            Python object to save (can contain memmap arrays and DaskMDTrajectory objects)
        save_path : str
            File path for saving (should end with .pkl or .npy)

        Returns
        -------
        None
            Saves object to disk using numpy.save with pickle support

        Examples
        --------
        >>> # Save TrajectoryData object
        >>> DataUtils.save_object(traj_data, 'analysis/results.pkl')

        >>> # Save any custom object with memmaps
        >>> DataUtils.save_object(my_analysis, 'outputs/analysis.pkl')
        """
        LoadAndSaveHelper.save_object(obj, save_path)

    @staticmethod
    def load_object(obj: Any, load_path: str) -> None:
        """
        Load data into existing Python object while restoring memmap properties.

        Parameters
        ----------
        obj : object
            Existing Python object to load data into (will be modified in-place)
        load_path : str
            Path to saved object file (.pkl or .npy)

        Returns
        -------
        None
            Modifies obj in-place, restoring attributes and memmap connections

        Examples
        --------
        >>> # Load into TrajectoryData object
        >>> traj = TrajectoryData()
        >>> DataUtils.load_object(traj, 'analysis/results.pkl')

        >>> # Load into custom object
        >>> my_obj = MyAnalysisClass()
        >>> DataUtils.load_object(my_obj, 'outputs/analysis.pkl')
        """       
        LoadAndSaveHelper.load_object(obj, load_path)

    @staticmethod
    def get_cache_file_path(cache_name: str, cache_path: str = "./cache") -> str:
        """
        Get cache file path from cache_path and cache_name.

        Parameters
        ----------
        cache_name : str
            Name for the cache file (e.g., 'pca.dat', 'kernel_pca.dat')
        cache_path : str, default="./cache"
            Base cache path (can be directory or full file path)

        Returns
        -------
        str
            Full path to the cache file

        Examples
        --------
        >>> # With directory cache_path
        >>> path = DataUtils.get_cache_file_path("pca.dat", "./cache")
        >>> print(path)  # "./cache/pca.dat"

        >>> # With full file cache_path
        >>> path = DataUtils.get_cache_file_path("pca.dat", "./cache/my_data.dat")
        >>> print(path)  # "./cache/my_data.dat"
        """
        if cache_path:
            # Check if cache_path is a directory or full file path
            if cache_path.endswith(".dat") or "." in os.path.basename(cache_path):
                # Full file path provided, use it directly
                cache_dir = os.path.dirname(cache_path)
                os.makedirs(cache_dir, exist_ok=True)
                return cache_path
            else:
                # Directory path provided, append cache_name
                os.makedirs(cache_path, exist_ok=True)
                return os.path.join(cache_path, cache_name)
        else:
            # Default cache path
            default_path = "./cache"
            os.makedirs(default_path, exist_ok=True)
            return os.path.join(default_path, cache_name)

    @staticmethod
    def is_memmap_view(array: Any) -> bool:
        """
        Check whether an array is backed by a numpy memmap (including views).

        Parameters
        ----------
        array : Any
            Array or view to check.

        Returns
        -------
        bool
            True if the array is a memmap or view on a memmap.
        """
        base = array
        seen = set()
        while base is not None and id(base) not in seen:
            if isinstance(base, np.memmap):
                return True
            seen.add(id(base))
            base = getattr(base, "base", None)
        return False

    @staticmethod
    def get_type_key(type_obj: Union[str, type, object]) -> str:
        """
        Get the type key from a type object.

        This utility method handles conversion of various type formats
        (instances, classes, strings) to their string identifier.
        It is specially used for the conventions inside this software.

        Parameters
        ----------
        type_obj : str, class, or instance
            Type object to get key for (e.g., decomposition type, feature type)

        Returns
        -------
        str
            Type key string identifier

        Examples
        --------
        >>> DataUtils.get_type_key("pca")
        'pca'
        >>> DataUtils.get_type_key(PCA())
        'pca'
        >>> DataUtils.get_type_key(PCA)
        'pca'
        """
        if isinstance(type_obj, str):
            return type_obj
        elif hasattr(type_obj, "get_type_name"):
            return type_obj.get_type_name()
        else:
            return str(type_obj)
