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

import os

import numpy as np


class DataUtils:
    """
    Utility class for saving and loading Python objects with memory-mapped array support.

    Provides methods to serialize and deserialize Python objects that contain
    memory-mapped numpy arrays while preserving memmap properties and file
    references. Works with any Python object, not just mdxplain classes.

    Examples:
    ---------
    >>> # Save any object with memmap support
    >>> DataUtils.save_object(my_object, 'data/my_object.pkl')

    >>> # Load into existing object
    >>> new_object = MyClass()
    >>> DataUtils.load_object(new_object, 'data/my_object.pkl')
    """

    @staticmethod
    def save_object(obj, save_path):
        """
        Save any Python object while preserving memory-mapped array properties.

        Parameters:
        -----------
        obj : object
            Python object to save (can contain memmap arrays)
        save_path : str
            File path for saving (should end with .pkl or .npy)

        Returns:
        --------
        None
            Saves object to disk using numpy.save with pickle support

        Examples:
        ---------
        >>> # Save TrajectoryData object
        >>> DataUtils.save_object(traj_data, 'analysis/results.pkl')

        >>> # Save any custom object with memmaps
        >>> DataUtils.save_object(my_analysis, 'outputs/analysis.pkl')
        """
        save_obj = DataUtils._prepare_save_object(obj)
        np.save(save_path, save_obj, allow_pickle=True)

    @staticmethod
    def load_object(obj, load_path):
        """
        Load data into existing Python object while restoring memmap properties.

        Parameters:
        -----------
        obj : object
            Existing Python object to load data into (will be modified in-place)
        load_path : str
            Path to saved object file (.pkl or .npy)

        Returns:
        --------
        None
            Modifies obj in-place, restoring attributes and memmap connections

        Examples:
        ---------
        >>> # Load into TrajectoryData object
        >>> traj = TrajectoryData()
        >>> DataUtils.load_object(traj, 'analysis/results.pkl')

        >>> # Load into custom object
        >>> my_obj = MyAnalysisClass()
        >>> DataUtils.load_object(my_obj, 'outputs/analysis.pkl')
        """
        loaded_obj = np.load(load_path, allow_pickle=True).item()
        DataUtils._restore_object_attributes(obj, loaded_obj)

    @staticmethod
    def _prepare_save_object(obj):
        """
        Prepare object for saving by converting memmaps to metadata.

        Prepare object attributes for saving.

        Parameters:
        -----------
        obj : object
            Object to prepare

        Returns:
        --------
        dict
            Dictionary with attributes, memmaps converted to metadata
        """
        save_obj = {}

        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(obj, attr_name)

            if isinstance(attr_value, np.memmap):
                save_obj[attr_name] = DataUtils._save_memmap_info(attr_value, attr_name)
            else:
                save_obj[attr_name] = attr_value

        return save_obj

    @staticmethod
    def _save_memmap_info(memmap_array, attr_name):
        """
        Save memmap metadata for later restoration.

        Parameters:
        -----------
        memmap_array : np.memmap
            Memory-mapped array to save metadata for
        attr_name : str
            Attribute name for error messages

        Returns:
        --------
        dict
            Metadata dictionary with shape, dtype, and file path
        """
        if not (hasattr(memmap_array, "filename") and memmap_array.filename):
            raise ValueError(
                f"Memmap {attr_name} has no filename - this should not happen!"
            )

        return {
            "_is_memmap": True,
            "dtype": memmap_array.dtype,
            "shape": memmap_array.shape,
            "mode": getattr(memmap_array, "mode", "r"),
            "original_path": memmap_array.filename,
        }

    @staticmethod
    def _restore_object_attributes(obj, loaded_obj):
        """
        Restore object attributes from loaded data.

        Parameters:
        -----------
        obj : object
            Target object to restore attributes into
        loaded_obj : dict
            Loaded data dictionary with attributes

        Returns:
        --------
        None
            Modifies obj in-place
        """
        for attr_name, attr_value in loaded_obj.items():
            if isinstance(attr_value, dict) and attr_value.get("_is_memmap", False):
                restored_memmap = DataUtils._restore_memmap(obj, attr_value, attr_name)
                setattr(obj, attr_name, restored_memmap)
            else:
                setattr(obj, attr_name, attr_value)

    @staticmethod
    def _restore_memmap(obj, memmap_info, attr_name):
        """
        Restore memmap from metadata.

        Parameters:
        -----------
        obj : object
            Target object for memmap restoration
        memmap_info : dict
            Metadata dictionary with memmap information
        attr_name : str
            Attribute name for the memmap

        Returns:
        --------
        np.memmap or None
            Restored memmap or None if file not found
        """
        original_path = memmap_info["original_path"]

        # Try to restore from original path first
        restored = DataUtils._try_restore_from_path(original_path, memmap_info)
        if restored is not None:
            return restored

        # Try alternative path if object supports it
        return DataUtils._try_restore_from_alternative_path(
            obj, attr_name, memmap_info, original_path
        )

    @staticmethod
    def _try_restore_from_path(path, memmap_info):
        """
        Try to restore memmap from given path.

        Parameters:
        -----------
        path : str
            File path to try for memmap restoration
        memmap_info : dict
            Metadata dictionary with memmap information

        Returns:
        --------
        np.memmap or None
            Restored memmap if file exists, None otherwise
        """
        if os.path.exists(path):
            return np.memmap(
                path,
                dtype=memmap_info["dtype"],
                mode="r",
                shape=tuple(memmap_info["shape"]),
            )
        return None

    @staticmethod
    def _try_restore_from_alternative_path(obj, attr_name, memmap_info, original_path):
        """
        Try to restore memmap from alternative path.

        Parameters:
        -----------
        obj : object
            Target object for memmap restoration
        attr_name : str
            Attribute name for the memmap
        memmap_info : dict
            Metadata dictionary with memmap information
        original_path : str
            Original file path that failed

        Returns:
        --------
        np.memmap or None
            Restored memmap from alternative path or None if not possible
        """
        if not DataUtils._check_supports_alternative_path(obj, attr_name):
            return None

        target_path = getattr(obj, f"{attr_name}_path")
        if DataUtils._check_is_invalid_alternative_path(target_path, original_path):
            return None

        return DataUtils._try_restore_from_path(target_path, memmap_info)

    @staticmethod
    def _check_supports_alternative_path(obj, attr_name):
        """
        Check if object supports alternative path restoration.

        Parameters:
        -----------
        obj : object
            Object to check for alternative path support
        attr_name : str
            Attribute name to check for path support

        Returns:
        --------
        bool
            True if object supports alternative path restoration
        """
        return (
            hasattr(obj, "use_memmap")
            and obj.use_memmap
            and hasattr(obj, f"{attr_name}_path")
        )

    @staticmethod
    def _check_is_invalid_alternative_path(target_path, original_path):
        """
        Check if alternative path is invalid.

        Parameters:
        -----------
        target_path : str
            Alternative path to check
        original_path : str
            Original path for comparison

        Returns:
        --------
        bool
            True if alternative path is invalid
        """
        return target_path == original_path or not os.path.exists(target_path)

    @staticmethod
    def get_cache_file_path(cache_name, cache_path="./cache"):
        """
        Get cache file path from cache_path and cache_name.

        Parameters:
        -----------
        cache_path : str or None
            Base cache path (can be directory or full file path)
        cache_name : str
            Name for the cache file (e.g., 'pca.dat', 'kernel_pca.dat')

        Returns:
        --------
        str
            Full path to the cache file

        Examples:
        ---------
        >>> # With directory cache_path
        >>> path = DataUtils.get_cache_file_path("./cache", "pca.dat")
        >>> print(path)  # "./cache/pca.dat"

        >>> # With full file cache_path
        >>> path = DataUtils.get_cache_file_path("./cache/my_data.dat", "pca.dat")
        >>> print(path)  # "./cache/my_data.dat"

        >>> # With None cache_path
        >>> path = DataUtils.get_cache_file_path(None, "pca.dat")
        >>> print(path)  # "./cache/pca.dat"
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

    @staticmethod
    def get_type_key(type_obj):
        """
        Get the type key from a type object.

        This utility method handles conversion of various type formats
        (instances, classes, strings) to their string identifier.
        It is specially used for the conventions inside this software.

        Parameters:
        -----------
        type_obj : str, class, or instance
            Type object to get key for (e.g., decomposition type, feature type)

        Returns:
        --------
        str
            Type key string identifier

        Examples:
        ---------
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
