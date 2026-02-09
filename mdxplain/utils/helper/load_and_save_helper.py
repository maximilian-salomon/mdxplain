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

"""
Helper class for saving and loading objects with memmap metadata.
"""

from typing import Any, Dict, Optional
import os
import pickle
import warnings

import numpy as np

from ..memmap_utils import MemmapUtils
from ..path_utils import PathUtils


class LoadAndSaveHelper:
    """
    Helper class for save/load operations with memmap support.
    """

    @staticmethod
    def save_object(obj: Any, save_path: str) -> None:
        """
        Save an object while preserving memmap metadata.

        Parameters
        ----------
        obj : Any
            Object to save.
        save_path : str
            Path to the output pickle file.

        Returns
        -------
        None
        """
        payload = LoadAndSaveHelper._prepare_save_object(obj)
        replacements: list = []
        visited: set = set()
        try:
            LoadAndSaveHelper._replace_memmaps_for_save(
                payload, replacements, visited
            )
            LoadAndSaveHelper._dump_pickle(payload, save_path)
        finally:
            LoadAndSaveHelper._restore_replacements(replacements)

    @staticmethod
    def load_object(obj: Any, load_path: str) -> None:
        """
        Load data into an object while restoring memmaps.

        Parameters
        ----------
        obj : Any
            Target object to populate.
        load_path : str
            Path to the input pickle file.

        Returns
        -------
        None
        """
        loaded_obj = LoadAndSaveHelper._load_pickle(load_path)
        LoadAndSaveHelper._restore_object_attributes(obj, loaded_obj)
        LoadAndSaveHelper._restore_memmaps_after_load(obj)

    @staticmethod
    def peek_cache_dir(load_path: str) -> Optional[str]:
        """
        Read and return ``cache_dir`` from a saved payload when available.

        Parameters
        ----------
        load_path : str
            Path to the saved pickle payload.

        Returns
        -------
        str or None
            Saved ``cache_dir`` value if present and non-empty, otherwise None.
        """
        payload = LoadAndSaveHelper._load_pickle(load_path)
        if not isinstance(payload, dict):
            return None
        candidate = payload.get("cache_dir")
        if not isinstance(candidate, str):
            return None
        candidate = candidate.strip()
        return candidate if candidate else None

    @staticmethod
    def _dump_pickle(payload: Dict[str, Any], save_path: str) -> None:
        """
        Dump a payload to a pickle file.

        Parameters
        ----------
        payload : dict
            Payload to serialize.
        save_path : str
            Path to the output pickle file.

        Returns
        -------
        None
        """
        save_path = PathUtils.prepare_file_path(
            save_path,
            create_parent=True,
            purpose="save path",
        )
        with open(save_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=4)

    @staticmethod
    def _load_pickle(load_path: str) -> Dict[str, Any]:
        """
        Load a payload from a pickle file.

        Parameters
        ----------
        load_path : str
            Path to the input pickle file.

        Returns
        -------
        dict
            Loaded payload.
        """
        load_path = PathUtils.prepare_file_path(
            load_path,
            create_parent=False,
            purpose="load path",
        )
        with open(load_path, "rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _prepare_save_object(obj: Any) -> Dict[str, Any]:
        """
        Prepare a save payload from public attributes.

        Parameters
        ----------
        obj : Any
            Object to prepare.

        Returns
        -------
        dict
            Attribute payload for saving.
        """
        # Persist only public instance state.
        # Private attributes are implementation details and intentionally
        # excluded from the serialized payload.
        return {
            key: value
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }

    @staticmethod
    def _restore_object_attributes(obj: Any, loaded_obj: Dict[str, Any]) -> None:
        """
        Restore object attributes from loaded payload.

        Parameters
        ----------
        obj : Any
            Target object to populate.
        loaded_obj : dict
            Loaded attribute payload.

        Returns
        -------
        None
        """
        for attr_name, attr_value in loaded_obj.items():
            setattr(obj, attr_name, attr_value)

    @staticmethod
    def _replace_memmaps_for_save(
        value: Any,
        replacements: list,
        visited: set,
        parent: Any = None,
        key: Any = None,
        is_attr: bool = False,
    ) -> None:
        """
        Recursively replace memmaps with metadata dicts.

        Parameters
        ----------
        value : Any
            Current value to process.
        replacements : list
            Collected replacements for later restoration.
        visited : set
            Set of visited object IDs.
        parent : Any, optional
            Parent container or object.
        key : Any, optional
            Index, key, or attribute name.
        is_attr : bool, default=False
            Whether the parent assignment is an attribute.

        Returns
        -------
        None
        """
        if LoadAndSaveHelper._replace_memmap_value(
            value, replacements, parent, key, is_attr
        ):
            return
        if LoadAndSaveHelper._is_atomic_value(value):
            return
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        if isinstance(value, dict):
            LoadAndSaveHelper._replace_in_mapping(
                value, replacements, visited
            )
        elif isinstance(value, list):
            LoadAndSaveHelper._replace_in_sequence(
                value, replacements, visited
            )
        elif isinstance(value, tuple):
            LoadAndSaveHelper._replace_in_tuple(
                value, replacements, visited, parent, key, is_attr
            )
        elif hasattr(value, "__dict__"):
            LoadAndSaveHelper._replace_in_object(
                value, replacements, visited
            )

    @staticmethod
    def _replace_memmap_value(
        value: Any,
        replacements: list,
        parent: Any,
        key: Any,
        is_attr: bool,
    ) -> bool:
        """
        Replace a memmap with metadata if needed.

        Parameters
        ----------
        value : Any
            Candidate value.
        replacements : list
            Collected replacements.
        parent : Any
            Parent container or object.
        key : Any
            Index, key, or attribute name.
        is_attr : bool
            Whether the parent assignment is an attribute.

        Returns
        -------
        bool
            True if a replacement happened.
        """
        if isinstance(value, dict) and value.get("_is_memmap", False):
            return True
        if not isinstance(value, np.memmap):
            return False
        meta = LoadAndSaveHelper._save_memmap_info(
            value, str(key or "memmap")
        )
        LoadAndSaveHelper._assign_value(parent, key, meta, is_attr)
        replacements.append((parent, key, value, is_attr))
        return True

    @staticmethod
    def _replace_in_mapping(
        mapping: dict, replacements: list, visited: set
    ) -> None:
        """
        Replace memmaps in a mapping in-place.

        Parameters
        ----------
        mapping : dict
            Mapping to process.
        replacements : list
            Collected replacements.
        visited : set
            Set of visited object IDs.

        Returns
        -------
        None
        """
        for child_key in list(mapping.keys()):
            LoadAndSaveHelper._replace_memmaps_for_save(
                mapping[child_key],
                replacements,
                visited,
                mapping,
                child_key,
                False,
            )

    @staticmethod
    def _replace_in_sequence(
        sequence: list, replacements: list, visited: set
    ) -> None:
        """
        Replace memmaps in a list in-place.

        Parameters
        ----------
        sequence : list
            List to process.
        replacements : list
            Collected replacements.
        visited : set
            Set of visited object IDs.

        Returns
        -------
        None
        """
        for idx in range(len(sequence)):
            LoadAndSaveHelper._replace_memmaps_for_save(
                sequence[idx],
                replacements,
                visited,
                sequence,
                idx,
                False,
            )

    @staticmethod
    def _replace_in_tuple(
        values: tuple,
        replacements: list,
        visited: set,
        parent: Any,
        key: Any,
        is_attr: bool,
    ) -> None:
        """
        Replace memmaps in a tuple by rebuilding it if needed.

        Parameters
        ----------
        values : tuple
            Tuple to process.
        replacements : list
            Collected replacements.
        visited : set
            Set of visited object IDs.
        parent : Any
            Parent container or object.
        key : Any
            Index, key, or attribute name.
        is_attr : bool
            Whether the parent assignment is an attribute.

        Returns
        -------
        None
        """
        new_items = list(values)
        changed = False
        for idx, item in enumerate(values):
            if isinstance(item, np.memmap):
                new_items[idx] = LoadAndSaveHelper._save_memmap_info(
                    item, str(idx)
                )
                changed = True
            else:
                LoadAndSaveHelper._replace_memmaps_for_save(
                    item, replacements, visited
                )
        if changed:
            LoadAndSaveHelper._assign_value(
                parent, key, tuple(new_items), is_attr
            )
            replacements.append((parent, key, values, is_attr))

    @staticmethod
    def _replace_in_object(
        obj: Any, replacements: list, visited: set
    ) -> None:
        """
        Replace memmaps in object attributes.

        Parameters
        ----------
        obj : Any
            Object to process.
        replacements : list
            Collected replacements.
        visited : set
            Set of visited object IDs.

        Returns
        -------
        None
        """
        for attr_name, attr_value in vars(obj).items():
            LoadAndSaveHelper._replace_memmaps_for_save(
                attr_value,
                replacements,
                visited,
                obj,
                attr_name,
                True,
            )

    @staticmethod
    def _restore_replacements(replacements: list) -> None:
        """
        Restore values replaced during memmap-safe saving.

        Parameters
        ----------
        replacements : list
            List of (parent, key, original_value, is_attr) tuples.

        Returns
        -------
        None
        """
        for parent, key, original_value, is_attr in reversed(replacements):
            LoadAndSaveHelper._assign_value(
                parent, key, original_value, is_attr
            )

    @staticmethod
    def _assign_value(
        parent: Any, key: Any, value: Any, is_attr: bool
    ) -> None:
        """
        Assign a value into a parent container or attribute.

        Parameters
        ----------
        parent : Any
            Parent container or object.
        key : Any
            Index, key, or attribute name.
        value : Any
            Value to assign.
        is_attr : bool
            Whether to set attribute on parent.

        Returns
        -------
        None
        """
        if parent is None:
            return
        if is_attr:
            setattr(parent, key, value)
        else:
            parent[key] = value

    @staticmethod
    def _is_atomic_value(value: Any) -> bool:
        """
        Check whether a value should not be traversed.

        Parameters
        ----------
        value : Any
            Value to check.

        Returns
        -------
        bool
            True if value should be treated as atomic.
        """
        if value is None:
            return True
        if isinstance(value, (str, bytes, int, float, bool, np.generic)):
            return True
        if isinstance(value, np.ndarray):
            return True
        if callable(value):
            return True
        return False

    @staticmethod
    def _restore_memmaps_after_load(obj: Any) -> None:
        """
        Restore nested memmaps after loading.

        Parameters
        ----------
        obj : Any
            Root object to process.

        Returns
        -------
        None
        """
        visited: set = set()
        LoadAndSaveHelper._restore_memmaps_recursive(
            obj, visited, None, None, False
        )

    @staticmethod
    def _restore_memmaps_recursive(
        value: Any,
        visited: set,
        parent: Any,
        key: Any,
        is_attr: bool,
    ) -> None:
        """
        Recursively restore memmaps from metadata.

        Parameters
        ----------
        value : Any
            Current value to process.
        visited : set
            Set of visited object IDs.
        parent : Any
            Parent container or object.
        key : Any
            Index, key, or attribute name.
        is_attr : bool
            Whether the parent assignment is an attribute.

        Returns
        -------
        None
        """
        if LoadAndSaveHelper._restore_memmap_value(
            value, parent, key, is_attr
        ):
            return
        if LoadAndSaveHelper._is_atomic_value(value):
            return
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        if isinstance(value, dict):
            for child_key in list(value.keys()):
                LoadAndSaveHelper._restore_memmaps_recursive(
                    value[child_key], visited, value, child_key, False
                )
        elif isinstance(value, list):
            for idx in range(len(value)):
                LoadAndSaveHelper._restore_memmaps_recursive(
                    value[idx], visited, value, idx, False
                )
        elif isinstance(value, tuple):
            LoadAndSaveHelper._restore_in_tuple(
                value, visited, parent, key, is_attr
            )
        elif hasattr(value, "__dict__"):
            for attr_name, attr_value in vars(value).items():
                LoadAndSaveHelper._restore_memmaps_recursive(
                    attr_value, visited, value, attr_name, True
                )

    @staticmethod
    def _restore_memmap_value(
        value: Any, parent: Any, key: Any, is_attr: bool
    ) -> bool:
        """
        Restore a single memmap value from metadata.

        Parameters
        ----------
        value : Any
            Candidate value.
        parent : Any
            Parent container or object.
        key : Any
            Index, key, or attribute name.
        is_attr : bool
            Whether the parent assignment is an attribute.

        Returns
        -------
        bool
            True if restoration happened.
        """
        if not (isinstance(value, dict) and value.get("_is_memmap", False)):
            return False
        restored = LoadAndSaveHelper._restore_memmap_from_info(
            value, parent, key, is_attr
        )
        LoadAndSaveHelper._assign_value(parent, key, restored, is_attr)
        return True

    @staticmethod
    def _restore_memmap_from_info(
        info: Dict[str, Any],
        parent: Any,
        key: Any,
        is_attr: bool,
    ) -> Optional[np.memmap]:
        """
        Restore memmap from metadata dict.

        Parameters
        ----------
        info : dict
            Memmap metadata dict.
        parent : Any
            Parent container or object.
        key : Any
            Index, key, or attribute name.
        is_attr : bool
            Whether the parent assignment is an attribute.

        Returns
        -------
        np.memmap or None
            Restored memmap or None if unavailable.
        """
        if is_attr and parent is not None:
            return LoadAndSaveHelper._restore_memmap(
                parent, info, str(key)
            )
        return LoadAndSaveHelper._try_restore_from_path(
            info["original_path"], info
        )

    @staticmethod
    def _restore_in_tuple(
        values: tuple,
        visited: set,
        parent: Any,
        key: Any,
        is_attr: bool,
    ) -> None:
        """
        Restore memmaps inside a tuple by rebuilding it if needed.

        Parameters
        ----------
        values : tuple
            Tuple to process.
        visited : set
            Set of visited object IDs.
        parent : Any
            Parent container or object.
        key : Any
            Index, key, or attribute name.
        is_attr : bool
            Whether the parent assignment is an attribute.

        Returns
        -------
        None
        """
        new_items = list(values)
        changed = False
        for idx, item in enumerate(values):
            if isinstance(item, dict) and item.get("_is_memmap", False):
                new_items[idx] = LoadAndSaveHelper._restore_memmap_from_info(
                    item, parent, idx, False
                )
                changed = True
            else:
                LoadAndSaveHelper._restore_memmaps_recursive(
                    item, visited, None, None, False
                )
        if changed:
            LoadAndSaveHelper._assign_value(
                parent, key, tuple(new_items), is_attr
            )

    @staticmethod
    def _save_memmap_info(memmap_array: np.memmap, attr_name: str) -> Dict[str, Any]:
        """
        Save memmap metadata for later restoration.

        Parameters
        ----------
        memmap_array : np.memmap
            Memory-mapped array to save metadata for.
        attr_name : str
            Attribute name for error messages.

        Returns
        -------
        dict
            Metadata dictionary with shape, dtype, and file path.
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
    def _restore_memmap(
        obj: Any, memmap_info: Dict[str, Any], attr_name: str
    ) -> Optional[np.memmap]:
        """
        Restore memmap from metadata.

        Parameters
        ----------
        obj : Any
            Target object for memmap restoration.
        memmap_info : dict
            Metadata dictionary with memmap information.
        attr_name : str
            Attribute name for the memmap.

        Returns
        -------
        np.memmap or None
            Restored memmap or None if file not found.
        """
        original_path: str = memmap_info["original_path"]
        restored = LoadAndSaveHelper._try_restore_from_path(
            original_path, memmap_info
        )
        if restored is not None:
            return restored

        alternative_path = LoadAndSaveHelper._resolve_alternative_path(
            obj, attr_name, original_path
        )
        if alternative_path is not None:
            restored = LoadAndSaveHelper._try_restore_from_path(
                alternative_path, memmap_info
            )
            if restored is not None:
                return restored

        LoadAndSaveHelper._warn_missing_memmap_path(
            attr_name=attr_name,
            original_path=original_path,
            alternative_path=alternative_path,
        )
        return None

    @staticmethod
    def _try_restore_from_path(
        path: str, memmap_info: Dict[str, Any]
    ) -> Optional[np.memmap]:
        """
        Try to restore memmap from given path.

        Parameters
        ----------
        path : str
            File path to try for memmap restoration.
        memmap_info : dict
            Metadata dictionary with memmap information.

        Returns
        -------
        np.memmap or None
            Restored memmap if file exists, None otherwise.
        """
        if os.path.exists(path):
            memmap_array = MemmapUtils.create_memmap(
                path=path,
                dtype=memmap_info["dtype"],
                mode="r",
                shape=tuple(memmap_info["shape"]),
                close_existing=False,
            )
            return memmap_array
        return None

    @staticmethod
    def _resolve_alternative_path(
        obj: Any,
        attr_name: str,
        original_path: str,
    ) -> Optional[str]:
        """
        Resolve alternative memmap restore path from object state.

        Parameters
        ----------
        obj : Any
            Target object for memmap restoration.
        attr_name : str
            Attribute name for the memmap.
        original_path : str
            Original file path that failed.

        Returns
        -------
        str or None
            Alternative path when available and valid, otherwise None.
        """
        if not LoadAndSaveHelper._check_supports_alternative_path(
            obj, attr_name
        ):
            return None
        target_path: str = getattr(obj, f"{attr_name}_path")
        if LoadAndSaveHelper._check_is_invalid_alternative_path(
            target_path, original_path
        ):
            return None
        return target_path

    @staticmethod
    def _warn_missing_memmap_path(
        attr_name: str,
        original_path: str,
        alternative_path: Optional[str],
    ) -> None:
        """
        Emit runtime warning when a saved memmap cannot be restored.

        Parameters
        ----------
        attr_name : str
            Attribute name currently being restored.
        original_path : str
            Original path from serialized metadata.
        alternative_path : str or None
            Alternative path considered during restore.

        Returns
        -------
        None
            Warns and continues loading with ``None`` for this attribute.
        """
        message = (
            f"Memmap restore skipped for '{attr_name}': "
            f"missing file '{original_path}'"
        )
        if alternative_path is not None:
            message += f" and alternative '{alternative_path}'"
        message += (
            ". Memmap-backed data expects the original cache files from the "
            "save environment. This can happen when loading from a different "
            "working directory or with relative cache paths. Run the load "
            "from the same location used for save_to_single_file, or request "
            "a sharable archive created via create_sharable_archive "
            "(archive includes cache data). Continuing load with attribute "
            "set to None."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _check_supports_alternative_path(obj: Any, attr_name: str) -> bool:
        """
        Check if object supports alternative path restoration.

        Parameters
        ----------
        obj : Any
            Object to check for alternative path support.
        attr_name : str
            Attribute name to check for path support.

        Returns
        -------
        bool
            True if object supports alternative path restoration.
        """
        return (
            hasattr(obj, "use_memmap")
            and obj.use_memmap
            and hasattr(obj, f"{attr_name}_path")
        )

    @staticmethod
    def _check_is_invalid_alternative_path(
        target_path: str, original_path: str
    ) -> bool:
        """
        Check if alternative path is invalid.

        Parameters
        ----------
        target_path : str
            Alternative path to check.
        original_path : str
            Original path for comparison.

        Returns
        -------
        bool
            True if alternative path is invalid.
        """
        return target_path == original_path or not os.path.exists(target_path)
