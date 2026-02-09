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
Helper for remapping cache-backed runtime bindings after archive load.

When loading from a sharable archive, cache files are extracted into a fresh
runtime cache scope. This helper updates the loaded object graph so memmap
and zarr-backed attributes point to that new runtime scope instead of stale
original paths.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING
import os

import numpy as np

from ...utils.memmap_utils import MemmapUtils
from ...utils.path_utils import PathUtils

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData


class CacheRemapHelper:
    """
    Pipeline-local helper for remapping known cache-backed bindings.

    Notes
    -----
    This helper is intentionally software-specific: it only remaps paths and
    bindings used by mdxplain pipeline state.
    """

    # Per-class mapping:
    # "<class name>" -> {"<path attr>": {"memmap_attrs": (...), "path_kind": ...}}
    # path_kind:
    # - "file":      cache_path points to one concrete memmap file.
    # - "directory": cache_path points to a directory containing memmap files.
    _CLASS_MEMMAP_BINDINGS: Dict[str, Dict[str, Dict[str, Any]]] = {
        "FeatureData": {
            "cache_path": {
                "memmap_attrs": ("data",),
                "path_kind": "file",
            },
            "reduced_cache_path": {
                "memmap_attrs": ("reduced_data",),
                "path_kind": "file",
            },
        },
        "DecompositionData": {
            "cache_path": {
                "memmap_attrs": ("data",),
                "path_kind": "directory",
            },
        },
        "ClusterData": {
            "cache_path": {
                "memmap_attrs": ("labels",),
                "path_kind": "directory",
            },
        },
    }
    _ZARR_TRAJECTORY_CLASS_NAME = "DaskMDTrajectory"

    @staticmethod
    def remap_pipeline_memmaps(
        pipeline_data: PipelineData,
        runtime_cache_dir: str,
    ) -> None:
        """
        Remap all known memmap bindings in loaded pipeline data.

        Parameters
        ----------
        pipeline_data : PipelineData
            Loaded pipeline state whose cache-backed paths/memmaps should be
            rebound to the current runtime cache scope.
        runtime_cache_dir : str
            Absolute or relative runtime cache directory used by the current
            pipeline instance.

        Returns
        -------
        None
            Updates objects in-place.
        """
        normalized_runtime_cache_dir = PathUtils.prepare_directory_path(
            runtime_cache_dir,
            create=True,
            purpose="runtime cache directory",
        )
        visited_object_ids: Set[int] = set()
        CacheRemapHelper._traverse_pipeline_state(
            current_node=pipeline_data,
            runtime_cache_dir=normalized_runtime_cache_dir,
            visited_object_ids=visited_object_ids,
        )

    @staticmethod
    def _traverse_pipeline_state(
        current_node: Any,
        runtime_cache_dir: str,
        visited_object_ids: Set[int],
    ) -> None:
        """
        Traverse the loaded pipeline object graph and remap known bindings.

        Parameters
        ----------
        current_node : Any
            Current node in the object graph traversal.
        runtime_cache_dir : str
            Runtime cache directory for remapped paths.
        visited_object_ids : set of int
            IDs of objects already visited to prevent cycles.

        Returns
        -------
        None
            Traverses and updates objects in-place.
        """
        if current_node is None or isinstance(
            current_node, (str, bytes, int, float, bool, np.generic)
        ):
            return
        if isinstance(current_node, np.ndarray):
            return

        current_node_id = id(current_node)
        if current_node_id in visited_object_ids:
            return
        visited_object_ids.add(current_node_id)

        if isinstance(current_node, dict):
            for child in current_node.values():
                CacheRemapHelper._traverse_pipeline_state(
                    child,
                    runtime_cache_dir,
                    visited_object_ids,
                )
            return
        if isinstance(current_node, list):
            for child in current_node:
                CacheRemapHelper._traverse_pipeline_state(
                    child,
                    runtime_cache_dir,
                    visited_object_ids,
                )
            return
        if isinstance(current_node, tuple):
            for child in current_node:
                CacheRemapHelper._traverse_pipeline_state(
                    child,
                    runtime_cache_dir,
                    visited_object_ids,
                )
            return
        if isinstance(current_node, set):
            for child in current_node:
                CacheRemapHelper._traverse_pipeline_state(
                    child,
                    runtime_cache_dir,
                    visited_object_ids,
                )
            return
        if not hasattr(current_node, "__dict__"):
            return

        CacheRemapHelper._remap_known_memmap_bindings_for_object(
            current_node, runtime_cache_dir
        )
        CacheRemapHelper._remap_known_zarr_bindings_for_object(
            current_node, runtime_cache_dir
        )
        CacheRemapHelper._remap_selection_matrix_cache_paths_if_present(
            current_node, runtime_cache_dir
        )
        for child in vars(current_node).values():
            CacheRemapHelper._traverse_pipeline_state(
                child,
                runtime_cache_dir,
                visited_object_ids,
            )

    @staticmethod
    def _remap_known_memmap_bindings_for_object(
        current_node: Any, runtime_cache_dir: str
    ) -> None:
        """
        Remap known memmap path/array attributes for supported mdxplain classes.

        Parameters
        ----------
        current_node : Any
            Candidate object whose class may hold remappable memmap bindings.
        runtime_cache_dir : str
            Runtime cache directory for remapped file paths.

        Returns
        -------
        None
            Updates matching attributes in-place.
        """
        class_name = current_node.__class__.__name__
        class_bindings = CacheRemapHelper._CLASS_MEMMAP_BINDINGS.get(class_name)
        if class_bindings is None:
            return

        for path_attr_name, binding_info in class_bindings.items():
            memmap_attr_names: Tuple[str, ...] = tuple(
                binding_info.get("memmap_attrs", ())
            )
            path_kind: str = str(binding_info.get("path_kind", "file"))
            remapped_cache_path = CacheRemapHelper._remap_cache_path_attribute(
                current_node,
                path_attr_name,
                runtime_cache_dir,
                path_kind=path_kind,
            )
            for memmap_attr_name in memmap_attr_names:
                CacheRemapHelper._reopen_memmap_array_attribute(
                    current_node,
                    memmap_attr_name,
                    remapped_cache_path,
                    path_attr_name=path_attr_name,
                    path_kind=path_kind,
                )

    @staticmethod
    def _remap_cache_path_attribute(
        current_node: Any,
        path_attr_name: str,
        runtime_cache_dir: str,
        *,
        path_kind: str,
    ) -> Optional[str]:
        """
        Remap one cache path attribute into the runtime cache directory.

        Parameters
        ----------
        current_node : Any
            Object containing the path attribute.
        path_attr_name : str
            Name of the path attribute to remap (e.g. ``cache_path``).
        runtime_cache_dir : str
            Runtime cache directory for the remapped path.
        path_kind : str
            Either ``"file"`` or ``"directory"``.

        Returns
        -------
        str or None
            Remapped file/directory path if attribute exists and is a
            non-empty string, otherwise None.
        """
        current_path = getattr(current_node, path_attr_name, None)
        if not isinstance(current_path, str) or not current_path:
            return None

        remapped_raw_path = os.path.join(
            runtime_cache_dir,
            os.path.basename(current_path),
        )
        if path_kind == "directory":
            remapped_path = PathUtils.prepare_directory_path(
                remapped_raw_path,
                create=True,
                purpose=f"{path_attr_name} remap directory",
            )
        else:
            remapped_path = PathUtils.prepare_file_path(
                remapped_raw_path,
                create_parent=True,
                purpose=f"{path_attr_name} remap file path",
            )
        setattr(current_node, path_attr_name, remapped_path)
        return remapped_path

    @staticmethod
    def _reopen_memmap_array_attribute(
        current_node: Any,
        memmap_attr_name: str,
        remapped_cache_path: Optional[str],
        *,
        path_attr_name: str,
        path_kind: str,
    ) -> None:
        """
        Reopen one memmap array attribute from a remapped cache path.

        Parameters
        ----------
        current_node : Any
            Object that may contain a memmap attribute.
        memmap_attr_name : str
            Attribute name expected to hold a memmap array.
        remapped_cache_path : str or None
            Remapped cache file/directory path associated with the array attribute.
        path_attr_name : str
            Path attribute on ``current_node`` associated with this memmap.
        path_kind : str
            Either ``"file"`` or ``"directory"``.

        Returns
        -------
        None
            Rebinds attribute in-place when it is a memmap.
        """
        if not remapped_cache_path:
            return
        current_value = getattr(current_node, memmap_attr_name, None)
        if not isinstance(current_value, np.memmap):
            return

        old_memmap = current_value
        memmap_filename = getattr(old_memmap, "filename", None)
        if not memmap_filename:
            return

        if path_kind == "directory":
            runtime_cache_dir = os.path.dirname(remapped_cache_path)
            old_parent_name = os.path.basename(
                os.path.dirname(memmap_filename)
            )
            remapped_directory = remapped_cache_path
            if old_parent_name:
                remapped_directory = PathUtils.prepare_directory_path(
                    os.path.join(runtime_cache_dir, old_parent_name),
                    create=True,
                    purpose=f"{path_attr_name} remap directory",
                )
                setattr(current_node, path_attr_name, remapped_directory)
            remapped_file_path = PathUtils.prepare_file_path(
                os.path.join(
                    remapped_directory,
                    os.path.basename(memmap_filename),
                ),
                create_parent=True,
                purpose=f"{memmap_attr_name} remap file path",
            )
        else:
            remapped_file_path = remapped_cache_path

        MemmapUtils.close_memmap_view(old_memmap)
        reopened_memmap = MemmapUtils.create_memmap(
            path=remapped_file_path,
            dtype=old_memmap.dtype,
            mode="r",
            shape=tuple(old_memmap.shape),
            close_existing=False,
        )
        setattr(current_node, memmap_attr_name, reopened_memmap)

    @staticmethod
    def _remap_known_zarr_bindings_for_object(
        current_node: Any, runtime_cache_dir: str
    ) -> None:
        """
        Remap Dask trajectory zarr cache bindings to runtime cache scope.

        Parameters
        ----------
        current_node : Any
            Candidate object in traversal.
        runtime_cache_dir : str
            Runtime cache directory for remapped zarr paths.

        Returns
        -------
        None
            Updates zarr binding and reloads trajectory caches in-place.
        """
        if (
            current_node.__class__.__name__
            != CacheRemapHelper._ZARR_TRAJECTORY_CLASS_NAME
        ):
            return

        current_zarr_path = getattr(current_node, "zarr_cache_path", None)
        if not isinstance(current_zarr_path, str) or not current_zarr_path:
            return

        remapped_zarr_path = PathUtils.prepare_directory_path(
            os.path.join(runtime_cache_dir, os.path.basename(current_zarr_path)),
            create=False,
            purpose="trajectory zarr cache path remap",
        )
        if not os.path.exists(remapped_zarr_path):
            raise FileNotFoundError(
                f"Expected remapped trajectory zarr cache at '{remapped_zarr_path}' "
                "after archive extraction, but it does not exist."
            )

        current_node.zarr_cache_path = remapped_zarr_path
        if hasattr(current_node, "_cache_dir"):
            current_node._cache_dir = os.path.dirname(remapped_zarr_path)
        join_stack_helper = getattr(current_node, "_join_stack_helper", None)
        if join_stack_helper is not None and hasattr(join_stack_helper, "cache_dir"):
            join_stack_helper.cache_dir = os.path.dirname(remapped_zarr_path)

        reload_from_cache = getattr(current_node, "_reload_from_cache", None)
        if callable(reload_from_cache):
            reload_from_cache()

    @staticmethod
    def _remap_selection_matrix_cache_paths_if_present(
        current_node: Any, runtime_cache_dir: str
    ) -> None:
        """
        Remap PipelineData selection-matrix cache path references.

        PipelineData stores selection matrix cache entries as
        ``_matrix_cache[key] = (memmap_path, frame_mapping)``.
        These path references must also point to the runtime cache scope.

        Parameters
        ----------
        current_node : Any
            Candidate object in traversal.
        runtime_cache_dir : str
            Runtime cache directory for remapped matrix paths.

        Returns
        -------
        None
            Updates ``_matrix_cache`` entries in-place when present.
        """
        if current_node.__class__.__name__ != "PipelineData":
            return

        matrix_cache = getattr(current_node, "_matrix_cache", None)
        if not isinstance(matrix_cache, dict):
            return

        for cache_key, cache_entry in list(matrix_cache.items()):
            if not (
                isinstance(cache_entry, tuple)
                and len(cache_entry) == 2
                and isinstance(cache_entry[0], str)
            ):
                continue
            existing_memmap_path, frame_mapping = cache_entry
            remapped_matrix_path = PathUtils.prepare_file_path(
                os.path.join(
                    runtime_cache_dir,
                    os.path.basename(existing_memmap_path),
                ),
                create_parent=True,
                purpose="selection matrix cache path remap",
            )
            matrix_cache[cache_key] = (remapped_matrix_path, frame_mapping)
