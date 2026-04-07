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
Matrix operations helper for feature selection system.

Provides efficient matrix construction directly with proper shape calculation,
memory management, and frame mapping instead of collecting and merging matrices.
"""
from __future__ import annotations

import hashlib
import os
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, TYPE_CHECKING

from ...utils.memmap_utils import MemmapUtils
from ...utils.path_utils import PathUtils
from ...utils.resource_utils import ResourceUtils

if TYPE_CHECKING:
    from ..entities.pipeline_data import PipelineData


class SelectionMatrixHelper:
    """
    Helper class for efficient matrix construction from selection data.
    
    Builds matrices directly with correct shape. Supports both regular arrays
    and memory-mapped files for large datasets.
    """

    @staticmethod
    def _determine_matrix_dtype(
        pipeline_data: PipelineData,
        feature_selector_name: str,
    ) -> np.dtype:
        """
        Determine minimal shared dtype required for all selected columns.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object.
        feature_selector_name : str
            Feature selector name.

        Returns
        -------
        np.dtype
            Minimal common dtype across all selected data sources.
        """
        selector_data = pipeline_data.selected_feature_data[feature_selector_name]
        all_results = selector_data.get_all_results()
        dtypes: List[np.dtype] = []

        for feature_type, selection_info in all_results.items():
            feature_data_dict = pipeline_data.feature_data.get(feature_type, {})
            trajectory_indices = selection_info.get("trajectory_indices", {})

            for traj_idx, traj_selection in trajectory_indices.items():
                feature_data = feature_data_dict.get(traj_idx)
                if feature_data is None:
                    continue

                flags = traj_selection.get("use_reduced", [])
                if not flags:
                    continue

                has_reduced = any(flags)
                has_original = any(not flag for flag in flags)

                if has_original and feature_data.data is not None:
                    dtypes.append(
                        SelectionMatrixHelper._resolve_effective_dtype(
                            feature_data.data,
                            feature_data.feature_metadata,
                        )
                    )

                if has_reduced:
                    if feature_data.reduced_data is not None:
                        dtypes.append(
                            SelectionMatrixHelper._resolve_effective_dtype(
                                feature_data.reduced_data,
                                feature_data.reduced_feature_metadata,
                            )
                        )
                    elif feature_data.data is not None:
                        dtypes.append(
                            SelectionMatrixHelper._resolve_effective_dtype(
                                feature_data.data,
                                feature_data.feature_metadata,
                            )
                        )

        if not dtypes:
            return np.dtype(pipeline_data.dtype)

        return np.result_type(*dtypes)

    @staticmethod
    def _resolve_effective_dtype(
        source_data: np.ndarray,
        metadata: Optional[Dict[str, Any]],
    ) -> np.dtype:
        """
        Resolve effective dtype after optional categorical conversion.

        Parameters
        ----------
        source_data : np.ndarray
            Source array.
        metadata : dict, optional
            Feature metadata for matrix mapping.

        Returns
        -------
        np.dtype
            Effective dtype for matrix storage.
        """
        source_dtype = np.dtype(source_data.dtype)
        if source_dtype.kind == "U":
            matrix_mapping = (metadata or {}).get("matrix_mapping", {})
            if matrix_mapping:
                return np.dtype(np.int8)
        return source_dtype

    @staticmethod
    def build_selection_matrix(
        pipeline_data: PipelineData,
        feature_selector_name: str,
        data_selector_name: Optional[str] = None,
        build_frame_mapping: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[int, Tuple[int, int]]]]:
        """
        Build selection matrix with efficient memory usage and caching.

        Uses caching when use_memmap=True to avoid rebuilding identical matrices.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object containing all data
        feature_selector_name : str
            Name of the feature selection
        data_selector_name : str, optional
            Name of data selector for frame filtering
        build_frame_mapping : bool, default=False
            Whether to build and return row-to-frame mapping.

        Returns
        -------
        Tuple[np.ndarray, Dict[int, Tuple[int, int]] or None]
            Complete matrix and optional frame mapping.
        """
        cache_key = pipeline_data._get_matrix_cache_key(
            feature_selector_name, data_selector_name
        )
        matrix_dtype = SelectionMatrixHelper._determine_matrix_dtype(
            pipeline_data, feature_selector_name
        )

        # Try loading from cache
        if pipeline_data.use_memmap and cache_key in pipeline_data._matrix_cache:
            result = SelectionMatrixHelper._load_from_cache(
                pipeline_data,
                cache_key,
                feature_selector_name,
                data_selector_name,
                matrix_dtype,
                build_frame_mapping,
            )
            if result is not None:
                return result

        # Cache miss - build new matrix
        return SelectionMatrixHelper._build_new_matrix(
            pipeline_data,
            cache_key,
            feature_selector_name,
            data_selector_name,
            matrix_dtype,
            build_frame_mapping,
        )
    
    @staticmethod
    def _calculate_matrix_shape(
        pipeline_data: PipelineData, feature_selector_name: str, data_selector_name: Optional[str]
    ) -> Tuple[int, int]:
        """
        Calculate final matrix shape efficiently.
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        feature_selector_name : str
            Selection name
        data_selector_name : str, optional
            Data selector name
            
        Returns
        -------
        Tuple[int, int]
            (n_rows, n_columns) for final matrix
        """
        # Get number of columns from FeatureSelectorData
        selector_data = pipeline_data.selected_feature_data[feature_selector_name]
        n_cols = selector_data.get_n_columns()
        
        if n_cols is None:
            raise ValueError(f"Selection '{feature_selector_name}' not processed yet. Run select() first.")
        
        # Calculate number of rows
        if data_selector_name is None:
            # Extract relevant trajectories from selector data
            all_results = selector_data.get_all_results()
            relevant_trajectories = set()
            for feature_type, selection_info in all_results.items():
                if "trajectory_indices" in selection_info:
                    relevant_trajectories.update(selection_info["trajectory_indices"].keys())
            
            # Sum frames only from selected trajectories
            n_rows = sum(
                pipeline_data.trajectory_data.trajectories[idx].n_frames 
                for idx in relevant_trajectories
                if idx < len(pipeline_data.trajectory_data.trajectories)
            )
        else:
            # Filtered frames - get from data selector
            data_selector = pipeline_data.data_selector_data[data_selector_name]
            n_rows = data_selector.n_selected_frames
        
        return n_rows, n_cols

    @staticmethod
    def _load_from_cache(
        pipeline_data: PipelineData,
        cache_key: str,
        feature_selector_name: str,
        data_selector_name: Optional[str],
        matrix_dtype: np.dtype,
        build_frame_mapping: bool,
    ) -> Optional[Tuple[np.ndarray, Optional[Dict[int, Tuple[int, int]]]]]:
        """
        Load matrix and frame mapping from cache.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        cache_key : str
            Cache key for lookup
        feature_selector_name : str
            Feature selector name
        data_selector_name : str, optional
            Data selector name
        matrix_dtype : np.dtype
            Matrix dtype expected for this selection.
        build_frame_mapping : bool
            Whether frame mapping is required for the caller.

        Returns
        -------
        Optional[Tuple[np.ndarray, Dict[int, Tuple[int, int]] or None]]
            Matrix and optional frame mapping, or None if cache invalid.
        """
        memmap_path, frame_mapping = pipeline_data._matrix_cache[cache_key]
        memmap_path = PathUtils.prepare_file_path(memmap_path)

        # Verify cached file exists
        if not os.path.exists(memmap_path):
            return None

        # Calculate expected shape
        n_rows, n_cols = SelectionMatrixHelper._calculate_matrix_shape(
            pipeline_data, feature_selector_name, data_selector_name
        )

        # Validate shape matches file size (prevent reading stale cache with wrong shape)
        file_size = os.path.getsize(memmap_path)
        dtype_size = np.dtype(matrix_dtype).itemsize
        expected_size = n_rows * n_cols * dtype_size

        if file_size != expected_size:
            # Shape mismatch - invalidate cache (will be overwritten on rebuild)
            del pipeline_data._matrix_cache[cache_key]
            return None

        # Load cached memmap
        matrix = MemmapUtils.create_memmap(
            path=memmap_path,
            dtype=matrix_dtype,
            mode="r+",
            shape=(n_rows, n_cols),
            close_existing=False,
        )

        if build_frame_mapping and frame_mapping is None:
            frame_mapping = SelectionMatrixHelper._build_frame_mapping_only(
                pipeline_data,
                feature_selector_name,
                data_selector_name,
            )
            pipeline_data._matrix_cache[cache_key] = (memmap_path, frame_mapping)

        if not build_frame_mapping:
            return matrix, None
        return matrix, frame_mapping

    @staticmethod
    def _build_new_matrix(
        pipeline_data: PipelineData,
        cache_key: str,
        feature_selector_name: str,
        data_selector_name: Optional[str],
        matrix_dtype: np.dtype,
        build_frame_mapping: bool,
    ) -> Tuple[np.ndarray, Optional[Dict[int, Tuple[int, int]]]]:
        """
        Build new matrix and frame mapping, store in cache.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        cache_key : str
            Cache key for storage
        feature_selector_name : str
            Feature selector name
        data_selector_name : str, optional
            Data selector name
        matrix_dtype : np.dtype
            Matrix dtype used for allocation.
        build_frame_mapping : bool
            Whether to build frame mapping.

        Returns
        -------
        Tuple[np.ndarray, Dict[int, Tuple[int, int]] or None]
            Matrix and optional frame mapping
        """
        # Calculate matrix shape
        n_rows, n_cols = SelectionMatrixHelper._calculate_matrix_shape(
            pipeline_data, feature_selector_name, data_selector_name
        )

        # Create matrix
        matrix, memmap_path = SelectionMatrixHelper._create_matrix(
            (n_rows, n_cols), pipeline_data.use_memmap,
            pipeline_data.cache_dir, feature_selector_name, data_selector_name, cache_key,
            matrix_dtype
        )

        # Fill matrix and create frame mapping
        frame_mapping = SelectionMatrixHelper._fill_matrix(
            matrix,
            pipeline_data,
            feature_selector_name,
            data_selector_name,
            build_frame_mapping,
        )

        # Store in cache if memmap enabled
        if pipeline_data.use_memmap and cache_key and memmap_path:
            pipeline_data._matrix_cache[cache_key] = (memmap_path, frame_mapping)

        return matrix, frame_mapping

    @staticmethod
    def _create_matrix(
        shape: Tuple[int, int],
        use_memmap: bool,
        cache_dir: str,
        feature_selector_name: str,
        data_selector_name: Optional[str],
        cache_key: str,
        dtype: type
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Create matrix with optimal memory management.

        Parameters
        ----------
        shape : Tuple[int, int]
            Matrix shape (rows, columns)
        use_memmap : bool
            Whether to use memory mapping
        cache_dir : str
            Cache directory for memmap files
        feature_selector_name : str
            Feature selector name
        data_selector_name : str, optional
            Data selector name
        cache_key : str
            Unique cache key for feature/data selector combination
        dtype : type
            Data type for matrix (float32 or float64)

        Returns
        -------
        Tuple[np.ndarray, Optional[str]]
            Matrix and memmap path (None if not using memmap)
        """
        if use_memmap:
            # Generate unique memmap path for each cache key to avoid collisions
            cache_filename = SelectionMatrixHelper._build_memmap_cache_filename(
                feature_selector_name, data_selector_name, cache_key
            )
            memmap_path = PathUtils.get_cache_file_path(
                cache_path=cache_dir,
                cache_name=cache_filename
            )

            # Create new memmap
            matrix = MemmapUtils.create_memmap(
                path=memmap_path,
                dtype=dtype,
                mode="w+",
                shape=shape,
            )

            return matrix, memmap_path

        return np.zeros(shape, dtype=dtype), None

    @staticmethod
    def _build_memmap_cache_filename(
        feature_selector_name: str, data_selector_name: Optional[str], cache_key: str
    ) -> str:
        """
        Build a collision-safe memmap filename for a selector/cache-key pair.

        Parameters
        ----------
        feature_selector_name : str
            Feature selector name (for readability in filename)
        data_selector_name : str, optional
            Data selector name (for readability in filename)
        cache_key : str
            Unique cache key used by PipelineData

        Returns
        -------
        str
            Stable memmap filename
        """
        safe_feature = SelectionMatrixHelper._sanitize_filename_component(
            feature_selector_name, "selector"
        )
        safe_data_selector = SelectionMatrixHelper._sanitize_filename_component(
            data_selector_name if data_selector_name is not None else "all_frames",
            "all_frames",
        )
        key_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
        return f"selection_matrix_{safe_feature}_{safe_data_selector}_{key_hash}.dat"

    @staticmethod
    def _sanitize_filename_component(
        value: str, fallback: str, limit: int = 40
    ) -> str:
        """
        Sanitize a filename component to a safe ASCII token.

        Parameters
        ----------
        value : str
            Raw component value to sanitize.
        fallback : str
            Replacement token used if sanitization yields an empty string.
        limit : int, default=40
            Maximum number of characters to keep.

        Returns
        -------
        str
            Sanitized component containing only ``[A-Za-z0-9_-]``.
        """
        safe = "".join(
            ch if (ch.isascii() and (ch.isalnum() or ch in {"_", "-"})) else "_"
            for ch in value
        ).strip("_")
        if not safe:
            safe = fallback
        return safe[:limit]

    @staticmethod
    def _fill_matrix(
        matrix: np.ndarray,
        pipeline_data: PipelineData,
        name: str,
        data_selector_name: Optional[str],
        build_frame_mapping: bool,
    ) -> Optional[Dict[int, Tuple[int, int]]]:
        """
        Fill matrix with selected feature data and optionally create frame mapping.
        
        Parameters
        ----------
        matrix : np.ndarray
            Pre-allocated matrix to fill
        pipeline_data : PipelineData
            Pipeline data object
        name : str
            Selection name
        data_selector_name : str, optional
            Data selector name
        build_frame_mapping : bool
            Whether to build frame mapping.
            
        Returns
        -------
        Dict[int, Tuple[int, int]] or None
            Optional frame mapping {global_idx: (traj_idx, local_idx)}.
        """
        current_row = 0
        is_memmap = isinstance(matrix, np.memmap)
        if is_memmap:
            ResourceUtils.tune_memmap(matrix, "sequential")
        mapping_traj_chunks: Optional[List[np.ndarray]] = [] if build_frame_mapping else None
        mapping_frame_chunks: Optional[List[np.ndarray]] = [] if build_frame_mapping else None

        selector_data = pipeline_data.selected_feature_data[name]
        all_results = selector_data.get_all_results()

        # Get frame selection (all frames or filtered) once.
        if data_selector_name is None:
            traj_frames = None
        else:
            traj_frames = pipeline_data.data_selector_data[
                data_selector_name
            ].get_trajectory_frames()

        # Extract relevant trajectories from feature selections
        relevant_trajectories = set()
        for feature_type, selection_info in all_results.items():
            if "trajectory_indices" in selection_info:
                relevant_trajectories.update(selection_info["trajectory_indices"].keys())

        # Fill matrix trajectory by trajectory (only selected trajectories)
        for traj_idx in sorted(relevant_trajectories):
            current_row = SelectionMatrixHelper._fill_trajectory_data(
                matrix=matrix,
                pipeline_data=pipeline_data,
                all_results=all_results,
                traj_idx=traj_idx,
                start_row=current_row,
                traj_frames=traj_frames,
                mapping_traj_chunks=mapping_traj_chunks,
                mapping_frame_chunks=mapping_frame_chunks,
                build_frame_mapping=build_frame_mapping,
                chunk_size=pipeline_data.chunk_size,
            )

        if is_memmap:
            ResourceUtils.tune_memmap(matrix, "random")
        if not build_frame_mapping:
            return None
        return SelectionMatrixHelper._materialize_frame_mapping(
            mapping_traj_chunks or [],
            mapping_frame_chunks or [],
        )

    @staticmethod
    def _fill_trajectory_data(
        matrix: np.ndarray,
        pipeline_data: PipelineData,
        all_results: Dict[str, Any],
        traj_idx: int,
        start_row: int,
        traj_frames: Optional[Dict[int, List[int]]],
        mapping_traj_chunks: Optional[List[np.ndarray]],
        mapping_frame_chunks: Optional[List[np.ndarray]],
        build_frame_mapping: bool,
        chunk_size: int,
    ) -> int:
        """
        Fill matrix with data from one trajectory.
        
        Parameters
        ----------
        matrix : np.ndarray
            Matrix to fill
        pipeline_data : PipelineData
            Pipeline data object
        all_results : dict
            All feature selection results
        traj_idx : int
            Trajectory index to process
        start_row : int
            Starting row index for this trajectory
        traj_frames : dict or None
            Optional pre-fetched frame indices per trajectory.
        mapping_traj_chunks : list[np.ndarray] or None
            Accumulator for per-row trajectory indices.
        mapping_frame_chunks : list[np.ndarray] or None
            Accumulator for per-row local frame indices.
        build_frame_mapping : bool
            Whether frame mapping should be collected.
        chunk_size : int
            Row chunk size forwarded to feature block copy.
            
        Returns
        -------
        int
            Next available row index
        """
        # Get frame indices for this trajectory
        if traj_frames is None:
            traj_data = pipeline_data.trajectory_data.trajectories[traj_idx]
            frame_indices = np.arange(traj_data.n_frames, dtype=np.int32)
            direct_row_slice = True
        else:
            frame_indices_list = traj_frames.get(traj_idx, [])
            if not frame_indices_list:
                return start_row
            frame_indices = np.asarray(frame_indices_list, dtype=np.int32)
            direct_row_slice = False

        if frame_indices.size == 0:
            return start_row  # No frames for this trajectory

        n_selected_rows = int(frame_indices.size)
        n_cols = int(matrix.shape[1])

        # Build one full row-chunk across all features in RAM, then write once.
        for row_chunk_start in range(0, n_selected_rows, chunk_size):
            row_chunk_end = min(row_chunk_start + chunk_size, n_selected_rows)
            frame_chunk = frame_indices[row_chunk_start:row_chunk_end]
            chunk_rows = int(frame_chunk.size)
            chunk_buffer = np.zeros((chunk_rows, n_cols), dtype=matrix.dtype)

            current_col = 0
            for feature_type, selection_info in all_results.items():
                current_col = SelectionMatrixHelper._fill_feature_data(
                    matrix=chunk_buffer,
                    pipeline_data=pipeline_data,
                    feature_type=feature_type,
                    selection_info=selection_info,
                    traj_idx=traj_idx,
                    frame_indices=frame_chunk,
                    direct_row_slice=direct_row_slice,
                    source_row_offset=row_chunk_start,
                    start_row=0,
                    start_col=current_col,
                )

            matrix[
                start_row + row_chunk_start:start_row + row_chunk_end,
                :,
            ] = chunk_buffer
            
            MemmapUtils.evict_memory_range(
                matrix,
                start_row + row_chunk_start,
                start_row + row_chunk_end,
            )

        if build_frame_mapping:
            if mapping_traj_chunks is None or mapping_frame_chunks is None:
                raise ValueError(
                    "Mapping chunk accumulators are required when build_frame_mapping=True."
                )
            mapping_traj_chunks.append(np.full(n_selected_rows, traj_idx, dtype=np.int32))
            mapping_frame_chunks.append(frame_indices)

        return start_row + n_selected_rows

    @staticmethod
    def _materialize_frame_mapping(
        mapping_traj_chunks: List[np.ndarray],
        mapping_frame_chunks: List[np.ndarray],
    ) -> Dict[int, Tuple[int, int]]:
        """
        Materialize compact mapping arrays into public mapping dictionary.

        Parameters
        ----------
        mapping_traj_chunks : list[np.ndarray]
            Trajectory index chunks.
        mapping_frame_chunks : list[np.ndarray]
            Local frame index chunks.

        Returns
        -------
        Dict[int, Tuple[int, int]]
            Mapping dictionary keyed by matrix row index.
        """
        if not mapping_traj_chunks:
            return {}
        traj_values = np.concatenate(mapping_traj_chunks, axis=0)
        frame_values = np.concatenate(mapping_frame_chunks, axis=0)
        return {
            row_idx: (int(traj_values[row_idx]), int(frame_values[row_idx]))
            for row_idx in range(int(traj_values.size))
        }

    @staticmethod
    def _build_frame_mapping_only(
        pipeline_data: PipelineData,
        feature_selector_name: str,
        data_selector_name: Optional[str],
    ) -> Dict[int, Tuple[int, int]]:
        """
        Build frame mapping without rebuilding the selection matrix.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object.
        feature_selector_name : str
            Feature selector name.
        data_selector_name : str, optional
            Data selector name.

        Returns
        -------
        Dict[int, Tuple[int, int]]
            Frame mapping for selected rows.
        """
        selector_data = pipeline_data.selected_feature_data[feature_selector_name]
        all_results = selector_data.get_all_results()

        if data_selector_name is None:
            traj_frames = None
        else:
            traj_frames = pipeline_data.data_selector_data[
                data_selector_name
            ].get_trajectory_frames()

        relevant_trajectories = set()
        for selection_info in all_results.values():
            if "trajectory_indices" in selection_info:
                relevant_trajectories.update(selection_info["trajectory_indices"].keys())

        mapping_traj_chunks: List[np.ndarray] = []
        mapping_frame_chunks: List[np.ndarray] = []
        for traj_idx in sorted(relevant_trajectories):
            if traj_frames is None:
                traj_data = pipeline_data.trajectory_data.trajectories[traj_idx]
                frame_indices = np.arange(traj_data.n_frames, dtype=np.int32)
            else:
                selected_frames = traj_frames.get(traj_idx, [])
                if not selected_frames:
                    continue
                frame_indices = np.asarray(selected_frames, dtype=np.int32)
            if frame_indices.size == 0:
                continue
            mapping_traj_chunks.append(
                np.full(int(frame_indices.size), traj_idx, dtype=np.int32)
            )
            mapping_frame_chunks.append(frame_indices)

        return SelectionMatrixHelper._materialize_frame_mapping(
            mapping_traj_chunks,
            mapping_frame_chunks,
        )

    @staticmethod
    def _fill_feature_data(
        matrix: np.ndarray,
        pipeline_data: PipelineData,
        feature_type: str,
        selection_info: Dict[str, Any],
        traj_idx: int,
        frame_indices: np.ndarray,
        direct_row_slice: bool,
        source_row_offset: int,
        start_row: int,
        start_col: int,
    ) -> int:
        """
        Fill matrix with data from one feature type.
        
        Parameters
        ----------
        matrix : np.ndarray
            Matrix to fill
        pipeline_data : PipelineData
            Pipeline data object
        feature_type : str
            Feature type name
        selection_info : dict
            Selection info for this feature
        traj_idx : int
            Trajectory index
        frame_indices : np.ndarray
            Frame indices to extract
        direct_row_slice : bool
            True when row chunks map directly to ``source_data[start:end]``.
        source_row_offset : int
            Row offset in source trajectory when ``direct_row_slice=True``.
        start_row : int
            Starting row in matrix
        start_col : int
            Starting column in matrix
            
        Returns
        -------
        int
            Next available column index
        """
        # Get trajectory-specific feature data
        feature_data_dict = pipeline_data.feature_data[feature_type]
        if traj_idx not in feature_data_dict:
            return start_col  # No feature data for this trajectory
        
        feature_data = feature_data_dict[traj_idx]
        
        # Get trajectory-specific selection indices
        trajectory_indices_data = selection_info.get("trajectory_indices", {})
        if traj_idx not in trajectory_indices_data:
            return start_col  # No selection for this trajectory
        
        traj_selection = trajectory_indices_data[traj_idx]
        indices = traj_selection.get("indices", [])
        use_reduced_flags = traj_selection.get("use_reduced", [])
        
        if not indices:
            return start_col

        original_src_cols: List[int] = []
        original_dst_offsets: List[int] = []
        reduced_src_cols: List[int] = []
        reduced_dst_offsets: List[int] = []

        for dst_offset, (src_col, use_reduced) in enumerate(
            zip(indices, use_reduced_flags)
        ):
            if use_reduced:
                reduced_src_cols.append(int(src_col))
                reduced_dst_offsets.append(dst_offset)
            else:
                original_src_cols.append(int(src_col))
                original_dst_offsets.append(dst_offset)

        if original_src_cols and feature_data.data is not None:
            SelectionMatrixHelper._copy_feature_block(
                matrix=matrix,
                source_data=feature_data.data,
                metadata=feature_data.feature_metadata,
                frame_indices=frame_indices,
                direct_row_slice=direct_row_slice,
                source_row_offset=source_row_offset,
                start_row=start_row,
                start_col=start_col,
                src_cols=np.asarray(original_src_cols, dtype=np.int32),
                dst_offsets=np.asarray(original_dst_offsets, dtype=np.int32),
            )

        if reduced_src_cols:
            SelectionMatrixHelper._copy_feature_block(
                matrix=matrix,
                source_data=feature_data.reduced_data,
                metadata=feature_data.reduced_feature_metadata,
                frame_indices=frame_indices,
                direct_row_slice=direct_row_slice,
                source_row_offset=source_row_offset,
                start_row=start_row,
                start_col=start_col,
                src_cols=np.asarray(reduced_src_cols, dtype=np.int32),
                dst_offsets=np.asarray(reduced_dst_offsets, dtype=np.int32),
            )

        return start_col + len(indices)

    @staticmethod
    def _copy_feature_block(
        matrix: np.ndarray,
        source_data: np.ndarray,
        metadata: Optional[Dict[str, Any]],
        frame_indices: np.ndarray,
        direct_row_slice: bool,
        source_row_offset: int,
        start_row: int,
        start_col: int,
        src_cols: np.ndarray,
        dst_offsets: np.ndarray,
    ) -> None:
        """
        Copy one feature block into output matrix using row chunks.

        Parameters
        ----------
        matrix : np.ndarray
            Destination matrix.
        source_data : np.ndarray
            Source feature matrix.
        metadata : dict, optional
            Metadata for optional categorical conversion.
        frame_indices : np.ndarray
            Global row indices in source trajectory matrix.
        direct_row_slice : bool
            True when rows can be sliced by ``start:end`` directly.
        source_row_offset : int
            Row offset in source trajectory when ``direct_row_slice=True``.
        start_row : int
            Destination row offset.
        start_col : int
            Destination column offset.
        src_cols : np.ndarray
            Source columns to copy.
        dst_offsets : np.ndarray
            Destination offsets relative to ``start_col``.
        """
        runs = SelectionMatrixHelper._build_contiguous_runs(src_cols, dst_offsets)
        is_memmap_source = MemmapUtils.is_memmap_view(source_data)
        if is_memmap_source:
            ResourceUtils.tune_memmap(source_data, "sequential")

        n_rows = int(frame_indices.size)
        out_row_slice = slice(start_row, start_row + n_rows)
        if direct_row_slice:
            row_block = source_data[source_row_offset:source_row_offset + n_rows]
        else:
            row_block = source_data[frame_indices]

        for src_start, src_end, dst_start, dst_end in runs:
            block = row_block[:, src_start:src_end]
            block = SelectionMatrixHelper._convert_char_to_int(
                block, metadata or {}
            )
            matrix[
                out_row_slice,
                slice(start_col + dst_start, start_col + dst_end),
            ] = block

        if is_memmap_source:
            ResourceUtils.tune_memmap(source_data, "random")
            ResourceUtils.tune_memmap(source_data, "dontneed")

    @staticmethod
    def _build_contiguous_runs(
        src_cols: np.ndarray,
        dst_offsets: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Build contiguous source/destination copy runs.

        Parameters
        ----------
        src_cols : np.ndarray
            Source column indices.
        dst_offsets : np.ndarray
            Destination offsets relative to output column start.

        Returns
        -------
        list[tuple[int, int, int, int]]
            List of ``(src_start, src_end, dst_start, dst_end)`` ranges with
            exclusive end indices.
        """
        if src_cols.size == 0:
            return []

        src = np.asarray(src_cols, dtype=np.int64)
        dst = np.asarray(dst_offsets, dtype=np.int64)

        if src.size == 1:
            return [(int(src[0]), int(src[0]) + 1, int(dst[0]), int(dst[0]) + 1)]

        contiguous = (np.diff(src) == 1) & (np.diff(dst) == 1)
        breaks = np.flatnonzero(~contiguous) + 1
        run_starts = np.concatenate((np.array([0], dtype=np.int64), breaks))
        run_ends = np.concatenate((breaks, np.array([src.size], dtype=np.int64)))

        return [
            (
                int(src[start_idx]),
                int(src[end_idx - 1]) + 1,
                int(dst[start_idx]),
                int(dst[end_idx - 1]) + 1,
            )
            for start_idx, end_idx in zip(run_starts, run_ends)
        ]

    @staticmethod
    def _convert_char_to_int(data_slice: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Convert char-encoded categorical data to integer if needed.

        Parameters
        ----------
        data_slice : numpy.ndarray
            Data slice to potentially convert
        metadata : dict
            Feature metadata containing matrix_mapping

        Returns
        -------
        numpy.ndarray
            Converted data (integer) or original data if no conversion needed

        Raises
        ------
        ValueError
            If data contains characters not present in matrix_mapping

        Notes
        -----
        Uses matrix_mapping from metadata for char => int conversion.
        Only converts if data is Unicode string dtype.
        """
        if data_slice.dtype.kind != 'U':  # Not Unicode string
            return data_slice

        matrix_mapping = metadata.get("matrix_mapping", {})
        if not matrix_mapping:
            return data_slice

        # Validate all chars are in mapping before conversion
        unique_chars = np.unique(data_slice)
        unknown_chars = [char for char in unique_chars if char not in matrix_mapping]

        if unknown_chars:
            raise ValueError(
                f"Unknown characters in data: {unknown_chars}. "
                f"Valid characters are: {list(matrix_mapping.keys())}"
            )

        # Vectorized conversion using numpy vectorize
        convert_func = np.vectorize(
            lambda x: matrix_mapping[x],
            otypes=[np.int8]
        )
        return convert_func(data_slice)
