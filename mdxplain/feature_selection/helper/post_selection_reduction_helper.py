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
Helper for applying post-selection reduction.

This module provides the PostSelectionReductionHelper class that applies
statistical reduction to feature selections. The reduction is applied ONLY
to the specific selection where it's defined, not to all selections of that
feature type.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np
import warnings
import os
import tempfile

from mdxplain.utils.cleanup_utils import CleanupUtils
from mdxplain.utils.resource_utils import ResourceUtils
from mdxplain.utils.memmap_utils import MemmapUtils
from mdxplain.utils.path_utils import PathUtils

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData
    from ...feature.feature_type.interfaces.calculator_base import CalculatorBase

_WARNED_LEGACY_CROSS_TRAJECTORY = False


class PostSelectionReductionHelper:
    """
    Helper for applying post-selection reduction.

    Important: Reduction is applied ONLY to the specific selection where it's defined,
    not to all selections of that feature type.

    This helper applies statistical reduction to features after initial selection.
    It uses the appropriate calculator for each feature type to compute reduction
    metrics and filters features based on threshold criteria.

    The reduction process:
    1. Get feature data for each trajectory
    2. Calculate which columns to remove using calculator
    3. Apply reduction mode (intersection/union/pooled or per_trajectory)
    4. Update trajectory_results with reduced indices
    """

    @staticmethod
    def apply_reduction(
        pipeline_data: PipelineData,
        feature_key: str,
        selection_dict: Dict[str, Any],
        trajectory_results: Dict[int, Dict],
        selected_traj_indices: List[int],
        use_memmap: bool = False,
        chunk_size: int = 2000,
        cache_dir: str = "./cache"
    ) -> Dict[int, Dict]:
        """
        Apply reduction to a specific selection.

        Process:
        1. Get feature data for each trajectory
        2. Calculate which columns to remove using calculator
        3. Apply reduction mode (intersection/union/pooled or per_trajectory)
        4. Update trajectory_results

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key (e.g., "distances", "contacts")
        selection_dict : dict
            Selection configuration with reduction config
        trajectory_results : dict
            Current selection results per trajectory
        selected_traj_indices : list
            Trajectory indices for this selection
        use_memmap : bool, default=False
            Whether to use memory-mapped files for large data processing
        chunk_size : int, default=2000
            Size of data chunks for memory-efficient processing
        cache_dir : str, default="./cache"
            Directory for temporary cache files

        Returns
        -------
        dict
            Updated trajectory_results with reduced indices

        Examples
        --------
        >>> results = PostSelectionReductionHelper.apply_reduction(
        ...     pipeline_data, "distances", selection_dict, results, [0, 1])
        """
        reduction_config = selection_dict.get("reduction")
        if not reduction_config:
            return trajectory_results

        mode = PostSelectionReductionHelper._resolve_reduction_mode(reduction_config)

        if mode == "pooled":
            return PostSelectionReductionHelper._apply_pooled_reduction(
                pipeline_data,
                feature_key,
                selection_dict,
                trajectory_results,
                selected_traj_indices,
                reduction_config,
                use_memmap,
                chunk_size,
                cache_dir,
            )

        reduced_indices_per_traj, temp_paths = PostSelectionReductionHelper._collect_reduction_results(
            pipeline_data,
            feature_key,
            selection_dict,
            trajectory_results,
            selected_traj_indices,
            reduction_config,
            use_memmap,
            chunk_size,
            cache_dir,
        )

        # Apply cross-trajectory logic
        PostSelectionReductionHelper._update_trajectory_results(
            trajectory_results,
            reduced_indices_per_traj,
            selected_traj_indices,
            selection_dict,
            mode,
        )

        # Clean up temporary files
        PostSelectionReductionHelper._cleanup_temp_files(temp_paths)

        return trajectory_results

    @staticmethod
    def _collect_reduction_results(
        pipeline_data: PipelineData,
        feature_key: str,
        selection_dict: Dict[str, Any],
        trajectory_results: Dict[int, Dict],
        selected_traj_indices: List[int],
        reduction_config: Dict[str, Any],
        use_memmap: bool,
        chunk_size: int,
        cache_dir: str,
    ) -> Tuple[Dict[int, List[int]], List[str]]:
        """
        Collect reduced indices and temp paths for per-trajectory reduction.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key
        selection_dict : dict
            Selection configuration
        trajectory_results : dict
            Current selection results per trajectory
        selected_traj_indices : list
            Trajectory indices for this selection
        reduction_config : dict
            Reduction configuration
        use_memmap : bool
            Whether to use memmap processing
        chunk_size : int
            Chunk size for processing
        cache_dir : str
            Cache directory for temporary files

        Returns
        -------
        tuple
            (reduced_indices_per_traj, temp_paths)
        """
        reduced_indices_per_traj: Dict[int, List[int]] = {}
        temp_paths: List[str] = []

        for traj_idx in selected_traj_indices:
            if traj_idx not in trajectory_results:
                continue

            kept_indices, temp_path = PostSelectionReductionHelper._process_trajectory(
                pipeline_data,
                feature_key,
                traj_idx,
                trajectory_results[traj_idx],
                selection_dict,
                reduction_config,
                use_memmap,
                chunk_size,
                cache_dir,
            )
            reduced_indices_per_traj[traj_idx] = kept_indices
            if temp_path:
                temp_paths.append(temp_path)

        return reduced_indices_per_traj, temp_paths

    @staticmethod
    def _resolve_reduction_mode(reduction_config: Dict[str, Any]) -> str:
        """
        Resolve reduction mode from configuration.

        Priority:
        1) Explicit mode flags (intersection/union/pooled)
        2) Legacy cross_trajectory flag

        Parameters
        ----------
        reduction_config : dict
            Reduction configuration

        Returns
        -------
        str
            Reduction mode ("intersection", "union", "pooled", or "per_trajectory")

        Notes
        -----
        Only one explicit mode flag may be True. When no explicit or legacy flag
        is provided, the default is per_trajectory.
        """
        mode = PostSelectionReductionHelper._resolve_explicit_reduction_mode(
            reduction_config
        )
        if mode != "per_trajectory":
            return mode
        legacy = reduction_config.get("cross_trajectory", None)
        if legacy is None:
            return mode
        return PostSelectionReductionHelper._resolve_legacy_reduction_mode(legacy)

    @staticmethod
    def _resolve_explicit_reduction_mode(reduction_config: Dict[str, Any]) -> str:
        """
        Resolve explicit cross-trajectory mode flags.

        Parameters
        ----------
        reduction_config : dict
            Reduction configuration

        Returns
        -------
        str
            Explicit mode or per_trajectory when no explicit flags are True
        """
        flags = {
            "intersection": reduction_config.get("cross_trajectory_intersection"),
            "union": reduction_config.get("cross_trajectory_union"),
            "pooled": reduction_config.get("cross_trajectory_pooled"),
        }
        enabled = [name for name, value in flags.items() if value]
        if len(enabled) > 1:
            raise ValueError(
                "Only one of cross_trajectory_intersection, cross_trajectory_union, "
                "or cross_trajectory_pooled can be True."
            )
        if enabled:
            return enabled[0]
        return "per_trajectory"

    @staticmethod
    def _resolve_legacy_reduction_mode(legacy: bool) -> str:
        """
        Resolve legacy cross_trajectory mode with deprecation handling.

        Parameters
        ----------
        legacy : bool
            Legacy cross_trajectory value

        Returns
        -------
        str
            Legacy reduction mode
        """
        PostSelectionReductionHelper._warn_legacy_cross_trajectory()
        return "intersection" if legacy else "per_trajectory"

    @staticmethod
    def _warn_legacy_cross_trajectory() -> None:
        """
        Emit a deprecation warning for legacy cross_trajectory usage.

        Returns
        -------
        None
        """
        global _WARNED_LEGACY_CROSS_TRAJECTORY
        if _WARNED_LEGACY_CROSS_TRAJECTORY:
            return
        warnings.warn(
            "cross_trajectory is deprecated; use cross_trajectory_intersection, "
            "cross_trajectory_union, or cross_trajectory_pooled instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        _WARNED_LEGACY_CROSS_TRAJECTORY = True

    @staticmethod
    def _apply_pooled_reduction(
        pipeline_data: PipelineData,
        feature_key: str,
        selection_dict: Dict[str, Any],
        trajectory_results: Dict[int, Dict],
        selected_traj_indices: List[int],
        reduction_config: Dict[str, Any],
        use_memmap: bool,
        chunk_size: int,
        cache_dir: str,
    ) -> Dict[int, Dict]:
        """
        Apply pooled reduction across trajectories.

        Delegates pooled metric computation to calculators, which should apply
        boundary-safe logic where needed.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key
        selection_dict : dict
            Selection configuration
        trajectory_results : dict
            Current selection results per trajectory
        selected_traj_indices : list
            Trajectory indices to include
        reduction_config : dict
            Reduction configuration
        use_memmap : bool
            Whether to use memory-mapped files
        chunk_size : int
            Chunk size for processing
        cache_dir : str
            Cache directory for temporary files

        Returns
        -------
        dict
            Updated trajectory_results
        """
        selected_segments, temp_paths = (
            PostSelectionReductionHelper._collect_pooled_segments(
                pipeline_data,
                feature_key,
                selection_dict,
                trajectory_results,
                selected_traj_indices,
                use_memmap,
                chunk_size,
                cache_dir,
            )
        )

        if not selected_segments:
            PostSelectionReductionHelper._cleanup_temp_files(temp_paths)
            return trajectory_results

        kept_local_indices = PostSelectionReductionHelper._get_pooled_kept_indices(
            pipeline_data,
            feature_key,
            reduction_config,
            selected_segments,
            use_memmap,
            chunk_size,
        )
        PostSelectionReductionHelper._apply_kept_indices_to_segments(
            trajectory_results,
            selected_segments,
            kept_local_indices,
            selection_dict["use_reduced"],
        )

        PostSelectionReductionHelper._cleanup_temp_files(temp_paths)
        return trajectory_results

    @staticmethod
    def _collect_pooled_segments(
        pipeline_data: PipelineData,
        feature_key: str,
        selection_dict: Dict[str, Any],
        trajectory_results: Dict[int, Dict],
        selected_traj_indices: List[int],
        use_memmap: bool,
        chunk_size: int,
        cache_dir: str,
    ) -> Tuple[List[Tuple[int, List[int], np.ndarray]], List[str]]:
        """
        Collect selected data segments for pooled reduction.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key
        selection_dict : dict
            Selection configuration
        trajectory_results : dict
            Current selection results per trajectory
        selected_traj_indices : list
            Trajectory indices to include
        use_memmap : bool
            Whether to use memmap extraction
        chunk_size : int
            Chunk size for extraction
        cache_dir : str
            Cache directory for temporary files

        Returns
        -------
        tuple
            (segments, temp_paths)
        """
        selected_segments: List[Tuple[int, List[int], np.ndarray]] = []
        temp_paths: List[str] = []
        n_cols: Optional[int] = None

        for traj_idx in selected_traj_indices:
            if traj_idx not in trajectory_results:
                continue
            feature_data = pipeline_data.feature_data[feature_key][traj_idx]
            selection_indices = trajectory_results[traj_idx]["indices"]
            selected_data, temp_path = PostSelectionReductionHelper._select_data(
                feature_data,
                selection_indices,
                selection_dict["use_reduced"],
                use_memmap,
                chunk_size,
                cache_dir,
            )
            n_cols = PostSelectionReductionHelper._update_pooled_shape(
                n_cols,
                selected_data,
                temp_paths,
            )
            selected_segments.append((traj_idx, selection_indices, selected_data))
            if temp_path:
                temp_paths.append(temp_path)

        return selected_segments, temp_paths

    @staticmethod
    def _update_pooled_shape(
        n_cols: Optional[int],
        selected_data: np.ndarray,
        temp_paths: List[str],
    ) -> int:
        """
        Validate pooled column counts across trajectories.

        Parameters
        ----------
        n_cols : int or None
            Current expected column count
        selected_data : np.ndarray
            Selected data for the current segment
        temp_paths : list
            Temporary file paths to clean on error

        Returns
        -------
        int
            Updated expected column count
        """
        if n_cols is None:
            return selected_data.shape[1]

        if selected_data.shape[1] != n_cols:
            PostSelectionReductionHelper._cleanup_temp_files(temp_paths)
            raise ValueError(
                "Pooled reduction requires the same number of selected features "
                "across trajectories. Use common_denominator=True or ensure "
                "consistent selections."
            )

        return n_cols

    @staticmethod
    def _get_pooled_kept_indices(
        pipeline_data: PipelineData,
        feature_key: str,
        reduction_config: Dict[str, Any],
        selected_segments: List[Tuple[int, List[int], np.ndarray]],
        use_memmap: bool,
        chunk_size: int,
    ) -> List[int]:
        """
        Determine kept indices from pooled metric values.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key
        reduction_config : dict
            Reduction configuration
        selected_segments : list
            Selected trajectory segments
        use_memmap : bool
            Whether to use memmap output
        chunk_size : int
            Chunk size for processing

        Returns
        -------
        list
            Kept local indices for pooled selection
        """
        metric = reduction_config.get("metric")
        calculator = pipeline_data.feature_data[feature_key][selected_segments[0][0]].feature_type.calculator
        segments = [segment for _, _, segment in selected_segments]
        params = {"chunk_size": chunk_size, "use_memmap": use_memmap}
        for key in [
            "transition_threshold",
            "window_size",
            "transition_mode",
            "lag_time",
        ]:
            value = reduction_config.get(key)
            if value is not None:
                params[key] = value
        mask = calculator.compute_pooled_selection_mask(
            segments,
            metric,
            threshold_min=reduction_config.get("threshold_min"),
            threshold_max=reduction_config.get("threshold_max"),
            **params,
        )
        kept_local_indices = np.where(mask.flatten())[0].tolist()
        PostSelectionReductionHelper._validate_selection_results(
            len(kept_local_indices),
            reduction_config.get("metric"),
            reduction_config.get("threshold_min"),
            reduction_config.get("threshold_max"),
        )
        return kept_local_indices

    @staticmethod
    def _apply_kept_indices_to_segments(
        trajectory_results: Dict[int, Dict],
        selected_segments: List[Tuple[int, List[int], np.ndarray]],
        kept_local_indices: List[int],
        use_reduced: bool,
    ) -> None:
        """
        Map pooled kept indices back to trajectory-specific indices.

        Parameters
        ----------
        trajectory_results : dict
            Results to update in-place
        selected_segments : list
            Selected trajectory segments
        kept_local_indices : list
            Kept local indices from pooled selection
        use_reduced : bool
            Whether reduced data is used

        Returns
        -------
        None
        """
        for traj_idx, selection_indices, _ in selected_segments:
            kept_original = PostSelectionReductionHelper._map_kept_indices(
                selection_indices, kept_local_indices
            )
            PostSelectionReductionHelper._set_trajectory_indices(
                trajectory_results,
                traj_idx,
                kept_original,
                use_reduced,
            )

    @staticmethod
    def _validate_selection_results(
        n_selected: int,
        metric_name: Optional[str],
        threshold_min: Optional[float],
        threshold_max: Optional[float],
    ) -> None:
        """
        Warn if no values were selected by thresholds.

        Parameters
        ----------
        n_selected : int
            Number of selected values
        metric_name : str or None
            Metric name for reporting
        threshold_min : float, optional
            Minimum threshold
        threshold_max : float, optional
            Maximum threshold

        Returns
        -------
        None
        """
        if n_selected != 0:
            return
        threshold_desc = f"min={threshold_min}, max={threshold_max}"
        warnings.warn(
            "No values found within the specified threshold criteria. "
            f"Metric: {metric_name}, Thresholds: {threshold_desc}"
        )

    @staticmethod
    def _process_trajectory(
        pipeline_data: PipelineData,
        feature_key: str,
        traj_idx: int,
        traj_results: Dict,
        selection_dict: Dict,
        reduction_config: Dict,
        use_memmap: bool,
        chunk_size: int,
        cache_dir: str
    ) -> Tuple[List[int], Optional[str]]:
        """
        Process reduction for single trajectory.

        Extracts selected columns, applies calculator reduction, and maps
        back to original indices for one trajectory.

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container
        feature_key : str
            Feature type key
        traj_idx : int
            Trajectory index
        traj_results : dict
            Results for this trajectory
        selection_dict : dict
            Selection configuration
        reduction_config : dict
            Reduction configuration
        use_memmap : bool
            Use memmap processing
        chunk_size : int
            Chunk size for processing
        cache_dir : str
            Cache directory

        Returns
        -------
        tuple
            (kept_original_indices, temp_path or None)
        """
        feature_data = pipeline_data.feature_data[feature_key][traj_idx]
        selection_indices = traj_results["indices"]
        use_reduced = selection_dict["use_reduced"]

        selected_data, temp_path = PostSelectionReductionHelper._select_data(
            feature_data, selection_indices, use_reduced, use_memmap, chunk_size, cache_dir
        )
        selected_feature_metadata = PostSelectionReductionHelper._get_selected_feature_metadata(
            feature_data, selection_indices, use_reduced, feature_key, traj_idx
        )

        calculator = feature_data.feature_type.calculator
        temp_output_path = PostSelectionReductionHelper._create_temp_output_path(
            use_memmap, cache_dir
        )
        result = PostSelectionReductionHelper._apply_calculator(
            calculator, selected_data, reduction_config,
            selected_feature_metadata, temp_output_path
        )

        kept_original = PostSelectionReductionHelper._map_kept_indices(
            selection_indices, result["indices"]
        )
        PostSelectionReductionHelper._cleanup_temp_output(result, temp_output_path)

        return kept_original, temp_path

    @staticmethod
    def _select_data(
        feature_data: Any,
        selection_indices: List[int],
        use_reduced: bool,
        use_memmap: bool,
        chunk_size: int,
        cache_dir: str
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Select columns from feature data, optionally using memmap extraction.

        Parameters
        ----------
        feature_data : Any
            Feature data container for the trajectory
        selection_indices : list
            Column indices to select
        use_reduced : bool
            Whether to use reduced data
        use_memmap : bool
            Whether to use memmap extraction
        chunk_size : int
            Chunk size for memmap extraction
        cache_dir : str
            Directory for temporary cache files

        Returns
        -------
        tuple
            (selected_data, temp_path or None)
        """
        data_matrix = feature_data.reduced_data if use_reduced else feature_data.data
        if use_memmap:
            return PostSelectionReductionHelper._extract_memmap_columns(
                data_matrix, selection_indices, chunk_size, cache_dir
            )
        return data_matrix[:, selection_indices], None

    @staticmethod
    def _get_selected_feature_metadata(
        feature_data: Any,
        selection_indices: List[int],
        use_reduced: bool,
        feature_key: str,
        traj_idx: int
    ) -> Optional[List[Any]]:
        """
        Extract feature metadata for selected columns.

        Parameters
        ----------
        feature_data : Any
            Feature data container for the trajectory
        selection_indices : list
            Column indices to select
        use_reduced : bool
            Whether to use reduced metadata
        feature_key : str
            Feature type key
        traj_idx : int
            Trajectory index

        Returns
        -------
        list or None
            Selected feature metadata or None if unavailable
        """
        source_metadata = PostSelectionReductionHelper._get_metadata_container(
            feature_data,
            use_reduced,
        )
        if source_metadata is None:
            return None
        features_list = PostSelectionReductionHelper._require_features_list(
            source_metadata,
            feature_key,
            traj_idx,
        )
        return PostSelectionReductionHelper._select_feature_metadata(
            features_list,
            selection_indices,
        )

    @staticmethod
    def _get_metadata_container(feature_data: Any, use_reduced: bool) -> Optional[Dict[str, Any]]:
        """
        Retrieve feature metadata container based on reduction flag.

        Parameters
        ----------
        feature_data : Any
            Feature data container
        use_reduced : bool
            Whether to use reduced metadata

        Returns
        -------
        dict or None
            Metadata container or None
        """
        return (
            feature_data.reduced_feature_metadata
            if use_reduced
            else feature_data.feature_metadata
        )

    @staticmethod
    def _require_features_list(
        source_metadata: Dict[str, Any],
        feature_key: str,
        traj_idx: int,
    ) -> Any:
        """
        Fetch the feature list from metadata or raise a descriptive error.

        Parameters
        ----------
        source_metadata : dict
            Metadata container
        feature_key : str
            Feature type key
        traj_idx : int
            Trajectory index

        Returns
        -------
        Any
            Feature list object
        """
        features_list = source_metadata.get("features")
        if features_list is None:
            raise ValueError(
                f"Feature metadata missing 'features' for '{feature_key}' "
                f"(trajectory {traj_idx})."
            )
        return features_list

    @staticmethod
    def _select_feature_metadata(
        features_list: Any,
        selection_indices: List[int],
    ) -> Any:
        """
        Select metadata entries by index.

        Parameters
        ----------
        features_list : Any
            Features list or array
        selection_indices : list
            Indices to select

        Returns
        -------
        Any
            Selected feature metadata
        """
        if isinstance(features_list, np.ndarray):
            return features_list[selection_indices]
        return [features_list[idx] for idx in selection_indices]

    @staticmethod
    def _create_temp_output_path(use_memmap: bool, cache_dir: str) -> Optional[str]:
        """
        Create a temporary memmap output path when memmap is enabled.

        Parameters
        ----------
        use_memmap : bool
            Whether to use memmap output
        cache_dir : str
            Directory for temporary cache files

        Returns
        -------
        str or None
            Temporary file path or None when memmap is disabled
        """
        if not use_memmap:
            return None
        cache_dir = PathUtils.prepare_directory_path(
            cache_dir,
            create=True,
            purpose="cache directory",
        )
        temp_fd, temp_path = tempfile.mkstemp(suffix=".dat", dir=cache_dir)
        os.close(temp_fd)
        return temp_path

    @staticmethod
    def _map_kept_indices(
        selection_indices: List[int],
        kept_local_indices: Any
    ) -> List[int]:
        """
        Map kept local indices back to original selection indices.

        Parameters
        ----------
        selection_indices : list
            Original selected column indices
        kept_local_indices : Any
            Indices retained after reduction (local to selection)

        Returns
        -------
        list
            Original indices retained after reduction
        """
        if isinstance(kept_local_indices, np.ndarray):
            kept_local_indices = kept_local_indices.tolist()
        selection_array = np.array(selection_indices)
        return selection_array[kept_local_indices].tolist()

    @staticmethod
    def _cleanup_temp_output(result: Dict, temp_output_path: Optional[str]) -> None:
        """
        Clean up temporary output memmap file.

        Parameters
        ----------
        result : dict
            Calculator result containing optional dynamic_data
        temp_output_path : str or None
            Temporary output path to delete

        Returns
        -------
        None
            Deletes temp output file if present
        """
        if not temp_output_path:
            return
        PostSelectionReductionHelper._flush_memmap(result.get("dynamic_data"))
        PostSelectionReductionHelper._unlink_path(temp_output_path)

    @staticmethod
    def _flush_memmap(data: Any) -> None:
        """
        Flush and close memmap data if applicable.

        Parameters
        ----------
        data : Any
            Data that may be a memmap

        Returns
        -------
        None
        """
        if not isinstance(data, np.memmap):
            return
        try:
            data.flush()
        except Exception:
            pass
        mm = getattr(data, "_mmap", None)
        if mm is None:
            return
        try:
            mm.close()
        except Exception:
            pass

    @staticmethod
    def _unlink_path(path: str) -> None:
        """
        Remove a file path if it exists.

        Parameters
        ----------
        path : str
            Path to remove

        Returns
        -------
        None
        """
        CleanupUtils.remove_file(
            path,
            missing_ok=True,
            ignore_errors=False,
            purpose="temporary reduction file",
        )

    @staticmethod
    def _update_trajectory_results(
        trajectory_results: Dict,
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
        selection_dict: Dict,
        mode: str
    ) -> None:
        """
        Update trajectory results based on reduction mode.

        Handles intersection, union, or per-trajectory reduction.

        Parameters
        ----------
        trajectory_results : dict
            Results to update in-place
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Selected trajectory indices
        selection_dict : dict
            Selection configuration
        mode : str
            Reduction mode: 'intersection', 'union', or 'per_trajectory'

        Returns
        -------
        None
            Updates trajectory_results in-place
        """
        if len(reduced_indices_per_traj) <= 1:
            PostSelectionReductionHelper._apply_direct_updates(
                trajectory_results,
                reduced_indices_per_traj,
                selection_dict["use_reduced"],
            )
            return

        handler = {
            "intersection": PostSelectionReductionHelper._apply_intersection,
            "union": PostSelectionReductionHelper._apply_union,
        }.get(mode)
        if handler is not None:
            handler(
                trajectory_results,
                reduced_indices_per_traj,
                selected_traj_indices,
                selection_dict,
            )
            return

        reduction_config = selection_dict.get("reduction", {})
        if reduction_config.get("cross_trajectory") is False:
            PostSelectionReductionHelper._warn_per_trajectory_reduction()
        PostSelectionReductionHelper._apply_direct_updates(
            trajectory_results,
            reduced_indices_per_traj,
            selection_dict["use_reduced"],
        )

    @staticmethod
    def _apply_direct_updates(
        trajectory_results: Dict,
        reduced_indices_per_traj: Dict,
        use_reduced: bool,
    ) -> None:
        """
        Apply per-trajectory reduced indices directly.

        Parameters
        ----------
        trajectory_results : dict
            Results to update in-place
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        use_reduced : bool
            Whether reduced data is used

        Returns
        -------
        None
        """
        for traj_idx, indices in reduced_indices_per_traj.items():
            PostSelectionReductionHelper._set_trajectory_indices(
                trajectory_results,
                traj_idx,
                indices,
                use_reduced,
            )

    @staticmethod
    def _set_trajectory_indices(
        trajectory_results: Dict,
        traj_idx: int,
        indices: List[int],
        use_reduced: bool,
    ) -> None:
        """
        Update one trajectory's indices and reduction flag.

        Parameters
        ----------
        trajectory_results : dict
            Results to update in-place
        traj_idx : int
            Trajectory index
        indices : list
            Selected indices
        use_reduced : bool
            Whether reduced data is used

        Returns
        -------
        None
        """
        trajectory_results[traj_idx]["indices"] = indices
        trajectory_results[traj_idx]["use_reduced"] = [use_reduced] * len(indices)

    @staticmethod
    def _warn_per_trajectory_reduction() -> None:
        """
        Warn about per-trajectory reduction behavior.

        Returns
        -------
        None
        """
        warnings.warn(
            "per-trajectory reduction: Features may differ between trajectories!"
        )

    @staticmethod
    def _apply_intersection(
        trajectory_results: Dict,
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
        selection_dict: Dict
    ) -> None:
        """
        Apply intersection of reduced features to all trajectories.

        Finds intersection of reduced indices across trajectories and
        updates all trajectories with common features only.

        Parameters
        ----------
        trajectory_results : dict
            Results to update
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Selected trajectory indices
        selection_dict : dict
            Selection configuration

        Returns
        -------
        None
            Updates trajectory_results in-place
        """
        common = PostSelectionReductionHelper._compute_common_indices(
            reduced_indices_per_traj,
            selected_traj_indices,
        )
        
        PostSelectionReductionHelper._apply_index_set(
            trajectory_results,
            common,
            selection_dict["use_reduced"],
        )

    @staticmethod
    def _apply_union(
        trajectory_results: Dict,
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
        selection_dict: Dict
    ) -> None:
        """
        Apply union of reduced indices across trajectories.

        Keeps features that pass in any trajectory, then applies the union
        to all trajectories to keep column counts consistent.

        Parameters
        ----------
        trajectory_results : dict
            Results to update in-place
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Selected trajectory indices
        selection_dict : dict
            Selection configuration

        Returns
        -------
        None
        """
        union_set = PostSelectionReductionHelper._compute_union_indices(
            reduced_indices_per_traj,
            selected_traj_indices,
        )
        PostSelectionReductionHelper._apply_index_set(
            trajectory_results,
            union_set,
            selection_dict["use_reduced"],
        )

    @staticmethod
    def _compute_common_indices(
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
    ) -> set:
        """
        Compute intersection of indices across selected trajectories.

        Parameters
        ----------
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Trajectory indices to consider

        Returns
        -------
        set
            Common indices
        """
        index_sets = PostSelectionReductionHelper._index_sets(
            reduced_indices_per_traj,
            selected_traj_indices,
        )
        if not index_sets:
            return set()
        return set.intersection(*index_sets)

    @staticmethod
    def _compute_union_indices(
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
    ) -> set:
        """
        Compute union of indices across selected trajectories.

        Parameters
        ----------
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Trajectory indices to consider

        Returns
        -------
        set
            Union of indices
        """
        index_sets = PostSelectionReductionHelper._index_sets(
            reduced_indices_per_traj,
            selected_traj_indices,
        )
        if not index_sets:
            return set()
        return set().union(*index_sets)

    @staticmethod
    def _index_sets(
        reduced_indices_per_traj: Dict,
        selected_traj_indices: List[int],
    ) -> List[set]:
        """
        Build list of index sets for selected trajectories.

        Parameters
        ----------
        reduced_indices_per_traj : dict
            Reduced indices per trajectory
        selected_traj_indices : list
            Trajectory indices to consider

        Returns
        -------
        list
            List of index sets
        """
        return [
            set(reduced_indices_per_traj[traj_idx])
            for traj_idx in selected_traj_indices
            if traj_idx in reduced_indices_per_traj
        ]

    @staticmethod
    def _apply_index_set(
        trajectory_results: Dict,
        index_set: set,
        use_reduced: bool,
    ) -> None:
        """
        Apply a kept index set to all trajectories.

        Parameters
        ----------
        trajectory_results : dict
            Results to update in-place
        index_set : set
            Allowed indices
        use_reduced : bool
            Whether reduced data is used

        Returns
        -------
        None
        """
        for traj_idx in trajectory_results:
            kept = PostSelectionReductionHelper._filter_indices(
                trajectory_results[traj_idx]["indices"],
                index_set,
            )
            PostSelectionReductionHelper._set_trajectory_indices(
                trajectory_results,
                traj_idx,
                kept,
                use_reduced,
            )

    @staticmethod
    def _filter_indices(indices: List[int], index_set: set) -> List[int]:
        """
        Filter indices to those contained in the index set.

        Parameters
        ----------
        indices : list
            Indices to filter
        index_set : set
            Allowed indices

        Returns
        -------
        list
            Filtered indices
        """
        return [idx for idx in indices if idx in index_set]

    @staticmethod
    def _extract_memmap_columns(
        data_matrix: np.ndarray,
        column_indices: List[int],
        chunk_size: int,
        cache_dir: str
    ) -> Tuple[np.ndarray, str]:
        """
        Extract selected columns using memmap for memory efficiency.

        Creates temporary memmap file and copies selected columns
        chunk-wise to minimize memory usage.

        Parameters
        ----------
        data_matrix : np.ndarray
            Full data matrix (can be memmap)
        column_indices : list
            Indices of columns to extract
        chunk_size : int
            Frame chunk size for processing
        cache_dir : str
            Directory for temporary cache files

        Returns
        -------
        tuple
            (selected_data, temp_path)

        Examples
        --------
        >>> selected, path = PostSelectionReductionHelper._extract_memmap_columns(
        ...     data_matrix, [0, 5, 10], chunk_size=500, cache_dir="./cache")
        >>> print(selected.shape)
        (n_frames, 3)
        """
        n_frames = data_matrix.shape[0]
        n_selected = len(column_indices)

        # Create temporary memmap file
        cache_dir = PathUtils.prepare_directory_path(
            cache_dir,
            create=True,
            purpose="cache directory",
        )
        temp_fd, temp_path = tempfile.mkstemp(suffix='.dat', dir=cache_dir)
        os.close(temp_fd)

        # Create memmap for output
        selected_data = MemmapUtils.create_memmap(
            path=temp_path,
            dtype=data_matrix.dtype,
            mode="w+",
            shape=(n_frames, n_selected),
        )

        # Copy chunk-wise
        is_memmap_data = MemmapUtils.is_memmap_view(data_matrix)
        if is_memmap_data:
            ResourceUtils.tune_memmap(data_matrix, "sequential")
        ResourceUtils.tune_memmap(selected_data, "sequential")
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk = data_matrix[start:end, :][:, column_indices]
            selected_data[start:end, :] = chunk
            selected_data.flush()

        ResourceUtils.tune_memmap(selected_data, "random")
        if is_memmap_data:
            ResourceUtils.tune_memmap(data_matrix, "random")
        return selected_data, temp_path

    @staticmethod
    def _apply_calculator(
        calculator: 'CalculatorBase',
        selected_data: np.ndarray,
        reduction_config: Dict,
        metadata: Dict,
        output_path: Optional[str]
    ) -> Dict:
        """
        Apply calculator with reduction parameters.

        Calls calculator's compute_dynamic_values with all necessary
        parameters for reduction calculation.

        Parameters
        ----------
        calculator : CalculatorBase
            Feature type specific calculator instance
        selected_data : np.ndarray
            Data matrix with only selected features
        reduction_config : dict
            Full reduction configuration
        metadata : dict
            Feature metadata
        output_path : str, optional
            Output path for calculator operations (memmap output)

        Returns
        -------
        dict
            Calculator result with indices of features that meet criteria

        Examples
        --------
        >>> result = PostSelectionReductionHelper._apply_calculator(
        ...     calculator, data, config, metadata, path)
        >>> kept_indices = result["indices"]
        """
        # Base parameters
        params = {
            "metric": reduction_config["metric"],
            "threshold_min": reduction_config.get("threshold_min"),
            "threshold_max": reduction_config.get("threshold_max"),
            "feature_metadata": metadata,
        }
        if output_path is not None:
            params["output_path"] = output_path

        # Optional transition parameters
        for key in ["transition_threshold", "window_size", "transition_mode", "lag_time"]:
            if key in reduction_config:
                params[key] = reduction_config[key]

        return calculator.compute_dynamic_values(selected_data, **params)

    @staticmethod
    def _cleanup_temp_files(temp_paths: List[str]) -> None:
        """
        Clean up temporary files.

        Parameters
        ----------
        temp_paths : list
            List of temporary file paths to remove

        Returns
        -------
        None
        """
        for path in temp_paths:
            if path:
                CleanupUtils.remove_file(
                    path,
                    missing_ok=True,
                    ignore_errors=False,
                    purpose="temporary reduction file",
                )
