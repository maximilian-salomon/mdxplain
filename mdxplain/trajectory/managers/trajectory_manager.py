# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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

"""Trajectory management module for loading and manipulating MD trajectory data."""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..entities.trajectory_data import TrajectoryData
from ..helper.nomenclature import Nomenclature
from ..helper.trajectory_loader import TrajectoryLoader
from ..helper.consistency_checker import TrajectoryConsistencyChecker
from ..helper.selection_resolver import TrajectorySelectionResolver
from ..helper.keyword_manager import KeywordManager
from ..helper.trajectory_processor import TrajectoryProcessor
from ..helper.trajectory_validation_helper import TrajectoryValidationHelper

import os


class TrajectoryManager:
    """Manager for pure trajectory data objects without feature dependencies."""

    def __init__(
        self,
        stride: int = 1,
        concat: bool = False,
        selection: Optional[str] = None,
        cache_dir: str = "./cache",
    ):
        """
        Initialize trajectory manager.

        Parameters:
        -----------
        stride : int, default=1
            Load every stride-th frame from trajectories. Use values > 1 to
            reduce memory usage and computation time by subsampling frames.
        concat : bool, default=False
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. Useful when dealing with trajectory splits.
        selection : str, optional
            MDTraj selection string to apply to all loaded trajectories.
            See: https://mdtraj.org/1.9.4/atom_selection.html

        Returns:
        --------
        None
            Initializes TrajectoryManager instance with default parameters

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        """
        self.default_stride = stride
        self.default_concat = concat
        self.default_selection = selection
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        if stride <= 0:
            raise ValueError("Stride must be a positive integer.")

    def load_trajectories(
        self,
        pipeline_data,
        data_input: Union[str, List[Any]],
        concat: Optional[bool] = None,
        stride: Optional[int] = 1,
        selection: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Load molecular dynamics trajectories from files or directories into PipelineData.

        This method handles loading of MD trajectories in various formats
        (e.g., .xtc, .dcd, .trr) along with their topology files. The loading
        is performed using the TrajectoryLoader class which supports automatic
        format detection and multiple trajectory handling.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.load_trajectories('../data')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.load_trajectories(pipeline_data, '../data')  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data container where trajectories will be stored
        data_input : str or list
            Path to directory containing trajectory files, or list of trajectory
            file paths. When a directory is provided, all supported trajectory
            files in that directory will be loaded.
        concat : bool, optional
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. If None, uses manager default.
        stride : int, optional
            Load every stride-th frame from trajectories. If None, uses manager default.
        selection : str, optional
            MDTraj selection string to apply to all loaded trajectories.
            If None, uses manager default.
        force : bool, default=False
            Whether to force loading even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns:
        --------
        None
            Loads trajectories into pipeline_data.trajectory_data and sets up topology/names

        Examples:
        ---------
        Pipeline mode (automatic injection):
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.load_trajectories('../data')

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')

        >>> # Load with keywords
        >>> traj_manager.load_trajectories(
        ...     pipeline_data, '../data', keywords_file='keywords.json'
        ... )

        Notes:
        -----
        - Supported formats depend on MDTraj capabilities
        - Topology files (.pdb, .gro, .psf) should be in the same directory
        - Large trajectories benefit from striding to reduce memory usage
        - Selection is applied to all trajectories after loading
        """
        # Check for existing features before loading new trajectories
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "load"
        )

        selection, concat, stride = self._prepare_parameters(selection, concat, stride)

        result = TrajectoryLoader.load_trajectories(data_input, concat, stride, selection)
        new_trajectory_data = TrajectoryData()
        new_trajectory_data.trajectories = result["trajectories"]
        new_trajectory_data.trajectory_names = result["names"]
        pipeline_data.trajectory_data = new_trajectory_data

        # Check trajectory consistency
        TrajectoryConsistencyChecker.check_trajectory_consistency(
            pipeline_data.trajectory_data
        )
        # Build and set frame keyword mapping
        frame_mapping = KeywordManager.build_frame_keyword_mapping(
            pipeline_data.trajectory_data
        )
        pipeline_data.trajectory_data.frame_keyword_mapping = frame_mapping

    def add_trajectory(
        self,
        pipeline_data,
        data_input: Union[str, List[Any]],
        concat: Optional[bool] = None,
        stride: Optional[int] = 1,
        selection: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Add molecular dynamics trajectories to TrajectoryData object.

        This method works like load_trajectories but appends new trajectories
        instead of replacing existing ones. Useful for loading additional
        trajectory data without losing previously loaded trajectories.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.add_trajectory('../data2')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.add_trajectory(pipeline_data, '../data2')  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        data_input : str or list
            Path to directory containing trajectory files, or list of trajectory
            file paths. When a directory is provided, all supported trajectory
            files in that directory will be loaded.
        concat : bool, optional
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. If None, uses manager default.
        stride : int, optional
            Load every stride-th frame from trajectories. If None, uses manager default.
        selection : str, optional
            MDTraj selection string to apply to all newly loaded trajectories.
            If None, uses manager default.
        force : bool, default=False
            Whether to force adding even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns:
        --------
        None
            Appends new trajectories to existing trajectory list in pipeline_data

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        >>> traj_manager.add_trajectory(traj_data, '../data2')

        Raises:
        -------
        ValueError
            If no trajectories are currently loaded

        Notes:
        -----
        - New trajectories are appended to existing ones
        - Trajectory names are also appended to maintain consistency
        - Selection is only applied to newly loaded trajectories
        - Existing trajectories remain unchanged
        """
        # Check for existing features before adding new trajectories
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "add"
        )

        selection, concat, stride = self._prepare_parameters(selection, concat, stride)

        # Load and process new trajectories
        result = TrajectoryLoader.load_trajectories(data_input, concat, stride, selection)
        new_trajectories, new_names = result["trajectories"], result["names"]

        # Add to existing trajectories
        pipeline_data.trajectory_data.trajectories.extend(new_trajectories)
        pipeline_data.trajectory_data.trajectory_names.extend(new_names)

        # Check trajectory consistency
        TrajectoryConsistencyChecker.check_trajectory_consistency(
            pipeline_data.trajectory_data
        )
        # Build and set frame keyword mapping
        frame_mapping = KeywordManager.build_frame_keyword_mapping(
            pipeline_data.trajectory_data
        )
        pipeline_data.trajectory_data.frame_keyword_mapping = frame_mapping

    def remove_trajectory(
        self,
        pipeline_data,
        trajs: Union[int, str, List[Union[int, str]]],
        force: bool = False,
    ) -> None:
        """
        Remove specified trajectories from the loaded trajectory list.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.remove_trajectory([0, 1])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.remove_trajectory(pipeline_data, [0, 1])  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        trajs : int, str, or list
            Trajectory index (int), name (str), or list of indices/names to remove.
            Can be mixed list of integers and strings.
        force : bool, default=False
            Whether to force removal even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns:
        --------
        None
            Removes trajectories from pipeline_data

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        >>> traj_manager.remove_trajectory(pipeline_data, [0, 1])

        Raises:
        -------
        ValueError
            If trajectories are not loaded, if trajs contains invalid indices/names
        """
        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("No trajectories loaded to remove.")

        # Check for existing features before removing trajectories
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "remove"
        )

        indices_to_remove = TrajectorySelectionResolver.prepare_removal_indices(
            pipeline_data.trajectory_data, trajs
        )
        TrajectoryProcessor.execute_removal(
            pipeline_data.trajectory_data, indices_to_remove
        )
        TrajectoryConsistencyChecker.check_trajectory_consistency(
            pipeline_data.trajectory_data
        )
        # Build and set frame keyword mapping
        frame_mapping = KeywordManager.build_frame_keyword_mapping(
            pipeline_data.trajectory_data
        )
        pipeline_data.trajectory_data.frame_keyword_mapping = frame_mapping

    def cut_traj(
        self,
        pipeline_data,
        cut: Optional[int] = None,
        stride: Optional[int] = 1,
        selection: Optional[List[Union[int, str]]] = None,
        force: bool = False,
    ) -> None:
        """
        Cut and/or stride trajectories.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.cut_traj(cut=1000)  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.cut_traj(pipeline_data, cut=1000)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        cut : int, optional
            Frame number after which to cut the trajectories.
            Frames after this index will be removed.
        stride : int, optional
            Take every stride-th frame. Use values > 1 to subsample frames.
            If None, uses manager default.
        selection : list, optional
            List of trajectory indices (int) or names (str) to process.
            If None, all trajectories will be processed.
        force : bool, default=False
            Whether to force cutting even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns:
        --------
        None
            Modifies trajectories in-place

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        >>> traj_manager.cut_traj(pipeline_data, cut=1000)

        >>> # Cut and stride specific trajectories by index
        >>> traj_manager.cut_traj(pipeline_data, cut=500, stride=5, selection=[0, 2, 4])

        >>> # Cut and stride specific trajectories by name
        >>> traj_manager.cut_traj(
        ...     pipeline_data, cut=500, stride=5,
        ...     selection=['system1_prot_traj1', 'system2_prot_traj2']
        ... )

        Raises:
        -------
        ValueError
            If trajectories are not loaded or if selection contains invalid indices/names
        """
        # Check for existing features before cutting trajectories
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "cut"
        )

        # Validate cut parameter
        if cut is not None:
            if cut < 0 or not isinstance(cut, int):
                raise ValueError("Cut parameter must be a non-negative integer.")

        # Validate and set defaults
        selection, _, stride = self._prepare_parameters(
            selection=selection, stride=stride
        )

        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("Trajectories must be loaded before cutting.")

        # Determine which trajectories to process
        indices_to_process = TrajectorySelectionResolver.get_indices_to_process(
            pipeline_data.trajectory_data, selection
        )

        # Process selected trajectories
        for idx in indices_to_process:
            processed_traj = TrajectoryProcessor.process_trajectory_cuts(
                pipeline_data.trajectory_data.trajectories[idx], cut, stride
            )
            pipeline_data.trajectory_data.trajectories[idx] = processed_traj

        # Build and set frame keyword mapping
        frame_mapping = KeywordManager.build_frame_keyword_mapping(
            pipeline_data.trajectory_data
        )
        pipeline_data.trajectory_data.frame_keyword_mapping = frame_mapping

    def select_atoms(
        self,
        pipeline_data,
        selection: str,
        trajs: Optional[List[Union[int, str]]] = None,
        force: bool = False,
    ) -> None:
        """
        Apply atom selection to trajectories using MDTraj selection syntax.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.select_atoms("protein")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.select_atoms(pipeline_data, "protein")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        selection : str
            MDTraj selection string (e.g., "protein", "backbone", "resid 10 to 50")
            See: https://mdtraj.org/1.9.4/atom_selection.html
        trajs : list, optional
            List of trajectory indices (int) or names (str) to process.
            If None, all trajectories will be processed.
        force : bool, default=False
            Whether to force atom selection even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns:
        --------
        None
            Applies atom selection to trajectories in-place

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        >>> traj_manager.select_atoms(pipeline_data, "protein")

        Raises:
        -------
        ValueError
            If trajectories are not loaded or if selection/trajs contain invalid values
        """
        # Check for existing features before applying atom selection
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "select_atoms"
        )

        # Validate and set defaults
        if selection is None:
            selection = self.default_selection

        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("Trajectories must be loaded before atom selection.")

        # Determine which trajectories to process
        indices_to_process = TrajectorySelectionResolver.get_indices_to_process(
            pipeline_data.trajectory_data, trajs
        )

        # Apply atom selection to each trajectory
        for idx in indices_to_process:
            traj = pipeline_data.trajectory_data.trajectories[idx]

            # Get atom indices matching the selection
            atom_indices = traj.topology.select(selection)

            # Apply atom slice to trajectory
            selected_traj = traj.atom_slice(atom_indices)

            # Update trajectory
            pipeline_data.trajectory_data.trajectories[idx] = selected_traj

    def add_labels(
        self,
        pipeline_data,
        fragment_definition: Optional[Union[str, Dict[str, Tuple[int, int]]]] = None,
        fragment_type: Optional[Union[str, Dict[str, str]]] = None,
        fragment_molecule_name: Optional[Union[str, Dict[str, str]]] = None,
        consensus: bool = False,
        aa_short: bool = False,
        try_web_lookup: bool = True,
        force: bool = False,
        **nomenclature_kwargs,
    ) -> None:
        """
        Add nomenclature labels to trajectory data.

        This method creates consensus labels for MD trajectories using different
        mdciao nomenclature systems (GPCR, CGN, KLIFS).

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.add_labels(fragment_definition="receptor")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.add_labels(pipeline_data, fragment_definition="receptor")  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        fragment_definition : str or dict, default None
            If string, uses that as fragment name for entire topology.
            If dict, maps fragment names to residue ranges: {"cgn_a": (0, 348), "beta2": (400, 684)}
            Only required when consensus=True.
        fragment_type : str or dict, default None
            If string, uses that nomenclature type for all fragments.
            If dict, maps fragment names to nomenclature types:
            {"cgn_a": "cgn", "beta2": "gpcr"}. Use mdciao nomenclature types.
            Allowed types: gpcr, cgn, klifs
            Only required when consensus=True.
        fragment_molecule_name : str or dict, default None
            If string, uses that molecule name for all fragments.
            If dict, maps fragment names to molecule names:
            {"cgn_a": "gnas2_bovin", "beta2": "adrb2_human"}.
            Use the UniProt entry name (not accession ID) for GPCR/CGN labelers,
            or KLIFS string for KLIFS labelers.
            See https://www.uniprot.org/help/difference_accession_entryname  # noqa: E501
            for UniProt naming conventions.
            See https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS  # noqa: E501
            for KLIFS naming conventions.
            Only required when consensus=True.
        consensus : bool, default False
            Whether to use consensus labeling (combines AA codes with nomenclature labels).
            If False, only returns amino acid labels without nomenclature.
        aa_short : bool, default True
            Whether to use short amino acid names (T vs THR)
        verbose : bool, default False
            Whether to enable verbose output from labelers
        try_web_lookup : bool, default True
            Whether to try web lookup for molecule data
        force : bool, default False
            Whether to force label addition even when labels already exist. When True,
            existing labels will be overwritten. When False, raises ValueError if labels exist.
        write_to_disk : bool, default False
            Whether to write cache files to disk
        cache_folder : str, default "./cache"
            Folder for cache files
        **nomenclature_kwargs
            Additional keyword arguments passed to the mdciao labelers

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If trajectories are not loaded
        ValueError
            If nomenclature labels already exist and force=False
        ValueError
            If fragment_definition is required when consensus=True
        ValueError
            If fragment_type is required when consensus=True
        ValueError
            If fragment_molecule_name is required when consensus=True
        ValueError
            If fragment_definition is not a string or dictionary
        ValueError
            If fragment_type is not a string or dictionary
        ValueError
            If fragment_molecule_name is not a string or dictionary

        Notes
        -----
        This class uses mdciao consensus nomenclature systems:
        https://proteinformatics.uni-leipzig.de/mdciao/api/generated/mdciao.nomenclature.html

        Supported fragment types:
        - gpcr: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerGPCR.html#mdciao.nomenclature.LabelerGPCR
        - cgn: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerCGN.html#mdciao.nomenclature.LabelerCGN
        - klifs: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS

        Examples:
        ---------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        >>> traj_manager.add_labels(
        ...     pipeline_data, fragment_definition="receptor", fragment_type="gpcr",
        ...     fragment_molecule_name="adrb2_human", consensus=True
        ... )

        >>> # Adding labels again requires force=True
        >>> traj_manager.add_labels(pipeline_data, force=True)  # Overwrites existing labels
        """
        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("Trajectories must be loaded before adding labels.")

        # Check if labels already exist and force is not set
        if pipeline_data.trajectory_data.res_label_data is not None and not force:
            raise ValueError(
                "Nomenclature labels already exist. Use force=True to overwrite existing labels, "
                "or use a fresh PipelineData object to avoid conflicts."
            )

        if self.cache_dir is None:
            write_to_disk = False
        else:
            write_to_disk = True

        # Use first trajectory topology for labeling
        nomenclature = Nomenclature(
            topology=pipeline_data.trajectory_data.trajectories[0].topology,
            fragment_definition=fragment_definition,
            fragment_type=fragment_type,
            fragment_molecule_name=fragment_molecule_name,
            consensus=consensus,
            aa_short=aa_short,
            try_web_lookup=try_web_lookup,
            write_to_disk=write_to_disk,
            cache_folder=self.cache_dir,
            **nomenclature_kwargs,
        )

        pipeline_data.trajectory_data.res_label_data = (
            nomenclature.create_trajectory_label_dicts()
        )

    def add_keywords(
        self,
        pipeline_data,
        trajectory_selector,
        keywords=None,
    ) -> None:
        """
        Add keywords to trajectories using flexible selectors.

        This method supports single trajectories, multiple trajectories, pattern matching,
        and bulk assignment using dictionaries. It provides a powerful interface for
        managing trajectory keywords in complex scenarios.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.add_keywords(0, ["system_A"])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.add_keywords(pipeline_data, 0, ["system_A"])  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        trajectory_selector : int, str, list, range, dict
            Flexible selector for trajectories:
            - int: single trajectory by index (e.g., 0)
            - str: single trajectory by name or pattern (e.g., "traj1", "system_*")
                Supports multiple string formats:
                - Range: "0-3", "id 0-3" → [0, 1, 2, 3]
                - Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
                - Single number: "7", "id 7" → [7]
                - Pattern: "system_*" → fnmatch pattern matching
            - list: multiple selectors (e.g., [0, 1, "special_traj", range(5,8)])
            - range: range of indices (e.g., range(0, 4))
            - dict: bulk assignment {selector: keywords, ...}
        keywords : list, optional
            List of keyword strings to add. Required when trajectory_selector is not dict.
            Ignored when trajectory_selector is dict.

        Returns:
        --------
        None
            Adds keywords to selected trajectories and rebuilds frame mapping

        Examples:
        ---------
        >>> # Single trajectory
        >>> traj_manager.add_keywords(pipeline_data, 0, ["system_A", "biased"])
        >>> traj_manager.add_keywords(pipeline_data, "traj1", ["system_B"])

        >>> # Multiple trajectories
        >>> traj_manager.add_keywords(pipeline_data, [0, 1, 2], ["control"])
        >>> traj_manager.add_keywords(pipeline_data, range(0, 4), ["batch_1"])

        >>> # Pattern matching
        >>> traj_manager.add_keywords(pipeline_data, "system_2_*", ["system_B"])

        >>> # Complex nested selectors
        >>> traj_manager.add_keywords(pipeline_data, [range(0,3), "system_2_*"], ["mixed"])

        >>> # Bulk assignment with dict
        >>> traj_manager.add_keywords(pipeline_data, {
        ...     [range(0,4)]: ["system_A", "unbiased"],
        ...     [range(4,8), "special_traj"]: ["control"],
        ...     "system_2_*": ["system_B", "production"]
        ... })

        Raises:
        -------
        ValueError
            If keywords is None when trajectory_selector is not dict
        ValueError
            If trajectory selector contains invalid indices or names
        """
        # Get assignments from KeywordManager
        assignments = KeywordManager.resolve_trajectory_selectors(
            pipeline_data.trajectory_data, trajectory_selector, keywords
        )

        # Apply keywords to trajectory data
        for indices, kw_list in assignments:
            if not isinstance(kw_list, list):
                raise ValueError("keywords must be a list of strings")

            for idx in indices:
                existing_keywords = (
                    pipeline_data.trajectory_data.get_trajectory_keywords(idx) or []
                )
                # Add new keywords to existing ones (avoiding duplicates)
                updated_keywords = list(existing_keywords)
                for keyword in kw_list:
                    if keyword not in updated_keywords:
                        updated_keywords.append(keyword)

                pipeline_data.trajectory_data.set_trajectory_keywords(
                    idx, updated_keywords
                )

        # Build and set frame keyword mapping
        frame_mapping = KeywordManager.build_frame_keyword_mapping(
            pipeline_data.trajectory_data
        )
        pipeline_data.trajectory_data.frame_keyword_mapping = frame_mapping

    def rename_trajectories(
        self, pipeline_data, name_mapping: Union[Dict[Union[int, str], str], List[str]]
    ) -> None:
        """
        Rename trajectory names.

        This method allows renaming trajectory names for better organization
        and more descriptive identification. Supports both dictionary-based
        mapping and positional list assignment.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.rename_trajectories({0: "new_name"})  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.rename_trajectories(pipeline_data, {0: "new_name"})  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object
        name_mapping : dict or list
            Mapping for trajectory names:
            - dict: {old_name_or_index: new_name, ...} for selective renaming
            - list: [new_name1, new_name2, ...] for positional assignment

        Returns:
        --------
        None
            Renames trajectories and rebuilds frame keyword mapping

        Examples:
        ---------
        >>> # Dictionary-based renaming
        >>> traj_manager.rename_trajectories(pipeline_data, {
        ...     0: "system_A_replicate_1",
        ...     "old_traj_name": "system_B_replicate_1",
        ...     2: "control_experiment"
        ... })

        >>> # Positional list assignment
        >>> traj_manager.rename_trajectories(pipeline_data, [
        ...     "system_A_rep1",
        ...     "system_A_rep2",
        ...     "system_B_rep1",
        ...     "control"
        ... ])

        Raises:
        -------
        ValueError
            If no trajectories are loaded, mapping is invalid, or references invalid trajectories
        """
        # Validate input
        TrajectoryProcessor.validate_name_mapping(
            pipeline_data.trajectory_data, name_mapping
        )

        # Get new names from appropriate processor
        if isinstance(name_mapping, list):
            new_names = TrajectoryProcessor.rename_with_list(
                pipeline_data.trajectory_data, name_mapping
            )
        else:
            new_names = TrajectoryProcessor.rename_with_dict(
                pipeline_data.trajectory_data, name_mapping
            )

        # Manager sets the new names on PipelineData
        pipeline_data.trajectory_data.trajectory_names = new_names

        # Rebuild frame keyword mapping (names might be used in keyword selectors)
        frame_mapping = KeywordManager.build_frame_keyword_mapping(
            pipeline_data.trajectory_data
        )
        pipeline_data.trajectory_data.frame_keyword_mapping = frame_mapping

    def reset_trajectory_data(self, pipeline_data) -> None:
        """
        Reset the trajectory data object to empty state.

        Warning:
        --------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.reset_trajectory_data()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.reset_trajectory_data(pipeline_data)  # pipeline_data required

        Parameters:
        -----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns:
        --------
        None
            Resets trajectory data to empty state

        Examples:
        ---------
        >>> traj_manager.reset_trajectory_data(pipeline_data)
        """
        pipeline_data.trajectory_data.reset()

    def _prepare_parameters(
        self,
        selection: Optional[str] = None,
        concat: Optional[bool] = None,
        stride: Optional[int] = 1,
    ) -> Tuple[Optional[str], bool, int]:
        """
        Prepare and validate parameters with defaults.

        Parameters:
        -----------
        selection : str or None
            Selection parameter to validate
        concat : bool or None
            Concat parameter to validate
        stride : int or None
            Stride parameter to validate

        Returns:
        --------
        Tuple[Optional[str], bool, int]
            Tuple of (selection, concat, stride) with defaults applied
        """
        if selection is None:
            selection = self.default_selection
        if concat is None:
            concat = self.default_concat
        if stride is None or stride <= 0:
            print("Stride must be a positive integer. Using default stride value.")
            stride = self.default_stride
        return selection, concat, stride
