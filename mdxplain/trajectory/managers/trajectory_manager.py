# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Cursor IDE (Claude Sonnet 4.0, occasional Claude Sonnet 3.7 and Gemini 2.5 Pro).
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

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import os
import mdtraj as md

if TYPE_CHECKING:
    from ...pipeline.entities.pipeline_data import PipelineData

from ..entities.trajectory_data import TrajectoryData
from ..entities.dask_md_trajectory import DaskMDTrajectory
from ..helper.metadata_helper.nomenclature_helper import NomenclatureHelper
from ..helper.process_helper.trajectory_load_helper import TrajectoryLoadHelper
from ..helper.metadata_helper.tag_helper import TagHelper
from ..helper.process_helper.trajectory_process_helper import TrajectoryProcessHelper
from ..helper.process_helper.label_operation_helper import LabelOperationHelper
from ..helper.validation_helper.trajectory_validation_helper import TrajectoryValidationHelper


class TrajectoryManager:
    """
    Manager for pure trajectory data objects without feature dependencies.

    Provides methods to load, add, remove, slice, and select atoms in MD trajectories.
    This manager operates on TrajectoryData objects and does not depend on
    any feature data. It is designed to be used both standalone and within
    a pipeline context.

    It handles various trajectory formats, automatic format detection, and
    provides a consistent interface for working with trajectory data.

    It can load multiple trajectories, apply selections, and manage
    memory-efficient representations using DaskMDTrajectory.

    It can load trajectories from directories and nested directories or lists of files,
    handle topology files, and apply MDTraj selection strings.
    """

    def __init__(
        self,
        stride: int = 1,
        concat: bool = False,
        selection: Optional[str] = None,
        cache_dir: str = "./cache",
        use_memmap: bool = False,
        chunk_size: int = 1000,
    ):
        """
        Initialize trajectory manager.

        Parameters
        ----------
        stride : int, default=1
            Load every stride-th frame from trajectories. Use values > 1 to
            reduce memory usage and computation time by subsampling frames.
        concat : bool, default=False
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. Useful when dealing with trajectory splits.
        selection : str, optional
            MDTraj selection string to apply to all loaded trajectories.
            See: https://mdtraj.org/1.9.4/atom_selection.html
        use_memmap : bool, default=False
            Whether to use memory-mapped DaskMDTrajectory for large files.
            When True, trajectories are loaded as DaskMDTrajectory objects
            for efficient memory usage.
        chunk_size : int, default=1000
            Chunk size for DaskMDTrajectory (only used when use_memmap=True).
            Number of frames per chunk for memory management.
        cache_dir : str, default="./cache"
            Directory for caching intermediate results and Zarr files.

        Returns
        -------
        None
            Initializes TrajectoryManager instance with default parameters

        Examples
        --------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        """
        self.default_stride = stride
        self.default_concat = concat
        self.default_selection = selection
        self.cache_dir = cache_dir
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        os.makedirs(self.cache_dir, exist_ok=True)

        if stride <= 0:
            raise ValueError("Stride must be a positive integer.")

    def load_trajectories(
        self,
        pipeline_data: PipelineData,
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
        is performed using the TrajectoryLoadHelper class which supports automatic
        format detection and multiple trajectory handling.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.load_trajectories('../data')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.load_trajectories(pipeline_data, '../data')  # pipeline_data required

        Parameters
        ----------
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

        Returns
        -------
        None
            Loads trajectories into pipeline_data.trajectory_data and sets up topology/names

        Examples
        --------
        Pipeline mode (automatic injection):
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.load_trajectories('../data')

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')

        >>> # Load with tags
        >>> traj_manager.load_trajectories(
        ...     pipeline_data, '../data', tags_file='tags.json'
        ... )

        Notes
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

        # Load trajectories using TrajectoryLoadHelper (handles memmap automatically)
        result = TrajectoryLoadHelper.load_trajectories(
            data_input, concat, stride, selection, 
            use_memmap=self.use_memmap, chunk_size=self.chunk_size, cache_dir=self.cache_dir
        )
        
        new_trajectory_data = TrajectoryData()
        new_trajectory_data.trajectories = result["trajectories"]
        new_trajectory_data.trajectory_names = result["names"]
        pipeline_data.trajectory_data = new_trajectory_data
        # Update memory estimate based on max atoms
        if pipeline_data.trajectory_data.trajectories:
            max_atoms = max(traj.n_atoms for traj in pipeline_data.trajectory_data.trajectories)
            pipeline_data.update_max_memory_from_trajectories(max_atoms)

    def add_trajectory(
        self,
        pipeline_data: PipelineData,
        data_input: Union[str, List[Any]],
        concat: Optional[bool] = None,
        stride: Optional[int] = 1,
        selection: Optional[str] = None,
    ) -> None:
        """
        Add molecular dynamics trajectories to TrajectoryData object.

        This method works like load_trajectories but appends new trajectories
        instead of replacing existing ones. Useful for loading additional
        trajectory data without losing previously loaded trajectories.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.add_trajectory('../data2')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.add_trajectory(pipeline_data, '../data2')  # pipeline_data required

        Parameters
        ----------
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
        
        Returns
        -------
        None
            Appends new trajectories to existing trajectory list in pipeline_data

        Examples
        --------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        >>> traj_manager.add_trajectory(traj_data, '../data2')

        Raises
        ------
        ValueError
            If no trajectories are currently loaded

        Notes
        -----
        - New trajectories are appended to existing ones
        - Trajectory names are also appended to maintain consistency
        - Selection is only applied to newly loaded trajectories
        - Existing trajectories remain unchanged
        """
        selection, concat, stride = self._prepare_parameters(selection, concat, stride)

        # Load and process new trajectories
        result = TrajectoryLoadHelper.load_trajectories(
            data_input, concat, stride, selection,
            use_memmap=self.use_memmap, chunk_size=self.chunk_size, cache_dir=self.cache_dir
        )
        new_trajectories, new_names = result["trajectories"], result["names"]

        # Add to existing trajectories
        pipeline_data.trajectory_data.trajectories.extend(new_trajectories)
        pipeline_data.trajectory_data.trajectory_names.extend(new_names)
        
        # Update memory estimate based on max atoms
        if pipeline_data.trajectory_data.trajectories:
            max_atoms = max(traj.n_atoms for traj in pipeline_data.trajectory_data.trajectories)
            pipeline_data.update_max_memory_from_trajectories(max_atoms)

    def remove_trajectory(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], str],
        force: bool = False,
    ) -> None:
        """
        Remove specified trajectories from the loaded trajectory list.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.remove_trajectory([0, 1])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.remove_trajectory(pipeline_data, [0, 1])  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        traj_selection : int, str, list, or "all"
            Trajectory selection (required). Options:

            - int: single trajectory by index (e.g., 0)
            - str: trajectory name or "all" for all trajectories
            - list: multiple indices/names (can be mixed)
            - "all": all loaded trajectories
        force : bool, default=False
            Whether to force removal even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Removes trajectories from pipeline_data

        Examples
        --------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        >>> traj_manager.remove_trajectory(pipeline_data, [0, 1])

        Raises
        ------
        ValueError
            If trajectories are not loaded, if trajs contains invalid indices/names
        """
        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("No trajectories loaded to remove.")

        # Get trajectory indices and check for features
        indices_to_remove = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)
        
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "remove", indices_to_remove
        )
        indices_to_remove.sort(reverse=True)  # Sort in reverse to avoid index shifting issues

        TrajectoryProcessHelper.execute_removal(
            pipeline_data.trajectory_data, indices_to_remove
        )

    def slice_traj(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], str],
        frames: Optional[Union[int, slice, List[int]]] = None,
        stride: Optional[int] = 1,
        cut: Optional[int] = None,
        data_selector: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Slice trajectories using frame ranges, stride, OR DataSelector.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.slice_traj(traj_selection="all", frames=1000)  # NO pipeline_data parameter
        >>> pipeline.trajectory.slice_traj(traj_selection="all", data_selector="folded_frames")  # Use DataSelector

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.slice_traj(pipeline_data, frames=1000, traj_selection="all")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        traj_selection : int, str, list, or "all"
            Selection of trajectories to process (required):

            - int: trajectory index
            - str: trajectory name or "all" for all trajectories
            - list: list of indices/names
            - "all": all loaded trajectories
        frames : int, slice, list, optional
            Frame specification for slicing:

            - int: include frames 0 to frames (e.g., frames=1000 → frames 0-999)
            - slice: direct slice object (e.g., slice(100, 500) → frames 100-499)
            - list: specific frame indices (e.g., [0, 10, 20, 30])

            Ignored if data_selector is provided.
        stride : int, optional
            Take every stride-th frame. Use values > 1 to subsample frames.
            If None, uses manager default.
        cut : int, optional
            Frame number after which to cut trajectories. Frames after this 
            index will be removed. Applied after frame selection and stride.
        data_selector : str, optional
            Name of DataSelector to use for frame selection.
            If provided, overrides frames/stride parameters and uses
            the selected frames from the DataSelector.
        force : bool, default=False
            Whether to force slicing even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Modifies trajectories in-place

        Examples
        --------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        >>> traj_manager.slice_traj(pipeline_data, traj_selection="all", frames=1000)

        >>> # Slice specific frames with stride for specific trajectories by index
        >>> traj_manager.slice_traj(pipeline_data, traj_selection=[0, 2, 4], frames=500, stride=5)

        >>> # Use slice object for precise frame ranges
        >>> traj_manager.slice_traj(pipeline_data, traj_selection="all", frames=slice(100, 500), stride=2)

        >>> # Select specific frame indices by name selection
        >>> traj_manager.slice_traj(
        ...     pipeline_data,
        ...     traj_selection=['system1_prot_traj1', 'system2_prot_traj2'],
        ...     frames=[0, 10, 20, 50, 100]
        ... )

        >>> # Cut trajectory after 1000 frames with stride
        >>> traj_manager.slice_traj(pipeline_data, traj_selection="all", cut=1000, stride=2)

        >>> # Use DataSelector to slice trajectories to folded frames only
        >>> traj_manager.slice_traj(pipeline_data, traj_selection="all", data_selector="folded_frames")

        Raises
        ------
        ValueError
            If trajectories are not loaded or if selection contains invalid indices/names
            or if DataSelector does not exist
        """
        # Validation: Cannot use both frames and data_selector
        if frames is not None and data_selector is not None:
            raise ValueError("Cannot specify both 'frames' and 'data_selector' parameters. Choose one.")
        
        # Get trajectory indices to process  
        indices_to_process = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)
        
        # Check if features exist for trajectories to be sliced
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "slice", indices_to_process
        )
        
        # Apply slicing using unified helper method
        TrajectoryProcessHelper.apply_slicing(
            pipeline_data, indices_to_process, frames, data_selector, stride, cut
        )
        
    
    def select_atoms(
        self,
        pipeline_data: PipelineData,
        selection: str,
        traj_selection: Union[int, str, List[Union[int, str]], "all"],
        force: bool = False,
    ) -> None:
        """
        Apply atom selection to trajectories using MDTraj selection syntax.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.select_atoms("protein", "all")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.select_atoms(pipeline_data, "protein", "all")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        traj_selection : int, str, list, or "all"
            Selection of trajectories to process:

            - int: trajectory index
            - str: trajectory name or "all" for all trajectories
            - list: list of indices/names
        selection : str
            MDTraj selection string (e.g., "protein", "backbone", "resid 10 to 50")
            See: https://mdtraj.org/1.9.4/atom_selection.html
        force : bool, default=False
            Whether to force atom selection even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Applies atom selection to trajectories in-place

        Examples
        --------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        >>> traj_manager.select_atoms(pipeline_data, "protein", "all")

        >>> # Select atoms from specific trajectories
        >>> traj_manager.select_atoms(pipeline_data, "protein", [0, 1, 2])

        Raises
        ------
        ValueError
            If trajectories are not loaded or if selection/trajs contain invalid values
        """
        # Get trajectory indices to process
        indices_to_process = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)

        # Validate and set default
        if selection is None:
            selection = self.default_selection

        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("Trajectories must be loaded before atom selection.")

        # Check if features exist for trajectories to be modified
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "select_atoms", indices_to_process
        )

        # Apply atom selection to selected trajectory
        original_residue_indices = {}
        for idx in indices_to_process:
            traj = pipeline_data.trajectory_data.trajectories[idx]

            # Get atom indices matching the selection
            atom_indices = traj.topology.select(selection)
            if len(atom_indices) == 0:
                raise ValueError(
                    f"Atom selection '{selection}' produced no atoms for trajectory {idx}"
                )

            # BEFORE reducing trajectory: Get original residue indices for labels
            original_residue_indices[idx] = np.unique([
                traj.topology.atom(atom_idx).residue.index 
                for atom_idx in atom_indices
            ])

            # Apply atom slice to trajectory
            selected_traj = traj.atom_slice(atom_indices)

            # Update trajectory
            pipeline_data.trajectory_data.trajectories[idx] = selected_traj
        
        # Apply same atom selection to labels for consistency (with original residue indices)
        LabelOperationHelper.apply_atom_selection_to_labels(
            pipeline_data.trajectory_data, indices_to_process, original_residue_indices
        )
        

    def add_labels(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], "all"],
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
        Add nomenclature labels to selected trajectories.

        This method creates consensus labels for MD trajectories using different
        mdciao nomenclature systems (GPCR, CGN, KLIFS). Different systems can have
        different nomenclatures by applying labels to specific trajectory selections.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.add_labels([0, 1], fragment_definition="receptor")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.add_labels(pipeline_data, [0, 1], fragment_definition="receptor")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        traj_selection : int, str, list, or "all"
            Selection of trajectories to add labels to:

            - int: trajectory index
            - str: trajectory name or "all" for all trajectories
            - list: list of indices/names
            - "all": all loaded trajectories
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
        '**'nomenclature_kwargs
            Additional keyword arguments passed to the mdciao labelers

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If trajectories are not loaded
        ValueError
            If traj_selection contains invalid indices or names
        ValueError
            If nomenclature labels already exist for selected trajectories and force=False
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
        This method wraps mdciao consensus nomenclature systems:
        https://proteinformatics.uni-leipzig.de/mdciao/api/generated/mdciao.nomenclature.html

        Supported fragment types:

        - gpcr: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerGPCR.html#mdciao.nomenclature.LabelerGPCR
        - cgn: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerCGN.html#mdciao.nomenclature.LabelerCGN
        - klifs: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS

        Examples
        --------
        >>> pipeline_data = PipelineData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(pipeline_data, '../data')
        
        >>> # Add labels to specific trajectories (different systems)
        >>> traj_manager.add_labels(
        ...     pipeline_data, [0, 1], fragment_definition="receptor", fragment_type="gpcr",
        ...     fragment_molecule_name="adrb2_human", consensus=True
        ... )
        >>> traj_manager.add_labels(
        ...     pipeline_data, [2, 3], fragment_definition="kinase", fragment_type="klifs",
        ...     fragment_molecule_name="abl1_human", consensus=True
        ... )
        
        >>> # Add labels to all trajectories
        >>> traj_manager.add_labels(pipeline_data, "all", fragment_definition="protein")

        >>> # Adding labels again requires force=True
        >>> traj_manager.add_labels(pipeline_data, "all", force=True)  # Overwrites existing labels
        """
        if not pipeline_data.trajectory_data.trajectories:
            raise ValueError("Trajectories must be loaded before adding labels.")

        # Determine which trajectories to process
        indices_to_process = pipeline_data.trajectory_data.get_trajectory_indices(traj_selection)

        # Check if labels already exist for selected trajectories
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "add_labels", indices_to_process
        )

        if self.cache_dir is None:
            write_to_disk = False
        else:
            write_to_disk = True

        # Create labels for each selected trajectory
        for idx in indices_to_process:
            # Use the specific trajectory's topology for labeling
            nomenclature = NomenclatureHelper(
                topology=pipeline_data.trajectory_data.trajectories[idx].topology,
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

            # Create labels for this specific trajectory
            trajectory_labels = nomenclature.create_trajectory_label_dicts()
            
            # Store labels with trajectory index as key (int, not str)
            pipeline_data.trajectory_data.res_label_data[idx] = trajectory_labels

    def add_tags(
        self,
        pipeline_data: PipelineData,
        trajectory_selector: Union[int, str, List[Any], range, Dict[Any, List[str]]],
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Add tags to trajectories using flexible selectors.

        This method supports single trajectories, multiple trajectories, pattern matching,
        and bulk assignment using dictionaries. It provides a powerful interface for
        managing trajectory tags in complex scenarios.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.add_tags(0, ["system_A"])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.add_tags(pipeline_data, 0, ["system_A"])  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        trajectory_selector : int, str, list, range, dict
            Flexible selector for trajectories:
            
            - int: single trajectory by index (e.g., 0)
            - str: single trajectory by name or pattern (e.g., "traj1", "system_*")
                Supports multiple string formats:
                
                * Range: "0-3", "id 0-3" → [0, 1, 2, 3]
                * Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
                * Single number: "7", "id 7" → [7]
                * Pattern: "system_*" → fnmatch pattern matching
            - list: multiple selectors (e.g., [0, 1, "special_traj", range(5,8)])
            - range: range of indices (e.g., range(0, 4))
            - dict: bulk assignment {selector: tags, ...}
        tags : list, optional
            List of tag strings to add. Required when trajectory_selector is not dict.
            Ignored when trajectory_selector is dict.

        Returns
        -------
        None
            Adds tags to selected trajectories and rebuilds frame mapping

        Examples
        --------
        >>> # Single trajectory
        >>> traj_manager.add_tags(pipeline_data, 0, ["system_A", "biased"])
        >>> traj_manager.add_tags(pipeline_data, "traj1", ["system_B"])

        >>> # Multiple trajectories
        >>> traj_manager.add_tags(pipeline_data, [0, 1, 2], ["control"])
        >>> traj_manager.add_tags(pipeline_data, range(0, 4), ["batch_1"])

        >>> # Pattern matching
        >>> traj_manager.add_tags(pipeline_data, "system_2_*", ["system_B"])

        >>> # Complex nested selectors
        >>> traj_manager.add_tags(pipeline_data, [range(0,3), "system_2_*"], ["mixed"])

        >>> # Bulk assignment with dict
        >>> traj_manager.add_tags(pipeline_data, {
        ...     [range(0,4)]: ["system_A", "unbiased"],
        ...     [range(4,8), "special_traj"]: ["control"],
        ...     "system_2_*": ["system_B", "production"]
        ... })

        Raises
        ------
        ValueError
            If tags is None when trajectory_selector is not dict
        ValueError
            If trajectory selector contains invalid indices or names
        """
        # Get assignments from TagHelper
        assignments = TagHelper.resolve_trajectory_selectors(
            pipeline_data.trajectory_data, trajectory_selector, tags
        )

        # Apply tags to trajectory data
        for indices, tag_list in assignments:
            if not isinstance(tag_list, list):
                raise ValueError("tags must be a list of strings")
            for idx in indices:
                existing_tags = pipeline_data.trajectory_data.get_trajectory_tags(idx) or []
                updated_tags = self._merge_tags(existing_tags, tag_list)
                pipeline_data.trajectory_data.set_trajectory_tags(idx, updated_tags)
    
    def set_tags(
        self,
        pipeline_data: PipelineData,
        trajectory_selector: Union[int, str, List[Any], range, Dict[Any, List[str]]],
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Set (replace) tags for trajectories using flexible selectors.

        This method completely replaces existing tags instead of merging them.
        It supports the same flexible selector system as add_tags() but provides
        replacement semantics for tag management scenarios where you need to 
        reset or completely change trajectory tags.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.set_tags(0, ["system_A"])  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.set_tags(pipeline_data, 0, ["system_A"])  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        trajectory_selector : int, str, list, range, dict
            Flexible selector for trajectories:
            
            - int: single trajectory by index (e.g., 0)
            - str: single trajectory by name or pattern (e.g., "traj1", "system_*")
                Supports multiple string formats:
                
                * Range: "0-3", "id 0-3" → [0, 1, 2, 3]
                * Comma list: "1,2,4,5", "id 1,2,4,5" → [1, 2, 4, 5]
                * Single number: "7", "id 7" → [7]
                * Pattern: "system_*" → fnmatch pattern matching
            - list: multiple selectors (e.g., [0, 1, "special_traj", range(5,8)])
            - range: range of indices (e.g., range(0, 4))
            - dict: bulk assignment {selector: tags, ...}
        tags : list, optional
            List of tag strings to set. Required when trajectory_selector is not dict.
            Ignored when trajectory_selector is dict.

        Returns
        -------
        None
            Sets tags for selected trajectories and rebuilds frame mapping

        Examples
        --------
        >>> # Replace tags for single trajectory
        >>> traj_manager.add_tags(pipeline_data, 0, ["old_tag", "other"])
        >>> traj_manager.set_tags(pipeline_data, 0, ["new_tag"])  # Replaces both old tags
        
        >>> # Reset multiple trajectories to same tags
        >>> traj_manager.set_tags(pipeline_data, [0, 1, 2], ["reset", "control"])

        >>> # Clear all tags (set to empty)
        >>> traj_manager.set_tags(pipeline_data, "all", [])

        >>> # Bulk replacement with dict
        >>> traj_manager.set_tags(pipeline_data, {
        ...     [0, 1]: ["system_A", "production"],
        ...     [2, 3]: ["system_B", "test"],
        ...     "control_*": ["control"]
        ... })

        Raises
        ------
        ValueError
            If tags is None when trajectory_selector is not dict
        ValueError
            If trajectory selector contains invalid indices or names
        """
        # Get assignments from TagHelper (same as add_tags)
        assignments = TagHelper.resolve_trajectory_selectors(
            pipeline_data.trajectory_data, trajectory_selector, tags
        )

        # Apply tag replacement directly (minimal implementation)
        for indices, tag_list in assignments:
            if not isinstance(tag_list, list):
                raise ValueError("tags must be a list of strings")
            for idx in indices:
                pipeline_data.trajectory_data.set_trajectory_tags(idx, tag_list)

    
    def _merge_tags(self, existing_tags: List[str], new_tags: List[str]) -> List[str]:
        """
        Merge new tags with existing tags, avoiding duplicates.

        Parameters
        ----------
        existing_tags : List[str]
            Current list of tags
        new_tags : List[str]
            New tags to add

        Returns
        -------
        list
            Merged list of tags without duplicates
        """
        updated_tags = list(existing_tags)
        for tag in new_tags:
            if tag not in updated_tags:
                updated_tags.append(tag)
        return updated_tags


    def rename_trajectories(
        self, pipeline_data: PipelineData, name_mapping: Union[Dict[Union[int, str], str], List[str]]
    ) -> None:
        """
        Rename trajectory names.

        This method allows renaming trajectory names for better organization
        and more descriptive identification. Supports both dictionary-based
        mapping and positional list assignment.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.rename_trajectories({0: "new_name"})  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.rename_trajectories(pipeline_data, {0: "new_name"})  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object
        name_mapping : dict or list
            Mapping for trajectory names:

            - dict: {old_name_or_index: new_name, ...} for selective renaming
            - list: [new_name1, new_name2, ...] for positional assignment

        Returns
        -------
        None
            Renames trajectories and rebuilds frame tag mapping

        Examples
        --------
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

        Raises
        ------
        ValueError
            If no trajectories are loaded, mapping is invalid, or references invalid trajectories
        """
        # Validate input
        TrajectoryProcessHelper.validate_name_mapping(
            pipeline_data.trajectory_data, name_mapping
        )

        # Get new names from appropriate processor
        if isinstance(name_mapping, list):
            new_names = TrajectoryProcessHelper.rename_with_list(
                pipeline_data.trajectory_data, name_mapping
            )
        else:
            new_names = TrajectoryProcessHelper.rename_with_dict(
                pipeline_data.trajectory_data, name_mapping
            )

        # Manager sets the new names on PipelineData
        pipeline_data.trajectory_data.trajectory_names = new_names

    def reset_trajectory_data(self, pipeline_data: PipelineData) -> None:
        """
        Reset the trajectory data object to empty state.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.reset_trajectory_data()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.reset_trajectory_data(pipeline_data)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data object

        Returns
        -------
        None
            Resets trajectory data to empty state

        Examples
        --------
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

        Parameters
        ----------
        selection : str or None
            Selection parameter to validate
        concat : bool or None
            Concat parameter to validate
        stride : int or None
            Stride parameter to validate

        Returns
        -------
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

    def save(self, pipeline_data: PipelineData, save_path: str) -> None:
        """
        Save trajectory data to disk.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.save('trajectory_backup.pkl')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.save(pipeline_data, 'trajectory_backup.pkl')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        save_path : str
            Path where to save the trajectory data

        Returns
        -------
        None
            Saves the trajectory data to the specified path

        Examples
        --------
        >>> trajectory_manager.save(pipeline_data, 'trajectory_backup.pkl')
        """
        pipeline_data.trajectory_data.save(save_path)

    def load(self, pipeline_data: PipelineData, load_path: str) -> None:
        """
        Load trajectory data from disk.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.load('trajectory_backup.pkl')  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.load(pipeline_data, 'trajectory_backup.pkl')  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container to load trajectory data into
        load_path : str
            Path to the saved trajectory data file

        Returns
        -------
        None
            Loads the trajectory data from the specified path

        Examples
        --------
        >>> trajectory_manager.load(pipeline_data, 'trajectory_backup.pkl')
        """
        pipeline_data.trajectory_data.load(load_path)

    def print_info(self, pipeline_data: PipelineData) -> None:
        """
        Print trajectory information.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.print_info()  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.print_info(pipeline_data)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data

        Returns
        -------
        None
            Prints trajectory information to console

        Examples
        --------
        >>> trajectory_manager.print_info(pipeline_data)
        === TrajectoryData ===
        Loaded 3 trajectories:
          [0] system1_prot_traj1: 1000 frames, tags: ['system_A', 'biased']
          [1] system1_prot_traj2: 1500 frames, tags: ['system_A', 'unbiased']
          [2] system2_prot_traj1: 800 frames, tags: ['system_B', 'biased']
        """
        pipeline_data.trajectory_data.print_info()

    def select_trajs(self, pipeline_data: PipelineData, data_selector: str) -> List[Union[DaskMDTrajectory, md.Trajectory]]:
        """
        Create new trajectory objects from DataSelector frames.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> selected = pipeline.trajectory.select_trajs("folded_frames")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> selected = manager.select_trajs(pipeline_data, "folded_frames")  # pipeline_data required
        
        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data with trajectories and DataSelector
        data_selector : str
            Name of DataSelector to use
            
        Returns
        -------
        List[Union[DaskMDTrajectory, md.Trajectory]]
            List of new trajectory objects with selected frames
            
        Examples
        --------
        >>> # Create new trajectories from DataSelector
        >>> selected = pipeline.trajectory.select_trajs("folded_frames")
        >>> print(f"Created {len(selected)} new trajectories")
        >>> 
        >>> # Use the returned trajectories for analysis
        >>> for traj in selected:
        ...     print(f"Trajectory has {traj.n_frames} frames")
        """
        selector_data = self._validate_data_selector(pipeline_data, data_selector)
        
        new_trajectories = []
        for traj_idx, frame_indices in selector_data.trajectory_frames.items():
            if not frame_indices:
                continue
                
            traj = pipeline_data.trajectory_data.trajectories[traj_idx]
            
            # Create new trajectory using slice() method (unified interface)
            new_traj = traj.slice(frame_indices)
            
            new_trajectories.append(new_traj)
            print(f"Created new trajectory from traj {traj_idx} with {len(frame_indices)} frames")

        print(f"Created {len(new_trajectories)} trajectories from DataSelector '{data_selector}'")
        return new_trajectories

    def superpose(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], str],
        reference_traj: int = 0,
        reference_frame: int = 0,
        atom_selection: str = "backbone",
    ) -> None:
        """
        Superpose selected trajectories to a reference frame (always in-place).

        This method aligns all frames of selected trajectories to a specific reference
        frame using MDTraj's superpose functionality. The operation is performed
        in-place, modifying the original trajectories. For memory-mapped trajectories,
        the alignment is performed chunk-wise to manage memory usage efficiently.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.superpose(traj_selection="all", reference_traj=0, reference_frame=0)  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.superpose(pipeline_data, traj_selection="all", reference_traj=0, reference_frame=0)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        traj_selection : int, str, list, or "all"
            Selection of trajectories to align (required):

            - int: single trajectory by index
            - str: trajectory name, tag, or pattern (e.g., "tag:system_A", "traj_*")
            - list: multiple indices/names/tags
            - "all": all loaded trajectories
        reference_traj : int, default=0
            Index of trajectory containing the reference frame
        reference_frame : int, default=0
            Frame index within reference trajectory to use as alignment reference
        atom_selection : str, default="backbone"
            MDTraj selection string for atoms to use in alignment calculation.
            Common selections: "backbone", "name CA", "protein", "resid 10 to 50"
            See: https://mdtraj.org/1.9.4/atom_selection.html

        Returns
        -------
        None
            Modifies trajectories in-place. No return value.

        Examples
        --------
        Basic alignment:
        >>> pipeline.trajectory.superpose(traj_selection="all")  # Align all to first frame of first trajectory

        Specific reference:
        >>> pipeline.trajectory.superpose(
        ...     traj_selection="all",
        ...     reference_traj=2,
        ...     reference_frame=100,
        ...     atom_selection="name CA"
        ... )

        Selective alignment:
        >>> pipeline.trajectory.superpose(
        ...     traj_selection=[0, 1, 3],
        ...     atom_selection="backbone and resid 50:150"
        ... )

        Tag-based selection:
        >>> pipeline.trajectory.superpose(
        ...     traj_selection="tag:wild_type",
        ...     reference_traj=0,
        ...     reference_frame=0
        ... )

        Notes
        -----
        - Dask trajectories (use_memmap=True) handle memory management automatically
        - The reference trajectory itself is also aligned to the reference frame
        - All trajectories must have compatible topology for alignment
        - Large trajectories may take significant time to align

        Raises
        ------
        ValueError
            If no trajectories are loaded
        ValueError
            If reference_traj index is invalid
        ValueError
            If reference_frame index is invalid for reference trajectory
        ValueError
            If traj_selection contains invalid indices/names
        ValueError
            If atom_selection produces no atoms or incompatible atom counts
        """
        # Validate all parameters using TrajectoryValidationHelper
        _, traj_indices, ref_frame, ref_atom_indices = (
            TrajectoryValidationHelper.validate_superpose_parameters(
                pipeline_data, reference_traj, reference_frame, atom_selection, traj_selection
            )
        )

        print(f"Superposing {len(traj_indices)} trajectories to reference frame "
              f"(traj {reference_traj}, frame {reference_frame}) using {len(ref_atom_indices)} atoms")

        # Align each selected trajectory
        for idx in traj_indices:
            trajectory = pipeline_data.trajectory_data.trajectories[idx]

            print(f"  Aligning trajectory {idx} ({trajectory.n_frames} frames)...")

            # Validate atom selection for this trajectory
            traj_atom_indices = trajectory.topology.select(atom_selection)
            if len(traj_atom_indices) != len(ref_atom_indices):
                raise ValueError(f"Atom count mismatch: trajectory {idx} has {len(traj_atom_indices)} atoms "
                                f"but reference has {len(ref_atom_indices)} atoms for selection '{atom_selection}'")

            # Standard in-memory trajectory: direct alignment
            trajectory.superpose(reference=ref_frame, atom_indices=traj_atom_indices)

            print(f"    ✓ Trajectory {idx} aligned successfully")

        print(f"Superposition completed successfully for {len(traj_indices)} trajectories")

    def center_coordinates(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], str],
        mass_weighted: bool = False,
        force: bool = False,
    ) -> None:
        """
        Center trajectory coordinates at the origin (always in-place).

        This method centers all frames of selected trajectories at the origin using
        either geometric centering (default) or mass-weighted centering. The operation
        is performed in-place, modifying the original trajectories. For memory-mapped
        trajectories, the centering is performed chunk-wise to manage memory efficiently.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.center_coordinates(traj_selection="all")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.center_coordinates(pipeline_data, traj_selection="all")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        traj_selection : int, str, list, or "all"
            Selection of trajectories to center (required):

            - int: single trajectory by index
            - str: trajectory name, tag, or pattern (e.g., "tag:system_A", "traj_*")
            - list: multiple indices/names/tags
            - "all": all loaded trajectories
        mass_weighted : bool, default=False
            Use mass-weighted centering instead of geometric centering.
            When True, the center of mass is used; when False, the geometric
            center (centroid) is used.
        force : bool, default=False
            Whether to force centering even when features have been calculated.
            When True, existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Modifies trajectories in-place. No return value.

        Examples
        --------
        Basic geometric centering:
        >>> pipeline.trajectory.center_coordinates(traj_selection="all")

        Mass-weighted centering:
        >>> pipeline.trajectory.center_coordinates(
        ...     traj_selection="all",
        ...     mass_weighted=True
        ... )

        Center specific trajectories:
        >>> pipeline.trajectory.center_coordinates(
        ...     traj_selection=[0, 1, 2],
        ...     mass_weighted=False
        ... )

        Tag-based selection:
        >>> pipeline.trajectory.center_coordinates(
        ...     traj_selection="tag:production",
        ...     mass_weighted=True
        ... )

        Notes
        -----
        - Dask trajectories (use_memmap=True) handle memory management automatically
        - Centering is useful before RMSD calculations or structural analysis
        - Mass-weighted centering is more physically meaningful for biomolecules
        - This operation modifies coordinates but preserves topology

        Raises
        ------
        ValueError
            If no trajectories are loaded
        ValueError
            If traj_selection contains invalid indices/names
        """
        # Get trajectory indices to process
        indices_to_process = pipeline_data.trajectory_data.get_trajectory_indices(
            traj_selection
        )

        # Check if features exist for trajectories to be centered
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "center_coordinates", indices_to_process
        )

        print(
            f"Centering {len(indices_to_process)} trajectories "
            f"({'mass-weighted' if mass_weighted else 'geometric'})"
        )

        # Center each selected trajectory
        for idx in indices_to_process:
            trajectory = pipeline_data.trajectory_data.trajectories[idx]

            print(f"  Centering trajectory {idx} ({trajectory.n_frames} frames)...")

            # Apply centering (in-place for both DaskMDTrajectory and md.Trajectory)
            trajectory.center_coordinates(mass_weighted=mass_weighted)

            print(f"    ✓ Trajectory {idx} centered successfully")

        print(
            f"Centering completed successfully for {len(indices_to_process)} trajectories"
        )

    def smooth(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], str],
        width: int,
        order: Optional[int] = None,
        atom_selection: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Apply Savitzky-Golay smoothing filter to trajectory coordinates (always in-place).

        This method smooths trajectory coordinates using a Savitzky-Golay filter,
        which fits successive sub-sets of adjacent data points with a low-degree
        polynomial by linear least squares. The operation is performed in-place,
        modifying the original trajectories. Smoothing can be applied to all atoms
        or a subset selected via atom_selection.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.smooth(traj_selection="all", width=5)  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.smooth(pipeline_data, traj_selection="all", width=5)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        traj_selection : int, str, list, or "all"
            Selection of trajectories to smooth (required):

            - int: single trajectory by index
            - str: trajectory name, tag, or pattern (e.g., "tag:system_A", "traj_*")
            - list: multiple indices/names/tags
            - "all": all loaded trajectories
        width : int
            Smoothing window width (must be odd). Larger values produce smoother
            trajectories but may lose important structural details.
        order : int, optional
            Polynomial order for Savitzky-Golay filter. If None, uses default
            from MDTraj implementation. Typical values: 2-4.
        atom_selection : str, optional
            MDTraj selection string for atoms to smooth. If None, smooths all atoms.
            Common selections: "backbone", "name CA", "protein", "resid 10 to 50"
            See: https://mdtraj.org/1.9.4/atom_selection.html
        force : bool, default=False
            Whether to force smoothing even when features have been calculated.
            When True, existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Modifies trajectories in-place. No return value.

        Examples
        --------
        Basic smoothing of all atoms:
        >>> pipeline.trajectory.smooth(traj_selection="all", width=5)

        Smooth with specific polynomial order:
        >>> pipeline.trajectory.smooth(
        ...     traj_selection="all",
        ...     width=7,
        ...     order=3
        ... )

        Smooth only backbone atoms:
        >>> pipeline.trajectory.smooth(
        ...     traj_selection=[0, 1, 2],
        ...     width=5,
        ...     atom_selection="backbone"
        ... )

        Smooth CA atoms for specific trajectories:
        >>> pipeline.trajectory.smooth(
        ...     traj_selection="tag:production",
        ...     width=3,
        ...     atom_selection="name CA"
        ... )

        Notes
        -----
        - Dask trajectories (use_memmap=True) handle memory management automatically
        - Smoothing reduces high-frequency noise but may obscure fast dynamics
        - Window width should be odd; even values will be adjusted internally
        - Larger windows create smoother trajectories but lose temporal resolution
        - This operation modifies coordinates but preserves topology

        Raises
        ------
        ValueError
            If no trajectories are loaded
        ValueError
            If traj_selection contains invalid indices/names
        ValueError
            If width is not positive
        ValueError
            If atom_selection is invalid or produces no atoms
        """
        # Get trajectory indices to process
        indices_to_process = pipeline_data.trajectory_data.get_trajectory_indices(
            traj_selection
        )

        # Check if features exist for trajectories to be smoothed
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "smooth", indices_to_process
        )

        print(
            f"Smoothing {len(indices_to_process)} trajectories "
            f"(width={width}, order={order}, atoms={'all' if atom_selection is None else atom_selection})"
        )

        # Smooth each selected trajectory
        for idx in indices_to_process:
            trajectory = pipeline_data.trajectory_data.trajectories[idx]

            print(f"  Smoothing trajectory {idx} ({trajectory.n_frames} frames)...")

            # Get atom indices if selection specified
            atom_indices = None
            if atom_selection is not None:
                atom_indices = trajectory.topology.select(atom_selection)
                if len(atom_indices) == 0:
                    raise ValueError(
                        f"Atom selection '{atom_selection}' produced no atoms for trajectory {idx}"
                    )

            # Apply smoothing (in-place for both types)
            trajectory.smooth(
                width=width, order=order, atom_indices=atom_indices, inplace=True
            )

            print(f"    ✓ Trajectory {idx} smoothed successfully")

        print(
            f"Smoothing completed successfully for {len(indices_to_process)} trajectories"
        )

    def image_molecules(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], str],
        anchor_molecules: Optional[np.ndarray] = None,
        other_molecules: Optional[np.ndarray] = None,
        make_whole: bool = True,
        force: bool = False,
    ) -> None:
        """
        Apply periodic boundary condition imaging to molecules (always in-place).

        This method recenters molecules and wraps them into the primary unit cell
        using MDTraj's image_molecules method. This is essential for visualizing
        trajectories with periodic boundary conditions correctly. The operation
        modifies coordinates but does not change the number of atoms or topology.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.image_molecules(traj_selection="all")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.image_molecules(pipeline_data, traj_selection="all")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        traj_selection : int, str, list, or "all"
            Selection of trajectories to image (required):

            - int: single trajectory by index
            - str: trajectory name, tag, or pattern (e.g., "tag:system_A", "traj_*")
            - list: multiple indices/names/tags
            - "all": all loaded trajectories
        anchor_molecules : np.ndarray, optional
            Indices of molecules to anchor at the origin. If None, uses all
            molecules. Molecules are typically defined by bonded groups in the
            topology.
        other_molecules : np.ndarray, optional
            Indices of other molecules to image relative to anchors. If None,
            uses all molecules not in anchor_molecules.
        make_whole : bool, default=True
            Make molecules whole across periodic boundary conditions before
            imaging. This ensures molecules are not split across the box boundary.
        force : bool, default=False
            Whether to force imaging even when features have been calculated.
            When True, existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Modifies trajectories in-place. No return value.

        Examples
        --------
        Basic imaging (default parameters):
        >>> pipeline.trajectory.image_molecules(traj_selection="all")

        Image without making molecules whole:
        >>> pipeline.trajectory.image_molecules(
        ...     traj_selection="all",
        ...     make_whole=False
        ... )

        Image with specific anchor molecules:
        >>> # Anchor protein (molecules 0-2), image solvent around it
        >>> protein_molecules = np.array([0, 1, 2])
        >>> pipeline.trajectory.image_molecules(
        ...     traj_selection=[0, 1],
        ...     anchor_molecules=protein_molecules,
        ...     make_whole=True
        ... )

        Tag-based selection:
        >>> pipeline.trajectory.image_molecules(
        ...     traj_selection="tag:production",
        ...     make_whole=True
        ... )

        Notes
        -----
        - Dask trajectories (use_memmap=True) handle memory management automatically
        - Requires trajectory to have periodic boundary condition information
        - Essential for correct visualization of periodic systems
        - Does not change topology or number of atoms
        - Make molecules whole first to prevent artifacts

        Raises
        ------
        ValueError
            If no trajectories are loaded
        ValueError
            If traj_selection contains invalid indices/names
        ValueError
            If trajectory lacks unit cell information
        """
        # Get trajectory indices to process
        indices_to_process = pipeline_data.trajectory_data.get_trajectory_indices(
            traj_selection
        )

        # Check if features exist for trajectories to be imaged
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "image_molecules", indices_to_process
        )

        print(
            f"Imaging {len(indices_to_process)} trajectories "
            f"(make_whole={make_whole})"
        )

        # Image each selected trajectory
        for idx in indices_to_process:
            trajectory = pipeline_data.trajectory_data.trajectories[idx]

            print(f"  Imaging trajectory {idx} ({trajectory.n_frames} frames)...")

            # Apply imaging (in-place for both DaskMDTrajectory and md.Trajectory)
            trajectory.image_molecules(
                anchor_molecules=anchor_molecules,
                other_molecules=other_molecules,
                make_whole=make_whole,
                inplace=True,
            )

            print(f"    ✓ Trajectory {idx} imaged successfully")

        print(
            f"Imaging completed successfully for {len(indices_to_process)} trajectories"
        )

    def remove_solvent(
        self,
        pipeline_data: PipelineData,
        traj_selection: Union[int, str, List[Union[int, str]], str],
        exclude: Optional[list] = None,
        force: bool = False,
    ) -> None:
        """
        Remove solvent atoms from trajectories (always in-place).

        This method removes solvent atoms using MDTraj's remove_solvent method,
        which identifies and removes common solvent molecules (water, ions, etc.).
        The operation modifies both coordinates AND topology, changing the number
        of atoms in the trajectory. Labels are automatically adjusted to match the
        new residue structure.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.remove_solvent(traj_selection="all")  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.remove_solvent(pipeline_data, traj_selection="all")  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        traj_selection : int, str, list, or "all"
            Selection of trajectories to process (required):

            - int: single trajectory by index
            - str: trajectory name, tag, or pattern (e.g., "tag:system_A", "traj_*")
            - list: multiple indices/names/tags
            - "all": all loaded trajectories
        exclude : list, optional
            List of solvent residue names to KEEP (not remove). Common values
            include ['HOH', 'WAT'] to keep water molecules while removing other
            solvents. If None, removes all recognized solvent molecules.
        force : bool, default=False
            Whether to force removal even when features have been calculated.
            When True, existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Modifies trajectories in-place. No return value.

        Examples
        --------
        Remove all solvent:
        >>> pipeline.trajectory.remove_solvent(traj_selection="all")

        Keep water, remove other solvent:
        >>> pipeline.trajectory.remove_solvent(
        ...     traj_selection="all",
        ...     exclude=['HOH', 'WAT']
        ... )

        Remove solvent from specific trajectories:
        >>> pipeline.trajectory.remove_solvent(
        ...     traj_selection=[0, 1, 2],
        ...     exclude=None
        ... )

        Tag-based selection:
        >>> pipeline.trajectory.remove_solvent(
        ...     traj_selection="tag:production"
        ... )

        Notes
        -----
        - Dask trajectories (use_memmap=True) handle memory management automatically
        - This operation CHANGES the number of atoms and topology
        - Labels are automatically filtered to match new residue structure
        - Features must be recalculated after this operation
        - Common solvent names: HOH, WAT, SOL, Na+, Cl-, etc.

        Raises
        ------
        ValueError
            If no trajectories are loaded
        ValueError
            If traj_selection contains invalid indices/names
        """
        # Get trajectory indices to process
        indices_to_process = pipeline_data.trajectory_data.get_trajectory_indices(
            traj_selection
        )

        # Check if features exist for trajectories to be modified
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "remove_solvent", indices_to_process
        )

        print(
            f"Removing solvent from {len(indices_to_process)} trajectories "
            f"(exclude={exclude if exclude else 'none'})"
        )

        # Track kept residue indices for label filtering
        kept_residue_indices = {}
        # Store lightweight residue structure info for mapping
        original_residue_info = {}

        # Process each selected trajectory
        for idx in indices_to_process:
            trajectory = pipeline_data.trajectory_data.trajectories[idx]

            print(
                f"  Removing solvent from trajectory {idx} "
                f"({trajectory.n_frames} frames, {trajectory.n_atoms} atoms)..."
            )

            # BEFORE removal: Store lightweight residue structure (name + atoms)
            # This allows us to map remaining residues back to their original indices
            # Note: resSeq is unreliable as MDTraj renumbers after remove_solvent()
            original_residue_info[idx] = [
                (res.name, tuple((a.name, a.element.symbol) for a in res.atoms))
                for res in trajectory.topology.residues
            ]

            # Store counts before modification
            original_n_atoms = trajectory.n_atoms
            original_n_residues = trajectory.n_residues

            # Apply solvent removal (Trajectory method decides what is solvent!)
            trajectory.remove_solvent(exclude=exclude, inplace=True)

            # AFTER removal: Map remaining residues back to their ORIGINAL indices
            # Helper handles the matching of residues by name + atom composition
            kept_residue_indices[idx] = LabelOperationHelper.map_residues_to_original_indices(
                original_residue_info[idx], trajectory.topology
            )

            atoms_removed = original_n_atoms - trajectory.n_atoms
            residues_removed = original_n_residues - trajectory.n_residues

            print(
                f"    ✓ Trajectory {idx} processed: "
                f"{original_n_atoms} → {trajectory.n_atoms} atoms, "
                f"{original_n_residues} → {trajectory.n_residues} residues "
                f"({atoms_removed} atoms, {residues_removed} residues removed)"
            )

        # Apply same removal to labels for consistency
        LabelOperationHelper.apply_atom_selection_to_labels(
            pipeline_data.trajectory_data, indices_to_process, kept_residue_indices
        )

        print(
            f"Solvent removal completed successfully for {len(indices_to_process)} trajectories"
        )

    def join(
        self,
        pipeline_data: PipelineData,
        target_traj: Union[int, str],
        source_traj: Union[int, str],
        check_topology: bool = True,
        remove_source: bool = True,
        new_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Join two trajectories along the frame axis (in-place on target).

        This method concatenates frames from source_traj to target_traj, creating
        a single trajectory with combined frames. The target trajectory is modified
        in-place to contain all frames. Optionally, the source trajectory can be
        removed after joining, and the target can be renamed.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.join(target_traj=0, source_traj=1)  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.join(pipeline_data, target_traj=0, source_traj=1)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        target_traj : int or str
            Target trajectory (receives joined frames). Can be index or name.
        source_traj : int or str
            Source trajectory (provides frames to join). Can be index or name.
        check_topology : bool, default=True
            Whether to check topology compatibility between trajectories.
            When True, raises error if atom counts differ.
        remove_source : bool, default=True
            Whether to remove source trajectory after joining.
            When True, source_traj is deleted from trajectory list.
        new_name : str, optional
            New name for target trajectory after joining.
            If None, keeps original target name.
        force : bool, default=False
            Whether to force join even when features have been calculated.
            When True, existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Modifies target trajectory in-place and optionally removes source.

        Examples
        --------
        Basic join (source removed):
        >>> pipeline.trajectory.join(target_traj=0, source_traj=1)

        Join and keep source:
        >>> pipeline.trajectory.join(
        ...     target_traj=0,
        ...     source_traj=1,
        ...     remove_source=False
        ... )

        Join with renaming:
        >>> pipeline.trajectory.join(
        ...     target_traj="system_A_rep1",
        ...     source_traj="system_A_rep2",
        ...     new_name="system_A_combined"
        ... )

        Join without topology check:
        >>> pipeline.trajectory.join(
        ...     target_traj=0,
        ...     source_traj=1,
        ...     check_topology=False
        ... )

        Notes
        -----
        - Dask trajectories (use_memmap=True) handle memory management automatically
        - Both trajectories must have compatible topologies (same atoms)
        - Time arrays are concatenated; check for time continuity separately
        - Labels are preserved for target trajectory only
        - Source trajectory labels are discarded

        Raises
        ------
        ValueError
            If no trajectories are loaded
        ValueError
            If target_traj or source_traj are invalid
        ValueError
            If target and source refer to same trajectory
        ValueError
            If topologies are incompatible (when check_topology=True)
        """
        # Resolve and validate trajectory indices
        target_idx, source_idx = TrajectoryValidationHelper.resolve_and_validate_traj_pair(
            pipeline_data.trajectory_data, target_traj, source_traj
        )

        # Check features for both trajectories
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "join", [target_idx, source_idx]
        )

        target_trajectory = pipeline_data.trajectory_data.trajectories[target_idx]
        source_trajectory = pipeline_data.trajectory_data.trajectories[source_idx]

        print(
            f"Joining trajectories: target={target_idx} ({target_trajectory.n_frames} frames) + "
            f"source={source_idx} ({source_trajectory.n_frames} frames)"
        )

        # Perform join operation
        joined_traj = target_trajectory.join(source_trajectory, check_topology=check_topology)

        # Update target trajectory with joined result
        pipeline_data.trajectory_data.trajectories[target_idx] = joined_traj

        print(
            f"  ✓ Joined trajectory now has {joined_traj.n_frames} frames "
            f"({target_trajectory.n_frames} + {source_trajectory.n_frames})"
        )

        # Optionally rename target
        if new_name is not None:
            pipeline_data.trajectory_data.trajectory_names[target_idx] = new_name
            print(f"  ✓ Renamed target trajectory to '{new_name}'")

        # Optionally remove source trajectory
        if remove_source:
            # Use remove_trajectory helper to handle cleanup
            TrajectoryProcessHelper.execute_removal(
                pipeline_data.trajectory_data, [source_idx]
            )
            print(f"  ✓ Removed source trajectory (index {source_idx})")

        print("Join operation completed successfully")

    def stack(
        self,
        pipeline_data: PipelineData,
        target_traj: Union[int, str],
        source_traj: Union[int, str],
        remove_source: bool = True,
        new_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Stack two trajectories along the atom axis (creates new trajectory).

        This method combines atoms from source_traj with target_traj, creating
        a single trajectory with combined atoms but requiring identical frame counts.
        This is useful for combining protein and ligand trajectories or merging
        different molecular components. The target trajectory is replaced with the
        stacked result, and optionally the source can be removed.

        Warning
        -------
        When using PipelineManager, do NOT provide the pipeline_data parameter.
        The PipelineManager automatically injects this parameter.

        Pipeline mode:
        >>> pipeline = PipelineManager()
        >>> pipeline.trajectory.stack(target_traj=0, source_traj=1)  # NO pipeline_data parameter

        Standalone mode:
        >>> pipeline_data = PipelineData()
        >>> manager = TrajectoryManager()
        >>> manager.stack(pipeline_data, target_traj=0, source_traj=1)  # pipeline_data required

        Parameters
        ----------
        pipeline_data : PipelineData
            Pipeline data container with trajectory data
        target_traj : int or str
            Target trajectory (receives stacked atoms). Can be index or name.
        source_traj : int or str
            Source trajectory (provides atoms to stack). Can be index or name.
        remove_source : bool, default=True
            Whether to remove source trajectory after stacking.
            When True, source_traj is deleted from trajectory list.
        new_name : str, optional
            New name for target trajectory after stacking.
            If None, keeps original target name.
        force : bool, default=False
            Whether to force stack even when features have been calculated.
            When True, existing features become invalid and should be recalculated.

        Returns
        -------
        None
            Replaces target trajectory with stacked result and optionally removes source.

        Examples
        --------
        Basic stack (source removed):
        >>> pipeline.trajectory.stack(target_traj=0, source_traj=1)

        Stack and keep source:
        >>> pipeline.trajectory.stack(
        ...     target_traj=0,
        ...     source_traj=1,
        ...     remove_source=False
        ... )

        Stack with renaming:
        >>> pipeline.trajectory.stack(
        ...     target_traj="protein",
        ...     source_traj="ligand",
        ...     new_name="complex"
        ... )

        Notes
        -----
        - Dask trajectories (use_memmap=True) handle memory management automatically
        - Both trajectories MUST have same number of frames
        - Topologies are merged to create combined system
        - Labels from both trajectories are combined and renumbered
        - Useful for combining protein + ligand, or multiple chains

        Raises
        ------
        ValueError
            If no trajectories are loaded
        ValueError
            If target_traj or source_traj are invalid
        ValueError
            If target and source refer to same trajectory
        ValueError
            If frame counts differ between trajectories
        """
        # Resolve and validate trajectory indices
        target_idx, source_idx = TrajectoryValidationHelper.resolve_and_validate_traj_pair(
            pipeline_data.trajectory_data, target_traj, source_traj
        )

        # Check features for both trajectories
        TrajectoryValidationHelper.check_features_before_trajectory_changes(
            pipeline_data, force, "stack", [target_idx, source_idx]
        )

        target_trajectory = pipeline_data.trajectory_data.trajectories[target_idx]
        source_trajectory = pipeline_data.trajectory_data.trajectories[source_idx]

        # Validate frame counts
        if target_trajectory.n_frames != source_trajectory.n_frames:
            raise ValueError(
                f"Trajectories must have same number of frames for stacking. "
                f"Target has {target_trajectory.n_frames}, source has {source_trajectory.n_frames}"
            )

        print(
            f"Stacking trajectories: target={target_idx} ({target_trajectory.n_atoms} atoms) + "
            f"source={source_idx} ({source_trajectory.n_atoms} atoms)"
        )

        # Store original atom/residue counts for label handling
        target_n_residues = target_trajectory.n_residues

        # Perform stack operation
        stacked_traj = target_trajectory.stack(source_trajectory)

        # Update target trajectory with stacked result
        pipeline_data.trajectory_data.trajectories[target_idx] = stacked_traj

        print(
            f"  ✓ Stacked trajectory now has {stacked_traj.n_atoms} atoms "
            f"({target_trajectory.n_atoms} + {source_trajectory.n_atoms})"
        )

        # Combine labels from both trajectories
        LabelOperationHelper.combine_stack_labels(
            pipeline_data.trajectory_data, target_idx, source_idx, target_n_residues
        )

        # Optionally rename target
        if new_name is not None:
            pipeline_data.trajectory_data.trajectory_names[target_idx] = new_name
            print(f"  ✓ Renamed target trajectory to '{new_name}'")

        # Optionally remove source trajectory
        if remove_source:
            # Use remove_trajectory helper to handle cleanup
            TrajectoryProcessHelper.execute_removal(
                pipeline_data.trajectory_data, [source_idx]
            )
            print(f"  ✓ Removed source trajectory (index {source_idx})")

        # Validate trajectory types

        print("Stack operation completed successfully")
