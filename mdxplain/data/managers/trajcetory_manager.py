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

from ..helper.nomenclature import Nomenclature
from ..helper.trajectory_loader import TrajectoryLoader


class TrajectoryManager:
    """Manager for trajectory data objects."""

    def __init__(self, stride=1, concat=False, selection=None):
        """
        Initialize trajectory manager.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        stride : int, default=1
            Load every stride-th frame from trajectories. Use values > 1 to
            reduce memory usage and computation time by subsampling frames.
        concat : bool, default=False
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. Useful when dealing with trajectory splits.
        selection : str, optional
            MDTraj selection string to apply to all loaded trajectories.
            See: https://mdtraj.org/1.9.4/atom_selection.html

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        """
        self.default_stride = stride
        self.default_concat = concat
        self.default_selection = selection

    def load_trajectories(
        self, traj_data, data_input, concat=False, stride=None, selection=None, force=False
    ):
        """
        Load molecular dynamics trajectories from files or directories into a TrajectoryData object.

        This method handles loading of MD trajectories in various formats
        (e.g., .xtc, .dcd, .trr) along with their topology files. The loading
        is performed using the TrajectoryLoader class which supports automatic
        format detection and multiple trajectory handling.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        data_input : str or list
            Path to directory containing trajectory files, or list of trajectory
            file paths. When a directory is provided, all supported trajectory
            files in that directory will be loaded.
        concat : bool, default=False
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. Useful when dealing with trajectory splits.
        stride : int, default=1
            Load every stride-th frame from trajectories. Use values > 1 to
            reduce memory usage and computation time by subsampling frames.
        selection : str, optional
            MDTraj selection string to apply to all loaded trajectories.
            See: https://mdtraj.org/1.9.4/atom_selection.html
        force : bool, default=False
            Whether to force loading even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')

        Notes:
        -----
        - Supported formats depend on MDTraj capabilities
        - Topology files (.pdb, .gro, .psf) should be in the same directory
        - Large trajectories benefit from striding to reduce memory usage
        - Selection is applied to all trajectories after loading
        """
        # Check for existing features before loading new trajectories
        self._check_features_before_trajectory_changes(traj_data, force, "load")
        
        selection, concat, stride = self._prepare_add_parameters(
            selection, concat, stride
        )

        result = TrajectoryLoader.load_trajectories(data_input, concat, stride)
        traj_data.trajectories = result["trajectories"]
        traj_data.trajectory_names = result["names"]

        # Apply atom selection if provided
        if selection is not None:
            self.select_atoms(traj_data, selection)

        # Check trajectory consistency
        self._check_trajectory_consistency(traj_data)

    def _check_features_before_trajectory_changes(self, traj_data, force, operation_name):
        """
        Check if features exist before changing trajectories and handle accordingly.
        
        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        force : bool
            Whether to force the operation
        operation_name : str
            Name of the operation for error messages ("load" or "add")
            
        Raises:
        -------
        ValueError
            If features exist and force=False
        """
        if not traj_data.features:
            return
            
        feature_list = list(traj_data.features.keys())
        
        if not force:
            raise ValueError(
                f"Cannot {operation_name} trajectories: {len(feature_list)} feature(s) already computed: {', '.join(feature_list)}. "
                f"{operation_name.capitalize()}ing new trajectories would invalidate these features. "
                f"Use force=True to proceed, or call reset_features() with FeatureManager to clear features first."
                f"The whole analysis is based on the trajectory data, so if you want to change this base, you need to make the analysis again."
                f"Maybe create a new TrajectoryData object and start from scratch, to not lose your results."
            )
        
        print(f"WARNING: {operation_name.capitalize()}ing new trajectories will invalidate {len(feature_list)} existing features. "
              f"Features must be recalculated and labels reset."
              f"The whole analysis is based on the trajectory data, so if you want to change this base, you need to make the analysis again."
              f"Maybe create a new TrajectoryData object and start from scratch, to not lose your results."
              )

    def add_trajectory(
        self, traj_data, data_input, concat=False, stride=None, selection=None, force=False
    ):
        """
        Add molecular dynamics trajectories to TrajectoryData object.

        This method works like load_trajectories but appends new trajectories
        instead of replacing existing ones. Useful for loading additional
        trajectory data without losing previously loaded trajectories.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        data_input : str or list
            Path to directory containing trajectory files, or list of trajectory
            file paths. When a directory is provided, all supported trajectory
            files in that directory will be loaded.
        concat : bool, default=False
            Whether to concatenate multiple trajectories per system into single
            trajectory objects. Useful when dealing with trajectory splits.
        stride : int, default=1
            Load every stride-th frame from trajectories. Use values > 1 to
            reduce memory usage and computation time by subsampling frames.
        selection : str, optional
            MDTraj selection string to apply to all newly loaded trajectories.
            See: https://mdtraj.org/1.9.4/atom_selection.html
        force : bool, default=False
            Whether to force loading even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

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
        self._check_features_before_trajectory_changes(traj_data, force, "add")
            
        # Prepare parameters and validate
        selection, concat, stride = self._prepare_add_parameters(
            selection, concat, stride
        )

        # Load and process new trajectories
        new_trajectories, new_names = self._load_new_trajectories(
            data_input, concat, stride
        )
        new_trajectories = self._apply_selection_to_new_trajectories(
            new_trajectories, selection
        )

        # Add to existing trajectories
        traj_data.trajectories.extend(new_trajectories)
        traj_data.trajectory_names.extend(new_names)

        # Check trajectory consistency
        self._check_trajectory_consistency(traj_data)

    def _prepare_add_parameters(self, selection, concat, stride):
        """Prepare and validate parameters for adding trajectories."""
        if selection is None:
            selection = self.default_selection
        if concat is None:
            concat = self.default_concat
        if stride is None:
            stride = self.default_stride
        return selection, concat, stride

    def _load_new_trajectories(self, data_input, concat, stride):
        """Load new trajectories and return trajectories and names."""
        result = TrajectoryLoader.load_trajectories(data_input, concat, stride)
        return result["trajectories"], result["names"]

    def _apply_selection_to_new_trajectories(self, new_trajectories, selection):
        """Apply atom selection to new trajectories if provided."""
        if selection is not None:
            for i, traj in enumerate(new_trajectories):
                atom_indices = traj.topology.select(selection)
                new_trajectories[i] = traj.atom_slice(atom_indices)
        return new_trajectories

    def _check_trajectory_consistency(self, traj_data):
        """
        Check if all trajectories have consistent atom and residue counts.

        Issues detailed warnings if inconsistencies are found, providing
        specific information about which trajectories differ and suggestions
        for resolution.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        None
            Prints warnings if trajectory inconsistencies are found
        """
        if not traj_data.trajectories or len(traj_data.trajectories) <= 1:
            return

        any_inconsistency = self._check_all_inconsistencies(traj_data)
        if any_inconsistency:
            self._print_consistency_suggestion()

    def _check_all_inconsistencies(self, traj_data):
        """Check for any type of trajectory inconsistency."""
        residue_inconsistent = self._find_residue_inconsistency(traj_data)
        if residue_inconsistent:
            return True

        atom_inconsistent = self._find_atom_inconsistency(traj_data)
        return atom_inconsistent

    def _print_consistency_suggestion(self):
        """Print suggestion for handling trajectory inconsistencies."""
        print(
            "\nSuggestion: Use select_atoms() with a selector that applies "
            "to all trajectories\nto find common ground for analysis, or use "
            "remove_trajectory() to exclude\nincompatible trajectories from "
            "the analysis.\n"
        )

    def _find_residue_inconsistency(self, traj_data):
        """
        Check for residue count inconsistencies between trajectories.

        All trajectories must have the same number of residues for some features.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        bool
            True if residue inconsistency found, False if all consistent
        """
        return self._find_count_inconsistency(
            traj_data,
            count_attr="n_residues",
            count_type="residue",
            plural_type="residues",
        )

    def _find_atom_inconsistency(self, traj_data):
        """
        Check for atom count inconsistencies between trajectories.

        All trajectories must have the same number of atoms for some features.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        bool
            True if atom inconsistency found, False if all consistent
        """
        return self._find_count_inconsistency(
            traj_data, count_attr="n_atoms", count_type="atom", plural_type="atoms"
        )

    def _find_count_inconsistency(self, traj_data, count_attr, count_type, plural_type):
        """
        Check for count inconsistencies between trajectories.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        count_attr : str
            Attribute name to check (e.g., 'n_residues', 'n_atoms')
        count_type : str
            Single form of count type (e.g., 'residue', 'atom')
        plural_type : str
            Plural form of count type (e.g., 'residues', 'atoms')

        Returns:
        --------
        bool
            True if inconsistency found, False if all consistent
        """
        if not traj_data.trajectories or len(traj_data.trajectories) <= 1:
            return False

        ref_count = getattr(traj_data.trajectories[0], count_attr)
        inconsistent = self._collect_inconsistent_trajectories(
            traj_data, count_attr, ref_count
        )

        if inconsistent:
            self._print_inconsistency_warning(
                traj_data, inconsistent, ref_count, count_type, plural_type
            )
            return True

        return False

    def _collect_inconsistent_trajectories(self, traj_data, count_attr, ref_count):
        """Collect trajectories with inconsistent counts."""
        inconsistent = []
        for i, traj in enumerate(traj_data.trajectories):
            count = getattr(traj, count_attr)
            if count != ref_count:
                inconsistent.append((i, traj_data.trajectory_names[i], count))
        return inconsistent

    def _print_inconsistency_warning(
        self, traj_data, inconsistent, ref_count, count_type, plural_type
    ):
        """
        Print warning message for count inconsistencies.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        inconsistent : list
            List of (index, name, count) tuples for inconsistent trajectories
        ref_count : int
            Reference count from first trajectory
        count_type : str
            Single form of count type
        plural_type : str
            Plural form of count type
        """
        inconsistent_list = "\n".join(
            [
                f"  [{idx}] {name}: {count} {plural_type}"
                for idx, name, count in inconsistent
            ]
        )
        print(
            f"\nWARNING: Inconsistent {count_type} counts detected!\n"
            f"Reference trajectory '{traj_data.trajectory_names[0]}' has "
            f"{ref_count} {plural_type}.\n"
            f"Trajectories with different {count_type} counts:\n"
            f"{inconsistent_list}\n\n"
            f"{count_type.capitalize()}-based feature calculations may fail or produce "
            f"incorrect results."
        )

    def remove_trajectory(self, traj_data, trajs, force=False):
        """
        Remove specified trajectories from the loaded trajectory list.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        trajs : int, str, or list
            Trajectory index (int), name (str), or list of indices/names to remove.
            Can be mixed list of integers and strings.
        force : bool, default=False
            Force removal even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        >>> traj_manager.remove_trajectory(traj_data, [0, 1])

        Raises:
        -------
        ValueError
            If trajectories are not loaded, if trajs contains invalid indices/names,
            or if features exist and force=False
        """
        if traj_data.trajectories is None:
            raise ValueError("No trajectories loaded to remove.")

        self._check_features_before_trajectory_changes(traj_data, force, "remove")
        indices_to_remove = self._prepare_removal_indices(traj_data, trajs)
        self._execute_removal(traj_data, indices_to_remove)
        self._check_trajectory_consistency(traj_data)

    def _prepare_removal_indices(self, traj_data, trajs):
        """
        Convert trajs input to sorted indices for removal.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        trajs : int, str, or list
            Trajectory index (int), name (str), or list of indices/names to remove.
            Can be mixed list of integers and strings.

        Returns:
        --------
        list
            List of trajectory indices to remove
        """
        if not isinstance(trajs, list):
            trajs = [trajs]
        indices_to_remove = self._resolve_selection(traj_data, trajs)
        indices_to_remove.sort(reverse=True)  # Avoid index shifting issues
        return indices_to_remove

    def _execute_removal(self, traj_data, indices_to_remove):
        """
        Remove trajectories and names, return count of removed items.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        indices_to_remove : list
            List of trajectory indices to remove
        """
        removed_trajectories = []
        for idx in indices_to_remove:
            removed_name = traj_data.trajectory_names[idx]
            removed_trajectories.append(f"{removed_name} (was at index {idx})")
            del traj_data.trajectories[idx]
            del traj_data.trajectory_names[idx]

        # Print removal summary
        for removed_info in removed_trajectories:
            print(f"Removed trajectory: {removed_info}")

        print(
            f"Removed {len(indices_to_remove)} trajectory(ies). "
            f"{len(traj_data.trajectories)} trajectory(ies) remaining."
        )
        return len(indices_to_remove)

    def cut_traj(self, traj_data, cut=None, stride=None, selection=None, force=False):
        """
        Cut and/or stride trajectories.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        cut : int, optional
            Frame number after which to cut the trajectories.
            Frames after this index will be removed.
        stride : int, default=1
            Take every stride-th frame. Use values > 1 to subsample frames.
        selection : list, optional
            List of trajectory indices (int) or names (str) to process.
            If None, all trajectories will be processed.
        force : bool, default=False
            Whether to force cutting even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        >>> traj_manager.cut_traj(traj_data, cut=1000)

        >>> # Cut and stride specific trajectories by index
        >>> traj_manager.cut_traj(traj_data, cut=500, stride=5, selection=[0, 2, 4])

        >>> # Cut and stride specific trajectories by name
        >>> traj_manager.cut_traj(
        ...     traj_data, cut=500, stride=5,
        ...     selection=['system1_prot_traj1', 'system2_prot_traj2']
        ... )

        Raises:
        -------
        ValueError
            If trajectories are not loaded or if selection contains invalid indices/names
        """
        self._check_features_before_trajectory_changes(traj_data, force, "cut")

        # Validate and set defaults
        selection, stride = self._prepare_cut_parameters(selection, stride)
        self._validate_trajectories_loaded(traj_data)

        # Determine which trajectories to process
        indices_to_process = self._get_indices_to_process(traj_data, selection)

        # Process selected trajectories
        for idx in indices_to_process:
            processed_traj = self._process_trajectory_cuts(
                traj_data.trajectories[idx], cut, stride
            )
            traj_data.trajectories[idx] = processed_traj

    def _prepare_cut_parameters(self, selection, stride):
        """Prepare and validate parameters for cutting."""
        if selection is None:
            selection = self.default_selection
        if stride is None:
            stride = self.default_stride
        return selection, stride

    def _validate_trajectories_loaded(self, traj_data):
        """Validate that trajectories are loaded."""
        if traj_data.trajectories is None:
            raise ValueError("Trajectories must be loaded before cutting.")

    def _get_indices_to_process(self, traj_data, selection):
        """Get list of trajectory indices to process."""
        if selection is None:
            return list(range(len(traj_data.trajectories)))
        else:
            return self._resolve_selection(traj_data, selection)

    def _process_trajectory_cuts(self, traj, cut, stride):
        """Apply cut and stride operations to a single trajectory."""
        # Apply cut if specified
        if cut is not None and cut < traj.n_frames:
            traj = traj[:cut]

        # Apply stride if not 1
        if stride > 1:
            traj = traj[::stride]

        return traj

    def _resolve_selection(self, traj_data, selection=None):
        """
        Resolve selection list to trajectory indices.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selection : list, optional
            List of trajectory indices (int) or names (str)

        Returns:
        --------
        list
            List of trajectory indices

        Raises:
        -------
        ValueError
            If selection contains invalid indices or names
        """
        if selection is None:
            selection = self.default_selection

        indices = []
        for item in selection:
            idx = self._resolve_single_selection_item(item, traj_data)
            indices.append(idx)

        return indices

    def _resolve_single_selection_item(self, item, traj_data):
        """Resolve a single selection item to trajectory index."""
        if isinstance(item, int):
            return self._resolve_index_selection(item, traj_data)
        elif isinstance(item, str):
            return self._resolve_name_selection(item, traj_data)
        else:
            raise ValueError(
                f"Selection item must be int (index) or str (name), got {type(item)}"
            )

    def _resolve_index_selection(self, item, traj_data):
        """Resolve integer index selection."""
        if 0 <= item < len(traj_data.trajectories):
            return item
        else:
            raise ValueError(
                f"Trajectory index {item} out of range (0-{len(traj_data.trajectories)-1})"
            )

    def _resolve_name_selection(self, item, traj_data):
        """Resolve string name selection."""
        if item in traj_data.trajectory_names:
            return traj_data.trajectory_names.index(item)
        else:
            raise ValueError(
                f"Trajectory name '{item}' not found. Available names: {traj_data.trajectory_names}"
            )

    def select_atoms(self, traj_data, selection=None, trajs=None, force=False):
        """
        Apply atom selection to trajectories using MDTraj selection syntax.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object
        selection : str, optional
            MDTraj selection string (e.g., "protein", "backbone", "resid 10 to 50")
            See: https://mdtraj.org/1.9.4/atom_selection.html
        trajs : list, optional
            List of trajectory indices (int) or names (str) to process.
            If None, all trajectories will be processed.
        force : bool, default=False
            Whether to force atom selection even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Examples:
        ---------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        >>> traj_manager.select_atoms(traj_data, "protein")

        Raises:
        -------
        ValueError
            If trajectories are not loaded or if selection/trajs contain invalid values
        """
        self._check_features_before_trajectory_changes(traj_data, force, "select_atoms")
        
        if selection is None:
            selection = self.default_selection

        if traj_data.trajectories is None:
            raise ValueError("Trajectories must be loaded before atom selection.")

        # Determine which trajectories to process
        if trajs is None:
            # Process all trajectories
            indices_to_process = list(range(len(traj_data.trajectories)))
        else:
            indices_to_process = self._resolve_selection(traj_data, trajs)

        # Apply atom selection to each trajectory
        for idx in indices_to_process:
            traj = traj_data.trajectories[idx]

            # Get atom indices matching the selection
            atom_indices = traj.topology.select(selection)

            # Apply atom slice to trajectory
            selected_traj = traj.atom_slice(atom_indices)

            # Update trajectory
            traj_data.trajectories[idx] = selected_traj

    def add_labels(
        self,
        traj_data,
        fragment_definition=None,
        fragment_type=None,
        fragment_molecule_name=None,
        consensus=False,
        aa_short=False,
        try_web_lookup=True,
        **nomenclature_kwargs,
    ):
        """
        Initialize the Nomenclature labeler.

        This class provides a unified interface to create consensus labels for MD trajectories
        using different mdciao nomenclature systems.

        Parameters
        ----------
        traj_data : TrajectoryData
            Trajectory data object
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
        write_to_disk : bool, default False
            Whether to write cache files to disk
        cache_folder : str, default "./cache"
            Folder for cache files
        **nomenclature_kwargs
            Additional keyword arguments passed to the mdciao labelers

        Notes
        -----
        This class uses mdciao consensus nomenclature systems:
        https://proteinformatics.uni-leipzig.de/mdciao/api/generated/mdciao.nomenclature.html

        Supported fragment types:
        - gpcr: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerGPCR.html#mdciao.nomenclature.LabelerGPCR
        - cgn: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerCGN.html#mdciao.nomenclature.LabelerCGN
        - klifs: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS

        Examples
        --------
        >>> traj_data = TrajectoryData()
        >>> traj_manager = TrajectoryManager()
        >>> traj_manager.load_trajectories(traj_data, '../data')
        >>> traj_manager.add_labels(
        ...     traj_data, fragment_definition="receptor", fragment_type="gpcr",
        ...     fragment_molecule_name="adrb2_human", consensus=True
        ... )
        """
        if traj_data.trajectories is None:
            raise ValueError("Trajectories must be loaded before adding labels.")
        if traj_data.cache_dir is None:
            write_to_disk = False
        else:
            write_to_disk = True

        # Use first trajectory topology for labeling
        nomenclature = Nomenclature(
            topology=traj_data.trajectories[0].topology,
            fragment_definition=fragment_definition,
            fragment_type=fragment_type,
            fragment_molecule_name=fragment_molecule_name,
            consensus=consensus,
            aa_short=aa_short,
            try_web_lookup=try_web_lookup,
            write_to_disk=write_to_disk,
            cache_folder=traj_data.cache_dir,
            **nomenclature_kwargs,
        )

        traj_data.res_label_data = nomenclature.create_trajectory_label_dicts()
    
    def reset_trajectory_data(self, traj_data):
        """
        Reset the trajectory data object.

        Parameters:
        -----------
        traj_data : TrajectoryData
            Trajectory data object

        Returns:
        --------
        None

        Notes:
        -----
        This method resets the trajectory data object to its initial state.
        """
        traj_data.trajectories = []
        traj_data.trajectory_names = []
        traj_data.res_label_data = None
        traj_data.features = {}
