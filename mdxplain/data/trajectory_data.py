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

"""
MD trajectory data container and analysis interface.

Container for MD trajectory data and analysis results. Uses a pluggable
Loader class for flexible loading strategies. Provides unified interface
for handling multiple trajectories and their derived features.
"""

from ..utils.data_utils import DataUtils
from ..utils.nomenclature import Nomenclature
from .feature_data import FeatureData
from .trajectory_loader import TrajectoryLoader


class TrajectoryData:
    """
    Container for MD trajectory data and analysis results.

    This class serves as the main interface for loading molecular dynamics trajectories
    and computing various features such as distances and contacts. It provides a
    unified interface for handling multiple trajectories and their derived features
    with memory-efficient processing capabilities.

    The class uses a pluggable architecture where different feature types can be
    added and computed independently or with dependencies on other features.

    Examples:
    ---------
    Basic usage with distances and contacts:

    >>> import mdxplain.data.feature_type as feature_type
    >>> traj = TrajectoryData()
    >>> traj.load_trajectories('../data')
    >>> traj.add_feature(feature_type.Distances())
    >>> traj.add_feature(feature_type.Contacts(cutoff=4.5))
    >>> distances = traj.get_feature(feature_type.Distances())
    >>> contacts = traj.get_feature(feature_type.Contacts())

    Memory-mapped processing for large datasets:

    >>> traj = TrajectoryData(use_memmap=True, cache_dir='./cache')
    >>> traj.load_trajectories('../large_data')
    >>> traj.add_feature(feature_type.Distances(), chunk_size=1000)

    Attributes:
    -----------
    use_memmap : bool
        Whether to use memory mapping for large datasets
    cache_dir : str or None
        Directory for cache files when using memory mapping
    trajectories : list or None
        List of loaded MD trajectory objects
    trajectory_names : list or None
        List of trajectory names
    features : dict
        Dictionary storing FeatureData instances by feature type string
    """

    def __init__(self, use_memmap=False, cache_dir=None):
        """
        Initialize trajectory data container.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets. When True,
            large arrays are stored on disk and accessed via memory mapping
            to reduce RAM usage.
        cache_dir : str, optional
            Directory for cache files when using memory mapping. If None
            and use_memmap is True, defaults to './cache'.

        Examples:
        ---------
        >>> # Standard initialization
        >>> traj = TrajectoryData()

        >>> # For large datasets with memory mapping
        >>> traj = TrajectoryData(use_memmap=True, cache_dir='/tmp/mdxplain_cache')
        """
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir

        if use_memmap and cache_dir is None:
            self.cache_dir = "./cache"

        self.trajectories = None
        self.trajectory_names = None  # List of trajectory names
        self.features = {}  # Dictionary to store FeatureData instances by feature type
        self.labels = None  # Labels for trajectory residues

    def add_feature(self, feature_type, cache_path=None, chunk_size=None, force=False):
        """
        Add and compute a feature for the loaded trajectories.

        This method creates a FeatureData instance for the specified feature type,
        handles dependency checking, and computes the feature data. Features with
        dependencies (like Contacts depending on Distances) will automatically
        use the required input data.

        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object (e.g., Distances(), Contacts()). The feature
            type determines what kind of analysis will be performed.
        cache_path : str, optional
            Specific cache path for this feature's data when using memory mapping.
            If None, uses the default cache directory structure.
        chunk_size : int, optional
            Chunk size for processing large datasets. Smaller chunks use less
            memory but may be slower. If None, uses automatic chunking.
        force : bool, default=False
            Whether to force recomputation of the feature even if it already exists.

        Raises:
        -------
        ValueError
            If the feature already exists with computed data, if required
            dependencies are missing, or if trajectories are not loaded.

        Examples:
        ---------
        >>> # Add basic distance feature
        >>> traj.add_feature(feature_type.Distances())

        >>> # Add contacts with custom cutoff
        >>> traj.add_feature(feature_type.Contacts(cutoff=3.5))

        >>> # Add feature with memory mapping and chunking
        >>> traj.add_feature(
        ...     feature_type.Distances(),
        ...     cache_path='/tmp/distances.dat',
        ...     chunk_size=500
        ... )
        """
        feature_key = feature_type.get_type_name()

        # Check if feature already exists
        if (feature_key in self.features and
            self.features[feature_key].data is not None and not force):
            raise ValueError(
                f"{feature_key.capitalize()} FeatureData already exists.")
        if (feature_key in self.features and
            self.features[feature_key].data is not None and force):
            print(f"WARNING: {feature_key.capitalize()} FeatureData already "
                  f"exists. Forcing recomputation.")
            self.features[feature_key].reset()

        # Check dependencies
        dependencies = feature_type.get_dependencies()
        for dep in dependencies:
            if dep not in self.features or self.features[dep].get_data() is None:
                raise ValueError(
                    f"Dependency '{dep}' must be computed before '{feature_key}'."
                )

        # Create FeatureData instance
        feature_data = FeatureData(
            feature_type=feature_type,
            use_memmap=self.use_memmap,
            cache_path=cache_path,
            chunk_size=chunk_size,
        )

        # If the feature type uses another feature as input,
        # compute the feature with the input feature data
        # Otherwise, compute the feature with the trajectories
        if feature_type.get_input() is not None:
            feature_data.compute(
                self.features[feature_type.get_input()].get_data(),
                self.features[feature_type.get_input()].get_feature_names(),
                labels=self.labels,
            )
        else:
            if self.trajectories is None:
                raise ValueError(
                    "Trajectories must be loaded before computing features."
                )
            feature_data.compute(
                self.trajectories, feature_names=None, labels=self.labels
            )

        # Store the feature data
        self.features[feature_key] = feature_data

    def get_feature(self, feature_type):
        """
        Retrieve a computed feature by its type.

        This method returns the FeatureData instance for a previously computed
        feature. The returned object provides access to the computed data,
        feature names, analysis methods, and data reduction capabilities.

        Parameters:
        -----------
        feature_type : FeatureTypeBase
            Feature type object (e.g., Distances(), Contacts()). Must be
            the same type as used when adding the feature.

        Returns:
        --------
        FeatureData
            The FeatureData instance containing computed data and analysis methods.

        Raises:
        -------
        ValueError
            If the requested feature type has not been computed yet.

        Examples:
        ---------
        >>> # Get distances feature
        >>> distances = traj.get_feature(feature_type.Distances())
        >>> distance_data = distances.get_data()
        >>> feature_names = distances.get_feature_names()

        >>> # Get contacts and apply analysis
        >>> contacts = traj.get_feature(feature_type.Contacts())
        >>> frequency = contacts.analysis.compute_frequency()

        >>> # Apply data reduction
        >>> contacts.reduce_data(
        ...     contacts.ReduceMetrics.FREQUENCY,
        ...     threshold_min=0.1,
        ...     threshold_max=0.9
        ... )
        """
        feature_data = self.features.get(feature_type.get_type_name())
        if feature_data is None:
            raise ValueError(
                f"Feature {feature_type.get_type_name()} not found.")
        return feature_data

    def load_trajectories(self, data_input, concat=False, stride=1, selection=None):
        """
        Load molecular dynamics trajectories from files or directories.

        This method handles loading of MD trajectories in various formats
        (e.g., .xtc, .dcd, .trr) along with their topology files. The loading
        is performed using the TrajectoryLoader class which supports automatic
        format detection and multiple trajectory handling.

        Parameters:
        -----------
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

        Examples:
        ---------
        >>> # Load from directory
        >>> traj.load_trajectories('../data')

        >>> # Load specific files with striding
        >>> traj.load_trajectories(['traj1.xtc', 'traj2.xtc'], stride=10)

        >>> # Load and concatenate trajectories per system
        >>> traj.load_trajectories('../data', concat=True, stride=5)

        >>> # Load only protein atoms
        >>> traj.load_trajectories('../data', selection="protein")

        >>> # Load backbone atoms with striding
        >>> traj.load_trajectories('../data', stride=10, selection="backbone")

        Notes:
        -----
        - Supported formats depend on MDTraj capabilities
        - Topology files (.pdb, .gro, .psf) should be in the same directory
        - Large trajectories benefit from striding to reduce memory usage
        - Selection is applied to all trajectories after loading
        """
        result = TrajectoryLoader.load_trajectories(
            data_input, concat, stride
        )
        self.trajectories = result["trajectories"]
        self.trajectory_names = result["names"]

        # Apply atom selection if provided
        if selection is not None:
            self.select_atoms(selection)

        # Check trajectory consistency
        self._check_trajectory_consistency()

    def add_trajectory(self, data_input, concat=False, stride=1, selection=None):
        """
        Add molecular dynamics trajectories to existing loaded trajectories.

        This method works like load_trajectories but appends new trajectories
        instead of replacing existing ones. Useful for loading additional
        trajectory data without losing previously loaded trajectories.

        Parameters:
        -----------
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

        Examples:
        ---------
        >>> # Load initial trajectories
        >>> traj.load_trajectories('../data1')

        >>> # Add more trajectories from another directory
        >>> traj.add_trajectory('../data2')

        >>> # Add trajectories with specific selection
        >>> traj.add_trajectory('../data3', selection="protein")

        >>> # Add with striding and concatenation
        >>> traj.add_trajectory('../data4', concat=True, stride=5)

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
        if self.trajectories is None:
            raise ValueError(
                "No trajectories currently loaded. Use load_trajectories() first."
            )

        # Load new trajectories
        result = TrajectoryLoader.load_trajectories(
            data_input, concat, stride
        )
        new_trajectories = result["trajectories"]
        new_names = result["names"]

        # Apply atom selection to new trajectories if provided
        if selection is not None:
            for i, traj in enumerate(new_trajectories):
                atom_indices = traj.topology.select(selection)
                new_trajectories[i] = traj.atom_slice(atom_indices)

        # Append to existing trajectories and names
        self.trajectories.extend(new_trajectories)
        self.trajectory_names.extend(new_names)

        # Check trajectory consistency
        self._check_trajectory_consistency()

    def _check_trajectory_consistency(self):
        """
        Check if all trajectories have consistent atom and residue counts.

        Issues detailed warnings if inconsistencies are found, providing
        specific information about which trajectories differ and suggestions
        for resolution.
        """
        if not self.trajectories or len(self.trajectories) <= 1:
            return
        atom_inconsistent = False
        residue_inconsistent = False
        # Check residue inconsistency first
        residue_inconsistent = self._find_residue_inconsistency()

        if not residue_inconsistent:
            atom_inconsistent = self._find_atom_inconsistency()

        if atom_inconsistent or residue_inconsistent:
            print("\nSuggestion: Use select_atoms() with a selector that applies "
                  "to all trajectories\nto find common ground for analysis, or use "
                  "remove_trajectory() to exclude\nincompatible trajectories from "
                  "the analysis.\n")

    def _find_residue_inconsistency(self):
        """
        Check for residue count inconsistencies between trajectories.

        All trajectories must have the same number of residues for some features.

        Returns:
        --------
        bool
            True if residue inconsistency found, False if all consistent
        """
        if not self.trajectories or len(self.trajectories) <= 1:
            return False

        ref_n_residues = self.trajectories[0].n_residues
        residue_inconsistent = []

        for i, traj in enumerate(self.trajectories):
            if traj.n_residues != ref_n_residues:
                residue_inconsistent.append(
                    (i, self.trajectory_names[i], traj.n_residues))

        if residue_inconsistent:
            inconsistent_list = "\n".join([
                f"  [{idx}] {name}: {n_residues} residues"
                for idx, name, n_residues in residue_inconsistent
            ])
            print(f"\nWARNING: Inconsistent residue counts detected!\n"
                  f"Reference trajectory '{self.trajectory_names[0]}' has "
                  f"{ref_n_residues} residues.\n"
                  f"Trajectories with different residue counts:\n"
                  f"{inconsistent_list}\n\n"
                  f"Residue-based feature calculations may fail or produce "
                  f"incorrect results.")
            return True

        return False

    def _find_atom_inconsistency(self):
        """
        Check for atom count inconsistencies between trajectories.

        All trajectories must have the same number of atoms for some features.

        Returns:
        --------
        bool
            True if atom inconsistency found, False if all consistent
        """
        if not self.trajectories or len(self.trajectories) <= 1:
            return False

        ref_n_atoms = self.trajectories[0].n_atoms
        atom_inconsistent = []

        for i, traj in enumerate(self.trajectories):
            if traj.n_atoms != ref_n_atoms:
                atom_inconsistent.append(
                    (i, self.trajectory_names[i], traj.n_atoms))

        if atom_inconsistent:
            inconsistent_list = "\n".join([
                f"  [{idx}] {name}: {n_atoms} atoms"
                for idx, name, n_atoms in atom_inconsistent
            ])
            print(f"\nWARNING: Inconsistent atom counts detected!\n"
                  f"Reference trajectory '{self.trajectory_names[0]}' has {ref_n_atoms} atoms.\n"
                  f"Trajectories with different atom counts:\n"
                  f"{inconsistent_list}\n\n"
                  f"Atom-based feature calculations may fail or produce incorrect results.")
            return True

        return False

    def _print_consistency_suggestions(self):
        """Print suggestions for resolving inconsistencies."""

    def remove_trajectory(self, trajs, force=False):
        """
        Remove specified trajectories from the loaded trajectory list.

        Parameters:
        -----------
        trajs : int, str, or list
            Trajectory index (int), name (str), or list of indices/names to remove.
            Can be mixed list of integers and strings.
        force : bool, default=False
            Force removal even when features have been calculated. When True,
            existing features become invalid and should be recalculated.

        Examples:
        ---------
        >>> # Remove single trajectory by index
        >>> traj.remove_trajectory(0)

        >>> # Remove single trajectory by name
        >>> traj.remove_trajectory('system1_prot')

        >>> # Remove multiple trajectories by indices
        >>> traj.remove_trajectory([0, 2, 4])

        >>> # Remove multiple trajectories by names
        >>> traj.remove_trajectory(['system1_prot', 'system2_complex'])

        >>> # Remove mixed indices and names
        >>> traj.remove_trajectory([0, 'system2_prot', 3])

        >>> # Force removal when features exist
        >>> traj.remove_trajectory([1, 2], force=True)

        Raises:
        -------
        ValueError
            If trajectories are not loaded, if trajs contains invalid indices/names,
            or if features exist and force=False
        """
        if self.trajectories is None:
            raise ValueError("No trajectories loaded to remove.")

        self._check_features_before_removal(force)
        indices_to_remove = self._prepare_removal_indices(trajs)
        self._execute_removal(indices_to_remove)
        self._handle_post_removal(force)

    def _check_features_before_removal(self, force):
        """Check if features exist and raise error if force=False."""
        if self.features and not force:
            feature_list = list(self.features.keys())
            raise ValueError(
                f"Cannot remove trajectories: {len(feature_list)} feature(s) "
                f"have been calculated.\nCalculated features: {', '.join(feature_list)}\n"
                f"Removing trajectories would invalidate these features.\n"
                f"Use force=True to proceed, then call reset_features() to clear "
                f"invalid features."
            )

    def _prepare_removal_indices(self, trajs):
        """Convert trajs input to sorted indices for removal."""
        if not isinstance(trajs, list):
            trajs = [trajs]
        indices_to_remove = self._resolve_selection(trajs)
        indices_to_remove.sort(reverse=True)  # Avoid index shifting issues
        return indices_to_remove

    def _execute_removal(self, indices_to_remove):
        """Remove trajectories and names, return count of removed items."""
        removed_trajectories = []
        for idx in indices_to_remove:
            removed_name = self.trajectory_names[idx]
            removed_trajectories.append(f"{removed_name} (was at index {idx})")
            del self.trajectories[idx]
            del self.trajectory_names[idx]

        # Print removal summary
        for removed_info in removed_trajectories:
            print(f"Removed trajectory: {removed_info}")

        print(f"Removed {len(indices_to_remove)} trajectory(ies). "
              f"{len(self.trajectories)} trajectory(ies) remaining.")
        return len(indices_to_remove)

    def _handle_post_removal(self, force):
        """Handle warnings and consistency checks after removal."""
        if self.features and force:
            self._warn_invalid_features()

        if self.trajectories:
            self._check_trajectory_consistency()

    def _warn_invalid_features(self):
        """Warn user about invalid features after forced removal."""
        feature_list = list(self.features.keys())
        print("\nWARNING: Feature data is now INVALID!")
        print(f"The following {len(feature_list)} feature(s) were calculated "
              f"on the original trajectory set:")
        print(f"  {', '.join(feature_list)}")
        print("These features no longer correspond to the current trajectories.")
        print("Call reset_features() to clear invalid features and "
              "recalculate from scratch.\n")

    def reset_features(self):
        """
        Reset all calculated features and clear feature data.

        This method removes all computed features and their associated data,
        requiring features to be recalculated from scratch. Use this method
        after trajectory modifications that invalidate existing features.

        Examples:
        ---------
        >>> # After removing trajectories with force=True
        >>> traj.remove_trajectory([1, 2], force=True)
        >>> traj.reset_features()

        >>> # Manual reset before recalculating features
        >>> traj.reset_features()
        >>> traj.add_feature(feature_type.Distances())

        Notes:
        -----
        - All feature data is permanently deleted
        - Memory-mapped feature files remain on disk but are no longer referenced
        - Features must be recalculated after reset
        """
        if not self.features:
            print("No features to reset.")
            return

        feature_list = list(self.features.keys())
        self.features.clear()

        print(f"Reset {len(feature_list)} feature(s): "
              f"{', '.join(feature_list)}")
        print("All feature data has been cleared. Features must be recalculated.")

    def cut_traj(self, cut=None, stride=1, selection=None):
        """
        Cut and/or stride trajectories.

        Parameters:
        -----------
        cut : int, optional
            Frame number after which to cut the trajectories.
            Frames after this index will be removed.
        stride : int, default=1
            Take every stride-th frame. Use values > 1 to subsample frames.
        selection : list, optional
            List of trajectory indices (int) or names (str) to process.
            If None, all trajectories will be processed.

        Examples:
        ---------
        >>> # Cut all trajectories after frame 1000
        >>> traj.cut_traj(cut=1000)

        >>> # Take every 10th frame from all trajectories
        >>> traj.cut_traj(stride=10)

        >>> # Cut and stride specific trajectories by index
        >>> traj.cut_traj(cut=500, stride=5, selection=[0, 2, 4])

        >>> # Cut and stride specific trajectories by name
        >>> traj.cut_traj(cut=500, stride=5, selection=['system1_prot_traj1', 'system2_prot_traj2'])

        >>> # Combine cut and stride
        >>> traj.cut_traj(cut=2000, stride=2)

        Raises:
        -------
        ValueError
            If trajectories are not loaded or if selection contains invalid indices/names
        """
        if self.trajectories is None:
            raise ValueError("Trajectories must be loaded before cutting.")

        # Determine which trajectories to process
        if selection is None:
            # Process all trajectories
            indices_to_process = list(range(len(self.trajectories)))
        else:
            indices_to_process = self._resolve_selection(selection)

        # Process selected trajectories
        for idx in indices_to_process:
            traj = self.trajectories[idx]

            # Apply cut if specified
            if cut is not None:
                if cut < traj.n_frames:
                    traj = traj[:cut]

            # Apply stride if not 1
            if stride > 1:
                traj = traj[::stride]

            # Update trajectory
            self.trajectories[idx] = traj

    def _resolve_selection(self, selection):
        """
        Resolve selection list to trajectory indices.

        Parameters:
        -----------
        selection : list
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
        indices = []

        for item in selection:
            if isinstance(item, int):
                if 0 <= item < len(self.trajectories):
                    indices.append(item)
                else:
                    raise ValueError(
                        f"Trajectory index {item} out of range "
                        f"(0-{len(self.trajectories)-1})")
            elif isinstance(item, str):
                if item in self.trajectory_names:
                    idx = self.trajectory_names.index(item)
                    indices.append(idx)
                else:
                    raise ValueError(
                        f"Trajectory name '{item}' not found. Available names: "
                        f"{self.trajectory_names}")
            else:
                raise ValueError(
                    f"Selection item must be int (index) or str (name), "
                    f"got {type(item)}")

        return indices

    def select_atoms(self, selection, trajs=None):
        """
        Apply atom selection to trajectories using MDTraj selection syntax.

        Parameters:
        -----------
        selection : str
            MDTraj selection string (e.g., "protein", "backbone", "resid 10 to 50")
            See: https://mdtraj.org/1.9.4/atom_selection.html
        trajs : list, optional
            List of trajectory indices (int) or names (str) to process.
            If None, all trajectories will be processed.

        Examples:
        ---------
        >>> # Select only protein atoms from all trajectories
        >>> traj.select_atoms("protein")

        >>> # Select backbone atoms from specific trajectories
        >>> traj.select_atoms("backbone", trajs=[0, 2])

        >>> # Select specific residues by name
        >>> traj.select_atoms("resname ALA or resname GLY", trajs=['system1_prot'])

        >>> # Select atoms within distance range
        >>> traj.select_atoms("name CA and resid 10 to 50")

        Raises:
        -------
        ValueError
            If trajectories are not loaded or if selection/trajs contain invalid values
        """
        if self.trajectories is None:
            raise ValueError(
                "Trajectories must be loaded before atom selection.")

        # Determine which trajectories to process
        if trajs is None:
            # Process all trajectories
            indices_to_process = list(range(len(self.trajectories)))
        else:
            indices_to_process = self._resolve_selection(trajs)

        # Apply atom selection to each trajectory
        for idx in indices_to_process:
            traj = self.trajectories[idx]

            # Get atom indices matching the selection
            atom_indices = traj.topology.select(selection)

            # Apply atom slice to trajectory
            selected_traj = traj.atom_slice(atom_indices)

            # Update trajectory
            self.trajectories[idx] = selected_traj

    def get_trajectory_names(self):
        """
        Get list of trajectory names.

        Returns:
        --------
        list or None
            List of trajectory names, or None if trajectories not loaded

        Examples:
        ---------
        >>> names = traj.get_trajectory_names()
        >>> print(names)
        ['system1_prot_traj1', 'system1_prot_traj2', 'system2_prot_traj1']
        """
        return self.trajectory_names

    def print_trajectory_info(self):
        """
        Print information about loaded trajectories.

        Examples:
        ---------
        >>> traj.print_trajectory_info()
        Loaded 3 trajectories:
          [0] system1_prot_traj1: 1000 frames
          [1] system1_prot_traj2: 1500 frames
          [2] system2_prot_traj1: 800 frames
        """
        if self.trajectories is None or self.trajectory_names is None:
            print("No trajectories loaded.")
            return

        print(f"Loaded {len(self.trajectories)} trajectories:")
        for i, (traj, name) in enumerate(zip(self.trajectories, self.trajectory_names)):
            print(f"  [{i}] {name}: {traj}")

    def save(self, save_path):
        """
        Save the complete TrajectoryData object to disk.

        Parameters:
        -----------
        save_path : str
            Path where to save the TrajectoryData object. Should have a
            .pkl or .npy extension. The directory will be created if it
            doesn't exist.

        Examples:
        ---------
        >>> # Save after computing features
        >>> traj.add_feature(feature_type.Distances())
        >>> traj.add_feature(feature_type.Contacts(cutoff=4.5))
        >>> traj.save('analysis_results/trajectory_data.pkl')

        >>> # Save with specific path structure
        >>> import os
        >>> save_dir = 'project_results/session_001'
        >>> os.makedirs(save_dir, exist_ok=True)
        >>> traj.save(f'{save_dir}/traj_analysis.pkl')

        Notes:
        -----
        - All computed features and their reduced versions are saved
        - Memory-mapped data files remain separate and are referenced
        - Trajectories are included in the saved object
        - Analysis method bindings are restored upon loading
        - Cache paths and memory mapping settings are preserved
        """
        DataUtils.save_object(self, save_path)

    def load(self, load_path):
        """
        Load a previously saved TrajectoryData object from disk.

        Parameters:
        -----------
        load_path : str
            Path to the saved TrajectoryData file (.pkl or .npy).
            The file must have been created using the save() method.

        Examples:
        ---------
        >>> # Load previously saved analysis
        >>> traj = TrajectoryData()
        >>> traj.load('analysis_results/trajectory_data.pkl')
        >>>
        >>> # Access loaded features immediately
        >>> distances = traj.get_feature(feature_type.Distances())
        >>> contacts = traj.get_feature(feature_type.Contacts())
        >>>
        >>> # Continue analysis where you left off
        >>> mean_distances = distances.analysis.compute_mean()

        >>> # Load and continue with different analysis
        >>> traj_loaded = TrajectoryData()
        >>> traj_loaded.load('project_results/session_001/traj_analysis.pkl')
        >>> contacts = traj_loaded.get_feature(feature_type.Contacts())
        >>> contacts.reduce_data(
        ...     metric=feature_type.Contacts.ReduceMetrics.STABILITY,
        ...     threshold_min=0.5
        ... )

        Raises:
        -------
        FileNotFoundError
            If the specified file doesn't exist
        ValueError
            If the file is corrupted or not a valid TrajectoryData save file

        Notes:
        -----
        - All previously computed features are restored
        - Analysis method bindings are automatically recreated
        - Memory mapping settings and cache paths are preserved
        - If memory-mapped data files are missing, an error will occur
        - Trajectories are fully restored with all metadata
        """
        DataUtils.load_object(self, load_path)

    def add_labels(
        self,
        fragment_definition=None,
        fragment_type=None,
        fragment_molecule_name=None,
        consensus=False,
        aa_short=True,
        try_web_lookup=True,
        **nomenclature_kwargs,
    ):
        """
        Initialize the Nomenclature labeler.

        This class provides a unified interface to create consensus labels for MD trajectories
        using different mdciao nomenclature systems.

        Parameters
        ----------
        fragment_definition : str or dict, default None
            If string, uses that as fragment name for entire topology.
            If dict, maps fragment names to residue ranges: {"cgn_a": (0, 348), "beta2": (400, 684)}
            Only required when consensus=True.
        fragment_type : str or dict, default None
            If string, uses that nomenclature type for all fragments.
            If dict, maps fragment names to nomenclature types: {"cgn_a": "cgn", "beta2": "gpcr"}
            Use mdciao nomenclature types.
            Allowed types: gpcr, cgn, klifs
            Only required when consensus=True.
        fragment_molecule_name : str or dict, default None
            If string, uses that molecule name for all fragments.
            If dict, maps fragment names to molecule names:
            {"cgn_a": "gnas2_bovin", "beta2": "adrb2_human"}
            Use the UniProt entry name (not accession ID) for GPCR/CGN labelers,
            or KLIFS string for KLIFS labelers.
            See https://www.uniprot.org/help/difference_accession_entryname for UniProt naming conventions.
            See https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS for KLIFS naming conventions.
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
        >>> # Amino acid labels only (no nomenclature)
        >>> traj.add_labels(consensus=False, aa_short=True)

        >>> # Simple nomenclature labeling
        >>> traj.add_labels(
        ...     fragment_definition="receptor",
        ...     fragment_type="gpcr",
        ...     fragment_molecule_name="adrb2_human",
        ...     consensus=True
        ... )

        >>> # Complex multi-fragment labeling
        >>> traj.add_labels(
        ...     fragment_definition={"gpcr": (0, 300), "g_protein": (300, 600)},
        ...     fragment_type={"gpcr": "gpcr", "g_protein": "cgn"},
        ...     fragment_molecule_name={"gpcr": "adrb2_human", "g_protein": "gnas_human"},
        ...     consensus=True
        ... )
        """
        if self.trajectories is None:
            raise ValueError(
                "Trajectories must be loaded before adding labels.")
        if self.cache_dir is None:
            write_to_disk = False
        else:
            write_to_disk = True

        # Use first trajectory topology for labeling
        nomenclature = Nomenclature(
            topology=self.trajectories[0].topology,
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

        self.labels = nomenclature.create_labels()
