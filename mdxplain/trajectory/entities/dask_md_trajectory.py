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
DaskMDTrajectory - Memory-efficient MDTraj-compatible trajectory class.

Provides the complete MDTraj.Trajectory interface with Dask/Zarr backend
for efficient processing of large trajectory files.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple
import os
import pickle
import shutil
import numpy as np
import zarr
import dask.array as da
import mdtraj as md
from tqdm import tqdm

from zarr.codecs import BloscCodec

# Default compression for all zarr operations
DEFAULT_COMPRESSOR = BloscCodec(cname='lz4', clevel=1)

from ..helper.dask_trajectory_helper.dask_trajectory_build_helper import DaskMDTrajectoryBuildHelper
from ..helper.dask_trajectory_helper.dask_trajectory_store_helper import DaskMDTrajectoryStoreHelper  
from ..helper.dask_trajectory_helper.dask_trajectory_join_stack_helper import DaskMDTrajectoryJoinStackHelper
from ..helper.dask_trajectory_helper.parallel_operations_helper import ParallelOperationsHelper


class DaskMDTrajectory:
    """
    Memory-efficient trajectory class compatible with MDTraj interface.
    
    Uses Dask arrays and Zarr storage for optimal memory usage while maintaining
    full compatibility with MDTraj operations and workflows.
    """
    
    def __init__(self, trajectory_file: str, topology_file: Optional[str] = None,
                 zarr_cache_path: Optional[str] = None, chunk_size: int = 1000,
                 n_workers: Optional[int] = None):
        """
        Initialize DaskMDTrajectory.
        
        Parameters:
        -----------
        trajectory_file : str
            Path to trajectory file (.xtc, .dcd, etc.)
        topology_file : str, optional
            Path to topology file (.pdb, .gro, etc.)
        zarr_cache_path : str, optional
            Custom path for Zarr cache file
        chunk_size : int, default=1000
            Number of frames per chunk (optimized for performance)
        n_workers : int, optional
            Number of parallel workers (defaults to CPU count)
"""        
        # Delegate complex initialization to builder (sets _cache_dir)
        self._builder = DaskMDTrajectoryBuildHelper()
        self._builder.initialize_instance(
            self, trajectory_file, topology_file, zarr_cache_path, 
            chunk_size, n_workers
        )
        
        # Initialize helper for join/stack operations with cache_dir (already set by builder)
        self._join_stack_helper = DaskMDTrajectoryJoinStackHelper(cache_dir=self._cache_dir)
        
        # Temp file tracking for cleanup
        self._is_temp_store = False  # Flag to track if this is a temporary store
        self._temp_zarr_path = None  # Path to temp zarr file for cleanup
    
    @classmethod
    def from_mdtraj(cls, mdtraj: md.Trajectory, 
                    zarr_cache_path: Optional[str] = None,
                    chunk_size: int = 1000,
                    n_workers: Optional[int] = None) -> DaskMDTrajectory:
        """
        Create DaskMDTrajectory from existing MDTraj trajectory.
        
        Parameters:
        -----------
        mdtraj : md.Trajectory
            MDTraj trajectory object to convert
        zarr_cache_path : str, optional
            Path for Zarr cache. If None, creates temporary cache.
        chunk_size : int, default=1000
            Number of frames per chunk for Dask arrays
        n_workers : int, optional
            Number of parallel workers (defaults to CPU count)
            
        Returns:
        --------
        DaskMDTrajectory
            New DaskMDTrajectory instance with data from MDTraj
            
        Examples:
        ---------
        >>> import mdtraj as md
        >>> traj = md.load('trajectory.xtc', top='topology.pdb')
        >>> dask_traj = DaskMDTrajectory.from_mdtraj(traj)
        >>> print(f"Converted {dask_traj.n_frames} frames")
        
        >>> # With custom cache path
        >>> dask_traj = DaskMDTrajectory.from_mdtraj(
        ...     traj, zarr_cache_path='/tmp/my_cache.zarr'
        ... )
        """
        # Create new instance
        instance = cls.__new__(cls)
        
        # Initialize basic attributes
        instance._is_temp_store = zarr_cache_path is None
        instance._temp_zarr_path = None
        
        # Delegate initialization to builder with mdtraj input
        instance._builder = DaskMDTrajectoryBuildHelper()
        instance._builder.initialize_from_mdtraj(
            instance, mdtraj, zarr_cache_path, chunk_size, n_workers
        )
        
        # Initialize helper for join/stack operations
        instance._join_stack_helper = DaskMDTrajectoryJoinStackHelper(
            cache_dir=instance._cache_dir
        )
        
        return instance
    
    # ============================================================================
    # MDTraj Properties Interface
    # ============================================================================
    
    @property
    def n_frames(self) -> int:
        """
        Number of frames in the trajectory.
        
        Returns:
        --------
        int
            Total number of frames
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> print(dask_traj.n_frames)
        1000
        """
        return self.metadata['n_frames']
    
    @property
    def n_atoms(self) -> int:
        """
        Number of atoms in the trajectory.
        
        Returns:
        --------
        int
            Total number of atoms
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> print(dask_traj.n_atoms)
        5000
        """
        return self.metadata['n_atoms']
    
    @property
    def n_residues(self) -> int:
        """
        Number of residues in the trajectory.
        
        Returns:
        --------
        int
            Total number of residues
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> print(dask_traj.n_residues)
        333
        """
        return self.metadata['n_residues']
    
    @property
    def topology(self) -> md.Topology:
        """
        System topology.
        
        Returns:
        --------
        md.Topology
            MDTraj topology object
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> topology = dask_traj.topology
        >>> print(topology.n_atoms)
        5000
        """
        return self._topology
    
    @property
    def top(self) -> md.Topology:
        """
        System topology (alias for topology property).
        
        Returns:
        --------
        md.Topology
            MDTraj topology object
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> top = dask_traj.top  # Same as dask_traj.topology
        >>> print(top.n_residues)
        333
        """
        return self._topology
    
    @property
    def time(self) -> np.ndarray:
        """
        Simulation time for each frame (lazy loaded).
        
        Returns:
        --------
        np.ndarray
            Array of simulation times with shape (n_frames,)
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> times = dask_traj.time
        >>> print(f"First frame: {times[0]} ps")
        First frame: 0.0 ps
        """
        if self._time_cache is None:
            print("⏱️  Loading time data...")
            self._time_cache = self._dask_time.compute()
        return self._time_cache
    
    @property
    def timestep(self) -> float:
        """
        Time between frames in picoseconds.
        
        Returns:
        --------
        float
            Time step in picoseconds
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> dt = dask_traj.timestep
        >>> print(f"Timestep: {dt} ps")
        Timestep: 1.0 ps
        """
        if self.n_frames > 1:
            return float(self.time[1] - self.time[0])
        return 1.0
    
    @property
    def xyz(self) -> np.ndarray:
        """
        Cartesian coordinates (lazy loaded with memory management).
        
        Returns:
        --------
        np.ndarray
            Coordinate array with shape (n_frames, n_atoms, 3)
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> coords = dask_traj.xyz  # Loads all coordinates into memory
        >>> print(coords.shape)
        (1000, 5000, 3)
        >>> # For large trajectories, use slicing:
        >>> coords_subset = dask_traj[0:100].xyz
        """
        # Check if we can safely load all coordinates
        coords_size_mb = (self.n_frames * self.n_atoms * 3 * 4) / (1024**2)
        
        # Memory check removed - user controls memory via indexing/slicing
        # Large arrays should use indexing: traj[start:end].xyz
        
        if self._xyz_cache is None:
            print(f"📊 Loading {coords_size_mb:.1f} MB coordinate data...")
            self._xyz_cache = self._dask_coords.compute()
        
        return self._xyz_cache
    
    @property
    def unitcell_vectors(self) -> Optional[np.ndarray]:
        """
        Unit cell vectors (lazy loaded).
        
        Returns:
        --------
        Optional[np.ndarray]
            Unit cell vectors array with shape (n_frames, 3, 3) or None if no unitcell
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> vectors = dask_traj.unitcell_vectors
        >>> if vectors is not None:
        ...     print(f"Unit cell shape: {vectors.shape}")
        """
        if not self._has_unitcell:
            return None
        return self._dask_unitcell_vectors.compute()
    
    @property
    def unitcell_lengths(self) -> Optional[np.ndarray]:
        """
        Unit cell lengths (lazy loaded).
        
        Returns:
        --------
        Optional[np.ndarray]
            Unit cell lengths array with shape (n_frames, 3) or None if no unitcell
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> lengths = dask_traj.unitcell_lengths
        >>> if lengths is not None:
        ...     print(f"Average box size: {lengths.mean(axis=0)} nm")
        """
        if not self._has_unitcell:
            return None
        return self._dask_unitcell_lengths.compute()
    
    @property
    def unitcell_angles(self) -> Optional[np.ndarray]:
        """
        Unit cell angles (lazy loaded).
        
        Returns:
        --------
        Optional[np.ndarray]
            Unit cell angles array with shape (n_frames, 3) or None if no unitcell
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> angles = dask_traj.unitcell_angles
        >>> if angles is not None:
        ...     print(f"Box angles: {angles[0]} degrees")
        """
        if not self._has_unitcell:
            return None
        return self._dask_unitcell_angles.compute()
    
    # ============================================================================
    # MDTraj Methods Interface
    # ============================================================================
    
    def atom_slice(
            self, 
            atom_indices: Union[np.ndarray, list]
    ) -> DaskMDTrajectory:
        """
        Create trajectory from subset of atoms.
        
        Parameters:
        -----------
        atom_indices : array_like
            Indices of atoms to keep
        inplace : bool, default=False
            Modify trajectory in place (creates new instance regardless)
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory with selected atoms
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> # Select first 100 atoms
        >>> small_traj = dask_traj.atom_slice(range(100))
        >>> print(f"Original: {dask_traj.n_atoms} atoms, Sliced: {small_traj.n_atoms} atoms")
        >>> # Select specific atom indices
        >>> ca_indices = dask_traj.topology.select('name CA')
        >>> ca_traj = dask_traj.atom_slice(ca_indices)
        """
        # Clean up current temp store before creating new one (chained cleanup)
        if self._is_temp_store:
            self.cleanup()
            
        atom_indices = np.asarray(atom_indices)
        new_zarr_store = self._parallel_ops.atom_slice(atom_indices)
        
        # Create new DaskMDTrajectory from processed store
        return self._create_from_zarr_store(new_zarr_store)
    
    def center_coordinates(self, mass_weighted: bool = False) -> DaskMDTrajectory:
        """
        Center trajectory frames at origin.
        
        Parameters:
        -----------
        mass_weighted : bool, default=False
            Use mass-weighted centering
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory with centered coordinates
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> # Center coordinates (geometric center)
        >>> centered = dask_traj.center_coordinates()
        >>> # Center coordinates using mass weighting
        >>> mass_centered = dask_traj.center_coordinates(mass_weighted=True)
        """
        # Clean up current temp store before creating new one (chained cleanup)
        if self._is_temp_store:
            self.cleanup()
            
        new_zarr_store = self._parallel_ops.center_coordinates(mass_weighted)
        return self._create_from_zarr_store(new_zarr_store)
    
    def superpose(
            self, 
            reference: Optional[Union[DaskMDTrajectory, md.Trajectory]] = None,
            frame: int = 0, 
            atom_indices: Optional[np.ndarray] = None
    ) -> DaskMDTrajectory:
        """
        Align trajectory to reference structure.
        
        Parameters:
        -----------
        reference : DaskMDTrajectory, md.Trajectory, optional
            Reference trajectory (if None, uses self as reference)
        frame : int, default=0
            Frame index from reference trajectory to use for alignment
        atom_indices : np.ndarray, optional
            Atoms to use for alignment
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory aligned to reference
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> # Align to own first frame
        >>> aligned = dask_traj.superpose()
        >>> # Align to own frame 10
        >>> aligned = dask_traj.superpose(frame=10)
        >>> # Align to external trajectory frame 0
        >>> other_traj = DaskMDTrajectory('other.xtc', 'topology.pdb')
        >>> aligned = dask_traj.superpose(reference=other_traj, frame=0)
        >>> # Align using only CA atoms
        >>> ca_indices = dask_traj.topology.select('name CA')
        >>> ca_aligned = dask_traj.superpose(atom_indices=ca_indices)
        
        Raises:
        -------
        ValueError
            If reference frame index is out of range
        """
        # Determine reference trajectory
        if reference is None:
            ref_trajectory = self
        else:
            ref_trajectory = reference
        
        # Extract reference frame as md.Trajectory
        ref_traj = ref_trajectory[frame]
        
        # Clean up current temp store before creating new one (chained cleanup)
        if self._is_temp_store:
            self.cleanup()
            
        # Pass reference trajectory to parallel operations
        new_zarr_store = self._parallel_ops.superpose(ref_traj, atom_indices)
        return self._create_from_zarr_store(new_zarr_store)
    
    def smooth(
            self, 
            width: int, 
            order: Optional[int] = None, 
            atom_indices: Optional[np.ndarray] = None
    ) -> DaskMDTrajectory:
        """
        Apply smoothing filter to trajectory.
        
        Parameters:
        -----------
        width : int
            Smoothing window width
        order : int, optional
            Polynomial order for Savitzky-Golay filter
        atom_indices : np.ndarray, optional
            Atoms to smooth (default: all)
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory with smoothed coordinates
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> # Apply smoothing with window width 5
        >>> smoothed = dask_traj.smooth(width=5)
        >>> # Smooth only backbone atoms
        >>> backbone = dask_traj.topology.select('backbone')
        >>> backbone_smoothed = dask_traj.smooth(width=3, atom_indices=backbone)
        """
        # Clean up current temp store before creating new one (chained cleanup)
        if self._is_temp_store:
            self.cleanup()
            
        new_zarr_store = self._parallel_ops.smooth(width, order, atom_indices)
        return self._create_from_zarr_store(new_zarr_store)
    
    def join(self, other: DaskMDTrajectory, check_topology: bool = True) -> DaskMDTrajectory:
        """
        Combine trajectories along frame axis.
        
        Parameters:
        -----------
        other : DaskMDTrajectory
            Trajectory to join
        check_topology : bool, default=True
            Check topology compatibility
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory with combined frames
        
        Examples:
        ---------
        >>> traj1 = DaskMDTrajectory('part1.xtc', 'topology.pdb')
        >>> traj2 = DaskMDTrajectory('part2.xtc', 'topology.pdb')
        >>> combined = traj1.join(traj2)
        >>> print(f"Combined: {combined.n_frames} frames")
        
        Raises:
        -------
        ValueError
            If trajectories have different number of atoms when check_topology=True
        """
        # Clean up current temp store before creating new one (chained cleanup)
        if self._is_temp_store:
            self.cleanup()
            
        return self._join_stack_helper.join_trajectories(self, other, check_topology)
    
    def stack(self, other: DaskMDTrajectory) -> DaskMDTrajectory:
        """
        Combine trajectories along atom axis.
        
        Parameters:
        -----------
        other : DaskMDTrajectory
            Trajectory to stack
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory with combined atoms
        
        Examples:
        ---------
        >>> protein = DaskMDTrajectory('protein.xtc', 'protein.pdb')
        >>> ligand = DaskMDTrajectory('ligand.xtc', 'ligand.pdb')
        >>> complex_traj = protein.stack(ligand)
        >>> print(f"Complex: {complex_traj.n_atoms} atoms")
        
        Raises:
        -------
        ValueError
            If trajectories have different number of frames
        """
        # Clean up current temp store before creating new one (chained cleanup)
        if self._is_temp_store:
            self.cleanup()
            
        return self._join_stack_helper.stack_trajectories(self, other)
    
    def slice(self, key: Union[int, slice, np.ndarray], return_dask: bool = True) -> Union['DaskMDTrajectory', md.Trajectory]:
        """
        Extract specific frames.
        
        Parameters:
        -----------
        key : int, slice, or array_like
            Frame indices to extract
        return_dask : bool, default=True
            If True, returns a new DaskMDTrajectory with sliced data
            If False, returns an md.Trajectory (original behavior)
            
        Returns:
        --------
        DaskMDTrajectory or md.Trajectory
            DaskMDTrajectory with selected frames (if return_dask=True) or
            MDTraj trajectory with selected frames (if return_dask=False)
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> # Extract first 100 frames as DaskMDTrajectory (default)
        >>> subset = dask_traj.slice(slice(0, 100))
        >>> # Extract specific frames as md.Trajectory
        >>> frames = dask_traj.slice([10, 50, 100, 200], return_dask=False)
        >>> print(f"Extracted {frames.n_frames} frames")
        
        Note:
        -----
        When return_dask=True, uses lazy Dask array slicing to avoid memory issues.
        The slicing operation is completely lazy until data is actually accessed.
        """
        if return_dask:
            return self._create_sliced_dask(key)
        else:
            return self[key]
    
    # ============================================================================
    # Indexing and Slicing Support
    # ============================================================================
    
    def __getitem__(self, key: Union[int, slice, np.ndarray]) -> md.Trajectory:
        """
        Index trajectory frames.
        
        Parameters:
        -----------
        key : int, slice, or array_like
            Frame indices
            
        Returns:
        --------
        md.Trajectory
            Trajectory object with selected frames
        """
        # Normalize single int to slice for consistent processing
        if isinstance(key, int):
            key = slice(key, key + 1)
        # Normalize lists to numpy arrays for consistent processing
        elif isinstance(key, list):
            key = np.asarray(key)
        
        # All key types now processed uniformly
        coords, time = self._compute_basic_arrays(key)
        unitcell_data = self._compute_unitcell_arrays(key)
        return self._create_md_trajectory(coords, time, unitcell_data)
    
    
    def _compute_basic_arrays(self, key: Union[int, slice, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute basic coordinate and time arrays.
        
        Parameters:
        -----------
        key : Union[int, slice, np.ndarray]
            Index specification for array slicing
        
        Returns:
        --------
        tuple
            Tuple containing (coordinates, time) arrays
        """
        coords = self._dask_coords[key].compute()
        time = self._dask_time[key].compute()
        return coords, time
    
    def _compute_unitcell_arrays(self, key: Union[int, slice, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute unitcell arrays if present.
        
        Parameters:
        -----------
        key : Union[int, slice, np.ndarray]
            Index specification for array slicing
        
        Returns:
        --------
        tuple
            Tuple containing (unitcell_vectors, unitcell_lengths, unitcell_angles) or (None, None, None)
        """
        if self._has_unitcell:
            unitcell_vectors = self._dask_unitcell_vectors[key].compute()
            unitcell_lengths = self._dask_unitcell_lengths[key].compute()
            unitcell_angles = self._dask_unitcell_angles[key].compute()
            return unitcell_vectors, unitcell_lengths, unitcell_angles
        return None, None, None
    
    def _create_md_trajectory(self, coords: np.ndarray, time: np.ndarray, unitcell_data: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]) -> md.Trajectory:
        """
        Create md.Trajectory from computed arrays.
        
        Parameters:
        -----------
        coords : np.ndarray
            Coordinate array with shape (n_frames, n_atoms, 3)
        time : np.ndarray
            Time array with shape (n_frames,)
        unitcell_data : tuple
            Tuple containing unitcell vectors, lengths, and angles
        
        Returns:
        --------
        md.Trajectory
            Constructed MDTraj trajectory object
        """
        _, unitcell_lengths, unitcell_angles = unitcell_data
        return md.Trajectory(
            xyz=coords,
            topology=self._topology,
            time=time,
            unitcell_lengths=unitcell_lengths,
            unitcell_angles=unitcell_angles
        )
    
    def __len__(self) -> int:
        """Return number of frames."""
        return self.n_frames
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"<DaskMDTrajectory with {self.n_frames} frames, {self.n_atoms} atoms, "
            f"{self.n_residues} residues, and PBC (openmm format)>"
        )
    
    # ============================================================================
    # Additional Methods
    # ============================================================================
    
    def memory_usage(self) -> dict:
        """
        Get memory usage information.
        
        Returns:
        --------
        dict
            Memory usage statistics with keys: coordinates_size_mb, zarr_cache_size_mb, chunk_size, n_workers
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> usage = dask_traj.memory_usage()
        >>> print(f"Trajectory size: {usage['coordinates_size_mb']:.1f} MB")
        >>> print(f"Cache size: {usage['zarr_cache_size_mb']:.1f} MB")
        """
        coords_size_mb = (self.n_frames * self.n_atoms * 3 * 4) / (1024**2)
        cache_size_mb = self.cache_manager._get_cache_size(self.zarr_cache_path)
        
        return {
            'coordinates_size_mb': coords_size_mb,
            'zarr_cache_size_mb': cache_size_mb,
            'chunk_size': self.chunk_size,
            'n_workers': self.n_workers
        }
    
    def cleanup(self):
        """
        Clean up resources and temporary files.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
            Clears caches and closes zarr store
        
        Examples:
        ---------
        >>> dask_traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> # Do some work...
        >>> dask_traj.cleanup()  # Clean up when done
        """
        # Clear caches
        self._xyz_cache = None
        self._time_cache = None
        
        # Close Zarr store
        if hasattr(self, '_zarr_store'):
            del self._zarr_store
            
        # Remove temporary zarr files if this is a temp store
        if self._is_temp_store and self._temp_zarr_path and os.path.exists(self._temp_zarr_path):
            print(f"Cleaning up temp zarr file: {self._temp_zarr_path}")
            try:
                shutil.rmtree(self._temp_zarr_path)
                self._temp_zarr_path = None
                self._is_temp_store = False
            except OSError as e:
                print(f"Warning: Could not remove temp zarr file {self._temp_zarr_path}: {e}")
                
    def __del__(self):
        """
        Automatic cleanup when object is destroyed.
        
        Only cleans up temporary stores to prevent accidental deletion of permanent caches.
        """
        # Only cleanup temp stores automatically
        if hasattr(self, '_is_temp_store') and self._is_temp_store:
            try:
                self.cleanup()
            except:
                # Suppress all exceptions in destructor to avoid issues during shutdown
                pass
    
    # ============================================================================
    # Private Methods
    # ============================================================================
    
    def _copy_trajectory_chunks(self, target_store: zarr.Group, frame_offset: int = 0):
        """
        Copy trajectory data chunk-wise to target zarr store.
        
        Parameters:
        -----------
        target_store : zarr.Group
            Target zarr store to copy data to
        frame_offset : int, default=0
            Frame offset in target store where to start copying
            
        Returns:
        --------
        None
            Copies coordinate, time, and unitcell data chunk-wise
        """
        # Calculate optimal chunk size for memory efficiency
        chunk_size = min(self.chunk_size, self.n_frames)
        n_chunks = (self.n_frames + chunk_size - 1) // chunk_size
        
        print(f"  📝 Copying {self.n_frames} frames in {n_chunks} chunks (offset: {frame_offset})")
        
        # Copy coordinate and time data chunk-wise
        for i in tqdm(range(n_chunks), desc=f"Copying trajectory chunks", unit="chunks"):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, self.n_frames)
            
            target_start = frame_offset + start_idx
            target_end = frame_offset + end_idx
            
            # Copy coordinates chunk
            coords_chunk = self._dask_coords[start_idx:end_idx].compute()
            target_store['coordinates'][target_start:target_end] = coords_chunk
            
            # Copy time chunk
            time_chunk = self._dask_time[start_idx:end_idx].compute()
            target_store['time'][target_start:target_end] = time_chunk
            
            # Copy unitcell data if present
            if self._has_unitcell:
                target_store['unitcell_vectors'][target_start:target_end] = \
                    self._dask_unitcell_vectors[start_idx:end_idx].compute()
                target_store['unitcell_lengths'][target_start:target_end] = \
                    self._dask_unitcell_lengths[start_idx:end_idx].compute()
                target_store['unitcell_angles'][target_start:target_end] = \
                    self._dask_unitcell_angles[start_idx:end_idx].compute()
    
    def _create_sliced_dask(self, key: Union[int, slice, np.ndarray]) -> 'DaskMDTrajectory':
        """
        Create new DaskMDTrajectory with sliced arrays (lazy operation).
        
        Parameters:
        -----------
        key : int, slice, or array_like
            Frame indices to extract
            
        Returns:
        --------
        DaskMDTrajectory
            New DaskMDTrajectory with only selected frames
            
        Examples:
        ---------
        >>> sliced = dask_traj._create_sliced_dask(slice(0, 100))
        >>> sliced = dask_traj._create_sliced_dask([10, 50, 100])
        """
        # Normalize indices to list
        if isinstance(key, int):
            indices = [key]
        elif isinstance(key, slice):
            indices = list(range(*key.indices(self.n_frames)))
        else:
            indices = list(np.asarray(key))
        
        if len(indices) == 0:
            raise ValueError("Cannot create trajectory with 0 frames")
        
        # Create new instance by copying all attributes
        new_instance = DaskMDTrajectory.__new__(DaskMDTrajectory)
        
        # Copy all basic attributes
        new_instance._zarr_store = self._zarr_store  
        new_instance._topology = self._topology
        new_instance.chunk_size = self.chunk_size
        new_instance.n_workers = self.n_workers
        new_instance._cache_dir = self._cache_dir
        new_instance._builder = self._builder
        
        # Slice the Dask arrays (lazy!)
        new_instance._dask_coords = self._dask_coords[indices]
        new_instance._dask_time = self._dask_time[indices]
        
        # Update metadata
        new_instance.metadata = self.metadata.copy()
        new_instance.metadata['n_frames'] = len(indices)
        
        # Handle unitcell data if present
        new_instance._has_unitcell = self._has_unitcell
        if self._has_unitcell:
            new_instance._dask_unitcell_vectors = self._dask_unitcell_vectors[indices]
            new_instance._dask_unitcell_lengths = self._dask_unitcell_lengths[indices]
            new_instance._dask_unitcell_angles = self._dask_unitcell_angles[indices]
        else:
            new_instance._dask_unitcell_vectors = None
            new_instance._dask_unitcell_lengths = None
            new_instance._dask_unitcell_angles = None
        
        # Initialize cache attributes
        new_instance._xyz_cache = None
        new_instance._time_cache = None
        new_instance._unitcell_lengths_cache = None
        new_instance._unitcell_angles_cache = None
        
        # Share helpers (use existing configuration)
        new_instance._parallel_ops = self._parallel_ops
        new_instance._join_stack_helper = self._join_stack_helper
        
        # Temp tracking
        new_instance._is_temp_store = False
        new_instance._temp_zarr_path = None
        
        print(f"✅ Sliced DaskMDTrajectory created with {len(indices)} frames (lazy)")
        
        return new_instance
    
    def _create_from_zarr_store(self, zarr_store: zarr.Group) -> DaskMDTrajectory:
        """
        Create new DaskMDTrajectory from Zarr store.
        
        Parameters:
        -----------
        zarr_store : zarr.Group
            Zarr group containing trajectory data
        
        Returns:
        --------
        DaskMDTrajectory
            New DaskMDTrajectory instance with data from zarr store
        """
        # Create builder instance directly (no dependency on self._builder)
        builder = DaskMDTrajectoryBuildHelper()
        config = builder.create_from_zarr_store(zarr_store, self.chunk_size, self.n_workers)
        
        # Create new instance
        new_instance = DaskMDTrajectory.__new__(DaskMDTrajectory)
        
        # Set attributes from config
        new_instance._zarr_store = config['zarr_store']
        new_instance.metadata = config['metadata']
        new_instance._topology = config['topology']
        new_instance.chunk_size = config['chunk_size']
        new_instance.n_workers = config['n_workers']
        new_instance._cache_dir = os.path.dirname(config['zarr_path']) if config['zarr_path'] else './cache'
        
        # Create Dask arrays
        new_instance._dask_coords = da.from_zarr(config['zarr_path'], component='coordinates')
        new_instance._dask_time = da.from_zarr(config['zarr_path'], component='time')
        
        # Optional unitcell data
        new_instance._has_unitcell = config['has_unitcell']
        if new_instance._has_unitcell:
            new_instance._dask_unitcell_vectors = da.from_zarr(config['zarr_path'], component='unitcell_vectors')
            new_instance._dask_unitcell_lengths = da.from_zarr(config['zarr_path'], component='unitcell_lengths')
            new_instance._dask_unitcell_angles = da.from_zarr(config['zarr_path'], component='unitcell_angles')
        
        # Initialize parallel operations
        new_instance._parallel_ops = ParallelOperationsHelper(
            zarr_path=config['zarr_path'], 
            topology=new_instance._topology, 
            n_workers=new_instance.n_workers, 
            chunk_size=config['chunk_size'],
            cache_dir=new_instance._cache_dir
        )
        
        # Initialize helper for join/stack operations
        new_instance._join_stack_helper = DaskMDTrajectoryJoinStackHelper(cache_dir=new_instance._cache_dir)
        
        # Cache for frequently accessed data
        new_instance._xyz_cache = None
        new_instance._time_cache = None
        
        # Mark as temp store for automatic cleanup (chained cleanup)
        new_instance._is_temp_store = True
        new_instance._temp_zarr_path = config['zarr_path']
        
        return new_instance
    
    def _create_slice(self, key: Union[slice, np.ndarray]) -> DaskMDTrajectory:
        """
        Create new DaskMDTrajectory from slice.
        
        Parameters:
        -----------
        key : Union[slice, np.ndarray]
            Slice or array indices to extract
        
        Returns:
        --------
        DaskMDTrajectory
            New DaskMDTrajectory instance with sliced data
        """
        # Delegate to store manager with cache_dir
        store_manager = DaskMDTrajectoryStoreHelper(cache_dir=self._cache_dir)
        zarr_store = store_manager.create_slice_store(self, key)
        return self._create_from_zarr_store(zarr_store)

    def save(self, filepath: str) -> None:
        """
        Save DaskMDTrajectory to file.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the trajectory. Directory will be created if needed.
            
        Returns:
        --------
        None
            Saves trajectory to disk using pickle
            
        Notes:
        ------
        - Creates parent directories if they don't exist
        - The zarr cache must remain available at its original location
        - Uses pickle to preserve the complete object state
        
        Examples:
        ---------
        >>> traj = DaskMDTrajectory('trajectory.xtc', 'topology.pdb')
        >>> traj.save('output/my_traj.pkl')
        """
        # Ensure parent directory exists
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> DaskMDTrajectory:
        """
        Load DaskMDTrajectory from file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved trajectory file
            
        Returns:
        --------
        DaskMDTrajectory
            Loaded trajectory object
            
        Raises:
        -------
        FileNotFoundError
            If the saved file does not exist
            
        Notes:
        ------
        The zarr cache must still exist at the original location.
        
        Examples:
        ---------
        >>> traj = DaskMDTrajectory.load('output/my_traj.pkl')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
