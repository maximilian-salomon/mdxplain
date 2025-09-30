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
Parallel operations for MDTraj methods using Dask arrays.

Implements memory-efficient, parallelized versions of common MDTraj operations.
"""

from typing import Optional, Tuple, Any
import multiprocessing
import os
import tempfile
import time

import numpy as np
import dask.array as da
import mdtraj as md
from tqdm import tqdm
import zarr
from zarr.codecs import BloscCodec

# Default compression for all zarr operations
DEFAULT_COMPRESSOR = BloscCodec(cname='lz4', clevel=1)

from .zarr_cache_helper import ZarrCacheHelper


class ParallelOperationsHelper:
    """
    Parallel implementations of MDTraj operations using Dask arrays.
    
    All operations respect memory constraints and process data in chunks.
    """
    
    def __init__(self, zarr_path: str, topology: md.Topology, 
                 n_workers: Optional[int] = None, chunk_size: int = 1000,
                 cache_dir: str = './cache'):
        """
        Initialize parallel operations.
        
        Parameters
        ----------
        zarr_path : str
            Path to Zarr store containing trajectory data
        topology : md.Topology
            MDTraj topology
        n_workers : int, optional
            Number of parallel workers (defaults to CPU count)
        chunk_size : int, default=1000
            Number of frames per chunk for memory management
        cache_dir : str, default='./cache'
            Directory for temporary files during operations

        Returns
        -------
        None
            Initializes parallel operations
        """
        
        self.zarr_path = zarr_path
        self.zarr_store = zarr.open(zarr_path, mode='r')
        self.topology = topology
        self.n_workers = n_workers if n_workers is not None else multiprocessing.cpu_count()
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        
        # Create Dask arrays from Zarr path 
        self.dask_coords = da.from_zarr(zarr_path, component='coordinates')
        self.dask_time = da.from_zarr(zarr_path, component='time')
        
        # Optional unitcell data
        self.has_unitcell = 'unitcell_vectors' in self.zarr_store
        if self.has_unitcell:
            self.dask_unitcell_vectors = da.from_zarr(zarr_path, component='unitcell_vectors')
            self.dask_unitcell_lengths = da.from_zarr(zarr_path, component='unitcell_lengths')
            self.dask_unitcell_angles = da.from_zarr(zarr_path, component='unitcell_angles')
        
        self.n_frames, self.n_atoms = self.dask_coords.shape[:2]
        
    def center_coordinates(self, mass_weighted: bool = False) -> zarr.Group:
        """
        Center coordinates at origin using real MDTraj method chunkwise.
        
        Parameters
        ----------
        mass_weighted : bool, default=False
            Use mass-weighted centering
            
        Returns
        -------
        zarr.Group
            New Zarr store with centered coordinates
            
        Examples
        --------
        >>> parallel_ops = ParallelOperationsHelper('trajectory.zarr', topology, chunk_size=500)
        >>> centered_store = parallel_ops.center_coordinates(mass_weighted=True)
        >>> print(f"Centered trajectory stored at {centered_store.path}")
        """
        print(f"🎯 Centering coordinates (mass_weighted={mass_weighted})...")
        
        # Create temporary file for result
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', dir=self.cache_dir)
        os.close(temp_fd)
        os.remove(temp_path)
        
        # Define operation function for centering
        def center_operation(chunk_traj: md.Trajectory, mass_weighted: bool) -> None:
            chunk_traj.center_coordinates(mass_weighted=mass_weighted)
        
        # Use generic helper method
        return self._process_trajectory_chunked(
            operation_func=center_operation,
            result_path=temp_path,
            result_shape=(self.n_frames, self.n_atoms, 3),
            mass_weighted=mass_weighted
        )
    
    def _process_frame_chunks(self, operation_func: callable, result_store: zarr.Group, **operation_kwargs) -> None:
        """
        Sub-helper: Frame-wise chunking for operations like center_coordinates, superpose, atom_slice.
        
        Parameters
        ----------
        operation_func : callable
            Function to apply to each frame chunk
        result_store : zarr.Group
            Pre-configured zarr store to write results to
        **operation_kwargs
            Additional arguments for operation_func

        Returns
        -------
        None
            Applies operation_func to each frame chunk and writes results to result_store
        """
        chunk_size = self._calculate_chunk_size()
        n_chunks = (self.n_frames + chunk_size - 1) // chunk_size
        
        # Process chunks and write directly to zarr
        for i in tqdm(range(n_chunks), desc="  📊 Processing frame chunks"):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, self.n_frames)
            
            # Load chunk
            chunk_coords = self.dask_coords[start_idx:end_idx].compute()
            chunk_time = self.dask_time[start_idx:end_idx].compute()
            
            # Create MDTraj trajectory
            chunk_traj = md.Trajectory(
                xyz=chunk_coords,
                topology=self.topology,
                time=chunk_time
            )
            
            # Apply operation
            operation_func(chunk_traj, **operation_kwargs)
            
            # Write directly to zarr (no RAM accumulation)
            result_store['coordinates'][start_idx:end_idx] = chunk_traj.xyz.astype(np.float32)
            result_store['time'][start_idx:end_idx] = chunk_time.astype(np.float32)
    
    def _process_atom_chunks(self, operation_func: callable, result_store: zarr.Group, atom_indices: np.ndarray, **operation_kwargs) -> None:
        """
        Sub-helper: Atom-wise chunking for operations like smooth.
        
        Parameters
        ----------
        operation_func : callable
            Function to apply to each atom chunk 
        result_store : zarr.Group
            Pre-configured zarr store to write results to
        atom_indices : np.ndarray
            Indices of atoms to process (operation decides what to do with them)
        **operation_kwargs
            Additional arguments for operation_func

        Returns
        -------
        None
            Applies operation_func to each atom chunk and writes results to result_store
        """
        # Process ALL atoms in chunks (no copying phase)
        atom_chunk_size = self._calculate_atom_chunk_size()
        n_atom_chunks = (self.n_atoms + atom_chunk_size - 1) // atom_chunk_size
        
        # Also fill time array once
        time_data = self.dask_time.compute()
        result_store['time'][:] = time_data.astype(np.float32)
        
        for i in tqdm(range(n_atom_chunks), desc="  📊 Processing atom chunks"):
            chunk_start = i * atom_chunk_size
            chunk_end = min(chunk_start + atom_chunk_size, self.n_atoms)
            current_atom_indices = np.arange(chunk_start, chunk_end)
            
            # Load coordinates for these atoms (all frames)
            chunk_coords = self.dask_coords[:, current_atom_indices, :].compute()
            
            # Create trajectory for this atom chunk
            chunk_topology = self.topology.subset(current_atom_indices)
            chunk_traj = md.Trajectory(xyz=chunk_coords, topology=chunk_topology, time=time_data)
            
            # Apply operation - operation decides what to do based on atom_indices
            operation_func(chunk_traj, target_atom_indices=atom_indices, current_atom_indices=current_atom_indices, **operation_kwargs)
            
            # Write processed coords back to zarr
            result_store['coordinates'][:, current_atom_indices, :] = chunk_traj.xyz.astype(np.float32)
    
    def _create_result_store(self, result_path: str, result_shape: tuple, chunk_size: int) -> zarr.Group:
        """
        Create zarr store with coordinate and time arrays.
        
        Parameters
        ----------
        result_path : str
            Path for result zarr store
        result_shape : tuple
            Shape of result coordinates (n_frames, n_atoms, 3)
        chunk_size : int
            Chunk size for zarr arrays
            
        Returns
        -------
        zarr.Group
            Configured zarr store with empty arrays
        """
        n_frames, n_atoms = result_shape[:2]
        result_store = zarr.open(result_path, mode='w')
        compressor = DEFAULT_COMPRESSOR
        
        # Create coordinate and time arrays
        result_store.create_array(
            'coordinates',
            shape=(n_frames, n_atoms, 3),
            chunks=(chunk_size, n_atoms, 3),
            dtype=np.float32,
            compressors=compressor
        )
        result_store.create_array(
            'time',
            shape=(n_frames,),
            chunks=(chunk_size,),
            dtype=np.float32,
            compressors=compressor
        )
        
        return result_store
    
    def _process_trajectory_chunked(self, operation_func: callable, result_path: str, result_shape: Tuple[int, int, int], 
                                   chunking_mode: str = 'frame', result_topology: Optional[md.Topology] = None,
                                   **operation_kwargs: Any) -> zarr.Group:
        """
        Flexible helper for memory-efficient chunk processing directly to Zarr.
        
        Parameters
        ----------
        operation_func : callable
            Function to apply to each chunk
        result_path : str
            Path for result zarr store
        result_shape : tuple
            Shape of result coordinates (n_frames, n_atoms, 3)
        chunking_mode : str, default='frame'
            Chunking strategy: 'frame', 'atom', or 'reshape'
        result_topology : md.Topology, optional
            New topology for reshape operations (e.g., atom_slice)
        **operation_kwargs
            Additional arguments for operation_func
            
        Returns
        -------
        zarr.Group
            Result zarr store with processed coordinates
        """
        chunk_size = self._calculate_chunk_size()
        n_frames, n_atoms = result_shape[:2]
        
        # Create result zarr store with arrays
        result_store = self._create_result_store(result_path, result_shape, chunk_size)
        
        # Delegate to appropriate sub-helper based on chunking mode
        if chunking_mode == 'frame' or chunking_mode == 'reshape':
            # Both use frame-wise chunking (reshape changes shape via result_shape/result_topology)
            self._process_frame_chunks(operation_func, result_store, **operation_kwargs)
        elif chunking_mode == 'atom':
            atom_indices = operation_kwargs.get('atom_indices', np.arange(self.n_atoms))
            # Remove atom_indices from kwargs to avoid duplicate parameter
            filtered_kwargs = {k: v for k, v in operation_kwargs.items() if k != 'atom_indices'}
            self._process_atom_chunks(operation_func, result_store, atom_indices, **filtered_kwargs)
        else:
            raise ValueError(f"Unknown chunking_mode: {chunking_mode}")
        
        # Add metadata and topology (use result_topology if provided)
        topology_to_use = result_topology if result_topology is not None else self.topology
        self._store_metadata_and_topology(result_store, n_frames, n_atoms, topology_to_use, DEFAULT_COMPRESSOR)
        
        return result_store
    
    def superpose(self, reference_traj: md.Trajectory,
                 atom_indices: Optional[np.ndarray] = None) -> zarr.Group:
        """
        Superpose trajectory to reference trajectory using real MDTraj method chunkwise.
        
        Parameters
        ----------
        reference_traj : md.Trajectory
            Reference trajectory (single frame) to align to
        atom_indices : np.ndarray, optional
            Atoms to use for alignment
            
        Returns
        -------
        zarr.Group
            New Zarr store with superposed coordinates
            
        Raises
        ------
        ValueError
            If reference trajectory has wrong number of atoms or frames
        OSError
            If cache directory is not writable
            
        Examples
        --------
        >>> parallel_ops = ParallelOperationsHelper('trajectory.zarr', topology)
        >>> # Create reference trajectory (single frame)
        >>> ref_traj = md.load_frame('reference.pdb', 0)
        >>> aligned_store = parallel_ops.superpose(ref_traj)
        >>> # Superpose using only backbone atoms
        >>> backbone_indices = topology.select('backbone')
        >>> aligned_store = parallel_ops.superpose(ref_traj, backbone_indices)
        """
        print(f"🎯 Superposing to reference trajectory...")
        
        # Create temporary file for result
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', dir=self.cache_dir)
        os.close(temp_fd)
        os.remove(temp_path)
        
        # Define operation function for superpose
        def superpose_operation(chunk_traj: md.Trajectory, reference_traj: md.Trajectory, atom_indices: Optional[np.ndarray]) -> None:
            chunk_traj.superpose(reference_traj, frame=0, atom_indices=atom_indices)
        
        # Use generic helper method
        return self._process_trajectory_chunked(
            operation_func=superpose_operation,
            result_path=temp_path,
            result_shape=(self.n_frames, self.n_atoms, 3),
            reference_traj=reference_traj,
            atom_indices=atom_indices
        )
    
    def smooth(self, width: int, order: Optional[int] = None, 
              atom_indices: Optional[np.ndarray] = None) -> zarr.Group:
        """
        Apply smoothing filter using real MDTraj method with atom-wise chunking.
        
        Processes atoms in chunks to reduce memory usage while applying smooth
        across all frames for each atom (as per MDTraj's algorithm).
        
        Parameters
        ----------
        width : int
            Smoothing window width
        order : int, optional
            Polynomial order for Savitzky-Golay filter
        atom_indices : np.ndarray, optional
            Atoms to smooth (default: all)
            
        Returns
        -------
        zarr.Group
            New Zarr store with smoothed coordinates
            
        Raises
        ------
        ValueError
            If width is invalid for smoothing algorithm
        OSError
            If cache directory is not writable
            
        Examples
        --------
        >>> parallel_ops = ParallelOperationsHelper('trajectory.zarr', topology)
        >>> # Apply smoothing with width 5 to all atoms
        >>> smoothed_store = parallel_ops.smooth(width=5)
        >>> # Apply smoothing only to protein atoms
        >>> protein_indices = topology.select('protein')
        >>> smoothed_store = parallel_ops.smooth(5, atom_indices=protein_indices)
        """
        print(f"🎯 Smoothing with width={width} (atom-wise chunking)...")
        
        # Handle default parameters
        if order is None:
            order = 3
        if atom_indices is None:
            atom_indices = np.arange(self.n_atoms)
        
        # Create temporary file for result
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', dir=self.cache_dir)
        os.close(temp_fd)
        os.remove(temp_path)
        
        # Define operation function for smoothing
        def smooth_operation(chunk_traj: md.Trajectory, target_atom_indices: np.ndarray, current_atom_indices: np.ndarray, width: int, order: int) -> None:
            # Find intersection of target atoms with current chunk atoms
            # Convert global atom indices to local indices within current chunk
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(current_atom_indices)}
            local_target_indices = []
            
            for target_idx in target_atom_indices:
                if target_idx in global_to_local:
                    local_target_indices.append(global_to_local[target_idx])
            
            if local_target_indices:  # Only smooth if there are target atoms in this chunk
                chunk_traj.smooth(width=width, order=order, atom_indices=list(local_target_indices), inplace=True)
        
        # Use generic helper with atom chunking mode
        return self._process_trajectory_chunked(
            operation_func=smooth_operation,
            result_path=temp_path,
            result_shape=(self.n_frames, self.n_atoms, 3),
            chunking_mode='atom',
            atom_indices=atom_indices,
            width=width,
            order=order
        )
    
    def atom_slice(self, atom_indices: np.ndarray) -> zarr.Group:
        """
        Create atom slice using real MDTraj method chunkwise.
        
        Parameters
        ----------
        atom_indices : np.ndarray
            Indices of atoms to keep
            
        Returns
        -------
        zarr.Group
            New Zarr store with selected atoms
            
        Raises
        ------
        IndexError
            If atom_indices contains invalid atom indices
        OSError
            If cache directory is not writable
            
        Examples
        --------
        >>> parallel_ops = ParallelOperationsHelper('trajectory.zarr', topology)
        >>> # Select only first 100 atoms
        >>> atom_indices = np.arange(100)
        >>> sliced_store = parallel_ops.atom_slice(atom_indices)
        >>> # Select only CA atoms
        >>> ca_indices = topology.select('name CA')
        >>> ca_store = parallel_ops.atom_slice(ca_indices)
        """
        print(f"🎯 Creating atom slice ({len(atom_indices)} atoms)...")
        
        # Create temporary file for result
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', dir=self.cache_dir)
        os.close(temp_fd)
        os.remove(temp_path)
        
        # Create new topology using MDTraj
        new_topology = self.topology.subset(atom_indices)
        
        # Define operation function for atom slicing
        def atom_slice_operation(chunk_traj: md.Trajectory, atom_indices: np.ndarray) -> None:
            # Apply slicing and modify trajectory in-place
            sliced_traj = chunk_traj.atom_slice(atom_indices)
            # IMPORTANT: Set topology first, then coordinates (MDTraj validation!)
            chunk_traj._topology = sliced_traj.topology
            chunk_traj._xyz = sliced_traj.xyz  # Use _xyz to bypass validation
        
        # Use generic helper method with reshape mode and new topology
        return self._process_trajectory_chunked(
            operation_func=atom_slice_operation,
            result_path=temp_path,
            result_shape=(self.n_frames, len(atom_indices), 3),  # Note: n_atoms changes!
            chunking_mode='reshape',
            result_topology=new_topology,  # This will be used for metadata
            atom_indices=atom_indices
        )
    
    def _create_coordinate_array(self, store: zarr.Group, coords: np.ndarray, 
                               n_atoms: int, compressor: BloscCodec) -> None:
        """
        Create coordinate array in zarr store.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to create array in
        coords : np.ndarray
            Coordinate data to store
        n_atoms : int
            Number of atoms for chunking
        compressor : object
            Compression codec for the array
            
        Returns
        -------
        None
            Creates coordinate array in-place in store
        """
        store.create_array(
            'coordinates',
            data=coords.astype(np.float32),
            chunks=(self.chunk_size, n_atoms, 3),
            compressors=compressor
        )
    
    def _create_time_array(self, store: zarr.Group, n_frames: int, compressor: BloscCodec) -> None:
        """
        Create time array in zarr store.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to create array in
        n_frames : int
            Number of frames for time array
        compressor : object
            Compression codec for the array
            
        Returns
        -------
        None
            Creates time array in-place in store
        """
        # Always use real time data - subset if needed
        original_time_data = self.dask_time.compute()
        
        if n_frames == self.n_frames:
            time_data = original_time_data
        else:
            # Use subset of original time data instead of artificial times
            time_data = original_time_data[:n_frames]
            
        store.create_array(
            'time',
            data=time_data,
            chunks=(self.chunk_size,),
            compressors=compressor
        )
    
    
    def _create_unitcell_arrays(self, store: zarr.Group, compressor: BloscCodec) -> None:
        """
        Create all unitcell arrays in zarr store.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to create arrays in
        compressor : object
            Compression codec for the arrays
            
        Returns
        -------
        None
            Creates unitcell_vectors, unitcell_lengths, unitcell_angles arrays
        """
        unitcell_data = {
            'unitcell_vectors': (self.dask_unitcell_vectors.compute(), (3, 3)),
            'unitcell_lengths': (self.dask_unitcell_lengths.compute(), (3,)),
            'unitcell_angles': (self.dask_unitcell_angles.compute(), (3,))
        }
        
        for name, (data, chunk_shape) in unitcell_data.items():
            store.create_array(
                name,
                data=data,
                chunks=(self.chunk_size,) + chunk_shape,
                compressors=compressor
            )
    
    def _store_metadata_and_topology(self, store: zarr.Group, n_frames: int, n_atoms: int, 
                                    topology: md.Topology, compressor: BloscCodec) -> None:
        """
        Store metadata and topology in zarr store.
        
        Parameters
        ----------
        store : zarr.Group
            Target zarr store to store metadata and topology in
        n_frames : int
            Number of frames in processed trajectory
        n_atoms : int
            Number of atoms in processed trajectory
        topology : md.Topology
            MDTraj topology object to store
        compressor : object
            Compression codec for topology storage
            
        Returns
        -------
        None
            Stores metadata attributes and topology in store
        """
        metadata = {
            'n_frames': n_frames,
            'n_atoms': n_atoms,
            'n_residues': topology.n_residues,
            'chunk_size': self.chunk_size,
            'created_time': time.time(),
            'has_unitcell': self.has_unitcell
        }
        
        store.attrs['metadata'] = metadata
        
        ZarrCacheHelper.store_topology(store, topology, compressor)
    
    def _calculate_chunk_size(self) -> int:
        """
        Calculate optimal frame chunk size.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            User-defined chunk size with reasonable bounds
        """
        # Use the configured chunk_size with bounds: at least 1 frame, at most all frames
        return max(1, min(self.n_frames, self.chunk_size))
    
    def _calculate_atom_chunk_size(self) -> int:
        """
        Calculate optimal atom chunk size based on frame chunk size memory footprint.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            Optimal atom chunk size between 10 and 200 atoms
        """
        # Memory für chunk_size frames für alle Atome
        memory_for_frame_chunk = self.chunk_size * self.n_atoms * 3 * 4  # bytes
        
        # Für smooth: Nutze dieselbe Memory für weniger Atome aber alle Frames  
        memory_per_atom_all_frames = self.n_frames * 3 * 4  # bytes pro Atom
        
        # Wie viele Atome passen in dasselbe Memory?
        atom_chunk_size = memory_for_frame_chunk // memory_per_atom_all_frames
        
        # Use reasonable bounds: at least 1 atom, at most all atoms
        return max(1, min(self.n_atoms, atom_chunk_size))
    
    def _create_new_zarr_store(self, new_coords: np.ndarray, 
                              new_topology: Optional[md.Topology] = None) -> zarr.Group:
        """
        Create new Zarr store with processed coordinates.
        
        Parameters
        ----------
        new_coords : np.ndarray
            Processed coordinate data to store
        new_topology : md.Topology, optional
            New topology if different from original (e.g., for atom slicing)
            
        Returns
        -------
        zarr.Group
            New temporary zarr store with processed data
            
        Examples
        --------
        >>> coords = np.random.rand(100, 50, 3)
        >>> store = parallel_ops._create_new_zarr_store(coords)
        >>> print(f"Created store with {coords.shape[0]} frames")
        """
        
        # Use temporary zarr store with secure creation
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', dir=self.cache_dir)
        os.close(temp_fd)  # Close file descriptor, but keep file
        os.remove(temp_path)  # Remove file so zarr can create directory
        new_store = zarr.open(temp_path, mode='w')
        
        n_frames_new, n_atoms_new = new_coords.shape[:2]
        
        compressor = DEFAULT_COMPRESSOR
        
        # Create core arrays
        self._create_coordinate_array(new_store, new_coords, n_atoms_new, compressor)
        self._create_time_array(new_store, n_frames_new, compressor)
        
        # Create unitcell arrays if needed
        if self.has_unitcell:
            self._create_unitcell_arrays(new_store, compressor)
        
        # Store metadata and topology
        topology_to_store = new_topology if new_topology else self.topology
        self._store_metadata_and_topology(new_store, n_frames_new, n_atoms_new, topology_to_store, compressor)
        
        return new_store
