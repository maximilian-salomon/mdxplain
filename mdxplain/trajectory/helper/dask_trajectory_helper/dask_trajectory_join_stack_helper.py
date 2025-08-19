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
DaskMDTrajectory Join/Stack Helper - Extracted trajectory combination operations.

Handles memory-efficient joining and stacking of DaskMDTrajectory instances
using secure temporary directories and chunk-wise data processing.
"""

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .zarr_cache_helper import ZarrCacheHelper

if TYPE_CHECKING:
    from ..entities.dask_md_trajectory import DaskMDTrajectory

# Default compression for all zarr operations
DEFAULT_COMPRESSOR = BloscCodec(cname='lz4', clevel=1)


class DaskMDTrajectoryJoinStackHelper:
    """
    Helper class for memory-efficient trajectory joining and stacking operations.
    
    Extracts the complex join/stack logic from DaskMDTrajectory to reduce
    file size and improve maintainability.
    """
    
    def __init__(self, cache_dir: str = './cache'):
        """
        Initialize DaskMDTrajectoryJoinStackHelper.
        
        Parameters:
        -----------
        cache_dir : str, default='./cache'
            Directory for temporary files during join/stack operations
        
        Returns:
        --------
        None
            Initializes helper instance
        """
        self.cache_dir = cache_dir
    
    def join_trajectories(self, traj1: 'DaskMDTrajectory', traj2: 'DaskMDTrajectory', 
                         check_topology: bool = True) -> 'DaskMDTrajectory':
        """
        Combine trajectories along frame axis using memory-efficient approach.
        
        Parameters:
        -----------
        traj1 : DaskMDTrajectory
            First trajectory (self)
        traj2 : DaskMDTrajectory
            Second trajectory to join
        check_topology : bool, default=True
            Check topology compatibility
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory with combined frames
            
        Raises:
        ------
        ValueError
            If trajectories have different number of atoms when check_topology=True
        OSError
            If temporary cache directory is not writable
            
        Examples:
        ---------
        >>> helper = DaskMDTrajectoryJoinStackHelper(cache_dir='./cache')
        >>> traj1 = DaskMDTrajectory('part1.xtc', 'topology.pdb')
        >>> traj2 = DaskMDTrajectory('part2.xtc', 'topology.pdb') 
        >>> combined = helper.join_trajectories(traj1, traj2)
        >>> print(f"Combined trajectory: {combined.n_frames} frames")
        """
        if check_topology and traj1.n_atoms != traj2.n_atoms:
            raise ValueError("Trajectories have different number of atoms")
        
        print(f"🔗 Joining trajectories: {traj1.n_frames} + {traj2.n_frames} = {traj1.n_frames + traj2.n_frames} frames")
        
        # Create secure temporary file for zarr store in cache directory
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', prefix='dask_join_', dir=self.cache_dir)
        os.close(temp_fd)  # Close file descriptor, we only need the path
        os.unlink(temp_path)  # Remove the file, zarr will create directory
        
        temp_store = zarr.open(temp_path, mode='w')
        
        # Calculate final dimensions
        final_frames = traj1.n_frames + traj2.n_frames
        
        # Create arrays with final size - no data loading yet
        self._create_combined_store_arrays(temp_store, final_frames, traj1.n_atoms, traj1.chunk_size)
        
        # Create unitcell arrays if needed
        if traj1._has_unitcell:
            self._create_unitcell_arrays(temp_store, final_frames, traj1.chunk_size)
        
        # Copy first trajectory chunk-wise (frame offset 0)
        traj1._copy_trajectory_chunks(temp_store, frame_offset=0)
        
        # Copy second trajectory chunk-wise (frame offset = first traj frames)
        traj2._copy_trajectory_chunks(temp_store, frame_offset=traj1.n_frames)
        
        # Copy metadata and topology
        new_metadata = traj1.metadata.copy()
        new_metadata['n_frames'] = final_frames
        temp_store.attrs['metadata'] = new_metadata
        
        # Store topology
        ZarrCacheHelper.store_topology(temp_store, traj1._topology, DEFAULT_COMPRESSOR)
        
        # Create new instance from store - temp_path will be used by new instance
        result = traj1._create_from_zarr_store(temp_store)
        
        return result
    
    def stack_trajectories(self, traj1: 'DaskMDTrajectory', traj2: 'DaskMDTrajectory') -> 'DaskMDTrajectory':
        """
        Combine trajectories along atom axis using memory-efficient approach.
        
        Parameters:
        -----------
        traj1 : DaskMDTrajectory
            First trajectory (self)
        traj2 : DaskMDTrajectory
            Second trajectory to stack
            
        Returns:
        --------
        DaskMDTrajectory
            New trajectory with combined atoms
            
        Raises:
        ------
        ValueError
            If trajectories have different number of frames
        OSError
            If temporary cache directory is not writable
            
        Examples:
        ---------
        >>> helper = DaskMDTrajectoryJoinStackHelper(cache_dir='./cache')
        >>> protein = DaskMDTrajectory('protein.xtc', 'protein.pdb')
        >>> ligand = DaskMDTrajectory('ligand.xtc', 'ligand.pdb')
        >>> complex_traj = helper.stack_trajectories(protein, ligand)
        >>> print(f"Complex: {complex_traj.n_atoms} atoms")
        """
        if traj1.n_frames != traj2.n_frames:
            raise ValueError("Trajectories have different number of frames")
        
        print(f"📚 Stacking trajectories: {traj1.n_atoms} + {traj2.n_atoms} = {traj1.n_atoms + traj2.n_atoms} atoms")
        
        # Create secure temporary file for zarr store in cache directory
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zarr', prefix='dask_stack_', dir=self.cache_dir)
        os.close(temp_fd)  # Close file descriptor, we only need the path
        os.unlink(temp_path)  # Remove the file, zarr will create directory
        
        temp_store = zarr.open(temp_path, mode='w')
        
        # Calculate final dimensions
        final_atoms = traj1.n_atoms + traj2.n_atoms
        
        # Create arrays with final size - no data loading yet
        self._create_combined_store_arrays(temp_store, traj1.n_frames, final_atoms, traj1.chunk_size)
        
        # Copy atom data chunk-wise (frame-by-frame for atom axis concatenation)
        self._stack_trajectories_chunked(temp_store, traj1, traj2)
        
        # Combine topologies (stack = combine atoms, join = combine frames)
        combined_topology = traj1._topology.stack(traj2._topology)
        
        # Copy metadata
        new_metadata = traj1.metadata.copy()
        new_metadata['n_atoms'] = final_atoms
        new_metadata['n_residues'] = combined_topology.n_residues
        temp_store.attrs['metadata'] = new_metadata
        
        # Store topology  
        ZarrCacheHelper.store_topology(temp_store, combined_topology, DEFAULT_COMPRESSOR)
        
        # Create new instance from store - temp_path will be used by new instance
        result = traj1._create_from_zarr_store(temp_store)
        
        return result
    
    def _create_combined_store_arrays(self, store: zarr.Group, n_frames: int, 
                                    n_atoms: int, chunk_size: int):
        """
        Create coordinate and time arrays in zarr store.
        
        Parameters:
        -----------
        store : zarr.Group
            Target zarr store
        n_frames : int
            Number of frames
        n_atoms : int
            Number of atoms  
        chunk_size : int
            Chunk size for arrays
            
        Returns:
        --------
        None
            Creates coordinate and time arrays in store
        """
        store.create_array(
            'coordinates',
            shape=(n_frames, n_atoms, 3),
            chunks=(chunk_size, n_atoms, 3),
            dtype=np.float32,
            compressors=DEFAULT_COMPRESSOR
        )
        store.create_array(
            'time',
            shape=(n_frames,),
            chunks=(chunk_size,),
            dtype=np.float32,
            compressors=DEFAULT_COMPRESSOR
        )
    
    def _create_unitcell_arrays(self, store: zarr.Group, n_frames: int, chunk_size: int):
        """
        Create unitcell arrays in zarr store.
        
        Parameters:
        -----------
        store : zarr.Group
            Target zarr store
        n_frames : int
            Number of frames
        chunk_size : int
            Chunk size for arrays
            
        Returns:
        --------
        None
            Creates unitcell_vectors, unitcell_lengths, unitcell_angles arrays
        """
        for array_name, shape_suffix in [('unitcell_vectors', (3, 3)), 
                                         ('unitcell_lengths', (3,)), 
                                         ('unitcell_angles', (3,))]:
            store.create_array(
                array_name,
                shape=(n_frames,) + shape_suffix,
                chunks=(chunk_size,) + shape_suffix,
                dtype=np.float32,
                compressors=DEFAULT_COMPRESSOR
            )
    
    def _stack_trajectories_chunked(self, store: zarr.Group, traj1: 'DaskMDTrajectory', 
                                  traj2: 'DaskMDTrajectory'):
        """
        Stack trajectories chunk-wise along atom axis.
        
        Parameters:
        -----------
        store : zarr.Group
            Target zarr store
        traj1 : DaskMDTrajectory
            First trajectory
        traj2 : DaskMDTrajectory
            Second trajectory
            
        Returns:
        --------
        None
            Copies stacked data chunk-wise to store
        """
        # We need half chunk size, cause we have 2 chunks in RAM (traj1 and traj2)
        chunk_size = min(np.floor(traj1.chunk_size / 2), traj1.n_frames)
        n_chunks = (traj1.n_frames + chunk_size - 1) // chunk_size
        
        print(f"  📝 Stacking {traj1.n_frames} frames in {n_chunks} chunks")
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, traj1.n_frames)
            
            # Get coordinate chunks from both trajectories
            coords_chunk1 = traj1._dask_coords[start_idx:end_idx].compute()
            coords_chunk2 = traj2._dask_coords[start_idx:end_idx].compute()
            
            # Concatenate along atom axis (axis=1) and store
            stacked_chunk = np.concatenate([coords_chunk1, coords_chunk2], axis=1)
            store['coordinates'][start_idx:end_idx] = stacked_chunk
            
            # Copy time from first trajectory (should be identical)
            if i == 0 or start_idx == 0:  # Only copy time once or per chunk
                time_chunk = traj1._dask_time[start_idx:end_idx].compute()
                store['time'][start_idx:end_idx] = time_chunk
