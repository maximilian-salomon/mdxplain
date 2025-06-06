"""
TrajectoryLoader - MD Trajectory Loading and Processing

Author: Maximilian Salomon
Version: 0.1.0
Created with assistance from Claude-4-Sonnet and Cursor AI.
"""

import mdtraj as md
import numpy as np
import os
import warnings
from tqdm import tqdm
from ..utils.DistanceCalculator import DistanceCalculator
from ..utils.ContactCalculator import ContactCalculator

class TrajectoryLoader:
    def __init__(self, data_input, use_memmap=False, cache_dir=None, concat=False):
        self.data_input = data_input
        self.use_memmap = use_memmap
        self.concat = concat
        
        if use_memmap:
            if cache_dir is None:
                cache_dir = "./cache"
            os.makedirs(cache_dir, exist_ok=True)
            self.distances_path = os.path.join(cache_dir, "distances.dat")
            self.contacts_path = os.path.join(cache_dir, "contacts.dat")
        
        self.trajectories = self.load_trajectories()
        self.distances = None
        self.contacts = None
        self.res_list = None

    def load_trajectories(self):
        if isinstance(self.data_input, list):
            if self.concat and len(self.data_input) > 1:
                print(f"Concatenating {len(self.data_input)} provided trajectories...")
                concatenated = self.data_input[0]
                for traj in tqdm(self.data_input[1:], desc="Concatenating"):
                    concatenated = concatenated.join(traj)
                print(f"Result: 1 concatenated trajectory with {concatenated.n_frames} frames")
                return [concatenated]
            return self.data_input
        
        if isinstance(self.data_input, str) and os.path.exists(self.data_input):
            trajectories, system_summary = self._load_from_directory()
            if self.concat:
                print(f"\nLoaded {len(system_summary)} systems with {len(trajectories)} concatenated trajectories:")
            else:
                print(f"\nLoaded {len(system_summary)} systems with {len(trajectories)} total trajectories:")
            for system, count in system_summary:
                if self.concat and count > 1:
                    print(f"  {system}: 1 concatenated trajectory ({count} files)")
                else:
                    print(f"  {system}: {count} trajectories")
            return trajectories
        
        warnings.warn(f"Invalid data input: {self.data_input}. Expected list of trajectories or valid path.")
        return []
    
    def _load_from_directory(self):
        """Load trajectories from directory structure."""
        trajectories = []
        system_summary = []
        subdirs = [d for d in os.listdir(self.data_input) if os.path.isdir(os.path.join(self.data_input, d))]
        
        # Check if we have subdirectories or direct files
        if subdirs:
            # Nested structure: process subdirectories
            for subdir in tqdm(subdirs, desc="Loading systems"):
                subdir_path = os.path.join(self.data_input, subdir)
                system_trajs = self._load_system_trajectories(subdir_path, subdir)
                
                if system_trajs:
                    trajectories.extend(system_trajs)
                    system_summary.append((subdir, len(system_trajs)))
        else:
            # Direct structure: process files in given directory
            system_trajs = self._load_system_trajectories(self.data_input, os.path.basename(self.data_input))
            if system_trajs:
                trajectories.extend(system_trajs)
                system_summary.append((os.path.basename(self.data_input), len(system_trajs)))
        
        return trajectories, system_summary
    
    def _load_system_trajectories(self, subdir_path, subdir_name):
        """Load all trajectories for a single system."""
        pdb_files = [f for f in os.listdir(subdir_path) if f.endswith('.pdb')]
        xtc_files = [f for f in os.listdir(subdir_path) if f.endswith('.xtc')]
        
        if len(pdb_files) > 1:
            raise ValueError(f"More than one PDB file found in {subdir_path}: {pdb_files}")
        
        if not (pdb_files and xtc_files):
            return []
        
        pdb_path = os.path.join(subdir_path, pdb_files[0])
        system_trajs = []
        
        for xtc in tqdm(xtc_files, desc=f"Loading {subdir_name}", leave=False):
            system_trajs.append(md.load(os.path.join(subdir_path, xtc), top=pdb_path))
        
        # Concatenate trajectories within this system if concat=True
        if self.concat and len(system_trajs) > 1:
            print(f"  Concatenating {len(system_trajs)} trajectories for {subdir_name}...")
            concatenated = system_trajs[0]
            for traj in system_trajs[1:]:
                concatenated = concatenated.join(traj)
            return [concatenated]  # Return as list with single concatenated trajectory
        
        return system_trajs

    def compute_distances(self, ref, batch_size=2500):
        """
        Compute pairwise distances using the DistanceCalculator utility class.
        
        Parameters:
        -----------
        ref : mdtraj.Trajectory
            Reference trajectory for residue information
        batch_size : int, default=2500
            Number of frames to process at once
        """
        self.distances, self.res_list = DistanceCalculator.compute_distances(
            trajectories=self.trajectories,
            ref=ref,
            batch_size=batch_size,
            use_memmap=self.use_memmap,
            distances_path=getattr(self, 'distances_path', None)
        )

    def compute_contacts(self, cutoff=4.5):
        """
        Compute contact maps using the ContactCalculator utility class.
        
        Parameters:
        -----------
        cutoff : float, default=4.5
            Distance cutoff for contacts (in Angstrom)
        """
        if self.distances is None:
            raise ValueError("Distances must be computed first. Call compute_distances() before compute_contacts().")
        
        self.contacts = ContactCalculator.compute_contacts(
            distances=self.distances,
            cutoff=cutoff,
            use_memmap=self.use_memmap,
            contacts_path=getattr(self, 'contacts_path', None)
        )