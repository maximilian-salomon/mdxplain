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

"""Factory for creating deterministic mock trajectories for testing."""

from __future__ import annotations
import numpy as np
import mdtraj as md
from typing import List, Optional


class MockAtom:
    """Mock atom with element and residue information."""
    
    def __init__(self, index: int, element_symbol: str, residue_index: int):
        """Initialize mock atom."""
        self.index = index
        self.element = MockElement(element_symbol)
        self.residue = MockResidue(residue_index)
        self.name = f"{element_symbol}{index}"


class MockElement:
    """Mock element with symbol."""
    
    def __init__(self, symbol: str):
        """Initialize mock element."""
        self.symbol = symbol


class MockResidue:
    """Mock residue with index and name."""
    
    def __init__(self, index: int, name: str = "ALA", chain=None):
        """Initialize mock residue."""
        self.index = index
        self.name = name
        self.resSeq = index + 1  # Residue sequence number (1-based)
        self.segment_id = ""
        self.chain = chain or MockChain(0)  # Default chain
        self.atoms = []  # List of atoms in this residue (filled by topology)
        
    def __str__(self):
        return f"{self.name}-{self.resSeq}"


class MockChain:
    """Mock chain for residue."""
    
    def __init__(self, index: int, residues=None):
        self.index = index
        self.chain_id = chr(65 + index)  # A, B, C, etc.
        self._residues = residues or []
    
    @property
    def residues(self):
        """Get residues in this chain."""
        return self._residues


class MockTopology:
    """Mock topology with deterministic atom and residue structure."""
    
    def __init__(self, n_atoms: int = 50, atoms_per_residue: int = 5):
        """
        Initialize mock topology.
        
        Parameters:
        -----------
        n_atoms : int, default=50
            Total number of atoms
        atoms_per_residue : int, default=5
            Number of atoms per residue
        """
        self.n_atoms = n_atoms
        self.n_residues = max(1, n_atoms // atoms_per_residue)
        
        # Create residues first
        self._residues = []
        for i in range(self.n_residues):
            residue = MockResidue(i)
            self._residues.append(residue)
        
        # Create atoms
        self._atoms = []
        for i in range(n_atoms):
            residue_index = i // atoms_per_residue
            if residue_index >= self.n_residues:
                residue_index = self.n_residues - 1
                
            # Cycle through common elements
            elements = ["C", "N", "O", "H", "S"]
            element = elements[i % len(elements)]
            
            atom = MockAtom(i, element, residue_index)
            atom.residue = self._residues[residue_index]  # Link to actual residue
            self._residues[residue_index].atoms.append(atom)  # Add atom to residue's atoms list
            self._atoms.append(atom)
    
    @property
    def atoms(self) -> List[MockAtom]:
        """Get list of atoms."""
        return self._atoms
    
    @property
    def residues(self) -> List[MockResidue]:
        """Get list of residues (MDTraj compatibility)."""
        return self._residues
    
    @property 
    def chains(self) -> List:
        """Get list of chains (MDTraj compatibility for torsions)."""
        return [MockChain(0, self._residues)]
    
    def atom(self, index: int) -> MockAtom:
        """Get atom by index."""
        return self._atoms[index]
    
    def select(self, selection_string: str) -> np.ndarray:
        """Select atoms using MDTraj-style selection strings."""
        if selection_string == 'all':
            return np.arange(self.n_atoms)
        elif selection_string == 'name CA':
            # Return indices of CA atoms (every 5th atom in our mock)
            ca_indices = []
            for i, atom in enumerate(self._atoms):
                if atom.name == 'CA' or atom.element.symbol == 'C':
                    ca_indices.append(i)
            return np.array(ca_indices)
        elif selection_string.startswith('index '):
            # Parse index selection like "index 0 1 2"
            indices_str = selection_string.replace('index ', '')
            indices = [int(x) for x in indices_str.split()]
            return np.array(indices)
        elif selection_string == 'backbone':
            # Return first few atoms as "backbone"
            return np.arange(min(4, self.n_atoms))
        else:
            # Default: return all atoms
            return np.arange(self.n_atoms)


class MockProteinTopology(MockTopology):
    """Mock protein topology with CA atoms."""
    
    def __init__(self, n_residues: int = 20):
        """
        Initialize protein-like topology with one CA per residue.
        
        Parameters:
        -----------
        n_residues : int, default=20
            Number of residues (each with one CA atom)
        """
        self.n_residues = n_residues
        self.n_atoms = n_residues  # One CA per residue
        
        # Create CA atoms
        self._atoms = []
        for i in range(n_residues):
            atom = MockAtom(i, "C", i)  # CA atoms
            atom.name = "CA"
            self._atoms.append(atom)


class MockTrajectory:
    """Mock trajectory with deterministic coordinates and MDTraj compatibility."""
    
    def __init__(self, xyz: np.ndarray, topology: MockTopology):
        """
        Initialize mock trajectory.
        
        Parameters:
        -----------
        xyz : np.ndarray
            Coordinates array with shape (n_frames, n_atoms, 3)
        topology : MockTopology
            Mock topology
        """
        self.xyz = xyz
        self.topology = topology
        self.n_frames = xyz.shape[0]
        self.n_atoms = xyz.shape[1]
        self.n_residues = topology.n_residues
        
        # For MDTraj compatibility
        self.time = np.arange(self.n_frames, dtype=np.float32)
        self.unitcell_lengths = None
        self.unitcell_angles = None
        self.unitcell_vectors = None
    
    def iterchunks(self, chunk_size: int = 100):
        """Iterate over trajectory in chunks."""
        for start in range(0, self.n_frames, chunk_size):
            end = min(start + chunk_size, self.n_frames)
            chunk_xyz = self.xyz[start:end]
            yield MockTrajectory(chunk_xyz, self.topology)
    
    def __getitem__(self, key):
        """Support slicing."""
        if isinstance(key, slice):
            return MockTrajectory(self.xyz[key], self.topology)
        elif isinstance(key, int):
            return MockTrajectory(self.xyz[key:key+1], self.topology)
        else:
            return MockTrajectory(self.xyz[key], self.topology)
    
    def atom_slice(self, atom_indices):
        """Create trajectory with subset of atoms (MDTraj compatibility)."""
        if isinstance(atom_indices, (list, np.ndarray)):
            atom_indices = np.array(atom_indices)
            new_xyz = self.xyz[:, atom_indices, :]
            
            # Create new topology with subset of atoms
            new_topology = MockTopology(n_atoms=len(atom_indices))
            for i, old_idx in enumerate(atom_indices):
                if old_idx < len(self.topology._atoms):
                    new_topology._atoms[i] = self.topology._atoms[old_idx]
            
            return MockTrajectory(new_xyz, new_topology)
        else:
            raise ValueError("atom_indices must be a list or array")
    
    def compute_distances(self, atom_pairs):
        """Compute distances between atom pairs (MDTraj compatibility)."""
        if isinstance(atom_pairs, (list, tuple)):
            atom_pairs = np.array(atom_pairs)
        
        n_pairs = len(atom_pairs)
        distances = np.zeros((self.n_frames, n_pairs))
        
        for frame in range(self.n_frames):
            for i, (atom1, atom2) in enumerate(atom_pairs):
                pos1 = self.xyz[frame, atom1]
                pos2 = self.xyz[frame, atom2]
                distances[frame, i] = np.linalg.norm(pos1 - pos2)
        
        return distances
    
    def __len__(self):
        """Return number of frames (MDTraj compatibility)."""
        return self.n_frames
    
    def compute_dssp(self, simplified=True):
        """Mock DSSP computation."""
        # Return mock DSSP data - simple pattern based on residue index
        n_residues = min(self.n_atoms, self.topology.n_residues)
        
        if simplified:
            # Simplified: H, E, C
            classes = ['H', 'E', 'C']
            dssp_data = np.zeros((self.n_frames, n_residues), dtype='U1')
            
            for frame in range(self.n_frames):
                for res in range(n_residues):
                    # Simple pattern: every 3rd residue is helix, every 3rd+1 is sheet, rest coil
                    class_idx = (res + frame // 10) % 3  # Changes slowly over time
                    dssp_data[frame, res] = classes[class_idx]
        else:
            # Full DSSP: H, B, E, G, I, T, S, C
            classes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']
            dssp_data = np.zeros((self.n_frames, n_residues), dtype='U1')
            
            for frame in range(self.n_frames):
                for res in range(n_residues):
                    class_idx = (res + frame // 5) % len(classes)
                    dssp_data[frame, res] = classes[class_idx]
        
        return dssp_data

class MockTrajectoryFactory:
    """Factory for creating deterministic mock trajectories."""
    
    @staticmethod
    def create_simple(n_frames: int = 100, n_atoms: int = 50, seed: int = 42) -> MockTrajectory:
        """
        Create simple mock trajectory with deterministic coordinates.
        
        Parameters:
        -----------
        n_frames : int, default=100
            Number of trajectory frames
        n_atoms : int, default=50
            Number of atoms
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Deterministic mock trajectory
        """
        np.random.seed(seed)
        
        # Create topology
        topology = MockTopology(n_atoms=n_atoms, atoms_per_residue=1)  # 1 atom per residue for testing
        
        # Create deterministic coordinates with small linear drift + noise
        xyz = np.zeros((n_frames, n_atoms, 3))
        
        # Base positions (grid-like arrangement)
        base_positions = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            base_positions[i] = [
                (i % 10) * 2.0,  # x: grid layout
                (i // 10) * 2.0,  # y: grid layout  
                0.0  # z: flat
            ]
        
        # Generate trajectory with drift and vibrations
        for frame in range(n_frames):
            # Linear drift over time
            drift = np.array([frame * 0.01, frame * 0.005, 0.0])
            
            # Small random vibrations (consistent per frame with seed)
            np.random.seed(seed + frame)
            vibrations = np.random.randn(n_atoms, 3) * 0.1
            
            xyz[frame] = base_positions + drift + vibrations
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod
    def create_protein_like(n_residues: int = 20, n_frames: int = 100, seed: int = 42) -> MockTrajectory:
        """
        Create protein-like trajectory with CA atoms in helical arrangement.
        
        Parameters:
        -----------
        n_residues : int, default=20
            Number of residues (CA atoms)
        n_frames : int, default=100
            Number of trajectory frames
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Deterministic protein-like trajectory
        """
        np.random.seed(seed)
        
        # Create protein topology
        topology = MockProteinTopology(n_residues=n_residues)
        
        # Create coordinates with helical structure
        xyz = np.zeros((n_frames, n_residues, 3))
        
        # Base helix parameters
        radius = 5.0
        rise_per_residue = 1.5
        turn_angle = 100.0 * np.pi / 180.0  # 100 degrees per residue
        
        for frame in range(n_frames):
            # Overall helix orientation changes slightly over time
            frame_rotation = frame * 0.02
            
            for residue in range(n_residues):
                # Helical coordinates
                angle = residue * turn_angle + frame_rotation
                
                xyz[frame, residue] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle), 
                    residue * rise_per_residue
                ]
                
                # Add small thermal vibrations
                np.random.seed(seed + frame * n_residues + residue)
                thermal_noise = np.random.randn(3) * 0.05
                xyz[frame, residue] += thermal_noise
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod
    def create_two_state(n_atoms: int = 30, n_frames: int = 200, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with two distinct conformational states.
        
        Parameters:
        -----------
        n_atoms : int, default=30
            Number of atoms
        n_frames : int, default=200
            Number of trajectory frames
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Two-state trajectory for clustering tests
        """
        np.random.seed(seed)
        
        topology = MockTopology(n_atoms=n_atoms, atoms_per_residue=1)  # 1 atom per residue for testing
        xyz = np.zeros((n_frames, n_atoms, 3))
        
        # Define two distinct states
        state_a_center = np.array([0.0, 0.0, 0.0])
        state_b_center = np.array([10.0, 0.0, 0.0])
        
        # Generate base coordinates for each state
        state_a_coords = np.random.randn(n_atoms, 3) * 2.0 + state_a_center
        state_b_coords = np.random.randn(n_atoms, 3) * 2.0 + state_b_center
        
        # Assign frames to states (first half A, second half B)
        for frame in range(n_frames):
            if frame < n_frames // 2:
                # State A with small fluctuations
                base_coords = state_a_coords
            else:
                # State B with small fluctuations  
                base_coords = state_b_coords
            
            # Add frame-specific noise
            np.random.seed(seed + frame)
            noise = np.random.randn(n_atoms, 3) * 0.5
            xyz[frame] = base_coords + noise
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod
    def create_with_contacts(n_atoms: int = 10, n_frames: int = 50, contact_cutoff: float = 4.0, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with known contact patterns.
        
        Parameters:
        -----------
        n_atoms : int, default=10
            Number of atoms
        n_frames : int, default=50
            Number of trajectory frames
        contact_cutoff : float, default=4.0
            Distance cutoff for contacts
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Trajectory with predictable contacts
        """
        np.random.seed(seed)
        
        topology = MockTopology(n_atoms=n_atoms, atoms_per_residue=1)  # 1 atom per residue for testing
        xyz = np.zeros((n_frames, n_atoms, 3))
        
        # Create initial configuration with known contacts
        # Atoms 0-1 always in contact (distance ~2.0)
        # Atoms 2-3 sometimes in contact (distance oscillates around cutoff)
        # Other atoms spread out
        
        for frame in range(n_frames):
            # Atoms 0-1: stable contact
            xyz[frame, 0] = [0.0, 0.0, 0.0]
            xyz[frame, 1] = [2.0, 0.0, 0.0]
            
            # Atoms 2-3: oscillating contact
            distance_23 = contact_cutoff + 1.0 * np.sin(frame * 0.3)
            xyz[frame, 2] = [10.0, 0.0, 0.0]
            xyz[frame, 3] = [10.0 + distance_23, 0.0, 0.0]
            
            # Other atoms: spread out, no contacts
            for i in range(4, n_atoms):
                xyz[frame, i] = [i * 6.0, i * 6.0, 0.0]
            
            # Add small noise
            np.random.seed(seed + frame)
            noise = np.random.randn(n_atoms, 3) * 0.1
            xyz[frame] += noise
        
        return MockTrajectory(xyz, topology)

    @staticmethod
    def create_fixed_distances(n_frames: int = 10, n_atoms: int = 4, distance: float = 5.0, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory where all atom-atom distances are exactly the specified value.
        Perfect for testing distance calculations with known results.
        
        Parameters:
        -----------
        n_frames : int, default=10
            Number of trajectory frames
        n_atoms : int, default=4  
            Number of atoms (should be small for exact distance control)
        distance : float, default=5.0
            Exact distance between all atom pairs (in Angstrom)
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Trajectory with fixed distances for testing
        """
        topology = MockTopology(n_atoms=n_atoms, atoms_per_residue=1)  # 1 atom per residue for testing
        xyz = np.zeros((n_frames, n_atoms, 3))
        
        # Place all atoms at vertices of a regular simplex in 3D space to ensure ALL pairwise distances are equal
        # For n atoms, place them at vertices of regular (n-1)-simplex
        if n_atoms == 1:
            # Single atom at origin
            for frame in range(n_frames):
                xyz[frame, 0] = [0.0, 0.0, 0.0]
        elif n_atoms == 2:
            # Two atoms on x-axis
            for frame in range(n_frames):
                xyz[frame, 0] = [-distance / 2.0 / 10.0, 0.0, 0.0]  # Convert Angstrom to nm
                xyz[frame, 1] = [distance / 2.0 / 10.0, 0.0, 0.0]   # Convert Angstrom to nm
        elif n_atoms == 3:
            # Equilateral triangle in x-y plane
            for frame in range(n_frames):
                h = distance * np.sqrt(3) / 2  # Height from center to vertex
                xyz[frame, 0] = [0.0, h / 10.0, 0.0]
                xyz[frame, 1] = [-distance / 2.0 / 10.0, -h / 3.0 / 10.0, 0.0]
                xyz[frame, 2] = [distance / 2.0 / 10.0, -h / 3.0 / 10.0, 0.0]
        elif n_atoms == 4:
            # Regular tetrahedron in 3D space
            for frame in range(n_frames):
                # Tetrahedron with edge length = distance
                # For tetrahedron with vertices at (±1, ±1, ±1), edge length = 2√2
                # Scale factor to get desired edge length
                a = distance / (2 * np.sqrt(2))
                xyz[frame, 0] = [a / 10.0, a / 10.0, a / 10.0]
                xyz[frame, 1] = [a / 10.0, -a / 10.0, -a / 10.0] 
                xyz[frame, 2] = [-a / 10.0, a / 10.0, -a / 10.0]
                xyz[frame, 3] = [-a / 10.0, -a / 10.0, a / 10.0]
        else:
            # For more atoms, approximate with regular polygon (not all distances equal)
            for frame in range(n_frames):
                for atom in range(n_atoms):
                    angle = 2 * np.pi * atom / n_atoms
                    radius = distance / (2 * np.sin(np.pi / n_atoms))
                    xyz[frame, atom] = [
                        radius * np.cos(angle) / 10.0,
                        radius * np.sin(angle) / 10.0, 
                        0.0
                    ]
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod
    def create_linear_coordinates(n_frames: int = 10, n_atoms: int = 3, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with linearly increasing coordinates for predictable calculations.
        
        Parameters:
        -----------
        n_frames : int, default=10
            Number of frames
        n_atoms : int, default=3
            Number of atoms  
        seed : int, default=42
            Random seed
            
        Returns:
        --------
        MockTrajectory
            Trajectory with linear coordinate patterns
        """
        topology = MockTopology(n_atoms=n_atoms, atoms_per_residue=1)  # 1 atom per residue for testing
        xyz = np.zeros((n_frames, n_atoms, 3))
        
        # Create linear pattern: frame 0=[0,0,0], frame 1=[1,1,1], etc.
        for frame in range(n_frames):
            for atom in range(n_atoms):
                xyz[frame, atom] = [
                    frame + atom,      # x increases by frame + atom
                    frame * 0.5,       # y increases by 0.5 per frame
                    atom * 2.0         # z increases by 2.0 per atom
                ]
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod  
    def create_binary_contacts(n_frames: int = 20, n_atoms: int = 3, cutoff: float = 3.0, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with alternating contact/no-contact pattern.
        Even frames: atoms close together (contacts)
        Odd frames: atoms far apart (no contacts)
        
        Parameters:
        -----------
        n_frames : int, default=20
            Number of frames (should be even)
        n_atoms : int, default=3
            Number of atoms
        cutoff : float, default=3.0
            Contact cutoff distance
        seed : int, default=42
            Random seed
            
        Returns:
        --------
        MockTrajectory
            Trajectory with predictable contact pattern
        """
        topology = MockTopology(n_atoms=n_atoms, atoms_per_residue=1)  # 1 atom per residue for testing
        xyz = np.zeros((n_frames, n_atoms, 3))
        
        # Close distance (within cutoff)
        close_distance = cutoff * 0.8  # 80% of cutoff
        # Far distance (beyond cutoff)  
        far_distance = cutoff * 1.5    # 150% of cutoff
        
        for frame in range(n_frames):
            if frame % 2 == 0:
                # Even frames: atoms are close (contacts present)
                for atom in range(n_atoms):
                    angle = 2 * np.pi * atom / n_atoms
                    radius = close_distance / (2 * np.sin(np.pi / n_atoms)) if n_atoms > 1 else 0.0
                    xyz[frame, atom] = [
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        0.0
                    ]
            else:
                # Odd frames: atoms are far (no contacts)
                for atom in range(n_atoms):
                    angle = 2 * np.pi * atom / n_atoms
                    radius = far_distance / (2 * np.sin(np.pi / n_atoms)) if n_atoms > 1 else 0.0
                    xyz[frame, atom] = [
                        radius * np.cos(angle),
                        radius * np.sin(angle), 
                        0.0
                    ]
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod
    def create_triangle_atoms(n_frames: int = 10, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with 3 atoms in a right triangle.
        Perfect for testing distances with known values: 3.0, 4.0, 5.0 Å.
        
        Parameters:
        -----------
        n_frames : int, default=10
            Number of trajectory frames
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Triangle atoms with distances 0-1=3.0, 0-2=4.0, 1-2=5.0 Å
        """
        np.random.seed(seed)
        topology = MockTopology(n_atoms=3, atoms_per_residue=1)
        xyz = np.zeros((n_frames, 3, 3))
        
        # Right triangle: (0,0,0), (3,0,0), (0,4,0)
        # Distances: 0-1=3, 0-2=4, 1-2=5 (Pythagorean triple)
        for frame in range(n_frames):
            xyz[frame, 0, :] = [0.0, 0.0, 0.0]  # Origin
            xyz[frame, 1, :] = [3.0, 0.0, 0.0]  # 3 units along x
            xyz[frame, 2, :] = [0.0, 4.0, 0.0]  # 4 units along y
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod
    def create_known_contacts(n_frames: int = 10, cutoff: float = 3.5, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with known contact pattern using triangle atoms.
        With cutoff=3.5: contacts 0-1=1 (3.0<3.5), 0-2=0 (4.0>3.5), 1-2=0 (5.0>3.5)
        
        Parameters:
        -----------
        n_frames : int, default=10
            Number of trajectory frames
        cutoff : float, default=3.5
            Contact cutoff distance
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Triangle atoms for predictable contact testing
        """
        # Reuse triangle atoms - they have perfect distances for contact testing
        return MockTrajectoryFactory.create_triangle_atoms(n_frames, seed)
    
    @staticmethod
    def create_moving_atoms(n_frames: int = 10, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with atoms moving predictably for coordinate tests.
        
        Parameters:
        -----------
        n_frames : int, default=10
            Number of trajectory frames
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Two atoms: one static at origin, one moving linearly
        """
        np.random.seed(seed)
        topology = MockTopology(n_atoms=2, atoms_per_residue=1)
        xyz = np.zeros((n_frames, 2, 3))
        
        for frame in range(n_frames):
            xyz[frame, 0, :] = [0.0, 0.0, 0.0]  # Static atom at origin
            xyz[frame, 1, :] = [frame * 0.1, 0.0, 0.0]  # Moving along x-axis
        
        return MockTrajectory(xyz, topology)
    
    @staticmethod
    def create_varied_coordinates(n_frames: int = 10, seed: int = 42) -> MockTrajectory:
        """
        Create trajectory with coordinates having known standard deviations.
        Atom 0: no movement (std=0), Atom 1: small movement (std≈0.3), Atom 2: large movement (std≈1.0)
        
        Parameters:
        -----------
        n_frames : int, default=10
            Number of trajectory frames
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        MockTrajectory
            Three atoms with different mobility patterns
        """
        np.random.seed(seed)
        topology = MockTopology(n_atoms=3, atoms_per_residue=1)
        xyz = np.zeros((n_frames, 3, 3))
        
        for frame in range(n_frames):
            # Atom 0: static at origin (std = 0.0)
            xyz[frame, 0, :] = [0.0, 0.0, 0.0]
            
            # Atom 1: small oscillation (std ≈ 0.3)
            xyz[frame, 1, :] = [
                0.3 * np.sin(frame * 0.5),
                0.3 * np.cos(frame * 0.5), 
                0.0
            ]
            
            # Atom 2: large linear movement (std ≈ 1.0)
            xyz[frame, 2, :] = [
                frame * 0.3,  # Linear increase
                frame * 0.2,
                0.0
            ]
        
        return MockTrajectory(xyz, topology)
