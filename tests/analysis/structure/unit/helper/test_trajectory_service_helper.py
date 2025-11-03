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

"""Unit tests for TrajectoryServiceHelper with concrete value verification."""

from unittest.mock import Mock
import numpy as np
import pytest
import mdtraj as md

from mdxplain.analysis.structure.helper.trajectory_service_helper import TrajectoryServiceHelper
from mdxplain.pipeline.entities.pipeline_data import PipelineData


def create_mock_trajectory(name: str, n_frames: int = 10, n_atoms: int = 8, n_residues: int = 4):
    """Create mock trajectory with specific properties for testing.

    Creates trajectory with deterministic atom/residue layout:
    - Atoms: [CA, CB] per residue
    - Residue names: ALA1, VAL2, LEU3, GLY4
    - Residue IDs: 1, 2, 3, 4
    """
    traj = Mock(spec=md.Trajectory)
    traj.name = name
    traj.n_frames = n_frames
    traj.n_atoms = n_atoms

    # Create mock topology
    topology = Mock(spec=md.Topology)
    topology.n_atoms = n_atoms
    topology.n_residues = n_residues

    # Create mock atoms with specific names and residue assignments
    atoms = []
    residues = []

    # Create residues first
    residue_names = ["ALA", "VAL", "LEU", "GLY"]
    for i in range(n_residues):
        residue = Mock()
        residue.name = residue_names[i]
        residue.resSeq = i + 1  # Residue IDs 1, 2, 3, 4
        residue.index = i
        residues.append(residue)

    # Create atoms (2 per residue: CA, CB)
    atom_names = ["CA", "CB"]
    for res_idx in range(n_residues):
        for atom_idx, atom_name in enumerate(atom_names):
            atom = Mock()
            atom.name = atom_name
            atom.index = res_idx * 2 + atom_idx
            atom.residue = residues[res_idx]
            atoms.append(atom)

    topology.atoms = atoms
    topology.residues = residues

    # Mock select method for atom selections
    def mock_select(selection_string):
        if selection_string == "name CA":
            return np.array([0, 2, 4, 6])  # CA atoms
        elif selection_string == "name CB":
            return np.array([1, 3, 5, 7])  # CB atoms
        elif selection_string == "resid 2":
            return np.array([2, 3])  # VAL2 atoms
        elif selection_string == "resid 1 to 3":
            return np.array([0, 1, 2, 3, 4, 5])  # First 3 residues
        elif selection_string == "name CA and resid 2 to 3":
            return np.array([2, 4])  # CA atoms in residues 2-3
        elif selection_string == "(name CA or name CB) and resid 1":
            return np.array([0, 1])  # Both atoms in residue 1
        elif selection_string == "name XYZ":
            return np.array([])  # Invalid selection
        else:
            return np.array(range(n_atoms))  # All atoms for unknown selections

    topology.select = Mock(side_effect=mock_select)
    traj.topology = topology

    return traj


def create_mock_pipeline_data():
    """Create consistent mock pipeline data for testing."""
    pipeline_data = Mock(spec=PipelineData)

    # Create 5 mock trajectories
    trajectories = []
    for i in range(5):
        traj = create_mock_trajectory(
            name=f"traj_{i}",
            n_frames=10,
            n_atoms=8,
            n_residues=4
        )
        trajectories.append(traj)

    # Create mock trajectory_data
    trajectory_data = Mock()
    trajectory_data.trajectories = trajectories
    trajectory_data.trajectory_names = [f"traj_{i}" for i in range(5)]

    # Mock get_trajectory_indices method
    def mock_get_trajectory_indices(selection):
        if selection == "all":
            return [0, 1, 2, 3, 4]
        elif isinstance(selection, int):
            if 0 <= selection < 5:
                return [selection]
            else:
                raise IndexError(f"Trajectory index {selection} out of range")
        elif isinstance(selection, str) and selection.startswith("traj_"):
            try:
                idx = int(selection.split("_")[1])
                if 0 <= idx < 5:
                    return [idx]
                else:
                    raise ValueError(f"No trajectory named '{selection}' found")
            except (ValueError, IndexError):
                raise ValueError(f"No trajectory named '{selection}' found")
        elif isinstance(selection, list):
            if not selection:
                return []  # Empty selection
            indices = []
            for item in selection:
                if isinstance(item, int):
                    if 0 <= item < 5:
                        indices.append(item)
                    else:
                        raise IndexError(f"Trajectory index {item} out of range")
                elif isinstance(item, str) and item.startswith("traj_"):
                    try:
                        idx = int(item.split("_")[1])
                        if 0 <= idx < 5:
                            indices.append(idx)
                        else:
                            raise ValueError(f"No trajectory named '{item}' found")
                    except (ValueError, IndexError):
                        raise ValueError(f"No trajectory named '{item}' found")
                else:
                    raise ValueError(f"Invalid selection item: {item}")
            return indices
        else:
            raise ValueError(f"Invalid trajectory selection: {selection}")

    trajectory_data.get_trajectory_indices = Mock(side_effect=mock_get_trajectory_indices)
    pipeline_data.trajectory_data = trajectory_data

    return pipeline_data


class TestTrajectoryServiceHelper:
    """Test TrajectoryServiceHelper with concrete value verification."""

    def setup_method(self):
        """Create deterministic test data with known indices and names."""
        self.pipeline_data = create_mock_pipeline_data()
        self.helper = TrajectoryServiceHelper(self.pipeline_data)
        self.mock_trajectories = self.pipeline_data.trajectory_data.trajectories

    def test_resolve_trajectory_indices_with_exact_indices(self):
        """Test that exact trajectory indices are resolved correctly."""
        # Test single index
        indices = self.helper._resolve_trajectory_indices(2)
        assert indices == [2]  # Exactly trajectory at index 2

        # Test list of indices
        indices = self.helper._resolve_trajectory_indices([0, 2, 4])
        assert indices == [0, 2, 4]  # Exactly these trajectories

        # Test "all"
        indices = self.helper._resolve_trajectory_indices("all")
        assert indices == [0, 1, 2, 3, 4]  # All 5 trajectories

    def test_resolve_trajectory_indices_by_name(self):
        """Test trajectory resolution by name returns correct indices."""
        # Single name
        indices = self.helper._resolve_trajectory_indices("traj_2")
        assert indices == [2]  # "traj_2" is at index 2

        # Multiple names
        indices = self.helper._resolve_trajectory_indices(["traj_1", "traj_3"])
        assert indices == [1, 3]  # Exact indices for these names

    def test_resolve_atom_indices_with_exact_selections(self):
        """Test atom selection returns exact atom indices."""
        trajectory = self.mock_trajectories[0]

        # "name CA" should return [0, 2, 4, 6]
        atoms = self.helper._resolve_atom_indices(trajectory, "name CA")
        np.testing.assert_array_equal(atoms, np.array([0, 2, 4, 6]))

        # "name CB" should return [1, 3, 5, 7]
        atoms = self.helper._resolve_atom_indices(trajectory, "name CB")
        np.testing.assert_array_equal(atoms, np.array([1, 3, 5, 7]))

        # "resid 2" should return atoms [2, 3] (VAL2)
        atoms = self.helper._resolve_atom_indices(trajectory, "resid 2")
        np.testing.assert_array_equal(atoms, np.array([2, 3]))

        # "resid 1 to 3" should return atoms [0, 1, 2, 3, 4, 5]
        atoms = self.helper._resolve_atom_indices(trajectory, "resid 1 to 3")
        np.testing.assert_array_equal(atoms, np.array([0, 1, 2, 3, 4, 5]))

        # "all" should return None (meaning all atoms)
        atoms = self.helper._resolve_atom_indices(trajectory, "all")
        assert atoms is None

    def test_get_trajectories_returns_exact_objects(self):
        """Test that exact trajectory objects are returned."""
        indices = [0, 3, 4]
        trajectories = self.helper._get_trajectories(indices)

        # Verify exact object identity
        assert len(trajectories) == 3
        assert trajectories[0] is self.mock_trajectories[0]  # Exact object
        assert trajectories[1] is self.mock_trajectories[3]
        assert trajectories[2] is self.mock_trajectories[4]

    def test_get_trajectory_names_exact_mapping(self):
        """Test exact name mapping for trajectory indices."""
        indices = [1, 2, 4]
        names = self.helper._get_trajectory_names(indices)

        assert names == ["traj_1", "traj_2", "traj_4"]  # Exact names

        # Test fallback for missing names (simulate shorter names list)
        original_names = self.pipeline_data.trajectory_data.trajectory_names
        self.pipeline_data.trajectory_data.trajectory_names = ["traj_0"]  # Only first name

        indices = [0, 1]  # Index 1 doesn't exist in shortened names
        names = self.helper._get_trajectory_names(indices)
        assert names == ["traj_0", "trajectory_1"]  # Fallback name

        # Restore original names
        self.pipeline_data.trajectory_data.trajectory_names = original_names

    def test_resolve_trajectories_and_atoms_complete_flow(self):
        """Test complete resolution flow with exact values."""
        trajs, names, atoms = self.helper.resolve_trajectories_and_atoms(
            traj_selection=[1, 3],
            atom_selection="name CA"
        )

        # Exact trajectory verification
        assert len(trajs) == 2
        assert trajs[0].name == "traj_1"  # Check trajectory property
        assert trajs[1].name == "traj_3"

        # Exact name verification
        assert names == ["traj_1", "traj_3"]

        # Exact atom indices
        np.testing.assert_array_equal(atoms, np.array([0, 2, 4, 6]))

    def test_build_result_map_per_trajectory_exact_mapping(self):
        """Test result map builds exact name-to-array mapping."""
        names = ["traj_0", "traj_2", "traj_4"]
        arrays = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
            np.array([0.7, 0.8, 0.9], dtype=np.float32)
        ]

        result = self.helper.build_result_map(names, arrays, cross_trajectory=False)

        # Exact key verification
        assert list(result.keys()) == ["traj_0", "traj_2", "traj_4"]

        # Exact value verification
        np.testing.assert_array_equal(result["traj_0"], np.array([0.1, 0.2, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(result["traj_2"], np.array([0.4, 0.5, 0.6], dtype=np.float32))
        np.testing.assert_array_equal(result["traj_4"], np.array([0.7, 0.8, 0.9], dtype=np.float32))

    def test_build_result_map_cross_trajectory_exact_key(self):
        """Test cross-trajectory result uses exact 'combined' key."""
        names = ["traj_1", "traj_3"]
        arrays = [np.array([1.5, 2.5], dtype=np.float32)]  # Single combined array

        result = self.helper.build_result_map(names, arrays, cross_trajectory=True)

        # Exact key and value
        assert list(result.keys()) == ["combined"]
        np.testing.assert_array_equal(result["combined"], np.array([1.5, 2.5], dtype=np.float32))

    def test_map_reference_to_local_index_exact_positions(self):
        """Test reference trajectory maps to exact local position."""
        # Reference 2 in selection [0, 2, 4] -> local index 1
        local = self.helper.map_reference_to_local_index([0, 2, 4], 2)
        assert local == 1  # Position 1 in the list

        # Reference 4 in selection [1, 3, 4] -> local index 2
        local = self.helper.map_reference_to_local_index([1, 3, 4], 4)
        assert local == 2

        # Reference 0 in selection [0, 1, 2] -> local index 0
        local = self.helper.map_reference_to_local_index([0, 1, 2], 0)
        assert local == 0

    def test_map_reference_not_in_selection_exact_error(self):
        """Test exact error when reference not in selection."""
        with pytest.raises(ValueError) as excinfo:
            self.helper.map_reference_to_local_index([1, 2], 3)

        # Exact error message
        assert str(excinfo.value) == "Reference trajectory 3 not in selected trajectories [1, 2]"

    def test_empty_trajectory_selection_exact_error(self):
        """Test exact error for empty trajectory selection."""
        with pytest.raises(ValueError) as excinfo:
            self.helper._resolve_trajectory_indices([])

        assert str(excinfo.value) == "No trajectories found for the requested selection."

    def test_invalid_atom_selection_exact_error(self):
        """Test exact error for invalid atom selection."""
        trajectory = self.mock_trajectories[0]

        with pytest.raises(ValueError) as excinfo:
            self.helper._resolve_atom_indices(trajectory, "name XYZ")  # No atoms named XYZ

        assert str(excinfo.value) == "Atom selection 'name XYZ' produced no atoms."

    def test_mixed_selection_types(self):
        """Test mixing indices and names in selection."""
        # Mix of index and name
        indices = self.helper._resolve_trajectory_indices([0, "traj_2", 4])
        assert indices == [0, 2, 4]  # Resolved to exact indices

    def test_complex_atom_selection_exact_results(self):
        """Test complex MDTraj selections return exact atoms."""
        trajectory = self.mock_trajectories[0]

        # "name CA and resid 2 to 3"
        atoms = self.helper._resolve_atom_indices(trajectory, "name CA and resid 2 to 3")
        np.testing.assert_array_equal(atoms, np.array([2, 4]))  # CA atoms in residues 2-3

        # "(name CA or name CB) and resid 1"
        atoms = self.helper._resolve_atom_indices(trajectory, "(name CA or name CB) and resid 1")
        np.testing.assert_array_equal(atoms, np.array([0, 1]))  # Both atoms in residue 1

    def test_trajectory_order_preservation(self):
        """Test that trajectory order is preserved in selection."""
        # Order should be preserved even if not ascending
        indices = self.helper._resolve_trajectory_indices([3, 1, 4, 0])
        assert indices == [3, 1, 4, 0]  # Exact order preserved

        trajs = self.helper._get_trajectories(indices)
        assert trajs[0].name == "traj_3"
        assert trajs[1].name == "traj_1"
        assert trajs[2].name == "traj_4"
        assert trajs[3].name == "traj_0"

    def test_invalid_trajectory_index_exact_error(self):
        """Test exact error for invalid trajectory index."""
        with pytest.raises(IndexError) as excinfo:
            self.helper._resolve_trajectory_indices(10)  # Only 5 trajectories (0-4)

        assert "Trajectory index 10 out of range" in str(excinfo.value)

    def test_invalid_trajectory_name_exact_error(self):
        """Test exact error for invalid trajectory name."""
        with pytest.raises(ValueError) as excinfo:
            self.helper._resolve_trajectory_indices("invalid_name")

        assert "Invalid trajectory selection: invalid_name" in str(excinfo.value)
