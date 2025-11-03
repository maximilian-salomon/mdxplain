# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.5).
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
Test file for TrajectoryManager.stack() label handling.

Tests verify that labels are correctly combined and renumbered when stacking trajectories:
- Label indices are renumbered correctly for stacked trajectory
- Stacking works with and without labels
- Source trajectory is removed when requested
- Topology changes are reflected correctly
"""

import pytest
import numpy as np
import mdtraj as md

from mdxplain.pipeline.entities.pipeline_data import PipelineData
from mdxplain.trajectory.managers.trajectory_manager import TrajectoryManager
from mdxplain.trajectory.entities.trajectory_data import TrajectoryData


def create_simple_topology(n_residues, resname, atoms_per_residue=2):
    """
    Create simple test topology.

    Parameters
    ----------
    n_residues : int
        Number of residues
    resname : str
        Residue name (e.g., 'ALA', 'GLY')
    atoms_per_residue : int
        Number of atoms per residue

    Returns
    -------
    md.Topology
        Simple topology with specified residues
    """
    topology = md.Topology()
    chain = topology.add_chain()

    for i in range(n_residues):
        residue = topology.add_residue(resname, chain, resSeq=i + 1)
        for j in range(atoms_per_residue):
            atom_name = "CA" if j == 0 else f"C{j}"
            topology.add_atom(atom_name, md.element.carbon, residue)

    return topology


def create_test_trajectory(n_frames, n_residues, resname="ALA", atoms_per_residue=2):
    """
    Create test trajectory with random coordinates.

    Parameters
    ----------
    n_frames : int
        Number of frames
    n_residues : int
        Number of residues
    resname : str
        Residue name
    atoms_per_residue : int
        Number of atoms per residue

    Returns
    -------
    md.Trajectory
        Test trajectory
    """
    topology = create_simple_topology(n_residues, resname, atoms_per_residue)
    n_atoms = n_residues * atoms_per_residue
    xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
    return md.Trajectory(xyz, topology)


class TestStackLabels:
    """Test class for stack() label handling functionality."""

    def test_stack_labels_renumbering(self):
        """Test that labels are correctly renumbered when stacking trajectories."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        # Create two trajectories (3 residues each, 2 atoms per residue)
        traj1 = create_test_trajectory(n_frames=2, n_residues=3, resname="ALA")
        traj2 = create_test_trajectory(n_frames=2, n_residues=3, resname="GLY")

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj1, traj2]
        pipeline_data.trajectory_data.trajectory_names = ["traj1", "traj2"]

        # Add labels to both trajectories
        pipeline_data.trajectory_data.res_label_data[0] = [
            {"index": 0, "resSeq": 1, "resname": "ALA", "label": "res1"},
            {"index": 1, "resSeq": 2, "resname": "ALA", "label": "res2"},
            {"index": 2, "resSeq": 3, "resname": "ALA", "label": "res3"},
        ]

        pipeline_data.trajectory_data.res_label_data[1] = [
            {"index": 0, "resSeq": 1, "resname": "GLY", "label": "gly1"},
            {"index": 1, "resSeq": 2, "resname": "GLY", "label": "gly2"},
            {"index": 2, "resSeq": 3, "resname": "GLY", "label": "gly3"},
        ]

        # Stack traj2 onto traj1
        traj_manager.stack(
            pipeline_data, target_traj=0, source_traj=1, remove_source=True
        )

        # Get stacked labels
        stacked_labels = pipeline_data.trajectory_data.res_label_data[0]

        # Verify total number of labels (3 + 3 = 6)
        assert len(stacked_labels) == 6, f"Expected 6 labels, got {len(stacked_labels)}"

        # Verify first 3 labels unchanged (from traj1)
        assert stacked_labels[0]["index"] == 0
        assert stacked_labels[0]["label"] == "res1"
        assert stacked_labels[1]["index"] == 1
        assert stacked_labels[1]["label"] == "res2"
        assert stacked_labels[2]["index"] == 2
        assert stacked_labels[2]["label"] == "res3"

        # Verify last 3 labels renumbered (from traj2, shifted by +3)
        assert (
            stacked_labels[3]["index"] == 3
        ), f"Expected index 3, got {stacked_labels[3]['index']}"
        assert stacked_labels[3]["label"] == "gly1"
        assert (
            stacked_labels[4]["index"] == 4
        ), f"Expected index 4, got {stacked_labels[4]['index']}"
        assert stacked_labels[4]["label"] == "gly2"
        assert (
            stacked_labels[5]["index"] == 5
        ), f"Expected index 5, got {stacked_labels[5]['index']}"
        assert stacked_labels[5]["label"] == "gly3"

        # Verify topology has 6 residues (3 + 3)
        stacked_traj = pipeline_data.trajectory_data.trajectories[0]
        assert (
            stacked_traj.n_residues == 6
        ), f"Expected 6 residues, got {stacked_traj.n_residues}"
        assert (
            stacked_traj.n_atoms == 12
        ), f"Expected 12 atoms, got {stacked_traj.n_atoms}"

        # Verify source trajectory was removed
        assert len(pipeline_data.trajectory_data.trajectories) == 1
        assert 1 not in pipeline_data.trajectory_data.res_label_data

    def test_stack_without_labels(self):
        """Test that stacking works when trajectories have no labels."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        # Create two trajectories without labels
        traj1 = create_test_trajectory(n_frames=2, n_residues=2, resname="ALA")
        traj2 = create_test_trajectory(n_frames=2, n_residues=2, resname="GLY")

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj1, traj2]
        pipeline_data.trajectory_data.trajectory_names = ["traj1", "traj2"]

        # No labels added - should still work
        traj_manager.stack(
            pipeline_data, target_traj=0, source_traj=1, remove_source=True
        )

        # Verify stacking worked
        stacked_traj = pipeline_data.trajectory_data.trajectories[0]
        assert stacked_traj.n_residues == 4, "Should have 4 residues after stacking"
        assert (
            len(pipeline_data.trajectory_data.trajectories) == 1
        ), "Source should be removed"

    def test_stack_keep_source(self):
        """Test that source trajectory is kept when remove_source=False."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        # Create two trajectories
        traj1 = create_test_trajectory(n_frames=2, n_residues=2, resname="ALA")
        traj2 = create_test_trajectory(n_frames=2, n_residues=2, resname="GLY")

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj1, traj2]
        pipeline_data.trajectory_data.trajectory_names = ["traj1", "traj2"]

        # Stack without removing source
        traj_manager.stack(
            pipeline_data, target_traj=0, source_traj=1, remove_source=False
        )

        # Verify both trajectories still exist
        assert (
            len(pipeline_data.trajectory_data.trajectories) == 2
        ), "Both trajectories should exist"
        assert (
            pipeline_data.trajectory_data.trajectories[0].n_residues == 4
        ), "Target should be stacked"
        assert (
            pipeline_data.trajectory_data.trajectories[1].n_residues == 2
        ), "Source should be unchanged"

    def test_stack_labels_with_partial_labels(self):
        """Test stacking when only one trajectory has labels."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        # Create two trajectories
        traj1 = create_test_trajectory(n_frames=2, n_residues=2, resname="ALA")
        traj2 = create_test_trajectory(n_frames=2, n_residues=2, resname="GLY")

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj1, traj2]
        pipeline_data.trajectory_data.trajectory_names = ["traj1", "traj2"]

        # Add labels only to first trajectory
        pipeline_data.trajectory_data.res_label_data[0] = [
            {"index": 0, "resSeq": 1, "resname": "ALA", "label": "res1"},
            {"index": 1, "resSeq": 2, "resname": "ALA", "label": "res2"},
        ]

        # Stack (traj2 has no labels)
        traj_manager.stack(
            pipeline_data, target_traj=0, source_traj=1, remove_source=True
        )

        # Verify only first trajectory's labels exist (second had none to add)
        stacked_labels = pipeline_data.trajectory_data.res_label_data.get(0, [])
        assert (
            len(stacked_labels) == 2
        ), "Should only have labels from first trajectory"
        assert stacked_labels[0]["index"] == 0
        assert stacked_labels[1]["index"] == 1

        # Verify topology is correct
        stacked_traj = pipeline_data.trajectory_data.trajectories[0]
        assert stacked_traj.n_residues == 4, "Should have 4 residues"
