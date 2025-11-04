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
Test file for TrajectoryManager.remove_solvent() label handling.

Tests verify that labels are correctly filtered and mapped when removing solvent:
- Labels correctly filtered to kept residues
- Original residue indices correctly mapped after removal
- Works with different solvent positions in different trajectories
- Works with and without labels
"""

import numpy as np
import mdtraj as md

from mdxplain.pipeline.entities.pipeline_data import PipelineData
from mdxplain.trajectory.manager.trajectory_manager import TrajectoryManager
from mdxplain.trajectory.entities.trajectory_data import TrajectoryData


def create_topology_with_solvent(protein_residues, solvent_positions):
    """
    Create topology with protein and solvent at specified final positions.

    Parameters
    ----------
    protein_residues : list of str
        Protein residue names in order
    solvent_positions : list of int
        Final positions where HOH should appear (0-indexed)

    Returns
    -------
    md.Topology
        Topology with protein and solvent residues
    """
    topology = md.Topology()
    chain = topology.add_chain()

    total_residues = len(protein_residues) + len(solvent_positions)
    residue_list = [None] * total_residues

    for pos in solvent_positions:
        residue_list[pos] = "HOH"

    protein_idx = 0
    for i in range(total_residues):
        if residue_list[i] is None:
            residue_list[i] = protein_residues[protein_idx]
            protein_idx += 1

    for i, resname in enumerate(residue_list):
        residue = topology.add_residue(resname, chain, resSeq=i + 1)
        if resname == "HOH":
            topology.add_atom("O", md.element.oxygen, residue)
        else:
            topology.add_atom("CA", md.element.carbon, residue)
            topology.add_atom("C", md.element.carbon, residue)

    return topology


def create_trajectory_with_solvent(
    n_frames, protein_residues, solvent_positions
):
    """
    Create test trajectory with protein and solvent.

    Parameters
    ----------
    n_frames : int
        Number of frames
    protein_residues : list of str
        Protein residue names
    solvent_positions : list of int
        Positions where HOH should be inserted

    Returns
    -------
    md.Trajectory
        Test trajectory with protein and solvent
    """
    topology = create_topology_with_solvent(protein_residues, solvent_positions)
    n_atoms = topology.n_atoms
    xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
    return md.Trajectory(xyz, topology)


class TestRemoveSolventLabels:
    """Test class for remove_solvent() label handling functionality."""

    def test_remove_solvent_filters_labels(self):
        """Test that labels are correctly filtered when removing solvent."""
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        protein_residues = ["ALA", "GLY", "ALA"]
        solvent_positions = [1, 3]

        traj = create_trajectory_with_solvent(
            n_frames=2, protein_residues=protein_residues,
            solvent_positions=solvent_positions
        )

        before_topology = list(traj.topology.residues)

        assert len(before_topology) == 5, (
            f"Setup check: Expected 5 residues before removal. "
            f"Got: {[(i, r.name) for i, r in enumerate(before_topology)]}"
        )
        assert before_topology[0].name == "ALA"
        assert before_topology[1].name == "HOH"
        assert before_topology[2].name == "GLY"
        assert before_topology[3].name == "HOH"
        assert before_topology[4].name == "ALA"

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj]
        pipeline_data.trajectory_data.trajectory_names = ["test"]

        pipeline_data.trajectory_data.res_label_data[0] = [
            {"index": 0, "resSeq": 1, "resname": "ALA", "label": "res1"},
            {"index": 1, "resSeq": 2, "resname": "HOH", "label": "solv1"},
            {"index": 2, "resSeq": 3, "resname": "GLY", "label": "res2"},
            {"index": 3, "resSeq": 4, "resname": "HOH", "label": "solv2"},
            {"index": 4, "resSeq": 5, "resname": "ALA", "label": "res3"},
        ]

        traj_manager.remove_solvent(pipeline_data, traj_selection=0)

        result_traj = pipeline_data.trajectory_data.trajectories[0]
        after_topology = list(result_traj.topology.residues)

        assert result_traj.n_residues == 3, (
            f"Topology check: Expected 3 residues after removal, got {result_traj.n_residues}. "
            f"Before: {[(r.name, r.resSeq) for r in before_topology]}, "
            f"After: {[(r.name, r.resSeq) for r in after_topology]}"
        )

        assert after_topology[0].name == "ALA"
        assert after_topology[1].name == "GLY"
        assert after_topology[2].name == "ALA"

        labels = pipeline_data.trajectory_data.res_label_data[0]

        assert len(labels) == 3, (
            f"Expected 3 labels, got {len(labels)}. Labels: {labels}"
        )

        assert labels[0]["index"] == 0
        assert labels[0]["label"] == "res1"
        assert labels[0]["resname"] == "ALA"

        assert labels[1]["index"] == 1
        assert labels[1]["label"] == "res2"
        assert labels[1]["resname"] == "GLY"

        assert labels[2]["index"] == 2
        assert labels[2]["label"] == "res3"
        assert labels[2]["resname"] == "ALA"

        assert result_traj.n_atoms == 6

    def test_remove_solvent_different_positions_per_trajectory(self):
        """Test label mapping with different solvent positions per trajectory."""
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        traj1 = create_trajectory_with_solvent(
            n_frames=2, protein_residues=["ALA", "GLY"],
            solvent_positions=[0, 2]
        )
        traj2 = create_trajectory_with_solvent(
            n_frames=2, protein_residues=["ALA", "GLY"],
            solvent_positions=[1, 3]
        )

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj1, traj2]
        pipeline_data.trajectory_data.trajectory_names = ["traj1", "traj2"]

        pipeline_data.trajectory_data.res_label_data[0] = [
            {"index": 0, "resSeq": 1, "resname": "HOH", "label": "solv1"},
            {"index": 1, "resSeq": 2, "resname": "ALA", "label": "ala1"},
            {"index": 2, "resSeq": 3, "resname": "HOH", "label": "solv2"},
            {"index": 3, "resSeq": 4, "resname": "GLY", "label": "gly1"},
        ]

        pipeline_data.trajectory_data.res_label_data[1] = [
            {"index": 0, "resSeq": 1, "resname": "ALA", "label": "ala2"},
            {"index": 1, "resSeq": 2, "resname": "HOH", "label": "solv3"},
            {"index": 2, "resSeq": 3, "resname": "GLY", "label": "gly2"},
            {"index": 3, "resSeq": 4, "resname": "HOH", "label": "solv4"},
        ]

        traj_manager.remove_solvent(pipeline_data, traj_selection=[0, 1])

        labels_0 = pipeline_data.trajectory_data.res_label_data[0]
        assert len(labels_0) == 2
        assert labels_0[0]["index"] == 0
        assert labels_0[0]["label"] == "ala1"
        assert labels_0[1]["index"] == 1
        assert labels_0[1]["label"] == "gly1"

        labels_1 = pipeline_data.trajectory_data.res_label_data[1]
        assert len(labels_1) == 2
        assert labels_1[0]["index"] == 0
        assert labels_1[0]["label"] == "ala2"
        assert labels_1[1]["index"] == 1
        assert labels_1[1]["label"] == "gly2"

        assert pipeline_data.trajectory_data.trajectories[0].n_residues == 2
        assert pipeline_data.trajectory_data.trajectories[1].n_residues == 2

    def test_remove_solvent_without_labels(self):
        """Test that remove_solvent works when trajectory has no labels."""
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        traj = create_trajectory_with_solvent(
            n_frames=2, protein_residues=["ALA", "GLY"],
            solvent_positions=[1]
        )

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj]
        pipeline_data.trajectory_data.trajectory_names = ["test"]

        traj_manager.remove_solvent(pipeline_data, traj_selection=0)

        result_traj = pipeline_data.trajectory_data.trajectories[0]
        assert result_traj.n_residues == 2
        assert 0 not in pipeline_data.trajectory_data.res_label_data

    def test_remove_solvent_with_exclude(self):
        """Test remove_solvent with exclude parameter keeps excluded residues."""
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        traj = create_trajectory_with_solvent(
            n_frames=2, protein_residues=["ALA", "GLY"],
            solvent_positions=[1, 3]
        )

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj]
        pipeline_data.trajectory_data.trajectory_names = ["test"]

        pipeline_data.trajectory_data.res_label_data[0] = [
            {"index": 0, "resSeq": 1, "resname": "ALA", "label": "ala1"},
            {"index": 1, "resSeq": 2, "resname": "HOH", "label": "wat1"},
            {"index": 2, "resSeq": 3, "resname": "GLY", "label": "gly1"},
            {"index": 3, "resSeq": 4, "resname": "HOH", "label": "wat2"},
        ]

        traj_manager.remove_solvent(
            pipeline_data, traj_selection=0, exclude=["HOH"]
        )

        labels = pipeline_data.trajectory_data.res_label_data[0]
        assert len(labels) == 4, (
            f"exclude=['HOH'] should KEEP HOH residues. "
            f"Expected 4 labels, got {len(labels)}"
        )
        assert labels[0]["label"] == "ala1"
        assert labels[1]["label"] == "wat1"
        assert labels[2]["label"] == "gly1"
        assert labels[3]["label"] == "wat2"

        result_traj = pipeline_data.trajectory_data.trajectories[0]
        assert result_traj.n_residues == 4
