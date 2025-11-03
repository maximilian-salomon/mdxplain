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

"""Tests for StructureCalculationHelper."""

import mdtraj as md
import numpy as np
import pytest

from mdxplain.analysis.structure.helper.structure_calculation_helper import StructureCalculationHelper


class TestStructureCalculationHelper:
    """Test class for StructureCalculationHelper."""

    def create_simple_topology(self, n_atoms=2):
        """Create simple topology for testing.

        Parameters
        ----------
        n_atoms : int, optional
            Number of atoms to create. Defaults to 2.

        Returns
        -------
        md.Topology
            Simple topology with carbon atoms.
        """
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        for i in range(n_atoms):
            topology.add_atom(f"C{i}", md.element.carbon, residue)
        return topology

    def create_test_trajectory(self, n_frames, n_atoms=2):
        """Create test trajectory with simple coordinates.

        Parameters
        ----------
        n_frames : int
            Number of frames to create.
        n_atoms : int, optional
            Number of atoms per frame. Defaults to 2.

        Returns
        -------
        md.Trajectory
            Test trajectory with incrementing Y coordinates.
        """
        topology = self.create_simple_topology(n_atoms)
        coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

        for frame in range(n_frames):
            for atom in range(n_atoms):
                coords[frame, atom] = [atom, frame, 0.0]

        time = np.arange(n_frames, dtype=np.float32)
        return md.Trajectory(coords, topology, time=time)

    def test_iterate_chunks_standard(self):
        """Test iterate_chunks with standard chunking."""
        trajectory = self.create_test_trajectory(n_frames=10)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=3))

        assert len(chunks) == 4
        assert chunks[0][1] == 0 and chunks[0][2] == 3  # First chunk: frames 0-2
        assert chunks[1][1] == 3 and chunks[1][2] == 6  # Second chunk: frames 3-5
        assert chunks[2][1] == 6 and chunks[2][2] == 9  # Third chunk: frames 6-8
        assert chunks[3][1] == 9 and chunks[3][2] == 10  # Fourth chunk: frame 9

        assert chunks[0][0].n_frames == 3
        assert chunks[1][0].n_frames == 3
        assert chunks[2][0].n_frames == 3
        assert chunks[3][0].n_frames == 1

    def test_iterate_chunks_exact_fit(self):
        """Test iterate_chunks with exact division."""
        trajectory = self.create_test_trajectory(n_frames=6)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=2))

        assert len(chunks) == 3
        assert chunks[0][1] == 0 and chunks[0][2] == 2
        assert chunks[1][1] == 2 and chunks[1][2] == 4
        assert chunks[2][1] == 4 and chunks[2][2] == 6

        for chunk, _, _ in chunks:
            assert chunk.n_frames == 2

    def test_iterate_chunks_larger_than_trajectory(self):
        """Test iterate_chunks with chunk size larger than trajectory."""
        trajectory = self.create_test_trajectory(n_frames=5)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=10))

        assert len(chunks) == 1
        assert chunks[0][1] == 0 and chunks[0][2] == 5
        assert chunks[0][0].n_frames == 5

    def test_iterate_chunks_single_frame(self):
        """Test iterate_chunks with single frame trajectory."""
        trajectory = self.create_test_trajectory(n_frames=1)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=5))

        assert len(chunks) == 1
        assert chunks[0][1] == 0 and chunks[0][2] == 1
        assert chunks[0][0].n_frames == 1

    def test_iterate_chunks_no_chunking(self):
        """Test iterate_chunks with chunk_size=0 (no chunking)."""
        trajectory = self.create_test_trajectory(n_frames=8)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=0))

        assert len(chunks) == 1
        assert chunks[0][1] == 0 and chunks[0][2] == 8
        assert chunks[0][0].n_frames == 8

    def test_iterate_chunks_none_size(self):
        """Test iterate_chunks with chunk_size=None."""
        trajectory = self.create_test_trajectory(n_frames=7)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=None))

        assert len(chunks) == 1
        assert chunks[0][1] == 0 and chunks[0][2] == 7
        assert chunks[0][0].n_frames == 7

    def test_iterate_chunks_negative_size_error(self):
        """Test that negative chunk_size raises ValueError."""
        trajectory = self.create_test_trajectory(n_frames=6)

        with pytest.raises(ValueError, match="chunk_size must be positive or None, got -1"):
            list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=-1))

    def test_iterate_chunks_empty_trajectory(self):
        """Test iterate_chunks with empty trajectory."""
        topology = self.create_simple_topology()
        coords = np.empty((0, 2, 3), dtype=np.float32)
        trajectory = md.Trajectory(coords, topology)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=5))

        assert len(chunks) == 0

    def test_iterate_chunks_verify_indices(self):
        """Test that iterate_chunks provides correct frame indices."""
        trajectory = self.create_test_trajectory(n_frames=7)

        chunks = list(StructureCalculationHelper.iterate_chunks(trajectory, chunk_size=3))

        assert len(chunks) == 3

        # First chunk: frames 0-2
        chunk, start, end = chunks[0]
        assert start == 0 and end == 3
        np.testing.assert_array_equal(chunk.xyz[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(chunk.xyz[2, 0], [0.0, 2.0, 0.0])

        # Second chunk: frames 3-5
        chunk, start, end = chunks[1]
        assert start == 3 and end == 6
        np.testing.assert_array_equal(chunk.xyz[0, 0], [0.0, 3.0, 0.0])
        np.testing.assert_array_equal(chunk.xyz[2, 0], [0.0, 5.0, 0.0])

        # Third chunk: frame 6
        chunk, start, end = chunks[2]
        assert start == 6 and end == 7
        np.testing.assert_array_equal(chunk.xyz[0, 0], [0.0, 6.0, 0.0])

    def test_stack_frames_two_frames(self):
        """Test stack_frames with two single frames."""
        topology = self.create_simple_topology()

        frame1 = md.Trajectory(
            np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
            topology,
            time=np.array([0.0])
        )
        frame2 = md.Trajectory(
            np.array([[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float32),
            topology,
            time=np.array([1.0])
        )

        stacked = StructureCalculationHelper.stack_frames([frame1, frame2])

        assert stacked.n_frames == 2
        assert stacked.n_atoms == 2
        np.testing.assert_array_equal(stacked.xyz[0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(stacked.xyz[1], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    def test_stack_frames_multiple(self):
        """Test stack_frames with multiple frames."""
        topology = self.create_simple_topology()

        frames = []
        for i in range(5):
            frame = md.Trajectory(
                np.array([[[0.0, float(i), 0.0], [1.0, float(i), 0.0]]], dtype=np.float32),
                topology,
                time=np.array([float(i)])
            )
            frames.append(frame)

        stacked = StructureCalculationHelper.stack_frames(frames)

        assert stacked.n_frames == 5
        assert stacked.n_atoms == 2

        for i in range(5):
            np.testing.assert_array_equal(
                stacked.xyz[i], [[0.0, float(i), 0.0], [1.0, float(i), 0.0]]
            )

    def test_stack_frames_single(self):
        """Test stack_frames with single frame (identity operation)."""
        topology = self.create_simple_topology()

        frame = md.Trajectory(
            np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
            topology,
            time=np.array([0.0])
        )

        stacked = StructureCalculationHelper.stack_frames([frame])

        assert stacked.n_frames == 1
        assert stacked.n_atoms == 2
        np.testing.assert_array_equal(stacked.xyz[0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_stack_frames_verify_coordinates(self):
        """Test that stack_frames correctly concatenates coordinates."""
        topology = self.create_simple_topology(n_atoms=1)

        frames = []
        expected_coords = []
        for i in range(3):
            coord = np.array([[[float(i), float(i*2), float(i*3)]]], dtype=np.float32)
            frame = md.Trajectory(coord, topology, time=np.array([float(i)]))
            frames.append(frame)
            expected_coords.append(coord[0])

        stacked = StructureCalculationHelper.stack_frames(frames)

        assert stacked.n_frames == 3
        expected_stacked = np.array(expected_coords, dtype=np.float32)
        np.testing.assert_array_equal(stacked.xyz, expected_stacked)

    def test_stack_frames_verify_time(self):
        """Test that stack_frames correctly concatenates time arrays."""
        topology = self.create_simple_topology()

        frames = []
        expected_times = []
        for i in range(4):
            time_val = float(i * 10)
            frame = md.Trajectory(
                np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
                topology,
                time=np.array([time_val])
            )
            frames.append(frame)
            expected_times.append(time_val)

        stacked = StructureCalculationHelper.stack_frames(frames)

        np.testing.assert_array_equal(stacked.time, np.array(expected_times))

    def test_stack_frames_empty_list_error(self):
        """Test stack_frames with empty frame list raises ValueError."""
        with pytest.raises(ValueError, match="Expected at least one frame to stack."):
            StructureCalculationHelper.stack_frames([])