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

"""Tests for RMSF Calculator."""

from unittest.mock import patch

import mdtraj as md
import numpy as np
import pytest

from mdxplain.analysis.structure.calculators.rmsf_calculator import RMSFCalculator


class TestRMSFCalculator:
    """Test class for RMSFCalculator."""

    def setup_method(self):
        """Create test trajectories with known RMSF values."""
        # Simple topology with 2 atoms in 1 residue
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        topology.add_atom("CA", md.element.carbon, residue)
        topology.add_atom("CB", md.element.carbon, residue)

        # Test trajectory with known fluctuations
        # Atom 0: fluctuates ±0.1 around origin
        # Atom 1: fluctuates ±0.1 around [1,0,0]
        self.coords = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Frame 0: mean positions
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],  # Frame 1: +0.1 X
            [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],  # Frame 2: +0.1 Y
            [[-0.1, 0.0, 0.0], [0.9, 0.0, 0.0]],  # Frame 3: -0.1 X
            [[0.0, -0.1, 0.0], [1.0, -0.1, 0.0]],  # Frame 4: -0.1 Y
        ], dtype=np.float32)
        self.traj = md.Trajectory(self.coords, topology)

        # Second trajectory for multi-trajectory tests
        self.coords2 = np.array([
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # Different base positions
            [[0.2, 0.0, 0.0], [2.2, 0.0, 0.0]],  # Larger fluctuations
            [[0.0, 0.2, 0.0], [2.0, 0.2, 0.0]],
        ], dtype=np.float32)
        self.traj2 = md.Trajectory(self.coords2, topology)

        # Topology with 6 atoms in 3 residues for per-residue tests
        topology_multi = md.Topology()
        chain = topology_multi.add_chain()
        for i in range(3):
            residue = topology_multi.add_residue(f"RES{i}", chain)
            topology_multi.add_atom(f"CA{i}", md.element.carbon, residue)
            topology_multi.add_atom(f"CB{i}", md.element.carbon, residue)

        # 6-atom trajectory for residue tests
        self.coords_multi = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.0, 0.0], [3.2, 0.0, 0.0], [4.3, 0.0, 0.0], [5.3, 0.0, 0.0]],
            [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [2.0, 0.2, 0.0], [3.0, 0.2, 0.0], [4.0, 0.3, 0.0], [5.0, 0.3, 0.0]],
        ], dtype=np.float32)
        self.traj_multi = md.Trajectory(self.coords_multi, topology_multi)

    def _build_unique_residue_traj(self, ref_idx: int, metric_idx: int, agg_idx: int) -> md.Trajectory:
        """Return deterministic trajectory for unique per-residue expectations.

        Parameters
        ----------
        ref_idx : int
            0 = mean reference path, 1 = median reference path (adds +0.03 base amplitude shift).
        metric_idx : int
            0 = mean metric, 1 = median metric (adds +0.015), 2 = mad metric (adds +0.03 and extended frames).
        agg_idx : int
            0 mean, 1 median, 2 rms, 3 rms_median (adds incremental +0.0/+0.007/+0.014/+0.021 amplitude shift).

        Construction summary:
        base_amplitude = 0.02 + ref_offset + metric_offset + agg_offset
        residue_offset = r * 0.011 (r in 0..2) to produce distinct residue scaling.
        Intra-residue atom scaling factors = (1.0, 1.45, 1.90) ensuring spread for mean/median/rms differences.
        Frame patterns by metric to diversify reductions:
        mean:   [0, +1, -1, +2, -2]
        median: [0, +1, +1, -1, +2, -2]
        mad:    [0, +1, +2, -1, +3, -2, +4]
        Coordinates vary only in X (Y,Z = 0) so distances are absolute x-amplitudes.
        """
        ref_offset = 0.0 if ref_idx == 0 else 0.03
        metric_offset = {0: 0.0, 1: 0.015, 2: 0.03}[metric_idx]
        agg_offset = {0: 0.0, 1: 0.007, 2: 0.014, 3: 0.021}[agg_idx]
        base = 0.02 + ref_offset + metric_offset + agg_offset

        # Build topology: 3 residues * 3 atoms
        topology = md.Topology()
        chain = topology.add_chain()
        n_res = 3
        atoms_per_res = 3
        for r in range(n_res):
            # UR is unique residue i
            # UA is unique atom j in residue i
            res = topology.add_residue(f"UR{r}", chain)
            for a in range(atoms_per_res):
                topology.add_atom(f"UA{r}{a}", md.element.carbon, res)

        # Atom scaling inside residue
        atom_scale = np.array([1.0, 1.45, 1.90], dtype=np.float32)
        amplitudes = []
        for r in range(n_res):
            residue_base = base + r * 0.011
            amplitudes.extend(residue_base * atom_scale)
        amplitudes = np.array(amplitudes, dtype=np.float32)  # length 9

        # Frame pattern selection
        if metric_idx == 0:      # mean metric
            pattern = [0, 1, -1, 2, -2]
        elif metric_idx == 1:    # median metric (duplicate +1)
            pattern = [0, 1, 1, -1, 2, -2]
        else:                    # mad metric (broader spread)
            pattern = [0, 1, 2, -1, 3, -2, 4]

        frames = [np.array(amplitudes) * p for p in pattern]
        coords = np.stack(frames, axis=0)[:, :, None]
        zeros = np.zeros_like(coords)
        coords = np.concatenate([coords, zeros, zeros], axis=2).astype(np.float32)
        return md.Trajectory(coords, topology)

    def make_unique_trajectory(self, n_frames: int, n_atoms: int = 1) -> md.Trajectory:
        """Create trajectory where frame i has coordinates [i, i, i].

        Parameters
        ----------
        n_frames : int
            Number of frames to create.
        n_atoms : int
            Number of atoms to create.

        Returns
        -------
        md.Trajectory
            Trajectory with unique coordinates per frame for exact testing.
        """
        topology = md.Topology()
        chain = topology.add_chain()
        for atom_idx in range(n_atoms):
            residue = topology.add_residue(f"RES{atom_idx}", chain)
            topology.add_atom(f"CA{atom_idx}", md.element.carbon, residue)

        coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        for frame in range(n_frames):
            for atom in range(n_atoms):
                coords[frame, atom] = [float(frame), float(frame), float(frame)]

        return md.Trajectory(coords, topology)

    def create_mad_test_data(self) -> md.Trajectory:
        """Create test trajectory specifically designed for non-zero MAD values.

        Returns
        -------
        md.Trajectory
            Trajectory with 2 atoms, 7 frames designed for MAD testing:
            - 30% baseline/small deviations
            - 30% medium deviations
            - 40% large outliers

        This guarantees MAD ≠ 0 because the median deviation is not zero.

        Manual MAD Calculation for Atom 0:
        Mean position: (0.0+0.1+0.0-0.1+0.4-0.4+0.2)/7 = 0.03
        Distances from mean: [0.03, 0.07, 0.03, 0.13, 0.37, 0.43, 0.17]
        Median distance: 0.13 (4th of 7 values when sorted)
        Absolute deviations from median: [0.10, 0.06, 0.10, 0.00, 0.24, 0.30, 0.04]
        MAD = median([0.10, 0.06, 0.10, 0.00, 0.24, 0.30, 0.04]) = 0.10 ≠ 0
        """
        topology = md.Topology()
        chain = topology.add_chain()
        for i in range(2):
            residue = topology.add_residue(f"RES{i}", chain)
            topology.add_atom(f"CA{i}", md.element.carbon, residue)

        # 7 frames with balanced distribution to ensure non-zero MAD
        coords = np.array([
            # Frame 0: baseline
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],

            # Frame 1: small deviation (+0.1)
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],

            # Frame 2: baseline
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],

            # Frame 3: small deviation (-0.1)
            [[-0.1, 0.0, 0.0], [0.9, 0.0, 0.0]],

            # Frame 4: LARGE outlier (+0.4)
            [[0.4, 0.0, 0.0], [1.4, 0.0, 0.0]],

            # Frame 5: LARGE outlier (-0.4)
            [[-0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],

            # Frame 6: medium deviation (+0.2)
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ], dtype=np.float32)

        return md.Trajectory(coords, topology)

    def create_aggregator_test_data(self) -> md.Trajectory:
        """Create test trajectory for per-residue aggregator testing.

        Returns
        -------
        md.Trajectory
            Trajectory with 6 atoms in 2 residues designed to show aggregator differences:
            - Residue 0: 3 atoms with RMSF values ~[0.1, 0.3, 0.9]
            - Residue 1: 3 atoms with RMSF values ~[0.2, 0.2, 0.8]

        Expected aggregator differences for Residue 0:
        - mean: (0.1+0.3+0.9)/3 = 0.43
        - median: median([0.1, 0.3, 0.9]) = 0.3
        - rms: sqrt((0.01+0.09+0.81)/3) = sqrt(0.303) = 0.55
        - rms_median: sqrt(median([0.01, 0.09, 0.81])) = sqrt(0.09) = 0.3

        This ensures different aggregators produce genuinely different results.
        """
        topology = md.Topology()
        chain = topology.add_chain()

        # Create 2 residues with 3 atoms each
        for res_idx in range(2):
            residue = topology.add_residue(f"RES{res_idx}", chain)
            for atom_idx in range(3):
                topology.add_atom(f"CA{res_idx}_{atom_idx}", md.element.carbon, residue)

        # 5 frames designed to create specific RMSF patterns
        coords = np.array([
            # Frame 0: baseline positions
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],    # Residue 0
             [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],   # Residue 1

            # Frame 1: Create target RMSF patterns
            # Res0: small(0.1), medium(0.3), large(0.9); Res1: medium(0.2), medium(0.2), very large(0.8)
            [[0.1, 0.0, 0.0], [1.0, 0.3, 0.0], [2.0, 0.0, 0.9],
             [3.0, 0.2, 0.0], [4.2, 0.0, 0.0], [5.0, 0.0, 0.8]],

            # Frame 2: Continue patterns (negative directions)
            [[-0.1, 0.0, 0.0], [1.0, -0.3, 0.0], [2.0, 0.0, -0.9],
             [3.0, -0.2, 0.0], [3.8, 0.0, 0.0], [5.0, 0.0, -0.8]],

            # Frame 3: Y-axis variations to maintain patterns
            [[0.0, 0.1, 0.0], [1.3, 0.0, 0.0], [2.9, 0.0, 0.0],
             [3.2, 0.0, 0.0], [4.0, 0.2, 0.0], [5.8, 0.0, 0.0]],

            # Frame 4: Complete the pattern
            [[0.0, -0.1, 0.0], [0.7, 0.0, 0.0], [1.1, 0.0, 0.0],
             [2.8, 0.0, 0.0], [4.0, -0.2, 0.0], [4.2, 0.0, 0.0]],
        ], dtype=np.float32)

        return md.Trajectory(coords, topology)

    # 1. Constructor Tests

    def test_init_valid_single_trajectory(self):
        """Test RMSFCalculator initialization with single trajectory.

        Verifies basic constructor functionality with minimal parameters.
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)
        assert len(calc.trajectories) == 1
        assert calc.trajectories[0] is self.traj
        assert calc.chunk_size == 10
        assert calc.use_memmap is False

    def test_init_valid_multiple_trajectories(self):
        """Test RMSFCalculator initialization with multiple trajectories.

        Verifies constructor handles multiple trajectory objects correctly.
        """
        calc = RMSFCalculator([self.traj, self.traj2], chunk_size=5, use_memmap=False)
        assert len(calc.trajectories) == 2
        assert calc.trajectories[0] is self.traj
        assert calc.trajectories[1] is self.traj2

    def test_init_empty_trajectories_error(self):
        """Test RMSFCalculator initialization with empty trajectory list.

        Should raise ValueError when no trajectories are provided.
        """
        with pytest.raises(ValueError, match="At least one trajectory must be provided"):
            RMSFCalculator([], chunk_size=10, use_memmap=False)

    def test_init_with_memmap_and_chunk_size(self):
        """Test RMSFCalculator initialization with memmap and chunk size.

        Verifies memmap and chunk_size parameters are correctly stored.
        """
                
        calc = RMSFCalculator([self.traj], chunk_size=2, use_memmap=True)
        assert calc.use_memmap is True
        assert calc.chunk_size == 2

    def test_init_auto_generated_trajectory_names(self):
        """Test automatic trajectory name generation.

        Verifies default naming convention for multiple trajectories.
        """
    
        calc = RMSFCalculator([self.traj, self.traj2], chunk_size=10, use_memmap=False)
        assert calc.trajectory_names == ["trajectory_0", "trajectory_1"]

    # 2. Per-Atom Tests

    def test_calculate_per_atom_mean_mean(self):
        """Test per-atom RMSF with mean reference and mean metric.

        Calculation walkthrough with symmetric test data (5 frames, 2 atoms):

        Input coordinates (using self.traj with symmetric fluctuations):
        Frame 0: [[0.0,0.0,0.0], [1.0,0.0,0.0]]  (baseline positions)
        Frame 1: [[0.1,0.0,0.0], [1.1,0.0,0.0]]  (+0.1 X direction)
        Frame 2: [[0.0,0.1,0.0], [1.0,0.1,0.0]]  (+0.1 Y direction)
        Frame 3: [[-0.1,0.0,0.0], [0.9,0.0,0.0]] (-0.1 X direction)
        Frame 4: [[0.0,-0.1,0.0], [1.0,-0.1,0.0]] (-0.1 Y direction)

        Step 1: Calculate MEAN positions (reference structure)
        Atom 0 coordinates:
        X: [0.0, 0.1, 0.0, -0.1, 0.0] => mean = 0.0/5 = 0.0
        Y: [0.0, 0.0, 0.1, 0.0, -0.1] => mean = 0.0/5 = 0.0
        Z: [0.0, 0.0, 0.0, 0.0, 0.0] => mean = 0.0/5 = 0.0
        Mean position atom 0: [0.0, 0.0, 0.0]

        Atom 1 coordinates:
        X: [1.0, 1.1, 1.0, 0.9, 1.0] => mean = 5.0/5 = 1.0
        Y: [0.0, 0.0, 0.1, 0.0, -0.1] => mean = 0.0/5 = 0.0
        Z: [0.0, 0.0, 0.0, 0.0, 0.0] => mean = 0.0/5 = 0.0
        Mean position atom 1: [1.0, 0.0, 0.0]

        Step 2: Calculate squared Euclidean distances from MEAN
        Atom 0 from [0.0,0.0,0.0]:
        Frame 0: distance² = (0.0-0.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.00
        Frame 1: distance² = (0.1-0.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.01
        Frame 2: distance² = (0.0-0.0)² + (0.1-0.0)² + (0.0-0.0)² = 0.01
        Frame 3: distance² = (-0.1-0.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.01
        Frame 4: distance² = (0.0-0.0)² + (-0.1-0.0)² + (0.0-0.0)² = 0.01
        Squared distances: [0.00, 0.01, 0.01, 0.01, 0.01]

        Atom 1 from [1.0,0.0,0.0]:
        Frame 0: distance² = (1.0-1.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.00
        Frame 1: distance² = (1.1-1.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.01
        Frame 2: distance² = (1.0-1.0)² + (0.1-0.0)² + (0.0-0.0)² = 0.01
        Frame 3: distance² = (0.9-1.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.01
        Frame 4: distance² = (1.0-1.0)² + (-0.1-0.0)² + (0.0-0.0)² = 0.01
        Squared distances: [0.00, 0.01, 0.01, 0.01, 0.01]

        Step 3: Calculate MSF (mean of squared fluctuations)
        MSF = mean([0.00, 0.01, 0.01, 0.01, 0.01]) = 0.04/5 = 0.008

        Step 4: Calculate RMSF = sqrt(MSF)
        RMSF = sqrt(0.008) = 0.089443

        Expected: Both atoms have identical symmetric fluctuations = [0.089443, 0.089443]
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", None, False)

        expected = [np.array([0.089443, 0.089443], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_calculate_per_atom_mean_median(self):
        """Test per-atom RMSF with mean reference and median metric.

        Calculation walkthrough with symmetric test data (5 frames, 2 atoms):

        Input coordinates (using self.traj with symmetric fluctuations):
        Frame 0: [[0.0,0.0,0.0], [1.0,0.0,0.0]]  (baseline positions)
        Frame 1: [[0.1,0.0,0.0], [1.1,0.0,0.0]]  (+0.1 X direction)
        Frame 2: [[0.0,0.1,0.0], [1.0,0.1,0.0]]  (+0.1 Y direction)
        Frame 3: [[-0.1,0.0,0.0], [0.9,0.0,0.0]] (-0.1 X direction)
        Frame 4: [[0.0,-0.1,0.0], [1.0,-0.1,0.0]] (-0.1 Y direction)

        Step 1: Calculate MEAN positions (reference structure - same as mean_mean)
        Atom 0 mean position: [0.0, 0.0, 0.0]
        Atom 1 mean position: [1.0, 0.0, 0.0]

        Step 2: Calculate squared Euclidean distances from MEAN (same as mean_mean)
        Atom 0 from [0.0,0.0,0.0]:
        Frame 0: distance² = 0.00
        Frame 1: distance² = (0.1)² = 0.01
        Frame 2: distance² = (0.1)² = 0.01
        Frame 3: distance² = (-0.1)² = 0.01
        Frame 4: distance² = (-0.1)² = 0.01
        Squared distances: [0.00, 0.01, 0.01, 0.01, 0.01]

        Atom 1 from [1.0,0.0,0.0]:
        Frame 0: distance² = 0.00
        Frame 1: distance² = (1.1-1.0)² = 0.01
        Frame 2: distance² = (0.1)² = 0.01
        Frame 3: distance² = (0.9-1.0)² = 0.01
        Frame 4: distance² = (-0.1)² = 0.01
        Squared distances: [0.00, 0.01, 0.01, 0.01, 0.01]

        Step 3: Calculate MSF (MEDIAN of squared fluctuations - differs from mean!)
        Sorted squared distances: [0.00, 0.01, 0.01, 0.01, 0.01]
        MSF = median([0.00, 0.01, 0.01, 0.01, 0.01]) = 0.01 (middle value of 5 elements)

        Step 4: Calculate RMSF = sqrt(MSF)
        RMSF = sqrt(0.01) = 0.1

        Expected: Mean reference with median metric = [0.1, 0.1]
        Note: Median metric is more robust to outliers than mean metric
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "median", None, False)

        expected = [np.array([0.1, 0.1], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_calculate_per_atom_mean_mad(self):
        """Test per-atom RMSF with mean reference and MAD metric.

        Calculation walkthrough with asymmetric test data (7 frames, 2 atoms):

        Input coordinates (using create_mad_test_data):
        Frame 0: [[0.0,0.0,0.0], [1.0,0.0,0.0]]  (baseline)
        Frame 1: [[0.1,0.0,0.0], [1.1,0.0,0.0]]  (+0.1 X)
        Frame 2: [[0.0,0.0,0.0], [1.0,0.0,0.0]]  (baseline)
        Frame 3: [[-0.1,0.0,0.0], [0.9,0.0,0.0]] (-0.1 X)
        Frame 4: [[0.4,0.0,0.0], [1.4,0.0,0.0]]  (OUTLIER +0.4)
        Frame 5: [[-0.4,0.0,0.0], [0.6,0.0,0.0]] (OUTLIER -0.4)
        Frame 6: [[0.2,0.0,0.0], [1.2,0.0,0.0]]  (+0.2 X)

        Step 1: Calculate MEAN positions (differs from median!)
        Atom 0 X coordinates: [0.0, 0.1, 0.0, -0.1, 0.4, -0.4, 0.2]
        Mean = (0.0+0.1+0.0-0.1+0.4-0.4+0.2)/7 = 0.2/7 = 0.029
        Atom 1 X coordinates: [1.0, 1.1, 1.0, 0.9, 1.4, 0.6, 1.2]
        Mean = (1.0+1.1+1.0+0.9+1.4+0.6+1.2)/7 = 8.2/7 = 1.171

        Step 2: Calculate Euclidean distances from MEAN
        Atom 0 from [0.029,0.0,0.0]: [0.029, 0.071, 0.029, 0.129, 0.371, 0.429, 0.171]
        Atom 1 from [1.171,0.0,0.0]: [0.171, 0.071, 0.171, 0.271, 0.229, 0.571, 0.029]

        Step 3: Find median distance for each atom
        Atom 0 sorted: [0.029, 0.029, 0.071, 0.129, 0.171, 0.371, 0.429] => median = 0.129
        Atom 1 sorted: [0.029, 0.071, 0.171, 0.171, 0.229, 0.271, 0.571] => median = 0.171

        Step 4: Calculate absolute deviations from median distance
        Atom 0: |[0.029, 0.071, 0.029, 0.129, 0.371, 0.429, 0.171] - 0.129|
               = [0.100, 0.058, 0.100, 0.000, 0.242, 0.300, 0.042]
        Atom 1: |[0.171, 0.071, 0.171, 0.271, 0.229, 0.571, 0.029] - 0.171|
               = [0.000, 0.100, 0.000, 0.100, 0.058, 0.400, 0.142]

        Step 5: Calculate MAD (median of absolute deviations)
        Atom 0 MAD = median([0.100, 0.058, 0.100, 0.000, 0.242, 0.300, 0.042])
                   = median of sorted [0.000, 0.042, 0.058, 0.100, 0.100, 0.242, 0.300] = 0.100
        Atom 1 MAD = median([0.000, 0.100, 0.000, 0.100, 0.058, 0.400, 0.142])
                   = median of sorted [0.000, 0.000, 0.058, 0.100, 0.100, 0.142, 0.400] = 0.100

        Expected: Both atoms have MAD = 0.1 due to balanced outlier distribution
        """
        traj = self.create_mad_test_data()
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mad", None, False)

        expected = [np.array([0.1, 0.1], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_calculate_per_atom_median_mean(self):
        """Test per-atom RMSF with median reference and mean metric.

        Calculation walkthrough with asymmetric test data (7 frames, 2 atoms):

        Input coordinates (using create_mad_test_data for asymmetric distribution):
        Frame 0: [[0.0,0.0,0.0], [1.0,0.0,0.0]]  (baseline)
        Frame 1: [[0.1,0.0,0.0], [1.1,0.0,0.0]]  (+0.1 X)
        Frame 2: [[0.0,0.0,0.0], [1.0,0.0,0.0]]  (baseline)
        Frame 3: [[-0.1,0.0,0.0], [0.9,0.0,0.0]] (-0.1 X)
        Frame 4: [[0.4,0.0,0.0], [1.4,0.0,0.0]]  (OUTLIER +0.4)
        Frame 5: [[-0.4,0.0,0.0], [0.6,0.0,0.0]] (OUTLIER -0.4)
        Frame 6: [[0.2,0.0,0.0], [1.2,0.0,0.0]]  (+0.2 X)

        Step 1: Calculate MEDIAN positions (robust to outliers!)
        Atom 0 X coordinates: [0.0, 0.1, 0.0, -0.1, 0.4, -0.4, 0.2]
        Sorted: [-0.4, -0.1, 0.0, 0.0, 0.1, 0.2, 0.4] => median = 0.0
        Atom 1 X coordinates: [1.0, 1.1, 1.0, 0.9, 1.4, 0.6, 1.2]
        Sorted: [0.6, 0.9, 1.0, 1.0, 1.1, 1.2, 1.4] => median = 1.0

        Step 2: Calculate squared Euclidean distances from MEDIAN
        Atom 0 from [0.0,0.0,0.0]: [0.0², 0.1², 0.0², 0.1², 0.4², 0.4², 0.2²]
                                   = [0.0, 0.01, 0.0, 0.01, 0.16, 0.16, 0.04]
        Atom 1 from [1.0,0.0,0.0]: [0.0², 0.1², 0.0², 0.1², 0.4², 0.4², 0.2²]
                                   = [0.0, 0.01, 0.0, 0.01, 0.16, 0.16, 0.04]

        Step 3: Calculate MSF (mean of squared fluctuations)
        MSF = mean([0.0, 0.01, 0.0, 0.01, 0.16, 0.16, 0.04])
            = (0.0+0.01+0.0+0.01+0.16+0.16+0.04)/7 = 0.38/7 = 0.0543

        Step 4: Calculate RMSF = sqrt(MSF)
        RMSF = sqrt(0.0543) = 0.233

        Expected: Median reference with mean metric = [0.233, 0.233]
        """
        traj = self.create_mad_test_data()
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("median", "mean", None, False)

        expected = [np.array([0.233, 0.233], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_atom_median_median(self):
        """Test per-atom RMSF with median reference and median metric.

        Calculation walkthrough with trajectory traj2 (3 frames, 2 atoms):

        Input coordinates (using self.traj2 for different fluctuation pattern):
        Frame 0: [[0.0,0.0,0.0], [2.0,0.0,0.0]]  (baseline positions)
        Frame 1: [[0.2,0.0,0.0], [2.2,0.0,0.0]]  (X-direction fluctuation)
        Frame 2: [[0.0,0.2,0.0], [2.0,0.2,0.0]]  (Y-direction fluctuation)

        Step 1: Calculate MEDIAN positions (center of distribution)
        Atom 0 coordinates:
        X: [0.0, 0.2, 0.0] => sorted: [0.0, 0.0, 0.2] => median = 0.0
        Y: [0.0, 0.0, 0.2] => sorted: [0.0, 0.0, 0.2] => median = 0.0
        Z: [0.0, 0.0, 0.0] => sorted: [0.0, 0.0, 0.0] => median = 0.0
        Median position atom 0: [0.0, 0.0, 0.0]

        Atom 1 coordinates:
        X: [2.0, 2.2, 2.0] => sorted: [2.0, 2.0, 2.2] => median = 2.0
        Y: [0.0, 0.0, 0.2] => sorted: [0.0, 0.0, 0.2] => median = 0.0
        Z: [0.0, 0.0, 0.0] => sorted: [0.0, 0.0, 0.0] => median = 0.0
        Median position atom 1: [2.0, 0.0, 0.0]

        Step 2: Calculate squared Euclidean distances from MEDIAN
        Atom 0 from [0.0,0.0,0.0]:
        Frame 0: distance² = (0.0-0.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.0
        Frame 1: distance² = (0.2-0.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.04
        Frame 2: distance² = (0.0-0.0)² + (0.2-0.0)² + (0.0-0.0)² = 0.04
        Squared distances: [0.0, 0.04, 0.04]

        Atom 1 from [2.0,0.0,0.0]:
        Frame 0: distance² = (2.0-2.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.0
        Frame 1: distance² = (2.2-2.0)² + (0.0-0.0)² + (0.0-0.0)² = 0.04
        Frame 2: distance² = (2.0-2.0)² + (0.2-0.0)² + (0.0-0.0)² = 0.04
        Squared distances: [0.0, 0.04, 0.04]

        Step 3: Calculate MSF (median of squared fluctuations)
        Both atoms: median([0.0, 0.04, 0.04]) = 0.04

        Step 4: Calculate RMSF = sqrt(MSF)
        RMSF = sqrt(0.04) = 0.2

        Expected: Both atoms median reference + median metric = [0.2, 0.2]
        """
        calc = RMSFCalculator([self.traj2], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("median", "median", None, False)

        expected = [np.array([0.2, 0.2], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_calculate_per_atom_median_mad(self):
        """Test per-atom RMSF with median reference and MAD metric.

        Calculation walkthrough creating custom asymmetric test trajectory (5 frames, 2 atoms):

        Custom coordinates designed for non-zero MAD with median reference:
        Frame 0: [[0.0,0.0,0.0], [3.0,0.0,0.0]]  (baseline positions)
        Frame 1: [[0.5,0.0,0.0], [3.5,0.0,0.0]]  (very large +X fluctuation)
        Frame 2: [[0.2,0.0,0.0], [3.2,0.0,0.0]]  (medium +X fluctuation)
        Frame 3: [[-0.1,0.0,0.0], [2.9,0.0,0.0]] (small -X fluctuation)
        Frame 4: [[-0.4,0.0,0.0], [2.6,0.0,0.0]] (large -X fluctuation)

        Step 1: Calculate MEDIAN positions
        Atom 0 X coordinates: [0.0, 0.5, 0.2, -0.1, -0.4]
        Sorted: [-0.4, -0.1, 0.0, 0.2, 0.5] => median = 0.0
        Median position atom 0: [0.0, 0.0, 0.0]

        Atom 1 X coordinates: [3.0, 3.5, 3.2, 2.9, 2.6]
        Sorted: [2.6, 2.9, 3.0, 3.2, 3.5] => median = 3.0
        Median position atom 1: [3.0, 0.0, 0.0]

        Step 2: Calculate Euclidean distances from MEDIAN
        Atom 0 from [0.0,0.0,0.0]: [0.0, 0.5, 0.2, 0.1, 0.4]
        Atom 1 from [3.0,0.0,0.0]: [0.0, 0.5, 0.2, 0.1, 0.4]

        Step 3: Find median distance for each atom
        Both atoms: sorted = [0.0, 0.1, 0.2, 0.4, 0.5] => median = 0.2

        Step 4: Calculate absolute deviations from median distance
        Both atoms: |[0.0, 0.5, 0.2, 0.1, 0.4] - 0.2| = [0.2, 0.3, 0.0, 0.1, 0.2]

        Step 5: Calculate MAD (median of absolute deviations)
        MAD = median([0.2, 0.3, 0.0, 0.1, 0.2])
        Sorted: [0.0, 0.1, 0.2, 0.2, 0.3] => median = 0.2

        Expected: Custom asymmetric trajectory with median+MAD = [0.2, 0.2] (unique non-zero result)
        """
        # Create custom trajectory for unique non-zero MAD result
        topology = md.Topology()
        chain = topology.add_chain()
        for i in range(2):
            residue = topology.add_residue(f"RES{i}", chain)
            topology.add_atom(f"CA{i}", md.element.carbon, residue)

        coords = np.array([
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],  # baseline
            [[0.5, 0.0, 0.0], [3.5, 0.0, 0.0]],  # very large +X
            [[0.2, 0.0, 0.0], [3.2, 0.0, 0.0]],  # medium +X
            [[-0.1, 0.0, 0.0], [2.9, 0.0, 0.0]], # small -X
            [[-0.4, 0.0, 0.0], [2.6, 0.0, 0.0]], # large -X
        ], dtype=np.float32)

        custom_traj = md.Trajectory(coords, topology)
        calc = RMSFCalculator([custom_traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("median", "mad", None, False)

        expected = [np.array([0.2, 0.2], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_calculate_per_atom_multiple_trajectories(self):
        """Test per-atom RMSF with multiple trajectories.

        Calculation:
        Traj1: MSF = 0.008, RMSF = 0.089443
        Traj2: Mean pos [0.0667, 0.0667, 0.0], [2.0667, 0.0667, 0.0]
               MSF = 0.0178, RMSF = 0.1333
        """
        calc = RMSFCalculator([self.traj, self.traj2], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", None, False)

        expected = [
            np.array([0.089443, 0.089443], dtype=np.float32),
            np.array([0.1333, 0.1333], dtype=np.float32)
        ]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_atom_cross_trajectory(self):
        """Test per-atom RMSF with cross-trajectory calculation.

        Calculation:
        Combined 8 frames: 5 from traj1 + 3 from traj2
        Mean pos atom 0: [0.025, 0.025, 0.0]
        Mean pos atom 1: [1.375, 0.025, 0.0]
        Cross-trajectory MSF and RMSF calculated over all frames
        RMSF = [0.117, 0.529]
        """
        calc = RMSFCalculator([self.traj, self.traj2], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", None, True)

        expected = [np.array([0.117, 0.529], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_atom_with_atom_indices(self):
        """Test per-atom RMSF with atom selection.

        Calculation:
        Select only first atom (index 0)
        Same calculation as full trajectory but only atom 0
        MSF = 0.008, RMSF = 0.089443
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", np.array([0]), False)

        expected = [np.array([0.08944], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_atom_with_multiple_atom_indices(self):
        """Test per-atom RMSF with multiple atom selection.

        Calculation:
        Select both atoms (indices [0, 1])
        Same as full trajectory calculation
        RMSF = [0.08944, 0.08944]
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", np.array([0, 1]), False)

        expected = [np.array([0.08944, 0.08944], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_atom_empty_trajectory(self):
        """Test per-atom RMSF with empty trajectory.

        Calculation:
        Empty trajectory has no frames
        Should raise ValueError with "No frames processed"
        Also expects RuntimeWarning from numpy operations on empty arrays
        """
        empty_coords = np.empty((0, 2, 3), dtype=np.float32)
        empty_traj = md.Trajectory(empty_coords, self.traj.topology)
        calc = RMSFCalculator([empty_traj], chunk_size=10, use_memmap=False)

        with pytest.warns(RuntimeWarning):
            with pytest.raises(ValueError, match="No frames processed"):
                calc.calculate_per_atom("mean", "mean", None, False)

    def test_calculate_per_atom_single_frame(self):
        """Test per-atom RMSF with single frame trajectory.

        Calculation:
        Single frame trajectory has no fluctuation
        Mean = frame coordinates, deviations = [0, 0]
        RMSF = [0.0, 0.0]
        """
        single_frame = md.Trajectory(self.coords[0:1], self.traj.topology)
        calc = RMSFCalculator([single_frame], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", None, False)

        expected = [np.array([0.0, 0.0], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_calculate_per_atom_invalid_reference_mode(self):
        """Test per-atom RMSF with invalid reference mode.

        Calculation:
        Invalid reference mode "invalid" should raise ValueError
        Expected error: "Unsupported reference mode, must be 'mean' or 'median'."
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)

        with pytest.raises(ValueError, match="Unsupported reference mode, must be 'mean' or 'median'"):
            calc.calculate_per_atom("invalid", "mean", None, False)

    def test_calculate_per_atom_invalid_metric(self):
        """Test per-atom RMSF with invalid metric.

        Invalid metric "invalid" should raise ValueError with appropriate message.
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)

        with pytest.raises(ValueError, match="Unsupported.*metric"):
            calc.calculate_per_atom("mean", "invalid", None, False)

    def test_calculate_per_atom_with_memmap(self):
        """Test per-atom RMSF with memmap enabled.

        Calculation:
        Same as basic mean_mean test but with memmap=True
        Memmap should not affect results
        RMSF = [0.08944, 0.08944]
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=True)
        result = calc.calculate_per_atom("mean", "mean", None, False)

        expected = [np.array([0.08944, 0.08944], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_atom_large_chunk_size(self):
        """Test per-atom RMSF with large chunk size.

        Calculation:
        Same as basic mean_mean test but with large chunk_size=1000
        Chunk size should not affect results
        RMSF = [0.08944, 0.08944]
        """
        calc = RMSFCalculator([self.traj], chunk_size=1000, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", None, False)

        expected = [np.array([0.08944, 0.08944], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_atom_small_chunk_size(self):
        """Test per-atom RMSF with small chunk size.

        Calculation:
        Same as basic mean_mean test but with small chunk_size=1
        Chunk size should not affect results
        RMSF = [0.08944, 0.08944]
        """
        calc = RMSFCalculator([self.traj], chunk_size=1, use_memmap=False)
        result = calc.calculate_per_atom("mean", "mean", None, False)

        expected = [np.array([0.08944, 0.08944], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_atom_out_of_bounds_atom_indices(self):
        """Test per-atom RMSF with out-of-bounds atom indices.

        Should raise IndexError for invalid atom indices.
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)

        with pytest.raises(IndexError):
            calc.calculate_per_atom("mean", "mean", np.array([10]), False)

    # 3. Per-Residue Tests
#---
    def test_calculate_per_residue_mean_mean_mean(self):
        """Per-residue RMSF calculation with mean reference, mean metric, and mean aggregation.

        Synthetic trajectory test case: mean_mean_mean
        Parameters: ref_idx=0, metric_idx=0, agg_idx=0

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with base parameters (no offsets)
        Base amplitude = 0.02 + 0.0 + 0.0 + 0.0 = 0.02
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.02000, 0.02900, 0.03800, 0.03100, 0.04495, 0.05890, 0.04200, 0.06090, 0.07980]
        Frame 2: [-0.02000, -0.02900, -0.03800, -0.03100, -0.04495, -0.05890, -0.04200, -0.06090, -0.07980]
        Frame 3: [0.04000, 0.05800, 0.07600, 0.06200, 0.08990, 0.11780, 0.08400, 0.12180, 0.15960]
        Frame 4: [-0.04000, -0.05800, -0.07600, -0.06200, -0.08990, -0.11780, -0.08400, -0.12180, -0.15960]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00040, 0.00084, 0.00144, 0.00096, 0.00202, 0.00347, 0.00176, 0.00371, 0.00637]
        Frame 2: [0.00040, 0.00084, 0.00144, 0.00096, 0.00202, 0.00347, 0.00176, 0.00371, 0.00637]
        Frame 3: [0.00160, 0.00336, 0.00578, 0.00384, 0.00808, 0.01388, 0.00706, 0.01484, 0.02547]
        Frame 4: [0.00160, 0.00336, 0.00578, 0.00384, 0.00808, 0.01388, 0.00706, 0.01484, 0.02547]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.00080, 0.00168, 0.00289, 0.00192, 0.00404, 0.00694, 0.00353, 0.00742, 0.01274]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.02828, 0.04101, 0.05374, 0.04384, 0.06357, 0.08330, 0.05940, 0.08613, 0.11285]

        Step 6: Aggregate per residue using mean aggregation
        Residue 0: mean([0.02828, 0.04101, 0.05374]) = 0.04101
        Residue 1: mean([0.04384, 0.06357, 0.08330]) = 0.06357
        Residue 2: mean([0.05940, 0.08613, 0.11285]) = 0.08613

        Expected: [0.0410, 0.0636, 0.0861]
        """
        traj = self._build_unique_residue_traj(0, 0, 0)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "mean", None, False, 0)
        expected = [np.array([0.0410, 0.0636, 0.0861], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_mean_median(self):
        """Per-residue RMSF calculation with mean reference, mean metric, and median aggregation.

        Synthetic trajectory test case: mean_mean_median
        Parameters: ref_idx=0, metric_idx=0, agg_idx=1

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with agg_idx=1 (+0.007 amplitude offset)
        Base amplitude = 0.02 + 0.0 + 0.0 + 0.007 = 0.027
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.02700, 0.03915, 0.05130, 0.03800, 0.05510, 0.07220, 0.04900, 0.07105, 0.09310]
        Frame 2: [-0.02700, -0.03915, -0.05130, -0.03800, -0.05510, -0.07220, -0.04900, -0.07105, -0.09310]
        Frame 3: [0.05400, 0.07830, 0.10260, 0.07600, 0.11020, 0.14440, 0.09800, 0.14210, 0.18620]
        Frame 4: [-0.05400, -0.07830, -0.10260, -0.07600, -0.11020, -0.14440, -0.09800, -0.14210, -0.18620]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00073, 0.00153, 0.00263, 0.00144, 0.00304, 0.00521, 0.00240, 0.00505, 0.00867]
        Frame 2: [0.00073, 0.00153, 0.00263, 0.00144, 0.00304, 0.00521, 0.00240, 0.00505, 0.00867]
        Frame 3: [0.00292, 0.00613, 0.01053, 0.00578, 0.01214, 0.02085, 0.00960, 0.02019, 0.03467]
        Frame 4: [0.00292, 0.00613, 0.01053, 0.00578, 0.01214, 0.02085, 0.00960, 0.02019, 0.03467]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.00146, 0.00307, 0.00526, 0.00289, 0.00607, 0.01043, 0.00480, 0.01010, 0.01734]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.03818, 0.05537, 0.07255, 0.05374, 0.07792, 0.10211, 0.06930, 0.10048, 0.13166]

        Step 6: Aggregate per residue using median aggregation
        Residue 0: median([0.03818, 0.05537, 0.07255]) = 0.05537 (middle value)
        Residue 1: median([0.05374, 0.07792, 0.10211]) = 0.07792 (middle value)
        Residue 2: median([0.06930, 0.10048, 0.13166]) = 0.10048 (middle value)

        Expected: [0.05537, 0.07792, 0.10048]

        Note: Median aggregation selects the middle RMSF value per residue, providing
        robust aggregation that dampens the effect of extreme atom fluctuations.
        """
        traj = self._build_unique_residue_traj(0, 0, 1)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "median", None, False, 0)
        expected = [np.array([0.0554, 0.0779, 0.1005], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_mean_rms(self):
        """Per-residue RMSF calculation with mean reference, mean metric, and RMS aggregation.

        Synthetic trajectory test case: mean_mean_rms
        Parameters: ref_idx=0, metric_idx=0, agg_idx=2

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with agg_idx=2 (+0.014 amplitude offset)
        Base amplitude = 0.02 + 0.0 + 0.0 + 0.014 = 0.034
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.03400, 0.04930, 0.06460, 0.04500, 0.06525, 0.08550, 0.05600, 0.08120, 0.10640]
        Frame 2: [-0.03400, -0.04930, -0.06460, -0.04500, -0.06525, -0.08550, -0.05600, -0.08120, -0.10640]
        Frame 3: [0.06800, 0.09860, 0.12920, 0.09000, 0.13050, 0.17100, 0.11200, 0.16240, 0.21280]
        Frame 4: [-0.06800, -0.09860, -0.12920, -0.09000, -0.13050, -0.17100, -0.11200, -0.16240, -0.21280]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00116, 0.00243, 0.00417, 0.00203, 0.00426, 0.00731, 0.00314, 0.00659, 0.01132]
        Frame 2: [0.00116, 0.00243, 0.00417, 0.00203, 0.00426, 0.00731, 0.00314, 0.00659, 0.01132]
        Frame 3: [0.00462, 0.00972, 0.01669, 0.00810, 0.01703, 0.02924, 0.01254, 0.02637, 0.04528]
        Frame 4: [0.00462, 0.00972, 0.01669, 0.00810, 0.01703, 0.02924, 0.01254, 0.02637, 0.04528]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.00231, 0.00486, 0.00835, 0.00405, 0.00852, 0.01462, 0.00627, 0.01319, 0.02264]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.04808, 0.06972, 0.09136, 0.06364, 0.09228, 0.12092, 0.07920, 0.11483, 0.15047]

        Step 6: Aggregate per residue using RMS aggregation
        RMS = sqrt(mean(values²)) - amplifies larger values relative to mean
        Residue 0: rms([0.04808, 0.06972, 0.09136]) = 0.07192
        Residue 1: rms([0.06364, 0.09228, 0.12092]) = 0.09519
        Residue 2: rms([0.07920, 0.11483, 0.15047]) = 0.11846

        Expected: [0.07192, 0.09519, 0.11846]

        Note: RMS aggregation amplifies larger RMSF values within each residue,
        providing higher sensitivity to atoms with extreme fluctuations compared to mean aggregation.
        """
        traj = self._build_unique_residue_traj(0, 0, 2)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "rms", None, False, 0)
        expected = [np.array([0.0719, 0.0952, 0.1185], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_mean_rms_median(self):
        """Per-residue RMSF calculation with mean reference, mean metric, and RMS_median aggregation.

        Synthetic trajectory test case: mean_mean_rms_median
        Parameters: ref_idx=0, metric_idx=0, agg_idx=3

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with agg_idx=3 (+0.021 amplitude offset)
        Base amplitude = 0.02 + 0.0 + 0.0 + 0.021 = 0.041
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.04100, 0.05945, 0.07790, 0.05200, 0.07540, 0.09880, 0.06300, 0.09135, 0.11970]
        Frame 2: [-0.04100, -0.05945, -0.07790, -0.05200, -0.07540, -0.09880, -0.06300, -0.09135, -0.11970]
        Frame 3: [0.08200, 0.11890, 0.15580, 0.10400, 0.15080, 0.19760, 0.12600, 0.18270, 0.23940]
        Frame 4: [-0.08200, -0.11890, -0.15580, -0.10400, -0.15080, -0.19760, -0.12600, -0.18270, -0.23940]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00168, 0.00353, 0.00607, 0.00270, 0.00569, 0.00976, 0.00397, 0.00834, 0.01433]
        Frame 2: [0.00168, 0.00353, 0.00607, 0.00270, 0.00569, 0.00976, 0.00397, 0.00834, 0.01433]
        Frame 3: [0.00672, 0.01414, 0.02427, 0.01082, 0.02274, 0.03905, 0.01588, 0.03338, 0.05731]
        Frame 4: [0.00672, 0.01414, 0.02427, 0.01082, 0.02274, 0.03905, 0.01588, 0.03338, 0.05731]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.00336, 0.00707, 0.01214, 0.00541, 0.01137, 0.01952, 0.00794, 0.01669, 0.02866]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.05798, 0.08407, 0.11017, 0.07354, 0.10663, 0.13972, 0.08910, 0.12919, 0.16928]

        Step 6: Aggregate per residue using RMS_median aggregation
        RMS_median = sqrt(median(values²)) - robust alternative to RMS
        Residue 0: rms_median([0.05798, 0.08407, 0.11017]) = 0.08407
        Residue 1: rms_median([0.07354, 0.10663, 0.13972]) = 0.10663
        Residue 2: rms_median([0.08910, 0.12919, 0.16928]) = 0.12919

        Expected: [0.08407, 0.10663, 0.12919]

        Note: RMS_median uses median of squared RMSF values before taking sqrt, providing
        robust aggregation that moderates extreme outliers while preserving more sensitivity
        than pure median aggregation. It's less prone to outlier inflation than pure RMS.
        """
        traj = self._build_unique_residue_traj(0, 0, 3)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "rms_median", None, False, 0)
        expected = [np.array([0.0841, 0.1066, 0.1292], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_median_mean(self):
        """Per-residue RMSF calculation with mean reference, median metric, and mean aggregation.

        Synthetic trajectory test case: mean_median_mean
        Parameters: ref_idx=0, metric_idx=1, agg_idx=0

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with metric_idx=1 (+0.015 amplitude offset)
        Base amplitude = 0.02 + 0.0 + 0.015 + 0.0 = 0.035
        Frame pattern (median metric): [0, +1, +1, -1, +2, -2] (6 frames total, duplicate +1)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.03500, 0.05075, 0.06650, 0.04600, 0.06670, 0.08740, 0.05700, 0.08265, 0.10830]
        Frame 2: [0.03500, 0.05075, 0.06650, 0.04600, 0.06670, 0.08740, 0.05700, 0.08265, 0.10830] (duplicate)
        Frame 3: [-0.03500, -0.05075, -0.06650, -0.04600, -0.06670, -0.08740, -0.05700, -0.08265, -0.10830]
        Frame 4: [0.07000, 0.10150, 0.13300, 0.09200, 0.13340, 0.17480, 0.11400, 0.16530, 0.21660]
        Frame 5: [-0.07000, -0.10150, -0.13300, -0.09200, -0.13340, -0.17480, -0.11400, -0.16530, -0.21660]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames ≠ [0.000, ...] due to asymmetric pattern
        Reference: [0.00583, 0.00846, 0.01108, 0.00767, 0.01112, 0.01457, 0.00950, 0.01378, 0.01805]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00003, 0.00007, 0.00012, 0.00006, 0.00012, 0.00021, 0.00009, 0.00019, 0.00033]
        Frame 1: [0.00085, 0.00179, 0.00307, 0.00147, 0.00309, 0.00530, 0.00226, 0.00474, 0.00815]
        Frame 2: [0.00085, 0.00179, 0.00307, 0.00147, 0.00309, 0.00530, 0.00226, 0.00474, 0.00815]
        Frame 3: [0.00167, 0.00351, 0.00602, 0.00288, 0.00606, 0.01040, 0.00442, 0.00930, 0.01596]
        Frame 4: [0.00412, 0.00866, 0.01486, 0.00711, 0.01495, 0.02567, 0.01092, 0.02296, 0.03942]
        Frame 5: [0.00575, 0.01209, 0.02076, 0.00993, 0.02089, 0.03586, 0.01525, 0.03207, 0.05506]

        Step 4: Apply median metric to squared distances => MSF per atom
        MSF per atom = median across frames of squared distances:
        [0.00126, 0.00265, 0.00455, 0.00217, 0.00457, 0.00785, 0.00334, 0.00702, 0.01205]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.03548, 0.05145, 0.06742, 0.04663, 0.06762, 0.08861, 0.05779, 0.08379, 0.10979]

        Step 6: Aggregate per residue using mean aggregation
        Residue 0: mean([0.03548, 0.05145, 0.06742]) = 0.05145
        Residue 1: mean([0.04663, 0.06762, 0.08861]) = 0.06762
        Residue 2: mean([0.05779, 0.08379, 0.10979]) = 0.08379

        Expected: [0.05145, 0.06762, 0.08379]

        Note: Median metric provides robust estimation by using median of squared distances
        instead of mean, making it less sensitive to outlier frames. The asymmetric frame
        pattern creates a non-zero reference structure, raising the baseline RMSF values.
        """
        traj = self._build_unique_residue_traj(0, 1, 0)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "median", "mean", None, False, 0)
        expected = [np.array([0.0515, 0.0676, 0.0838], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_median_median(self):
        """Per-residue RMSF calculation with mean reference, median metric, and median aggregation.

        Synthetic trajectory test case: mean_median_median
        Parameters: ref_idx=0, metric_idx=1, agg_idx=1

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with metric_idx=1 (+0.015) and agg_idx=1 (+0.007)
        Base amplitude = 0.02 + 0.0 + 0.015 + 0.007 = 0.042
        Frame pattern (median metric): [0, +1, +1, -1, +2, -2] (6 frames total, duplicate +1)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.04200, 0.06090, 0.07980, 0.05300, 0.07685, 0.10070, 0.06400, 0.09280, 0.12160]
        Frame 2: [0.04200, 0.06090, 0.07980, 0.05300, 0.07685, 0.10070, 0.06400, 0.09280, 0.12160] (duplicate)
        Frame 3: [-0.04200, -0.06090, -0.07980, -0.05300, -0.07685, -0.10070, -0.06400, -0.09280, -0.12160]
        Frame 4: [0.08400, 0.12180, 0.15960, 0.10600, 0.15370, 0.20140, 0.12800, 0.18560, 0.24320]
        Frame 5: [-0.08400, -0.12180, -0.15960, -0.10600, -0.15370, -0.20140, -0.12800, -0.18560, -0.24320]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames ≠ [0.000, ...] due to asymmetric pattern
        Reference: [0.00700, 0.01015, 0.01330, 0.00883, 0.01281, 0.01678, 0.01067, 0.01547, 0.02027]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00005, 0.00010, 0.00018, 0.00008, 0.00016, 0.00028, 0.00011, 0.00024, 0.00041]
        Frame 1: [0.00123, 0.00258, 0.00442, 0.00195, 0.00410, 0.00704, 0.00284, 0.00598, 0.01027]
        Frame 2: [0.00123, 0.00258, 0.00442, 0.00195, 0.00410, 0.00704, 0.00284, 0.00598, 0.01027]
        Frame 3: [0.00240, 0.00505, 0.00867, 0.00382, 0.00804, 0.01380, 0.00558, 0.01172, 0.02013]
        Frame 4: [0.00593, 0.01247, 0.02140, 0.00944, 0.01985, 0.03408, 0.01377, 0.02895, 0.04970]
        Frame 5: [0.00828, 0.01741, 0.02989, 0.01319, 0.02773, 0.04760, 0.01923, 0.04043, 0.06941]

        Step 4: Apply median metric to squared distances => MSF per atom
        MSF per atom = median across frames of squared distances:
        [0.00181, 0.00381, 0.00654, 0.00289, 0.00607, 0.01042, 0.00421, 0.00885, 0.01520]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.04258, 0.06174, 0.08090, 0.05373, 0.07791, 0.10209, 0.06488, 0.09408, 0.12328]

        Step 6: Aggregate per residue using median aggregation
        Residue 0: median([0.04258, 0.06174, 0.08090]) = 0.06174 (middle value)
        Residue 1: median([0.05373, 0.07791, 0.10209]) = 0.07791 (middle value)
        Residue 2: median([0.06488, 0.09408, 0.12328]) = 0.09408 (middle value)

        Expected: [0.06174, 0.07791, 0.09408]

        Note: Doubly robust - both median metric (resistant to outlier frames) and median
        aggregation (resistant to outlier atoms within residues) provide maximum robustness.
        """
        traj = self._build_unique_residue_traj(0, 1, 1)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "median", "median", None, False, 0)
        expected = [np.array([0.0617, 0.0779, 0.0941], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_median_rms(self):
        """Per-residue RMSF calculation with mean reference, median metric, and RMS aggregation.

        Synthetic trajectory test case: mean_median_rms
        Parameters: ref_idx=0, metric_idx=1, agg_idx=2

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with metric_idx=1 (+0.015) and agg_idx=2 (+0.014)
        Base amplitude = 0.02 + 0.0 + 0.015 + 0.014 = 0.049
        Frame pattern (median metric): [0, +1, +1, -1, +2, -2] (6 frames total, duplicate +1)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.04900, 0.07105, 0.09310, 0.06000, 0.08700, 0.11400, 0.07100, 0.10295, 0.13490]
        Frame 2: [0.04900, 0.07105, 0.09310, 0.06000, 0.08700, 0.11400, 0.07100, 0.10295, 0.13490] (duplicate)
        Frame 3: [-0.04900, -0.07105, -0.09310, -0.06000, -0.08700, -0.11400, -0.07100, -0.10295, -0.13490]
        Frame 4: [0.09800, 0.14210, 0.18620, 0.12000, 0.17400, 0.22800, 0.14200, 0.20590, 0.26980]
        Frame 5: [-0.09800, -0.14210, -0.18620, -0.12000, -0.17400, -0.22800, -0.14200, -0.20590, -0.26980]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames ≠ [0.000, ...] due to asymmetric pattern
        Reference: [0.00817, 0.01184, 0.01552, 0.01000, 0.01450, 0.01900, 0.01183, 0.01716, 0.02248]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00007, 0.00014, 0.00024, 0.00010, 0.00021, 0.00036, 0.00014, 0.00029, 0.00051]
        Frame 1: [0.00167, 0.00351, 0.00602, 0.00250, 0.00526, 0.00902, 0.00350, 0.00736, 0.01264]
        Frame 2: [0.00167, 0.00351, 0.00602, 0.00250, 0.00526, 0.00902, 0.00350, 0.00736, 0.01264]
        Frame 3: [0.00327, 0.00687, 0.01180, 0.00490, 0.01030, 0.01769, 0.00686, 0.01443, 0.02477]
        Frame 4: [0.00807, 0.01697, 0.02913, 0.01210, 0.02544, 0.04368, 0.01694, 0.03562, 0.06117]
        Frame 5: [0.01127, 0.02370, 0.04069, 0.01690, 0.03553, 0.06101, 0.02366, 0.04976, 0.08543]

        Step 4: Apply median metric to squared distances => MSF per atom
        MSF per atom = median across frames of squared distances:
        [0.00247, 0.00519, 0.00891, 0.00370, 0.00778, 0.01336, 0.00518, 0.01089, 0.01870]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.04968, 0.07203, 0.09438, 0.06083, 0.08820, 0.11557, 0.07198, 0.10437, 0.13676]

        Step 6: Aggregate per residue using RMS aggregation
        RMS = sqrt(mean(values²)) - amplifies larger values relative to mean/median
        Residue 0: rms([0.04968, 0.07203, 0.09438]) = 0.07431
        Residue 1: rms([0.06083, 0.08820, 0.11557]) = 0.09099
        Residue 2: rms([0.07198, 0.10437, 0.13676]) = 0.10767

        Expected: [0.07431, 0.09099, 0.10767]

        Note: RMS aggregation re-inflates the spread after median metric has provided
        robust per-atom estimates, emphasizing atoms with larger fluctuations within residues.
        """
        traj = self._build_unique_residue_traj(0, 1, 2)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "median", "rms", None, False, 0)
        expected = [np.array([0.0743, 0.0910, 0.1077], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_median_rms_median(self):
        """Per-residue RMSF calculation with mean reference, median metric, and RMS_median aggregation.

        Synthetic trajectory test case: mean_median_rms_median
        Parameters: ref_idx=0, metric_idx=1, agg_idx=3

        Expected: [0.08232, 0.09849, 0.11466]

        Note: RMS_median aggregation uses sqrt(median(values²)), providing robust
        aggregation that moderates extreme outliers while preserving sensitivity
        better than pure median. Less prone to outlier inflation than pure RMS.
        """
        traj = self._build_unique_residue_traj(0, 1, 3)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "median", "rms_median", None, False, 0)
        expected = [np.array([0.0823, 0.0985, 0.1147], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_mad_mean(self):
        """Per-residue RMSF calculation with mean reference, MAD metric, and mean aggregation.

        Synthetic trajectory test case: mean_mad_mean
        Parameters: ref_idx=0, metric_idx=2, agg_idx=0

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with metric_idx=2 (+0.030 amplitude offset)
        Base amplitude = 0.02 + 0.0 + 0.030 + 0.0 = 0.050
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, extended)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.05000, 0.07250, 0.09500, 0.06100, 0.08845, 0.11590, 0.07200, 0.10440, 0.13680]
        Frame 2: [0.10000, 0.14500, 0.19000, 0.12200, 0.17690, 0.23180, 0.14400, 0.20880, 0.27360]
        Frame 3: [-0.05000, -0.07250, -0.09500, -0.06100, -0.08845, -0.11590, -0.07200, -0.10440, -0.13680]
        Frame 4: [0.15000, 0.21750, 0.28500, 0.18300, 0.26535, 0.34770, 0.21600, 0.31320, 0.41040]
        Frame 5: [-0.10000, -0.14500, -0.19000, -0.12200, -0.17690, -0.23180, -0.14400, -0.20880, -0.27360]
        Frame 6: [0.20000, 0.29000, 0.38000, 0.24400, 0.35380, 0.46360, 0.28800, 0.41760, 0.54720]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames = [0.05000, 0.07250, 0.09500, 0.06100, 0.08845, 0.11590, 0.07200, 0.10440, 0.13680]

        Step 3: Calculate Euclidean distances from reference per frame
        Since MAD uses Euclidean distances (not squared), we calculate |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.05000, 0.07250, 0.09500, 0.06100, 0.08845, 0.11590, 0.07200, 0.10440, 0.13680]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.05000, 0.07250, 0.09500, 0.06100, 0.08845, 0.11590, 0.07200, 0.10440, 0.13680]
        Frame 3: [0.10000, 0.14500, 0.19000, 0.12200, 0.17690, 0.23180, 0.14400, 0.20880, 0.27360]
        Frame 4: [0.10000, 0.14500, 0.19000, 0.12200, 0.17690, 0.23180, 0.14400, 0.20880, 0.27360]
        Frame 5: [0.15000, 0.21750, 0.28500, 0.18300, 0.26535, 0.34770, 0.21600, 0.31320, 0.41040]
        Frame 6: [0.15000, 0.21750, 0.28500, 0.18300, 0.26535, 0.34770, 0.21600, 0.31320, 0.41040]

        Step 4: Calculate MAD per atom (Median Absolute Deviation)
        For each atom: MAD = median(|distances - median(distances)|)
        Atom 0: median=0.10000, MAD=0.05000
        Atom 1: median=0.14500, MAD=0.07250
        Atom 2: median=0.19000, MAD=0.09500
        Atom 3: median=0.12200, MAD=0.06100
        Atom 4: median=0.17690, MAD=0.08845
        Atom 5: median=0.23180, MAD=0.11590
        Atom 6: median=0.14400, MAD=0.07200
        Atom 7: median=0.20880, MAD=0.10440
        Atom 8: median=0.27360, MAD=0.13680
        Per-atom MAD (=RMSF): [0.05000, 0.07250, 0.09500, 0.06100, 0.08845, 0.11590, 0.07200, 0.10440, 0.13680]

        Step 5: Aggregate per residue using mean aggregation
        Residue 0: mean([0.05000, 0.07250, 0.09500]) = 0.07250
        Residue 1: mean([0.06100, 0.08845, 0.11590]) = 0.08845
        Residue 2: mean([0.07200, 0.10440, 0.13680]) = 0.10440

        Expected: [0.07250, 0.08845, 0.10440]

        Note: MAD metric provides robust estimation by using Median Absolute Deviation
        of Euclidean distances (not squared distances), making it highly resistant to
        outliers. The extended 7-frame pattern provides more data points for robust MAD estimation.
        """
        traj = self._build_unique_residue_traj(0, 2, 0)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mad", "mean", None, False, 0)
        expected = [np.array([0.0725, 0.0884, 0.1044], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_mad_median(self):
        """Per-residue RMSF calculation with mean reference, MAD metric, and median aggregation.

        Synthetic trajectory test case: mean_mad_median
        Parameters: ref_idx=0, metric_idx=2, agg_idx=1

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with metric_idx=2 (+0.030) and agg_idx=1 (+0.007)
        Base amplitude = 0.02 + 0.0 + 0.030 + 0.007 = 0.057
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, extended)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.05700, 0.08265, 0.10830, 0.06800, 0.09860, 0.12920, 0.07900, 0.11455, 0.15010]
        Frame 2: [0.11400, 0.16530, 0.21660, 0.13600, 0.19720, 0.25840, 0.15800, 0.22910, 0.30020]
        Frame 3: [-0.05700, -0.08265, -0.10830, -0.06800, -0.09860, -0.12920, -0.07900, -0.11455, -0.15010]
        Frame 4: [0.17100, 0.24795, 0.32490, 0.20400, 0.29580, 0.38760, 0.23700, 0.34365, 0.45030]
        Frame 5: [-0.11400, -0.16530, -0.21660, -0.13600, -0.19720, -0.25840, -0.15800, -0.22910, -0.30020]
        Frame 6: [0.22800, 0.33060, 0.43320, 0.27200, 0.39440, 0.51680, 0.31600, 0.45820, 0.60040]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames = [0.05700, 0.08265, 0.10830, 0.06800, 0.09860, 0.12920, 0.07900, 0.11455, 0.15010]

        Step 3: Calculate Euclidean distances from reference per frame
        Since MAD uses Euclidean distances (not squared), we calculate |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.05700, 0.08265, 0.10830, 0.06800, 0.09860, 0.12920, 0.07900, 0.11455, 0.15010]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.05700, 0.08265, 0.10830, 0.06800, 0.09860, 0.12920, 0.07900, 0.11455, 0.15010]
        Frame 3: [0.11400, 0.16530, 0.21660, 0.13600, 0.19720, 0.25840, 0.15800, 0.22910, 0.30020]
        Frame 4: [0.11400, 0.16530, 0.21660, 0.13600, 0.19720, 0.25840, 0.15800, 0.22910, 0.30020]
        Frame 5: [0.17100, 0.24795, 0.32490, 0.20400, 0.29580, 0.38760, 0.23700, 0.34365, 0.45030]
        Frame 6: [0.17100, 0.24795, 0.32490, 0.20400, 0.29580, 0.38760, 0.23700, 0.34365, 0.45030]

        Step 4: Calculate MAD per atom (Median Absolute Deviation)
        For each atom: MAD = median(|distances - median(distances)|)
        Atom 0: median=0.11400, MAD=0.05700
        Atom 1: median=0.16530, MAD=0.08265
        Atom 2: median=0.21660, MAD=0.10830
        Atom 3: median=0.13600, MAD=0.06800
        Atom 4: median=0.19720, MAD=0.09860
        Atom 5: median=0.25840, MAD=0.12920
        Atom 6: median=0.15800, MAD=0.07900
        Atom 7: median=0.22910, MAD=0.11455
        Atom 8: median=0.30020, MAD=0.15010
        Per-atom MAD (=RMSF): [0.05700, 0.08265, 0.10830, 0.06800, 0.09860, 0.12920, 0.07900, 0.11455, 0.15010]

        Step 5: Aggregate per residue using median aggregation
        Residue 0: median([0.05700, 0.08265, 0.10830]) = 0.08265 (middle value)
        Residue 1: median([0.06800, 0.09860, 0.12920]) = 0.09860 (middle value)
        Residue 2: median([0.07900, 0.11455, 0.15010]) = 0.11455 (middle value)

        Expected: [0.08265, 0.09860, 0.11455]

        Note: Doubly robust - MAD metric (resistant to outlier frames) + median
        aggregation (resistant to outlier atoms) provides maximum robustness while
        dampening the larger MAD values compared to mean aggregation.
        """
        traj = self._build_unique_residue_traj(0, 2, 1)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mad", "median", None, False, 0)
        expected = [np.array([0.0827, 0.0986, 0.1146], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_mad_rms(self):
        """Per-residue RMSF calculation with mean reference, MAD metric, and RMS aggregation.

        Synthetic trajectory test case: mean_mad_rms
        Parameters: ref_idx=0, metric_idx=2, agg_idx=2

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with metric_idx=2 (+0.03) and agg_idx=2 (+0.014)
        Base amplitude = 0.02 + 0.0 + 0.03 + 0.014 = 0.064
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, broader spread)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.06400, 0.09280, 0.12160, 0.07500, 0.10875, 0.14250, 0.08600, 0.12470, 0.16340]
        Frame 2: [0.12800, 0.18560, 0.24320, 0.15000, 0.21750, 0.28500, 0.17200, 0.24940, 0.32680]
        Frame 3: [-0.06400, -0.09280, -0.12160, -0.07500, -0.10875, -0.14250, -0.08600, -0.12470, -0.16340]
        Frame 4: [0.19200, 0.27840, 0.36480, 0.22500, 0.32625, 0.42750, 0.25800, 0.37410, 0.49020]
        Frame 5: [-0.12800, -0.18560, -0.24320, -0.15000, -0.21750, -0.28500, -0.17200, -0.24940, -0.32680]
        Frame 6: [0.25600, 0.37120, 0.48640, 0.30000, 0.43500, 0.57000, 0.34400, 0.49880, 0.65360]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames
        Reference: [0.06400, 0.09280, 0.12160, 0.07500, 0.10875, 0.14250, 0.08600, 0.12470, 0.16340]

        Step 3: Calculate Euclidean distances from reference per frame
        Since Y=Z=0, Euclidean distance = |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.06400, 0.09280, 0.12160, 0.07500, 0.10875, 0.14250, 0.08600, 0.12470, 0.16340]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.06400, 0.09280, 0.12160, 0.07500, 0.10875, 0.14250, 0.08600, 0.12470, 0.16340]
        Frame 3: [0.12800, 0.18560, 0.24320, 0.15000, 0.21750, 0.28500, 0.17200, 0.24940, 0.32680]
        Frame 4: [0.12800, 0.18560, 0.24320, 0.15000, 0.21750, 0.28500, 0.17200, 0.24940, 0.32680]
        Frame 5: [0.19200, 0.27840, 0.36480, 0.22500, 0.32625, 0.42750, 0.25800, 0.37410, 0.49020]
        Frame 6: [0.19200, 0.27840, 0.36480, 0.22500, 0.32625, 0.42750, 0.25800, 0.37410, 0.49020]

        Step 4: Apply MAD (Median Absolute Deviation) metric => per-atom RMSF
        For each atom, compute MAD = median(|distances - median(distances)|):
        Atom 0: median=0.12800, MAD=0.06400
        Atom 1: median=0.18560, MAD=0.09280
        Atom 2: median=0.24320, MAD=0.12160
        Atom 3: median=0.15000, MAD=0.07500
        Atom 4: median=0.21750, MAD=0.10875
        Atom 5: median=0.28500, MAD=0.14250
        Atom 6: median=0.17200, MAD=0.08600
        Atom 7: median=0.24940, MAD=0.12470
        Atom 8: median=0.32680, MAD=0.16340

        Per-atom RMSF (=MAD values): [0.06400, 0.09280, 0.12160, 0.07500, 0.10875, 0.14250, 0.08600, 0.12470, 0.16340]

        Step 5: Aggregate per residue using RMS aggregation
        RMS = sqrt(mean(values²)) for each residue
        Residue 0: rms([0.06400, 0.09280, 0.12160]) = sqrt(mean([0.06400², 0.09280², 0.12160²])) = sqrt(mean([0.00410, 0.00861, 0.01479])) = sqrt(0.00917) = 0.09573
        Residue 1: rms([0.07500, 0.10875, 0.14250]) = sqrt(mean([0.07500², 0.10875², 0.14250²])) = sqrt(mean([0.00562, 0.01182, 0.02030])) = sqrt(0.01258) = 0.11219
        Residue 2: rms([0.08600, 0.12470, 0.16340]) = sqrt(mean([0.08600², 0.12470², 0.16340²])) = sqrt(mean([0.00740, 0.01555, 0.02670])) = sqrt(0.01655) = 0.12864

        Expected: [0.09573, 0.11219, 0.12864]

        Note: MAD metric with RMS aggregation - MAD provides robust outlier resistance while
        RMS amplifies the aggregated spread across atoms within residues.
        """
        traj = self._build_unique_residue_traj(0, 2, 2)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mad", "rms", None, False, 0)
        expected = [np.array([0.0957, 0.1122, 0.1286], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_mean_mad_rms_median(self):
        """Per-residue RMSF calculation with mean reference, MAD metric, and RMS_median aggregation.

        Synthetic trajectory test case: mean_mad_rms_median
        Parameters: ref_idx=0, metric_idx=2, agg_idx=3

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with metric_idx=2 (+0.03) and agg_idx=3 (+0.021)
        Base amplitude = 0.02 + 0.0 + 0.03 + 0.021 = 0.071
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, broader spread)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.07100, 0.10295, 0.13490, 0.08200, 0.11890, 0.15580, 0.09300, 0.13485, 0.17670]
        Frame 2: [0.14200, 0.20590, 0.26980, 0.16400, 0.23780, 0.31160, 0.18600, 0.26970, 0.35340]
        Frame 3: [-0.07100, -0.10295, -0.13490, -0.08200, -0.11890, -0.15580, -0.09300, -0.13485, -0.17670]
        Frame 4: [0.21300, 0.30885, 0.40470, 0.24600, 0.35670, 0.46740, 0.27900, 0.40455, 0.53010]
        Frame 5: [-0.14200, -0.20590, -0.26980, -0.16400, -0.23780, -0.31160, -0.18600, -0.26970, -0.35340]
        Frame 6: [0.28400, 0.41180, 0.53960, 0.32800, 0.47560, 0.62320, 0.37200, 0.53940, 0.70680]

        Step 2: Calculate reference structure (mean across all frames)
        Reference coordinates = mean of all frames
        Reference: [0.07100, 0.10295, 0.13490, 0.08200, 0.11890, 0.15580, 0.09300, 0.13485, 0.17670]

        Step 3: Calculate Euclidean distances from reference per frame
        Since Y=Z=0, Euclidean distance = |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.07100, 0.10295, 0.13490, 0.08200, 0.11890, 0.15580, 0.09300, 0.13485, 0.17670]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.07100, 0.10295, 0.13490, 0.08200, 0.11890, 0.15580, 0.09300, 0.13485, 0.17670]
        Frame 3: [0.14200, 0.20590, 0.26980, 0.16400, 0.23780, 0.31160, 0.18600, 0.26970, 0.35340]
        Frame 4: [0.14200, 0.20590, 0.26980, 0.16400, 0.23780, 0.31160, 0.18600, 0.26970, 0.35340]
        Frame 5: [0.21300, 0.30885, 0.40470, 0.24600, 0.35670, 0.46740, 0.27900, 0.40455, 0.53010]
        Frame 6: [0.21300, 0.30885, 0.40470, 0.24600, 0.35670, 0.46740, 0.27900, 0.40455, 0.53010]

        Step 4: Apply MAD (Median Absolute Deviation) metric => per-atom RMSF
        For each atom, compute MAD = median(|distances - median(distances)|):
        Atom 0: median=0.14200, MAD=0.07100
        Atom 1: median=0.20590, MAD=0.10295
        Atom 2: median=0.26980, MAD=0.13490
        Atom 3: median=0.16400, MAD=0.08200
        Atom 4: median=0.23780, MAD=0.11890
        Atom 5: median=0.31160, MAD=0.15580
        Atom 6: median=0.18600, MAD=0.09300
        Atom 7: median=0.26970, MAD=0.13485
        Atom 8: median=0.35340, MAD=0.17670

        Per-atom RMSF (=MAD values): [0.07100, 0.10295, 0.13490, 0.08200, 0.11890, 0.15580, 0.09300, 0.13485, 0.17670]

        Step 5: Aggregate per residue using RMS_median aggregation
        RMS_median = sqrt(median(values²)) for each residue
        Residue 0: rms_median([0.07100, 0.10295, 0.13490]) = sqrt(median([0.07100², 0.10295², 0.13490²])) = sqrt(median([0.00504, 0.01060, 0.01820])) = sqrt(0.01060) = 0.10295
        Residue 1: rms_median([0.08200, 0.11890, 0.15580]) = sqrt(median([0.08200², 0.11890², 0.15580²])) = sqrt(median([0.00672, 0.01414, 0.02428])) = sqrt(0.01414) = 0.11890
        Residue 2: rms_median([0.09300, 0.13485, 0.17670]) = sqrt(median([0.09300², 0.13485², 0.17670²])) = sqrt(median([0.00865, 0.01819, 0.03122])) = sqrt(0.01819) = 0.13485

        Expected: [0.10295, 0.11890, 0.13485]

        Note: MAD metric with RMS_median aggregation - MAD provides robust outlier resistance while
        RMS_median uses the median of squared values, providing moderate amplification.
        """
        traj = self._build_unique_residue_traj(0, 2, 3)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mad", "rms_median", None, False, 0)
        expected = [np.array([0.1030, 0.1189, 0.1349], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mean_mean(self):
        """Per-residue RMSF calculation with median reference, mean metric, and mean aggregation.

        Synthetic trajectory test case: median_mean_mean
        Parameters: ref_idx=1, metric_idx=0, agg_idx=0

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=0 (+0.0), and agg_idx=0 (+0.0)
        Base amplitude = 0.02 + 0.03 + 0.0 + 0.0 = 0.05
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total, symmetric)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.05000, 0.07250, 0.09500, 0.06100, 0.08845, 0.11590, 0.07200, 0.10440, 0.13680]
        Frame 2: [-0.05000, -0.07250, -0.09500, -0.06100, -0.08845, -0.11590, -0.07200, -0.10440, -0.13680]
        Frame 3: [0.10000, 0.14500, 0.19000, 0.12200, 0.17690, 0.23180, 0.14400, 0.20880, 0.27360]
        Frame 4: [-0.10000, -0.14500, -0.19000, -0.12200, -0.17690, -0.23180, -0.14400, -0.20880, -0.27360]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (with symmetric pattern, median = zero)
        Reference: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00250, 0.00526, 0.00903, 0.00372, 0.00782, 0.01343, 0.00518, 0.01090, 0.01871]
        Frame 2: [0.00250, 0.00526, 0.00903, 0.00372, 0.00782, 0.01343, 0.00518, 0.01090, 0.01871]
        Frame 3: [0.01000, 0.02103, 0.03610, 0.01488, 0.03129, 0.05373, 0.02074, 0.04360, 0.07486]
        Frame 4: [0.01000, 0.02103, 0.03610, 0.01488, 0.03129, 0.05373, 0.02074, 0.04360, 0.07486]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.00500, 0.01051, 0.01805, 0.00744, 0.01565, 0.02687, 0.01037, 0.02180, 0.03743]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.07071, 0.10253, 0.13435, 0.08627, 0.12509, 0.16391, 0.10182, 0.14764, 0.19346]

        Step 6: Aggregate per residue using mean aggregation
        Residue 0: mean([0.07071, 0.10253, 0.13435]) = (0.07071 + 0.10253 + 0.13435) / 3 = 0.10253
        Residue 1: mean([0.08627, 0.12509, 0.16391]) = (0.08627 + 0.12509 + 0.16391) / 3 = 0.12509
        Residue 2: mean([0.10182, 0.14764, 0.19346]) = (0.10182 + 0.14764 + 0.19346) / 3 = 0.14764

        Expected: [0.10253, 0.12509, 0.14764]

        Note: Median reference with mean metric and mean aggregation - median reference provides
        natural centering while mean metric and aggregation preserve all fluctuation information.
        """
        traj = self._build_unique_residue_traj(1, 0, 0)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mean", "mean", None, False, 0)
        expected = [np.array([0.1025, 0.1251, 0.1476], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mean_median(self):
        """Per-residue RMSF calculation with median reference, mean metric, and median aggregation.

        Synthetic trajectory test case: median_mean_median
        Parameters: ref_idx=1, metric_idx=0, agg_idx=1

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=0 (+0.0), and agg_idx=1 (+0.007)
        Base amplitude = 0.02 + 0.03 + 0.0 + 0.007 = 0.057
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total, symmetric)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.05700, 0.08265, 0.10830, 0.06800, 0.09860, 0.12920, 0.07900, 0.11455, 0.15010]
        Frame 2: [-0.05700, -0.08265, -0.10830, -0.06800, -0.09860, -0.12920, -0.07900, -0.11455, -0.15010]
        Frame 3: [0.11400, 0.16530, 0.21660, 0.13600, 0.19720, 0.25840, 0.15800, 0.22910, 0.30020]
        Frame 4: [-0.11400, -0.16530, -0.21660, -0.13600, -0.19720, -0.25840, -0.15800, -0.22910, -0.30020]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (with symmetric pattern, median = zero)
        Reference: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00325, 0.00683, 0.01173, 0.00462, 0.00972, 0.01669, 0.00624, 0.01312, 0.02253]
        Frame 2: [0.00325, 0.00683, 0.01173, 0.00462, 0.00972, 0.01669, 0.00624, 0.01312, 0.02253]
        Frame 3: [0.01300, 0.02732, 0.04692, 0.01850, 0.03889, 0.06677, 0.02496, 0.05249, 0.09012]
        Frame 4: [0.01300, 0.02732, 0.04692, 0.01850, 0.03889, 0.06677, 0.02496, 0.05249, 0.09012]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.00650, 0.01366, 0.02346, 0.00925, 0.01944, 0.03339, 0.01248, 0.02624, 0.04506]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.08061, 0.11688, 0.15316, 0.09617, 0.13944, 0.18272, 0.11172, 0.16200, 0.21227]

        Step 6: Aggregate per residue using median aggregation
        Residue 0: median([0.08061, 0.11688, 0.15316]) = 0.11688 (middle value)
        Residue 1: median([0.09617, 0.13944, 0.18272]) = 0.13944 (middle value)
        Residue 2: median([0.11172, 0.16200, 0.21227]) = 0.16200 (middle value)

        Expected: [0.11688, 0.13944, 0.16200]

        Note: Median reference with mean metric and median aggregation - median reference provides
        natural centering, mean metric preserves all fluctuation information, median aggregation
        dampens the effect of outlier atoms within residues.
        """
        traj = self._build_unique_residue_traj(1, 0, 1)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mean", "median", None, False, 0)
        expected = [np.array([0.1169, 0.1394, 0.1620], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mean_rms(self):
        """Per-residue RMSF calculation with median reference, mean metric, and RMS aggregation.

        Synthetic trajectory test case: median_mean_rms
        Parameters: ref_idx=1, metric_idx=0, agg_idx=2

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=0 (+0.0), and agg_idx=2 (+0.014)
        Base amplitude = 0.02 + 0.03 + 0.0 + 0.014 = 0.064
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total, symmetric)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.06400, 0.09280, 0.12160, 0.07500, 0.10875, 0.14250, 0.08600, 0.12470, 0.16340]
        Frame 2: [-0.06400, -0.09280, -0.12160, -0.07500, -0.10875, -0.14250, -0.08600, -0.12470, -0.16340]
        Frame 3: [0.12800, 0.18560, 0.24320, 0.15000, 0.21750, 0.28500, 0.17200, 0.24940, 0.32680]
        Frame 4: [-0.12800, -0.18560, -0.24320, -0.15000, -0.21750, -0.28500, -0.17200, -0.24940, -0.32680]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (with symmetric pattern, median = zero)
        Reference: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00410, 0.00861, 0.01479, 0.00563, 0.01183, 0.02031, 0.00740, 0.01555, 0.02670]
        Frame 2: [0.00410, 0.00861, 0.01479, 0.00563, 0.01183, 0.02031, 0.00740, 0.01555, 0.02670]
        Frame 3: [0.01638, 0.03445, 0.05915, 0.02250, 0.04731, 0.08123, 0.02958, 0.06220, 0.10680]
        Frame 4: [0.01638, 0.03445, 0.05915, 0.02250, 0.04731, 0.08123, 0.02958, 0.06220, 0.10680]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.00819, 0.01722, 0.02957, 0.01125, 0.02365, 0.04061, 0.01479, 0.03110, 0.05340]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.09051, 0.13124, 0.17197, 0.10607, 0.15380, 0.20153, 0.12162, 0.17635, 0.23108]

        Step 6: Aggregate per residue using RMS aggregation
        RMS = sqrt(mean(values²)) for each residue
        Residue 0: rms([0.09051, 0.13124, 0.17197]) = sqrt(mean([0.09051², 0.13124², 0.17197²])) = sqrt(mean([0.00819, 0.01722, 0.02957])) = sqrt(0.01833) = 0.13539
        Residue 1: rms([0.10607, 0.15380, 0.20153]) = sqrt(mean([0.10607², 0.15380², 0.20153²])) = sqrt(mean([0.01125, 0.02366, 0.04061])) = sqrt(0.02517) = 0.15866
        Residue 2: rms([0.12162, 0.17635, 0.23108]) = sqrt(mean([0.12162², 0.17635², 0.23108²])) = sqrt(mean([0.01479, 0.03110, 0.05340])) = sqrt(0.03310) = 0.18193

        Expected: [0.13539, 0.15866, 0.18193]

        Note: Median reference with mean metric and RMS aggregation - median reference provides
        natural centering, mean metric preserves all fluctuation information, RMS aggregation
        amplifies the spread across atoms within residues.
        """
        traj = self._build_unique_residue_traj(1, 0, 2)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mean", "rms", None, False, 0)
        expected = [np.array([0.1354, 0.1587, 0.1819], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mean_rms_median(self):
        """Per-residue RMSF calculation with median reference, mean metric, and RMS_median aggregation.

        Synthetic trajectory test case: median_mean_rms_median
        Parameters: ref_idx=1, metric_idx=0, agg_idx=3

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=0 (+0.0), and agg_idx=3 (+0.021)
        Base amplitude = 0.02 + 0.03 + 0.0 + 0.021 = 0.071
        Frame pattern (mean metric): [0, +1, -1, +2, -2] (5 frames total, symmetric)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.07100, 0.10295, 0.13490, 0.08200, 0.11890, 0.15580, 0.09300, 0.13485, 0.17670]
        Frame 2: [-0.07100, -0.10295, -0.13490, -0.08200, -0.11890, -0.15580, -0.09300, -0.13485, -0.17670]
        Frame 3: [0.14200, 0.20590, 0.26980, 0.16400, 0.23780, 0.31160, 0.18600, 0.26970, 0.35340]
        Frame 4: [-0.14200, -0.20590, -0.26980, -0.16400, -0.23780, -0.31160, -0.18600, -0.26970, -0.35340]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (with symmetric pattern, median = zero)
        Reference: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.00504, 0.01060, 0.01820, 0.00672, 0.01414, 0.02427, 0.00865, 0.01818, 0.03122]
        Frame 2: [0.00504, 0.01060, 0.01820, 0.00672, 0.01414, 0.02427, 0.00865, 0.01818, 0.03122]
        Frame 3: [0.02016, 0.04239, 0.07279, 0.02690, 0.05655, 0.09709, 0.03460, 0.07274, 0.12489]
        Frame 4: [0.02016, 0.04239, 0.07279, 0.02690, 0.05655, 0.09709, 0.03460, 0.07274, 0.12489]

        Step 4: Apply mean metric to squared distances => MSF per atom
        MSF per atom = mean across frames of squared distances:
        [0.01008, 0.02120, 0.03640, 0.01345, 0.02827, 0.04855, 0.01730, 0.03637, 0.06245]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.10041, 0.14559, 0.19078, 0.11597, 0.16815, 0.22033, 0.13152, 0.19071, 0.24989]

        Step 6: Aggregate per residue using RMS_median aggregation
        RMS_median = sqrt(median(values²)) for each residue
        Residue 0: rms_median([0.10041, 0.14559, 0.19078]) = sqrt(median([0.10041², 0.14559², 0.19078²])) = sqrt(median([0.01008, 0.02120, 0.03640])) = sqrt(0.02120) = 0.14559
        Residue 1: rms_median([0.11597, 0.16815, 0.22033]) = sqrt(median([0.11597², 0.16815², 0.22033²])) = sqrt(median([0.01345, 0.02827, 0.04855])) = sqrt(0.02827) = 0.16815
        Residue 2: rms_median([0.13152, 0.19071, 0.24989]) = sqrt(median([0.13152², 0.19071², 0.24989²])) = sqrt(median([0.01730, 0.03637, 0.06245])) = sqrt(0.03637) = 0.19071

        Expected: [0.14559, 0.16815, 0.19071]

        Note: Median reference with mean metric and RMS_median aggregation - median reference provides
        natural centering, mean metric preserves all fluctuation information, RMS_median uses the
        median of squared values for moderate amplification with outlier resistance.
        """
        traj = self._build_unique_residue_traj(1, 0, 3)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mean", "rms_median", None, False, 0)
        expected = [np.array([0.1456, 0.1681, 0.1907], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_median_mean(self):
        """Per-residue RMSF calculation with median reference, median metric, and mean aggregation.

        Synthetic trajectory test case: median_median_mean
        Parameters: ref_idx=1, metric_idx=1, agg_idx=0

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=1 (+0.015), and agg_idx=0 (+0.0)
        Base amplitude = 0.02 + 0.03 + 0.015 + 0.0 = 0.065
        Frame pattern (median metric): [0, +1, +1, -1, +2, -2] (6 frames total, duplicate +1)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.06500, 0.09425, 0.12350, 0.07600, 0.11020, 0.14440, 0.08700, 0.12615, 0.16530]
        Frame 2: [0.06500, 0.09425, 0.12350, 0.07600, 0.11020, 0.14440, 0.08700, 0.12615, 0.16530] (duplicate)
        Frame 3: [-0.06500, -0.09425, -0.12350, -0.07600, -0.11020, -0.14440, -0.08700, -0.12615, -0.16530]
        Frame 4: [0.13000, 0.18850, 0.24700, 0.15200, 0.22040, 0.28880, 0.17400, 0.25230, 0.33060]
        Frame 5: [-0.13000, -0.18850, -0.24700, -0.15200, -0.22040, -0.28880, -0.17400, -0.25230, -0.33060]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (asymmetric due to duplicate +1 frame)
        Reference: [0.03250, 0.04713, 0.06175, 0.03800, 0.05510, 0.07220, 0.04350, 0.06307, 0.08265]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00106, 0.00222, 0.00381, 0.00144, 0.00304, 0.00521, 0.00189, 0.00398, 0.00683]
        Frame 1: [0.00106, 0.00222, 0.00381, 0.00144, 0.00304, 0.00521, 0.00189, 0.00398, 0.00683]
        Frame 2: [0.00106, 0.00222, 0.00381, 0.00144, 0.00304, 0.00521, 0.00189, 0.00398, 0.00683] (duplicate)
        Frame 3: [0.00951, 0.01999, 0.03432, 0.01300, 0.02732, 0.04692, 0.01703, 0.03581, 0.06148]
        Frame 4: [0.00951, 0.01999, 0.03432, 0.01300, 0.02732, 0.04692, 0.01703, 0.03581, 0.06148]
        Frame 5: [0.02641, 0.05552, 0.09533, 0.03610, 0.07590, 0.13032, 0.04731, 0.09946, 0.17078]

        Step 4: Apply median metric to squared distances => MSF per atom
        MSF per atom = median across frames of squared distances:
        [0.00528, 0.01110, 0.01907, 0.00722, 0.01518, 0.02606, 0.00946, 0.01989, 0.03416]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.07267, 0.10537, 0.13808, 0.08497, 0.12321, 0.16144, 0.09727, 0.14104, 0.18481]

        Step 6: Aggregate per residue using mean aggregation
        Residue 0: mean([0.07267, 0.10537, 0.13808]) = (0.07267 + 0.10537 + 0.13808) / 3 = 0.10537
        Residue 1: mean([0.08497, 0.12321, 0.16144]) = (0.08497 + 0.12321 + 0.16144) / 3 = 0.12321
        Residue 2: mean([0.09727, 0.14104, 0.18481]) = (0.09727 + 0.14104 + 0.18481) / 3 = 0.14104

        Expected: [0.10537, 0.12321, 0.14104]

        Note: Double robust median approach - both median reference and median metric provide
        excellent outlier resistance, while mean aggregation preserves all per-atom information.
        """
        traj = self._build_unique_residue_traj(1, 1, 0)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "median", "mean", None, False, 0)
        expected = [np.array([0.1054, 0.1232, 0.1410], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_median_median(self):
        """Per-residue RMSF calculation with median reference, median metric, and median aggregation.

        Synthetic trajectory test case: median_median_median
        Parameters: ref_idx=1, metric_idx=1, agg_idx=1

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=1 (+0.015), and agg_idx=1 (+0.007)
        Base amplitude = 0.02 + 0.03 + 0.015 + 0.007 = 0.072
        Frame pattern (median metric): [0, +1, +1, -1, +2, -2] (6 frames total, duplicate +1)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.07200, 0.10440, 0.13680, 0.08300, 0.12035, 0.15770, 0.09400, 0.13630, 0.17860]
        Frame 2: [0.07200, 0.10440, 0.13680, 0.08300, 0.12035, 0.15770, 0.09400, 0.13630, 0.17860] (duplicate)
        Frame 3: [-0.07200, -0.10440, -0.13680, -0.08300, -0.12035, -0.15770, -0.09400, -0.13630, -0.17860]
        Frame 4: [0.14400, 0.20880, 0.27360, 0.16600, 0.24070, 0.31540, 0.18800, 0.27260, 0.35720]
        Frame 5: [-0.14400, -0.20880, -0.27360, -0.16600, -0.24070, -0.31540, -0.18800, -0.27260, -0.35720]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (asymmetric due to duplicate +1 frame)
        Reference: [0.03600, 0.05220, 0.06840, 0.04150, 0.06017, 0.07885, 0.04700, 0.06815, 0.08930]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00130, 0.00272, 0.00468, 0.00172, 0.00362, 0.00622, 0.00221, 0.00464, 0.00797]
        Frame 1: [0.00130, 0.00272, 0.00468, 0.00172, 0.00362, 0.00622, 0.00221, 0.00464, 0.00797]
        Frame 2: [0.00130, 0.00272, 0.00468, 0.00172, 0.00362, 0.00622, 0.00221, 0.00464, 0.00797] (duplicate)
        Frame 3: [0.01166, 0.02452, 0.04211, 0.01550, 0.03259, 0.05596, 0.01988, 0.04180, 0.07177]
        Frame 4: [0.01166, 0.02452, 0.04211, 0.01550, 0.03259, 0.05596, 0.01988, 0.04180, 0.07177]
        Frame 5: [0.03240, 0.06812, 0.11696, 0.04306, 0.09053, 0.15543, 0.05522, 0.11611, 0.19936]

        Step 4: Apply median metric to squared distances => MSF per atom
        MSF per atom = median across frames of squared distances:
        [0.00648, 0.01362, 0.02339, 0.00861, 0.01811, 0.03109, 0.01105, 0.02322, 0.03987]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.08050, 0.11672, 0.15295, 0.09280, 0.13456, 0.17631, 0.10510, 0.15239, 0.19968]

        Step 6: Aggregate per residue using median aggregation
        Residue 0: median([0.08050, 0.11672, 0.15295]) = 0.11672 (middle value)
        Residue 1: median([0.09280, 0.13456, 0.17631]) = 0.13456 (middle value)
        Residue 2: median([0.10510, 0.15239, 0.19968]) = 0.15239 (middle value)

        Expected: [0.11672, 0.13456, 0.15239]

        Note: Triple robust median approach - median reference, median metric, and median aggregation
        all provide excellent outlier resistance. This is the most robust RMSF calculation possible.
        """
        traj = self._build_unique_residue_traj(1, 1, 1)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "median", "median", None, False, 0)
        expected = [np.array([0.1167, 0.1346, 0.1524], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_median_rms(self):
        """Per-residue RMSF calculation with median reference, median metric, and RMS aggregation.

        Synthetic trajectory test case: median_median_rms
        Parameters: ref_idx=1, metric_idx=1, agg_idx=2

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=1 (+0.015), and agg_idx=2 (+0.014)
        Base amplitude = 0.02 + 0.03 + 0.015 + 0.014 = 0.079
        Frame pattern (median metric): [0, +1, +1, -1, +2, -2] (6 frames total, duplicate +1)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.07900, 0.11455, 0.15010, 0.09000, 0.13050, 0.17100, 0.10100, 0.14645, 0.19190]
        Frame 2: [0.07900, 0.11455, 0.15010, 0.09000, 0.13050, 0.17100, 0.10100, 0.14645, 0.19190] (duplicate)
        Frame 3: [-0.07900, -0.11455, -0.15010, -0.09000, -0.13050, -0.17100, -0.10100, -0.14645, -0.19190]
        Frame 4: [0.15800, 0.22910, 0.30020, 0.18000, 0.26100, 0.34200, 0.20200, 0.29290, 0.38380]
        Frame 5: [-0.15800, -0.22910, -0.30020, -0.18000, -0.26100, -0.34200, -0.20200, -0.29290, -0.38380]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (asymmetric due to duplicate +1 frame)
        Reference: [0.03950, 0.05728, 0.07505, 0.04500, 0.06525, 0.08550, 0.05050, 0.07323, 0.09595]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00156, 0.00328, 0.00563, 0.00203, 0.00426, 0.00731, 0.00255, 0.00536, 0.00921]
        Frame 1: [0.00156, 0.00328, 0.00563, 0.00203, 0.00426, 0.00731, 0.00255, 0.00536, 0.00921]
        Frame 2: [0.00156, 0.00328, 0.00563, 0.00203, 0.00426, 0.00731, 0.00255, 0.00536, 0.00921] (duplicate)
        Frame 3: [0.01404, 0.02952, 0.05069, 0.01823, 0.03832, 0.06579, 0.02295, 0.04826, 0.08286]
        Frame 4: [0.01404, 0.02952, 0.05069, 0.01823, 0.03832, 0.06579, 0.02295, 0.04826, 0.08286]
        Frame 5: [0.03901, 0.08201, 0.14081, 0.05063, 0.10644, 0.18276, 0.06376, 0.13405, 0.23016]

        Step 4: Apply median metric to squared distances => MSF per atom
        MSF per atom = median across frames of squared distances:
        [0.00780, 0.01640, 0.02816, 0.01013, 0.02129, 0.03655, 0.01275, 0.02681, 0.04603]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.08832, 0.12807, 0.16782, 0.10062, 0.14590, 0.19118, 0.11292, 0.16374, 0.21455]

        Step 6: Aggregate per residue using RMS aggregation
        RMS = sqrt(mean(values²)) for each residue
        Residue 0: rms([0.08832, 0.12807, 0.16782]) = sqrt(mean([0.08832², 0.12807², 0.16782²])) = sqrt(mean([0.00780, 0.01640, 0.02816])) = sqrt(0.01745) = 0.13212
        Residue 1: rms([0.10062, 0.14590, 0.19118]) = sqrt(mean([0.10062², 0.14590², 0.19118²])) = sqrt(mean([0.01012, 0.02129, 0.03655])) = sqrt(0.02265) = 0.15051
        Residue 2: rms([0.11292, 0.16374, 0.21455]) = sqrt(mean([0.11292², 0.16374², 0.21455²])) = sqrt(mean([0.01275, 0.02681, 0.04603])) = sqrt(0.02853) = 0.16891

        Expected: [0.13212, 0.15051, 0.16891]

        Note: Median reference and median metric provide excellent outlier resistance, while
        RMS aggregation amplifies the spread across atoms within residues.
        """
        traj = self._build_unique_residue_traj(1, 1, 2)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "median", "rms", None, False, 0)
        expected = [np.array([0.1321, 0.1505, 0.1689], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_median_rms_median(self):
        """Per-residue RMSF calculation with median reference, median metric, and RMS_median aggregation.

        Synthetic trajectory test case: median_median_rms_median
        Parameters: ref_idx=1, metric_idx=1, agg_idx=3

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=1 (+0.015), and agg_idx=3 (+0.021)
        Base amplitude = 0.02 + 0.03 + 0.015 + 0.021 = 0.086
        Frame pattern (median metric): [0, +1, +1, -1, +2, -2] (6 frames total, duplicate +1)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.08600, 0.12470, 0.16340, 0.09700, 0.14065, 0.18430, 0.10800, 0.15660, 0.20520]
        Frame 2: [0.08600, 0.12470, 0.16340, 0.09700, 0.14065, 0.18430, 0.10800, 0.15660, 0.20520] (duplicate)
        Frame 3: [-0.08600, -0.12470, -0.16340, -0.09700, -0.14065, -0.18430, -0.10800, -0.15660, -0.20520]
        Frame 4: [0.17200, 0.24940, 0.32680, 0.19400, 0.28130, 0.36860, 0.21600, 0.31320, 0.41040]
        Frame 5: [-0.17200, -0.24940, -0.32680, -0.19400, -0.28130, -0.36860, -0.21600, -0.31320, -0.41040]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames (asymmetric due to duplicate +1 frame)
        Reference: [0.04300, 0.06235, 0.08170, 0.04850, 0.07033, 0.09215, 0.05400, 0.07830, 0.10260]

        Step 3: Calculate squared distances from reference per frame
        Since Y=Z=0, squared distance = (X - X_ref)² for each atom

        Squared distance matrix per frame:
        Frame 0: [0.00185, 0.00389, 0.00667, 0.00235, 0.00495, 0.00849, 0.00292, 0.00613, 0.01053]
        Frame 1: [0.00185, 0.00389, 0.00667, 0.00235, 0.00495, 0.00849, 0.00292, 0.00613, 0.01053]
        Frame 2: [0.00185, 0.00389, 0.00667, 0.00235, 0.00495, 0.00849, 0.00292, 0.00613, 0.01053] (duplicate)
        Frame 3: [0.01664, 0.03499, 0.06007, 0.02117, 0.04451, 0.07642, 0.02624, 0.05518, 0.09474]
        Frame 4: [0.01664, 0.03499, 0.06007, 0.02117, 0.04451, 0.07642, 0.02624, 0.05518, 0.09474]
        Frame 5: [0.04623, 0.09719, 0.16687, 0.05881, 0.12364, 0.21229, 0.07290, 0.15327, 0.26317]

        Step 4: Apply median metric to squared distances => MSF per atom
        MSF per atom = median across frames of squared distances:
        [0.00925, 0.01944, 0.03337, 0.01176, 0.02473, 0.04246, 0.01458, 0.03065, 0.05263]

        Step 5: Calculate per-atom RMSF = sqrt(MSF)
        Per-atom RMSF = sqrt(MSF):
        [0.09615, 0.13942, 0.18269, 0.10845, 0.15725, 0.20605, 0.12075, 0.17508, 0.22942]

        Step 6: Aggregate per residue using RMS_median aggregation
        RMS_median = sqrt(median(values²)) for each residue
        Residue 0: rms_median([0.09615, 0.13942, 0.18269]) = sqrt(median([0.09615², 0.13942², 0.18269²])) = sqrt(median([0.00925, 0.01944, 0.03337])) = sqrt(0.01944) = 0.13942
        Residue 1: rms_median([0.10845, 0.15725, 0.20605]) = sqrt(median([0.10845², 0.15725², 0.20605²])) = sqrt(median([0.01176, 0.02473, 0.04246])) = sqrt(0.02473) = 0.15725
        Residue 2: rms_median([0.12075, 0.17508, 0.22942]) = sqrt(median([0.12075², 0.17508², 0.22942²])) = sqrt(median([0.01458, 0.03065, 0.05263])) = sqrt(0.03065) = 0.17508

        Expected: [0.13942, 0.15725, 0.17508]

        Note: Median reference and median metric provide excellent outlier resistance, while
        RMS_median aggregation uses the median of squared values for moderate amplification with additional robustness.
        """
        traj = self._build_unique_residue_traj(1, 1, 3)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "median", "rms_median", None, False, 0)
        expected = [np.array([0.1394, 0.1573, 0.1751], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mad_mean(self):
        """Per-residue RMSF calculation with median reference, MAD metric, and mean aggregation.

        Synthetic trajectory test case: median_mad_mean
        Parameters: ref_idx=1, metric_idx=2, agg_idx=0

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=2 (+0.03), and agg_idx=0 (+0.0)
        Base amplitude = 0.02 + 0.03 + 0.03 + 0.0 = 0.08
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, broader spread)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.08000, 0.11600, 0.15200, 0.09100, 0.13195, 0.17290, 0.10200, 0.14790, 0.19380]
        Frame 2: [0.16000, 0.23200, 0.30400, 0.18200, 0.26390, 0.34580, 0.20400, 0.29580, 0.38760]
        Frame 3: [-0.08000, -0.11600, -0.15200, -0.09100, -0.13195, -0.17290, -0.10200, -0.14790, -0.19380]
        Frame 4: [0.24000, 0.34800, 0.45600, 0.27300, 0.39585, 0.51870, 0.30600, 0.44370, 0.58140]
        Frame 5: [-0.16000, -0.23200, -0.30400, -0.18200, -0.26390, -0.34580, -0.20400, -0.29580, -0.38760]
        Frame 6: [0.32000, 0.46400, 0.60800, 0.36400, 0.52780, 0.69160, 0.40800, 0.59160, 0.77520]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames
        Reference: [0.08000, 0.11600, 0.15200, 0.09100, 0.13195, 0.17290, 0.10200, 0.14790, 0.19380]

        Step 3: Calculate Euclidean distances from reference per frame
        Since Y=Z=0, Euclidean distance = |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.08000, 0.11600, 0.15200, 0.09100, 0.13195, 0.17290, 0.10200, 0.14790, 0.19380]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.08000, 0.11600, 0.15200, 0.09100, 0.13195, 0.17290, 0.10200, 0.14790, 0.19380]
        Frame 3: [0.16000, 0.23200, 0.30400, 0.18200, 0.26390, 0.34580, 0.20400, 0.29580, 0.38760]
        Frame 4: [0.16000, 0.23200, 0.30400, 0.18200, 0.26390, 0.34580, 0.20400, 0.29580, 0.38760]
        Frame 5: [0.24000, 0.34800, 0.45600, 0.27300, 0.39585, 0.51870, 0.30600, 0.44370, 0.58140]
        Frame 6: [0.24000, 0.34800, 0.45600, 0.27300, 0.39585, 0.51870, 0.30600, 0.44370, 0.58140]

        Step 4: Apply MAD (Median Absolute Deviation) metric => per-atom RMSF
        For each atom, compute MAD = median(|distances - median(distances)|):
        Atom 0: median=0.16000, MAD=0.08000
        Atom 1: median=0.23200, MAD=0.11600
        Atom 2: median=0.30400, MAD=0.15200
        Atom 3: median=0.18200, MAD=0.09100
        Atom 4: median=0.26390, MAD=0.13195
        Atom 5: median=0.34580, MAD=0.17290
        Atom 6: median=0.20400, MAD=0.10200
        Atom 7: median=0.29580, MAD=0.14790
        Atom 8: median=0.38760, MAD=0.19380

        Per-atom RMSF (=MAD values): [0.08000, 0.11600, 0.15200, 0.09100, 0.13195, 0.17290, 0.10200, 0.14790, 0.19380]

        Step 5: Aggregate per residue using mean aggregation
        Mean aggregation for each residue
        Residue 0: mean([0.08000, 0.11600, 0.15200]) = (0.08000 + 0.11600 + 0.15200) / 3 = 0.11600
        Residue 1: mean([0.09100, 0.13195, 0.17290]) = (0.09100 + 0.13195 + 0.17290) / 3 = 0.13195
        Residue 2: mean([0.10200, 0.14790, 0.19380]) = (0.10200 + 0.14790 + 0.19380) / 3 = 0.14790

        Expected: [0.11600, 0.13195, 0.14790]

        Note: Median reference provides natural centering, MAD metric offers robust outlier resistance,
        mean aggregation preserves all per-atom information within residues.
        """
        traj = self._build_unique_residue_traj(1, 2, 0)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mad", "mean", None, False, 0)
        expected = [np.array([0.1160, 0.1319, 0.1479], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mad_median(self):
        """Per-residue RMSF calculation with median reference, MAD metric, and median aggregation.

        Synthetic trajectory test case: median_mad_median
        Parameters: ref_idx=1, metric_idx=2, agg_idx=1

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=2 (+0.03), and agg_idx=1 (+0.007)
        Base amplitude = 0.02 + 0.03 + 0.03 + 0.007 = 0.087
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, broader spread)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.08700, 0.12615, 0.16530, 0.09800, 0.14210, 0.18620, 0.10900, 0.15805, 0.20710]
        Frame 2: [0.17400, 0.25230, 0.33060, 0.19600, 0.28420, 0.37240, 0.21800, 0.31610, 0.41420]
        Frame 3: [-0.08700, -0.12615, -0.16530, -0.09800, -0.14210, -0.18620, -0.10900, -0.15805, -0.20710]
        Frame 4: [0.26100, 0.37845, 0.49590, 0.29400, 0.42630, 0.55860, 0.32700, 0.47415, 0.62130]
        Frame 5: [-0.17400, -0.25230, -0.33060, -0.19600, -0.28420, -0.37240, -0.21800, -0.31610, -0.41420]
        Frame 6: [0.34800, 0.50460, 0.66120, 0.39200, 0.56840, 0.74480, 0.43600, 0.63220, 0.82840]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames
        Reference: [0.08700, 0.12615, 0.16530, 0.09800, 0.14210, 0.18620, 0.10900, 0.15805, 0.20710]

        Step 3: Calculate Euclidean distances from reference per frame
        Since Y=Z=0, Euclidean distance = |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.08700, 0.12615, 0.16530, 0.09800, 0.14210, 0.18620, 0.10900, 0.15805, 0.20710]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.08700, 0.12615, 0.16530, 0.09800, 0.14210, 0.18620, 0.10900, 0.15805, 0.20710]
        Frame 3: [0.17400, 0.25230, 0.33060, 0.19600, 0.28420, 0.37240, 0.21800, 0.31610, 0.41420]
        Frame 4: [0.17400, 0.25230, 0.33060, 0.19600, 0.28420, 0.37240, 0.21800, 0.31610, 0.41420]
        Frame 5: [0.26100, 0.37845, 0.49590, 0.29400, 0.42630, 0.55860, 0.32700, 0.47415, 0.62130]
        Frame 6: [0.26100, 0.37845, 0.49590, 0.29400, 0.42630, 0.55860, 0.32700, 0.47415, 0.62130]

        Step 4: Apply MAD (Median Absolute Deviation) metric => per-atom RMSF
        For each atom, compute MAD = median(|distances - median(distances)|):
        Atom 0: median=0.17400, MAD=0.08700
        Atom 1: median=0.25230, MAD=0.12615
        Atom 2: median=0.33060, MAD=0.16530
        Atom 3: median=0.19600, MAD=0.09800
        Atom 4: median=0.28420, MAD=0.14210
        Atom 5: median=0.37240, MAD=0.18620
        Atom 6: median=0.21800, MAD=0.10900
        Atom 7: median=0.31610, MAD=0.15805
        Atom 8: median=0.41420, MAD=0.20710

        Per-atom RMSF (=MAD values): [0.08700, 0.12615, 0.16530, 0.09800, 0.14210, 0.18620, 0.10900, 0.15805, 0.20710]

        Step 5: Aggregate per residue using median aggregation
        Residue 0: median([0.08700, 0.12615, 0.16530]) = 0.12615 (middle value)
        Residue 1: median([0.09800, 0.14210, 0.18620]) = 0.14210 (middle value)
        Residue 2: median([0.10900, 0.15805, 0.20710]) = 0.15805 (middle value)

        Expected: [0.12615, 0.14210, 0.15805]

        Note: Triple robust approach - median reference, MAD metric, and median aggregation
        all provide maximum outlier resistance. Slightly dampened compared to mean aggregation.
        """
        traj = self._build_unique_residue_traj(1, 2, 1)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mad", "median", None, False, 0)
        expected = [np.array([0.1262, 0.1421, 0.1580], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mad_rms(self):
        """Per-residue RMSF calculation with median reference, MAD metric, and RMS aggregation.

        Synthetic trajectory test case: median_mad_rms
        Parameters: ref_idx=1, metric_idx=2, agg_idx=2

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=2 (+0.03), and agg_idx=2 (+0.014)
        Base amplitude = 0.02 + 0.03 + 0.03 + 0.014 = 0.094
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, broader spread)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.09400, 0.13630, 0.17860, 0.10500, 0.15225, 0.19950, 0.11600, 0.16820, 0.22040]
        Frame 2: [0.18800, 0.27260, 0.35720, 0.21000, 0.30450, 0.39900, 0.23200, 0.33640, 0.44080]
        Frame 3: [-0.09400, -0.13630, -0.17860, -0.10500, -0.15225, -0.19950, -0.11600, -0.16820, -0.22040]
        Frame 4: [0.28200, 0.40890, 0.53580, 0.31500, 0.45675, 0.59850, 0.34800, 0.50460, 0.66120]
        Frame 5: [-0.18800, -0.27260, -0.35720, -0.21000, -0.30450, -0.39900, -0.23200, -0.33640, -0.44080]
        Frame 6: [0.37600, 0.54520, 0.71440, 0.42000, 0.60900, 0.79800, 0.46400, 0.67280, 0.88160]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames
        Reference: [0.09400, 0.13630, 0.17860, 0.10500, 0.15225, 0.19950, 0.11600, 0.16820, 0.22040]

        Step 3: Calculate Euclidean distances from reference per frame
        Since Y=Z=0, Euclidean distance = |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.09400, 0.13630, 0.17860, 0.10500, 0.15225, 0.19950, 0.11600, 0.16820, 0.22040]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.09400, 0.13630, 0.17860, 0.10500, 0.15225, 0.19950, 0.11600, 0.16820, 0.22040]
        Frame 3: [0.18800, 0.27260, 0.35720, 0.21000, 0.30450, 0.39900, 0.23200, 0.33640, 0.44080]
        Frame 4: [0.18800, 0.27260, 0.35720, 0.21000, 0.30450, 0.39900, 0.23200, 0.33640, 0.44080]
        Frame 5: [0.28200, 0.40890, 0.53580, 0.31500, 0.45675, 0.59850, 0.34800, 0.50460, 0.66120]
        Frame 6: [0.28200, 0.40890, 0.53580, 0.31500, 0.45675, 0.59850, 0.34800, 0.50460, 0.66120]

        Step 4: Apply MAD (Median Absolute Deviation) metric => per-atom RMSF
        For each atom, compute MAD = median(|distances - median(distances)|):
        Atom 0: median=0.18800, MAD=0.09400
        Atom 1: median=0.27260, MAD=0.13630
        Atom 2: median=0.35720, MAD=0.17860
        Atom 3: median=0.21000, MAD=0.10500
        Atom 4: median=0.30450, MAD=0.15225
        Atom 5: median=0.39900, MAD=0.19950
        Atom 6: median=0.23200, MAD=0.11600
        Atom 7: median=0.33640, MAD=0.16820
        Atom 8: median=0.44080, MAD=0.22040

        Per-atom RMSF (=MAD values): [0.09400, 0.13630, 0.17860, 0.10500, 0.15225, 0.19950, 0.11600, 0.16820, 0.22040]

        Step 5: Aggregate per residue using RMS aggregation
        RMS = sqrt(mean(values²)) for each residue
        Residue 0: rms([0.09400, 0.13630, 0.17860]) = sqrt(mean([0.09400², 0.13630², 0.17860²])) = sqrt(mean([0.00884, 0.01858, 0.03190])) = sqrt(0.01977) = 0.14061
        Residue 1: rms([0.10500, 0.15225, 0.19950]) = sqrt(mean([0.10500², 0.15225², 0.19950²])) = sqrt(mean([0.01103, 0.02318, 0.03980])) = sqrt(0.02467) = 0.15707
        Residue 2: rms([0.11600, 0.16820, 0.22040]) = sqrt(mean([0.11600², 0.16820², 0.22040²])) = sqrt(mean([0.01346, 0.02829, 0.04858])) = sqrt(0.03011) = 0.17353

        Expected: [0.14061, 0.15707, 0.17353]

        Note: Median reference and MAD metric provide excellent outlier resistance, while
        RMS aggregation amplifies the dispersion across atoms within residues.
        """
        traj = self._build_unique_residue_traj(1, 2, 2)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mad", "rms", None, False, 0)
        expected = [np.array([0.1406, 0.1571, 0.1735], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_median_mad_rms_median(self):
        """Per-residue RMSF calculation with median reference, MAD metric, and RMS_median aggregation.

        Synthetic trajectory test case: median_mad_rms_median
        Parameters: ref_idx=1, metric_idx=2, agg_idx=3

        Detailed Step-by-Step Calculation:

        Step 1: Generate synthetic trajectory with ref_idx=1 (+0.03), metric_idx=2 (+0.03), and agg_idx=3 (+0.021)
        Base amplitude = 0.02 + 0.03 + 0.03 + 0.021 = 0.101
        Frame pattern (MAD metric): [0, +1, +2, -1, +3, -2, +4] (7 frames total, broader spread)
        Coordinates vary only in X-direction; Y and Z are zero.
        9 atoms arranged as: 3 residues x 3 atoms per residue
        Atom scaling per residue: [1.0, 1.45, 1.90] for intra-residue diversity

        Frame coordinates (X-axis only, Y=Z=0):
        Frame 0: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 1: [0.10100, 0.14645, 0.19190, 0.11200, 0.16240, 0.21280, 0.12300, 0.17835, 0.23370]
        Frame 2: [0.20200, 0.29290, 0.38380, 0.22400, 0.32480, 0.42560, 0.24600, 0.35670, 0.46740]
        Frame 3: [-0.10100, -0.14645, -0.19190, -0.11200, -0.16240, -0.21280, -0.12300, -0.17835, -0.23370]
        Frame 4: [0.30300, 0.43935, 0.57570, 0.33600, 0.48720, 0.63840, 0.36900, 0.53505, 0.70110]
        Frame 5: [-0.20200, -0.29290, -0.38380, -0.22400, -0.32480, -0.42560, -0.24600, -0.35670, -0.46740]
        Frame 6: [0.40400, 0.58580, 0.76760, 0.44800, 0.64960, 0.85120, 0.49200, 0.71340, 0.93480]

        Step 2: Calculate reference structure (median across all frames)
        Reference coordinates = median of all frames
        Reference: [0.10100, 0.14645, 0.19190, 0.11200, 0.16240, 0.21280, 0.12300, 0.17835, 0.23370]

        Step 3: Calculate Euclidean distances from reference per frame
        Since Y=Z=0, Euclidean distance = |X - X_ref| for each atom

        Euclidean distance matrix per frame:
        Frame 0: [0.10100, 0.14645, 0.19190, 0.11200, 0.16240, 0.21280, 0.12300, 0.17835, 0.23370]
        Frame 1: [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        Frame 2: [0.10100, 0.14645, 0.19190, 0.11200, 0.16240, 0.21280, 0.12300, 0.17835, 0.23370]
        Frame 3: [0.20200, 0.29290, 0.38380, 0.22400, 0.32480, 0.42560, 0.24600, 0.35670, 0.46740]
        Frame 4: [0.20200, 0.29290, 0.38380, 0.22400, 0.32480, 0.42560, 0.24600, 0.35670, 0.46740]
        Frame 5: [0.30300, 0.43935, 0.57570, 0.33600, 0.48720, 0.63840, 0.36900, 0.53505, 0.70110]
        Frame 6: [0.30300, 0.43935, 0.57570, 0.33600, 0.48720, 0.63840, 0.36900, 0.53505, 0.70110]

        Step 4: Apply MAD (Median Absolute Deviation) metric => per-atom RMSF
        For each atom, compute MAD = median(|distances - median(distances)|):
        Atom 0: median=0.20200, MAD=0.10100
        Atom 1: median=0.29290, MAD=0.14645
        Atom 2: median=0.38380, MAD=0.19190
        Atom 3: median=0.22400, MAD=0.11200
        Atom 4: median=0.32480, MAD=0.16240
        Atom 5: median=0.42560, MAD=0.21280
        Atom 6: median=0.24600, MAD=0.12300
        Atom 7: median=0.35670, MAD=0.17835
        Atom 8: median=0.46740, MAD=0.23370

        Per-atom RMSF (=MAD values): [0.10100, 0.14645, 0.19190, 0.11200, 0.16240, 0.21280, 0.12300, 0.17835, 0.23370]

        Step 5: Aggregate per residue using RMS_median aggregation
        RMS_median = sqrt(median(values²)) for each residue
        Residue 0: rms_median([0.10100, 0.14645, 0.19190]) = sqrt(median([0.10100², 0.14645², 0.19190²])) = sqrt(median([0.01020, 0.02145, 0.03683])) = sqrt(0.02145) = 0.14645
        Residue 1: rms_median([0.11200, 0.16240, 0.21280]) = sqrt(median([0.11200², 0.16240², 0.21280²])) = sqrt(median([0.01254, 0.02637, 0.04528])) = sqrt(0.02637) = 0.16240
        Residue 2: rms_median([0.12300, 0.17835, 0.23370]) = sqrt(median([0.12300², 0.17835², 0.23370²])) = sqrt(median([0.01513, 0.03181, 0.05462])) = sqrt(0.03181) = 0.17835

        Expected: [0.14645, 0.16240, 0.17835]

        Note: Median reference and MAD metric provide excellent outlier resistance, while
        RMS_median uses the median of squared values for moderate amplification with additional robustness.
        """
        traj = self._build_unique_residue_traj(1, 2, 3)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("median", "mad", "rms_median", None, False, 0)
        expected = [np.array([0.1465, 0.1624, 0.1784], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
#---
    def test_calculate_per_residue_multiple_trajectories(self):
        """Test per-residue RMSF with multiple trajectories.

        Each trajectory computed separately.
        """
        calc = RMSFCalculator([self.traj_multi, self.traj_multi], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "mean", None, False, 0)

        expected = [
            np.array([0.0667, 0.1333, 0.2000], dtype=np.float32),
            np.array([0.0667, 0.1333, 0.2000], dtype=np.float32)
        ]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_residue_cross_trajectory(self):
        """Test per-residue RMSF with cross-trajectory calculation.

        All trajectories combined into single RMSF profile.
        """
        calc = RMSFCalculator([self.traj_multi, self.traj_multi], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "mean", None, True, 0)

        expected = [np.array([0.0667, 0.1333, 0.2000], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_residue_with_atom_indices(self):
        """Test per-residue RMSF with atom selection.

        Select subset of atoms for calculation.
        """
        calc = RMSFCalculator([self.traj_multi], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "mean", np.array([0, 2, 4]), False, 0)

        # Only atoms 0, 2, 4 selected (one per residue)
        expected = [np.array([0.0667, 0.1333, 0.2000], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_residue_different_reference_trajectory(self):
        """Test per-residue RMSF with different reference trajectory.

        Use second trajectory for residue topology.
        """
        calc = RMSFCalculator([self.traj_multi, self.traj_multi], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "mean", None, False, 1)

        expected = [
            np.array([0.0667, 0.1333, 0.2000], dtype=np.float32),
            np.array([0.0667, 0.1333, 0.2000], dtype=np.float32)
        ]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_residue_invalid_aggregator(self):
        """Test per-residue RMSF with invalid aggregator.

        Should raise ValueError for unknown aggregator.
        """
        calc = RMSFCalculator([self.traj_multi], chunk_size=10, use_memmap=False)

        with pytest.raises(ValueError, match="Unsupported residue aggregator"):
            calc.calculate_per_residue("mean", "mean", "invalid", None, False, 0)

    def test_calculate_per_residue_invalid_reference_trajectory_index(self):
        """Test per-residue RMSF with invalid reference trajectory index.

        Should raise ValueError for out-of-bounds index.
        """
        calc = RMSFCalculator([self.traj_multi], chunk_size=10, use_memmap=False)

        with pytest.raises(IndexError, match="list index out of range"):
            calc.calculate_per_residue("mean", "mean", "mean", None, False, 5)

    def test_calculate_per_residue_single_residue(self):
        """Test per-residue RMSF with single residue.

        Single residue should work correctly.
        """
        calc = RMSFCalculator([self.traj], chunk_size=10, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "mean", None, False, 0)

        expected = [np.array([0.08944], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_calculate_per_residue_with_memmap(self):
        """Test per-residue RMSF with memmap enabled.

        Memmap should not affect results.
        """
        calc = RMSFCalculator([self.traj_multi], chunk_size=10, use_memmap=True)
        result = calc.calculate_per_residue("mean", "mean", "mean", None, False, 0)

        expected = [np.array([0.0667, 0.1333, 0.2000], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_calculate_per_residue_small_chunk_size(self):
        """Test per-residue RMSF with small chunk size.

        Small chunk size should not affect results.
        """
        calc = RMSFCalculator([self.traj_multi], chunk_size=1, use_memmap=False)
        result = calc.calculate_per_residue("mean", "mean", "mean", None, False, 0)

        expected = [np.array([0.0667, 0.1333, 0.2000], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    # 4. Chunking Verification Tests

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_atom_batching_single_trajectory(self, mock_squared_chunks):
        """Test atom chunking divides atoms into correct batches.

        Creates trajectory with 6 atoms and atom_batch_size=4, should create 2 batches.
        """
        # Mock return: squared deviations for each atom batch
        mock_squared_chunks.side_effect = [
            np.ones((10, 4), dtype=np.float32),  # First batch: 4 atoms
            np.ones((10, 2), dtype=np.float32),  # Second batch: 2 atoms
        ]

        # Create trajectory: 10 frames, 6 atoms, atom_batch_size=4
        traj = self.make_unique_trajectory(10, 6)
        calc = RMSFCalculator([traj], chunk_size=3, use_memmap=False)

        # Force small atom_batch_size by patching the default
        with patch.object(calc, '_determine_atom_batch_size', return_value=4):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Should be called twice: batch 1 (atoms 0-3) and batch 2 (atoms 4-5)
        assert mock_squared_chunks.call_count == 2

        # Check first batch: atoms 0-3
        call_args_1 = mock_squared_chunks.call_args_list[0]
        assert call_args_1.args[2] == 0  # atom_start
        assert call_args_1.args[3] == 4  # atom_end

        # Check second batch: atoms 4-5
        call_args_2 = mock_squared_chunks.call_args_list[1]
        assert call_args_2.args[2] == 4  # atom_start
        assert call_args_2.args[3] == 6  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_multiple_trajectories_exact_frames(self, mock_squared_chunks):
        """Test chunking processes multiple trajectories with correct atom batching.

        Calculation documentation:
        - 2 trajectories: traj1 (8 frames, 2 atoms), traj2 (6 frames, 2 atoms)
        - atom_batch_size=1 => 2 atom batches per trajectory
        - Total calls: 2 trajectories x 2 atom batches = 4 calls
        - Expected atom ranges: [0:1], [1:2] for each trajectory
        """
        mock_squared_chunks.side_effect = [
            np.ones((8, 1), dtype=np.float32),  # traj1, atoms 0:1
            np.ones((8, 1), dtype=np.float32),  # traj1, atoms 1:2
            np.ones((6, 1), dtype=np.float32),  # traj2, atoms 0:1
            np.ones((6, 1), dtype=np.float32),  # traj2, atoms 1:2
        ]

        traj1 = self.make_unique_trajectory(8, 2)
        traj2 = self.make_unique_trajectory(6, 2)
        calc = RMSFCalculator([traj1, traj2], chunk_size=3, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=1):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 4 calls total (2 trajectories x 2 atom batches each)
        assert mock_squared_chunks.call_count == 4

        # Verify atom ranges for each trajectory
        call_args_list = mock_squared_chunks.call_args_list

        # Trajectory 1 calls (first 2 calls)
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 1  # atom_end
        assert call_args_list[1].args[2] == 1  # atom_start
        assert call_args_list[1].args[3] == 2  # atom_end

        # Trajectory 2 calls (last 2 calls)
        assert call_args_list[2].args[2] == 0  # atom_start
        assert call_args_list[2].args[3] == 1  # atom_end
        assert call_args_list[3].args[2] == 1  # atom_start
        assert call_args_list[3].args[3] == 2  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_cross_trajectory_exact_frames(self, mock_squared_chunks):
        """Test chunking for cross-trajectory calculation with atom batching.

        Calculation documentation:
        - Cross-trajectory mode: trajectories combined (8+6=14 total frames)
        - 2 atoms total, atom_batch_size=1 => 2 atom batches
        - Total calls: 1 combined trajectory x 2 atom batches = 2 calls
        - Expected atom ranges: [0:1], [1:2]
        """
        # Provide more return values in case cross-trajectory calls more times
        mock_squared_chunks.return_value = np.ones((14, 1), dtype=np.float32)

        traj1 = self.make_unique_trajectory(8, 2)
        traj2 = self.make_unique_trajectory(6, 2)
        calc = RMSFCalculator([traj1, traj2], chunk_size=5, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=1):
            calc.calculate_per_atom("mean", "mean", None, True)

        # Verify that cross-trajectory chunking was called
        assert mock_squared_chunks.call_count >= 2

        # Verify that chunking calls were made (exact count may vary for cross-trajectory)
        call_args_list = mock_squared_chunks.call_args_list
        assert len(call_args_list) >= 2

        # Verify that atom batching occurred (different atom ranges in calls)
        atom_starts = [call.args[2] for call in call_args_list]
        atom_ends = [call.args[3] for call in call_args_list]
        assert len(set(atom_starts)) >= 2 or len(set(atom_ends)) >= 2  # Different ranges used

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_per_residue_exact_frames(self, mock_squared_chunks):
        """Test chunking for per-residue calculations with atom batching.

        Calculation documentation:
        - 1 trajectory: 12 frames, 6 atoms
        - atom_batch_size=3 => 2 atom batches (6 atoms / 3 = 2)
        - Per-residue uses same chunking as per-atom internally
        - Expected atom ranges: [0:3], [3:6]
        """
        mock_squared_chunks.side_effect = [
            np.ones((12, 3), dtype=np.float32),  # atoms 0:3
            np.ones((12, 3), dtype=np.float32),  # atoms 3:6
        ]

        traj = self.make_unique_trajectory(12, 6)
        calc = RMSFCalculator([traj], chunk_size=5, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=3):
            calc.calculate_per_residue("mean", "mean", "mean", None, False, 0)

        # Verify 2 calls (1 trajectory x 2 atom batches)
        assert mock_squared_chunks.call_count == 2

        # Verify atom ranges
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 3  # atom_end
        assert call_args_list[1].args[2] == 3  # atom_start
        assert call_args_list[1].args[3] == 6  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_with_memmap_enabled(self, mock_squared_chunks):
        """Test chunking with memmap enabled preserves chunking behavior.

        Calculation documentation:
        - 1 trajectory: 10 frames, 3 atoms
        - atom_batch_size=2 => 2 atom batches (3 atoms: [0:2], [2:3])
        - Memmap setting should not affect atom chunking behavior
        """
        mock_squared_chunks.side_effect = [
            np.ones((10, 2), dtype=np.float32),  # atoms 0:2
            np.ones((10, 1), dtype=np.float32),  # atoms 2:3
        ]

        traj = self.make_unique_trajectory(10, 3)
        calc = RMSFCalculator([traj], chunk_size=4, use_memmap=True)

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify chunking works with memmap
        assert mock_squared_chunks.call_count == 2
        assert calc.use_memmap is True

        # Verify atom ranges
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 2  # atom_end
        assert call_args_list[1].args[2] == 2  # atom_start
        assert call_args_list[1].args[3] == 3  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_small_chunk_size(self, mock_squared_chunks):
        """Test chunking with very small chunk size and frame chunking.

        Calculation documentation:
        - 1 trajectory: 5 frames, 2 atoms
        - chunk_size=1 => frame chunking creates very small frame chunks
        - atom_batch_size=1 => 2 atom batches
        - Both frame chunking and atom chunking should work together
        """
        mock_squared_chunks.side_effect = [
            np.ones((5, 1), dtype=np.float32),  # atoms 0:1
            np.ones((5, 1), dtype=np.float32),  # atoms 1:2
        ]

        traj = self.make_unique_trajectory(5, 2)
        calc = RMSFCalculator([traj], chunk_size=1, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=1):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 2 calls (1 trajectory x 2 atom batches)
        assert mock_squared_chunks.call_count == 2

        # Verify atom ranges
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 1  # atom_end
        assert call_args_list[1].args[2] == 1  # atom_start
        assert call_args_list[1].args[3] == 2  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_large_chunk_size(self, mock_squared_chunks):
        """Test chunking with very large chunk size processes all frames together.

        Calculation documentation:
        - 1 trajectory: 10 frames, 3 atoms
        - chunk_size=1000 => all frames processed in one chunk
        - atom_batch_size=2 => 2 atom batches (3 atoms: [0:2], [2:3])
        - Large chunk size should not affect atom batching
        """
        mock_squared_chunks.side_effect = [
            np.ones((10, 2), dtype=np.float32),  # atoms 0:2, all frames
            np.ones((10, 1), dtype=np.float32),  # atoms 2:3, all frames
        ]

        traj = self.make_unique_trajectory(10, 3)
        calc = RMSFCalculator([traj], chunk_size=1000, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 2 calls (1 trajectory x 2 atom batches)
        assert mock_squared_chunks.call_count == 2

        # Verify atom ranges
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 2  # atom_end
        assert call_args_list[1].args[2] == 2  # atom_start
        assert call_args_list[1].args[3] == 3  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_exact_division(self, mock_squared_chunks):
        """Test chunking when frames divide exactly by chunk size.

        Calculation documentation:
        - 1 trajectory: 12 frames, 3 atoms
        - chunk_size=4 => 12/4 = 3 frame chunks (exact division)
        - atom_batch_size=3 => 1 atom batch (all atoms together)
        - Should handle exact frame division correctly
        """
        mock_squared_chunks.side_effect = [
            np.ones((12, 3), dtype=np.float32),  # atoms 0:3, all frames
        ]

        traj = self.make_unique_trajectory(12, 3)
        calc = RMSFCalculator([traj], chunk_size=4, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=3):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 1 call (1 trajectory x 1 atom batch)
        assert mock_squared_chunks.call_count == 1

        # Verify atom range covers all atoms
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 3  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_inexact_division(self, mock_squared_chunks):
        """Test chunking when frames don't divide exactly by chunk size.

        Calculation documentation:
        - 1 trajectory: 13 frames, 3 atoms
        - chunk_size=5 => 13/5 = 2 full chunks + 3 remainder frames
        - atom_batch_size=2 => 2 atom batches (3 atoms: [0:2], [2:3])
        - Should handle inexact frame division correctly
        """
        mock_squared_chunks.side_effect = [
            np.ones((13, 2), dtype=np.float32),  # atoms 0:2, all frames
            np.ones((13, 1), dtype=np.float32),  # atoms 2:3, all frames
        ]

        traj = self.make_unique_trajectory(13, 3)
        calc = RMSFCalculator([traj], chunk_size=5, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 2 calls (1 trajectory x 2 atom batches)
        assert mock_squared_chunks.call_count == 2

        # Verify atom ranges
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 2  # atom_end
        assert call_args_list[1].args[2] == 2  # atom_start
        assert call_args_list[1].args[3] == 3  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_single_frame_trajectory(self, mock_squared_chunks):
        """Test chunking with single frame trajectory.

        Calculation documentation:
        - 1 trajectory: 1 frame, 3 atoms
        - Single frame => RMSF should be zeros (no fluctuation possible)
        - atom_batch_size=3 => 1 atom batch (all atoms together)
        - Should handle single frame correctly
        """
        mock_squared_chunks.side_effect = [
            np.zeros((1, 3), dtype=np.float32),  # atoms 0:3, single frame => zeros
        ]

        traj = self.make_unique_trajectory(1, 3)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=3):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 1 call (1 trajectory x 1 atom batch)
        assert mock_squared_chunks.call_count == 1

        # Verify atom range covers all atoms
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 3  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_with_atom_indices(self, mock_squared_chunks):
        """Test chunking with atom indices selection.

        Calculation documentation:
        - 1 trajectory: 10 frames, 5 atoms total
        - atom_indices=[1, 3] => only process atoms 1 and 3 (2 selected atoms)
        - atom_batch_size=1 => 2 atom batches for the selected atoms
        - Atom selection affects which atoms are processed but chunking still applies
        """
        mock_squared_chunks.side_effect = [
            np.ones((10, 1), dtype=np.float32),  # first selected atom batch
            np.ones((10, 1), dtype=np.float32),  # second selected atom batch
        ]

        traj = self.make_unique_trajectory(10, 5)
        calc = RMSFCalculator([traj], chunk_size=3, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=1):
            calc.calculate_per_atom("mean", "mean", np.array([1, 3]), False)

        # Verify 2 calls (chunking still applies to selected atoms)
        assert mock_squared_chunks.call_count == 2

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_different_reference_modes(self, mock_squared_chunks):
        """Test chunking with different reference modes.

        Calculation documentation:
        - 1 trajectory: 8 frames, 3 atoms
        - Different reference modes (mean, median) should not affect chunking
        - atom_batch_size=3 => 1 atom batch per calculation
        - 2 calculations => 2 calls total
        """
        mock_squared_chunks.side_effect = [
            np.ones((8, 3), dtype=np.float32),  # first call: mean reference
            np.ones((8, 3), dtype=np.float32),  # second call: median reference
        ]

        traj = self.make_unique_trajectory(8, 3)
        calc = RMSFCalculator([traj], chunk_size=3, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=3):
            # Test both reference modes
            calc.calculate_per_atom("mean", "mean", None, False)
            calc.calculate_per_atom("median", "mean", None, False)

        # Verify 2 calls (1 per calculation, reference mode doesn't affect chunking)
        assert mock_squared_chunks.call_count == 2

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_different_metrics(self, mock_squared_chunks):
        """Test chunking with different metrics.

        Calculation documentation:
        - 1 trajectory: 6 frames, 2 atoms
        - Different metrics (mean, median, MAD) should not affect chunking
        - atom_batch_size=2 => 1 atom batch per calculation
        - 3 calculations => 3 calls total
        """
        mock_squared_chunks.side_effect = [
            np.ones((6, 2), dtype=np.float32),  # first call: mean metric
            np.ones((6, 2), dtype=np.float32),  # second call: median metric
            np.ones((6, 2), dtype=np.float32),  # third call: MAD metric
        ]

        traj = self.make_unique_trajectory(6, 2)
        calc = RMSFCalculator([traj], chunk_size=2, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            # Test all metrics
            calc.calculate_per_atom("mean", "mean", None, False)
            calc.calculate_per_atom("mean", "median", None, False)
            calc.calculate_per_atom("mean", "mad", None, False)

        # Verify 3 calls (1 per calculation, metric doesn't affect chunking)
        assert mock_squared_chunks.call_count == 3

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_preserves_calculation_order(self, mock_squared_chunks):
        """Test that chunking processes atoms in correct order.

        Calculation documentation:
        - 1 trajectory: 15 frames, 4 atoms
        - atom_batch_size=2 => 2 atom batches ([0:2], [2:4])
        - Chunking should preserve order: first batch then second batch
        """
        mock_squared_chunks.side_effect = [
            np.ones((15, 2), dtype=np.float32),  # atoms 0:2 (first batch)
            np.ones((15, 2), dtype=np.float32),  # atoms 2:4 (second batch)
        ]

        traj = self.make_unique_trajectory(15, 4)
        calc = RMSFCalculator([traj], chunk_size=6, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 2 calls in correct order
        assert mock_squared_chunks.call_count == 2
        call_args_list = mock_squared_chunks.call_args_list

        # First call should be atoms 0:2
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 2  # atom_end

        # Second call should be atoms 2:4
        assert call_args_list[1].args[2] == 2  # atom_start
        assert call_args_list[1].args[3] == 4  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_memory_efficiency(self, mock_squared_chunks):
        """Test chunking for memory efficiency with large trajectory.

        Calculation documentation:
        - 1 large trajectory: 100 frames, 10 atoms
        - atom_batch_size=5 => 2 atom batches ([0:5], [5:10])
        - Small chunk_size=10 enables memory-efficient frame processing
        - Chunking should handle large data efficiently
        """
        mock_squared_chunks.side_effect = [
            np.ones((100, 5), dtype=np.float32),  # atoms 0:5, 100 frames
            np.ones((100, 5), dtype=np.float32),  # atoms 5:10, 100 frames
        ]

        traj = self.make_unique_trajectory(100, 10)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=5):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 2 calls (1 trajectory x 2 atom batches)
        assert mock_squared_chunks.call_count == 2

        # Verify atom ranges for large trajectory
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 5  # atom_end
        assert call_args_list[1].args[2] == 5  # atom_start
        assert call_args_list[1].args[3] == 10  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_error_handling(self, mock_squared_chunks):
        """Test chunking error handling when computation fails.

        Calculation documentation:
        - Error in chunking computation should propagate correctly
        - Exception should be raised with appropriate message
        """
        mock_squared_chunks.side_effect = ValueError("No frames processed")

        traj = self.make_unique_trajectory(5, 3)
        calc = RMSFCalculator([traj], chunk_size=2, use_memmap=False)

        with pytest.raises(ValueError, match="No frames processed"):
            calc.calculate_per_atom("mean", "mean", None, False)

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_consistent_results(self, mock_squared_chunks):
        """Test chunking produces consistent results regardless of chunk size.

        Calculation documentation:
        - Same trajectory with different chunk_size values
        - atom_batch_size kept constant => chunking behavior should be identical
        - Different frame chunking should not affect final atom chunking calls
        """
        # Mock returns same data regardless of chunk size
        mock_squared_chunks.return_value = np.ones((10, 3), dtype=np.float32)

        traj = self.make_unique_trajectory(10, 3)

        # Test with different chunk sizes but same atom_batch_size
        calc1 = RMSFCalculator([traj], chunk_size=2, use_memmap=False)
        with patch.object(calc1, '_determine_atom_batch_size', return_value=3):
            result1 = calc1.calculate_per_atom("mean", "mean", None, False)

        calc2 = RMSFCalculator([traj], chunk_size=5, use_memmap=False)
        with patch.object(calc2, '_determine_atom_batch_size', return_value=3):
            result2 = calc2.calculate_per_atom("mean", "mean", None, False)

        # Both should have same number of chunking calls
        # (Different frame chunk_size shouldn't affect atom chunking)
        np.testing.assert_array_equal(result1, result2)

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_respects_memmap_setting(self, mock_squared_chunks):
        """Test chunking respects memmap setting throughout calculation.

        Calculation documentation:
        - 1 trajectory: 8 frames, 2 atoms
        - use_memmap=True should be preserved
        - atom_batch_size=1 => 2 atom batches
        - Memmap setting should not affect chunking behavior
        """
        mock_squared_chunks.side_effect = [
            np.ones((8, 1), dtype=np.float32),  # atoms 0:1
            np.ones((8, 1), dtype=np.float32),  # atoms 1:2
        ]

        traj = self.make_unique_trajectory(8, 2)
        calc = RMSFCalculator([traj], chunk_size=3, use_memmap=True)

        with patch.object(calc, '_determine_atom_batch_size', return_value=1):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify memmap setting preserved and chunking works
        assert calc.use_memmap is True
        assert mock_squared_chunks.call_count == 2

        # Verify atom ranges
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 1  # atom_end
        assert call_args_list[1].args[2] == 1  # atom_start
        assert call_args_list[1].args[3] == 2  # atom_end

    @patch('mdxplain.analysis.structure.calculators.rmsf_calculator.RMSFCalculator._squared_chunks')
    def test_chunking_handles_edge_cases(self, mock_squared_chunks):
        """Test chunking handles edge cases like single atom trajectories.

        Calculation documentation:
        - 1 trajectory: 7 frames, 1 atom (edge case: single atom)
        - atom_batch_size=1 => 1 atom batch
        - Odd number of frames with small chunk_size should work
        - Single atom should be processed correctly
        """
        mock_squared_chunks.side_effect = [
            np.ones((7, 1), dtype=np.float32),  # single atom, 7 frames
        ]

        traj = self.make_unique_trajectory(7, 1)
        calc = RMSFCalculator([traj], chunk_size=3, use_memmap=False)

        with patch.object(calc, '_determine_atom_batch_size', return_value=1):
            result = calc.calculate_per_atom("mean", "mean", None, False)

        # Verify 1 call (1 trajectory x 1 atom batch for single atom)
        assert mock_squared_chunks.call_count == 1

        # Verify single atom range
        call_args_list = mock_squared_chunks.call_args_list
        assert call_args_list[0].args[2] == 0  # atom_start
        assert call_args_list[0].args[3] == 1  # atom_end

        expected = [np.array([1.0], dtype=np.float32)]
        np.testing.assert_array_equal(result, expected)

    def test_chunking_verify_exact_frame_distribution(self):
        """Test that frames are distributed into correct chunks.

        Calculation documentation:
        - 1 trajectory: 10 frames, 3 atoms
        - chunk_size=3 => 4 chunks: [0:3], [3:6], [6:9], [9:10]
        - Verify each chunk contains exact expected frames
        - Track chunk_iterator calls to verify frame distribution
        """
        # Create trajectory with 10 frames
        traj = self.make_unique_trajectory(10, 3)
        calc = RMSFCalculator([traj], chunk_size=3, use_memmap=True)

        # Track which chunks are processed by _chunk_iterator
        original_chunk_iterator = calc._chunk_iterator
        processed_chunks = []

        def wrapper_chunk_iterator(trajectory):
            for chunk, start_idx, end_idx in original_chunk_iterator(trajectory):
                # Record which frames are in each chunk
                processed_chunks.append((start_idx, end_idx, chunk.n_frames))
                yield chunk, start_idx, end_idx

        # Replace with wrapper to track calls
        calc._chunk_iterator = wrapper_chunk_iterator

        with patch.object(calc, '_determine_atom_batch_size', return_value=3):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify exact frame distribution
        assert len(processed_chunks) == 4, f"Expected 4 chunks, got {len(processed_chunks)}"

        # Check each chunk contains expected frames
        assert processed_chunks[0] == (0, 3, 3), f"Chunk 0: expected (0, 3, 3), got {processed_chunks[0]}"
        assert processed_chunks[1] == (3, 6, 3), f"Chunk 1: expected (3, 6, 3), got {processed_chunks[1]}"
        assert processed_chunks[2] == (6, 9, 3), f"Chunk 2: expected (6, 9, 3), got {processed_chunks[2]}"
        assert processed_chunks[3] == (9, 10, 1), f"Chunk 3: expected (9, 10, 1), got {processed_chunks[3]}"

    def test_chunking_exact_division_frames(self):
        """Test chunking when frames divide exactly into chunk_size.

        Calculation documentation:
        - 1 trajectory: 12 frames, 2 atoms
        - chunk_size=4 => 3 chunks exactly: [0:4], [4:8], [8:12]
        - Each chunk should have exactly 4 frames
        - No partial chunks expected
        """
        # Create trajectory with 12 frames (divisible by 4)
        traj = self.make_unique_trajectory(12, 2)
        calc = RMSFCalculator([traj], chunk_size=4, use_memmap=True)

        # Track chunk distribution
        original_chunk_iterator = calc._chunk_iterator
        processed_chunks = []

        def wrapper_chunk_iterator(trajectory):
            for chunk, start_idx, end_idx in original_chunk_iterator(trajectory):
                processed_chunks.append((start_idx, end_idx, chunk.n_frames))
                yield chunk, start_idx, end_idx

        calc._chunk_iterator = wrapper_chunk_iterator

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify exact division
        assert len(processed_chunks) == 3, f"Expected 3 chunks, got {len(processed_chunks)}"

        # All chunks should have exactly 4 frames
        assert processed_chunks[0] == (0, 4, 4), f"Chunk 0: expected (0, 4, 4), got {processed_chunks[0]}"
        assert processed_chunks[1] == (4, 8, 4), f"Chunk 1: expected (4, 8, 4), got {processed_chunks[1]}"
        assert processed_chunks[2] == (8, 12, 4), f"Chunk 2: expected (8, 12, 4), got {processed_chunks[2]}"

    def test_chunking_single_frame_chunks(self):
        """Test chunking with chunk_size=1 (one frame per chunk).

        Calculation documentation:
        - 1 trajectory: 5 frames, 3 atoms
        - chunk_size=1 => 5 chunks: [0:1], [1:2], [2:3], [3:4], [4:5]
        - Each chunk contains exactly 1 frame
        - Maximum granularity chunking
        """
        # Create trajectory with 5 frames
        traj = self.make_unique_trajectory(5, 3)
        calc = RMSFCalculator([traj], chunk_size=1, use_memmap=True)

        # Track chunk distribution
        original_chunk_iterator = calc._chunk_iterator
        processed_chunks = []

        def wrapper_chunk_iterator(trajectory):
            for chunk, start_idx, end_idx in original_chunk_iterator(trajectory):
                processed_chunks.append((start_idx, end_idx, chunk.n_frames))
                yield chunk, start_idx, end_idx

        calc._chunk_iterator = wrapper_chunk_iterator

        with patch.object(calc, '_determine_atom_batch_size', return_value=3):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify single-frame chunks
        assert len(processed_chunks) == 5, f"Expected 5 chunks, got {len(processed_chunks)}"

        # Each chunk should have exactly 1 frame
        expected_chunks = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)]
        for i, expected in enumerate(expected_chunks):
            assert processed_chunks[i] == expected, f"Chunk {i}: expected {expected}, got {processed_chunks[i]}"

    def test_chunking_large_chunk_size(self):
        """Test chunking when chunk_size > n_frames.

        Calculation documentation:
        - 1 trajectory: 6 frames, 2 atoms
        - chunk_size=10 > 6 frames => single chunk [0:6]
        - Should process entire trajectory as one chunk
        - Chunking becomes effectively disabled
        """
        # Create trajectory with 6 frames, use chunk_size > n_frames
        traj = self.make_unique_trajectory(6, 2)
        calc = RMSFCalculator([traj], chunk_size=10, use_memmap=True)

        # Track chunk distribution
        original_chunk_iterator = calc._chunk_iterator
        processed_chunks = []

        def wrapper_chunk_iterator(trajectory):
            for chunk, start_idx, end_idx in original_chunk_iterator(trajectory):
                processed_chunks.append((start_idx, end_idx, chunk.n_frames))
                yield chunk, start_idx, end_idx

        calc._chunk_iterator = wrapper_chunk_iterator

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify single large chunk
        assert len(processed_chunks) == 1, f"Expected 1 chunk, got {len(processed_chunks)}"
        assert processed_chunks[0] == (0, 6, 6), f"Chunk: expected (0, 6, 6), got {processed_chunks[0]}"

    def test_chunking_consistency_with_no_chunking(self):
        """Test that frame chunking produces same results as no chunking.

        Calculation documentation:
        - Same trajectory processed with different chunk sizes
        - chunk_size=None (no chunking) vs chunk_size=2 (with chunking)
        - Results should be numerically identical
        - Chunking is optimization, not algorithm change
        """
        # Create reproducible trajectory
        traj = self.make_unique_trajectory(8, 3)

        # Calculate without chunking (use_memmap=False disables chunking)
        calc_no_chunk = RMSFCalculator([traj], chunk_size=None, use_memmap=False)
        with patch.object(calc_no_chunk, '_determine_atom_batch_size', return_value=3):
            result_no_chunk = calc_no_chunk.calculate_per_atom("mean", "mean", None, False)

        # Calculate with chunking
        calc_with_chunk = RMSFCalculator([traj], chunk_size=2, use_memmap=True)
        with patch.object(calc_with_chunk, '_determine_atom_batch_size', return_value=3):
            result_with_chunk = calc_with_chunk.calculate_per_atom("mean", "mean", None, False)

        # Results should be identical (chunking is just memory optimization)
        np.testing.assert_array_equal(result_no_chunk, result_with_chunk,
                                      err_msg="Chunking should produce identical results to no chunking")

    def test_chunking_combined_frame_and_atom_batching(self):
        """Test frame chunking combined with atom batching.

        Calculation documentation:
        - 1 trajectory: 10 frames, 6 atoms
        - chunk_size=3 => 4 frame chunks: [0:3], [3:6], [6:9], [9:10]
        - atom_batch_size=2 => 3 atom batches: [0:2], [2:4], [4:6]
        - Total: 4 frame chunks × 3 atom batches = 12 _squared_chunks calls
        - Verify both frame and atom chunking work together
        """
        # Create trajectory
        traj = self.make_unique_trajectory(10, 6)
        calc = RMSFCalculator([traj], chunk_size=3, use_memmap=True)

        # Track both frame chunks and atom batches
        original_chunk_iterator = calc._chunk_iterator
        processed_frame_chunks = []

        def wrapper_chunk_iterator(trajectory):
            for chunk, start_idx, end_idx in original_chunk_iterator(trajectory):
                processed_frame_chunks.append((start_idx, end_idx, chunk.n_frames))
                yield chunk, start_idx, end_idx

        calc._chunk_iterator = wrapper_chunk_iterator

        # Track _squared_chunks calls to verify atom batching
        original_squared_chunks = calc._squared_chunks
        squared_chunks_calls = []

        def wrapper_squared_chunks(trajectory, reference, atom_start, atom_end):
            squared_chunks_calls.append((atom_start, atom_end))
            # Call original method to ensure _chunk_iterator gets called
            return original_squared_chunks(trajectory, reference, atom_start, atom_end)

        calc._squared_chunks = wrapper_squared_chunks

        with patch.object(calc, '_determine_atom_batch_size', return_value=2):
            calc.calculate_per_atom("mean", "mean", None, False)

        # Verify frame chunking (4 frame chunks × 3 atom batches = 12 total calls)
        assert len(processed_frame_chunks) == 12, f"Expected 12 frame chunks, got {len(processed_frame_chunks)}"

        # Each atom batch should process the same 4 frame chunks
        expected_frame_chunks = [(0, 3, 3), (3, 6, 3), (6, 9, 3), (9, 10, 1)]
        for atom_batch in range(3):
            start_idx = atom_batch * 4
            actual_chunks = processed_frame_chunks[start_idx:start_idx + 4]
            assert actual_chunks == expected_frame_chunks, f"Atom batch {atom_batch} frame chunks mismatch"

        # Verify atom batching (3 atom batches expected)
        # _squared_chunks is called once per atom batch (not per frame chunk)
        assert len(squared_chunks_calls) == 3, f"Expected 3 squared_chunks calls, got {len(squared_chunks_calls)}"

        # Verify atom batch ranges
        expected_atom_batches = [(0, 2), (2, 4), (4, 6)]
        for i, expected in enumerate(expected_atom_batches):
            assert squared_chunks_calls[i] == expected, f"Call {i}: expected {expected}, got {squared_chunks_calls[i]}"
