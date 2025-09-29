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
"""Unit tests for residue aggregation helper functions."""

import numpy as np
import pytest

from mdxplain.analysis.structure.helpers.residue_aggregation_helper import (
    aggregate_residues_jit,
    AGGREGATOR_CODES
)


class TestResidueAggregationHelper:
    """Test residue aggregation functions with simple, calculable data."""

    def test_aggregate_mean_simple(self):
        """
        Test mean aggregation with simple data.

        2 residues with 2 atoms each, known values.
        """
        atom_values = np.array([1.0, 3.0, 2.0, 4.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 2, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["mean"]
        )

        expected = np.array([2.0, 3.0])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_mean_single_atom_per_residue(self):
        """
        Test mean aggregation with single atom per residue.

        Should return original values.
        """
        atom_values = np.array([1.5, 2.5, 3.5])
        group_indices = np.array([0, 1, 2])
        group_boundaries = np.array([0, 1, 2, 3])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["mean"]
        )

        expected = np.array([1.5, 2.5, 3.5])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_mean_unequal_residues(self):
        """
        Test mean aggregation with unequal number of atoms per residue.

        Residue 0: 1 atom, Residue 1: 3 atoms.
        """
        atom_values = np.array([10.0, 1.0, 2.0, 3.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 1, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["mean"]
        )

        expected = np.array([10.0, 2.0])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_median_simple(self):
        """
        Test median aggregation with simple data.

        2 residues with 2 atoms each, known values.
        """
        atom_values = np.array([1.0, 3.0, 2.0, 4.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 2, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["median"]
        )

        expected = np.array([2.0, 3.0])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_median_single_atom_per_residue(self):
        """
        Test median aggregation with single atom per residue.

        Should return original values.
        """
        atom_values = np.array([1.5, 2.5, 3.5])
        group_indices = np.array([0, 1, 2])
        group_boundaries = np.array([0, 1, 2, 3])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["median"]
        )

        expected = np.array([1.5, 2.5, 3.5])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_median_unequal_residues(self):
        """
        Test median aggregation with unequal number of atoms per residue.

        Residue 0: 1 atom, Residue 1: 3 atoms.
        """
        atom_values = np.array([10.0, 1.0, 2.0, 3.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 1, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["median"]
        )

        expected = np.array([10.0, 2.0])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_rms_simple(self):
        """
        Test RMS aggregation with simple data.

        2 residues with 2 atoms each. RMS = sqrt(mean(values²)).
        Residue 0: [1,3] -> sqrt((1²+3²)/2) = sqrt(5) ≈ 2.236
        Residue 1: [2,4] -> sqrt((2²+4²)/2) = sqrt(10) ≈ 3.162
        """
        atom_values = np.array([1.0, 3.0, 2.0, 4.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 2, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms"]
        )

        expected = np.array([np.sqrt(5), np.sqrt(10)])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_rms_single_atom_per_residue(self):
        """
        Test RMS aggregation with single atom per residue.

        Should return original values (RMS of single value is the value itself).
        """
        atom_values = np.array([1.5, 2.5, 3.5])
        group_indices = np.array([0, 1, 2])
        group_boundaries = np.array([0, 1, 2, 3])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms"]
        )

        expected = np.array([1.5, 2.5, 3.5])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_rms_unequal_residues(self):
        """
        Test RMS aggregation with unequal number of atoms per residue.

        Residue 0: 1 atom [10] -> RMS = 10
        Residue 1: 3 atoms [1,2,3] -> RMS = sqrt((1²+2²+3²)/3) = sqrt(14/3) ≈ 2.160
        """
        atom_values = np.array([10.0, 1.0, 2.0, 3.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 1, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms"]
        )

        expected = np.array([10.0, np.sqrt(14/3)])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_rms_median_simple(self):
        """
        Test RMS-median aggregation with simple data.

        2 residues with 2 atoms each. RMS-median = sqrt(median(values²)).
        Residue 0: [1,3] -> sqrt(median([1²,3²])) = sqrt(median([1,9])) = sqrt(5) ≈ 2.236
        Residue 1: [2,4] -> sqrt(median([2²,4²])) = sqrt(median([4,16])) = sqrt(10) ≈ 3.162
        """
        atom_values = np.array([1.0, 3.0, 2.0, 4.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 2, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms_median"]
        )

        # Median of [1²,3²] = median([1,9]) = 5.0
        # Median of [2²,4²] = median([4,16]) = 10.0
        expected = np.array([np.sqrt(5), np.sqrt(10)])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_rms_median_single_atom_per_residue(self):
        """
        Test RMS-median aggregation with single atom per residue.

        Should return original values (median of single squared value is the value squared).
        """
        atom_values = np.array([1.5, 2.5, 3.5])
        group_indices = np.array([0, 1, 2])
        group_boundaries = np.array([0, 1, 2, 3])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms_median"]
        )

        expected = np.array([1.5, 2.5, 3.5])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregate_rms_median_unequal_residues(self):
        """
        Test RMS-median aggregation with unequal number of atoms per residue.

        Residue 0: 1 atom [10] -> sqrt(median([10²])) = 10
        Residue 1: 3 atoms [1,2,3] -> sqrt(median([1²,2²,3²])) = sqrt(median([1,4,9])) = sqrt(4) = 2
        """
        atom_values = np.array([10.0, 1.0, 2.0, 3.0])
        group_indices = np.array([0, 1, 2, 3])
        group_boundaries = np.array([0, 1, 4])

        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms_median"]
        )

        # Median of [1²,2²,3²] = median([1,4,9]) = 4
        expected = np.array([10.0, 2.0])

        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_aggregator_codes_mapping(self):
        """
        Test that aggregator codes are correctly defined.

        Validates the mapping dictionary used by JIT function.
        """
        expected_codes = {
            "mean": 0,
            "median": 1,
            "rms": 2,
            "rms_median": 3,
        }

        assert AGGREGATOR_CODES == expected_codes

    def test_edge_case_single_residue_single_atom(self):
        """
        Test minimal case: single residue with single atom.

        This is the smallest possible input for the aggregation function.
        """
        atom_values = np.array([5.0])
        group_indices = np.array([0])
        group_boundaries = np.array([0, 1])

        # Test all aggregation methods
        for method, code in AGGREGATOR_CODES.items():
            result = aggregate_residues_jit(
                atom_values, group_indices, group_boundaries, code
            )

            expected = np.array([5.0])
            np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_edge_case_large_residue(self):
        """
        Test residue with many atoms (10 atoms).

        Tests performance and correctness with larger groups.
        """
        # 10 atoms: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        atom_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        group_indices = np.arange(10)
        group_boundaries = np.array([0, 10])

        # Test mean: (1+2+...+10)/10 = 55/10 = 5.5
        result_mean = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["mean"]
        )
        np.testing.assert_array_almost_equal(result_mean, [5.5], decimal=6)

        # Test median: median([1,2,3,4,5,6,7,8,9,10]) = (5+6)/2 = 5.5
        result_median = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["median"]
        )
        np.testing.assert_array_almost_equal(result_median, [5.5], decimal=6)

        # Test RMS: sqrt((1²+2²+...+10²)/10) = sqrt(385/10) = sqrt(38.5) ≈ 6.205
        result_rms = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms"]
        )
        expected_rms = np.sqrt(385/10)
        np.testing.assert_array_almost_equal(result_rms, [expected_rms], decimal=6)

    def test_edge_case_zero_and_negative_values(self):
        """
        Test aggregation with zero and negative values.

        Validates correct handling of special numeric values.
        """
        # Mix of negative, zero, and positive values
        atom_values = np.array([-2.0, 0.0, 2.0, -1.0, 1.0, 0.0])
        group_indices = np.array([0, 1, 2, 3, 4, 5])
        group_boundaries = np.array([0, 3, 6])

        # Residue 0: [-2, 0, 2]
        # Residue 1: [-1, 1, 0]

        # Test mean: [-2+0+2]/3 = 0, [-1+1+0]/3 = 0
        result_mean = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["mean"]
        )
        np.testing.assert_array_almost_equal(result_mean, [0.0, 0.0], decimal=6)

        # Test median: median([-2,0,2]) = 0, median([-1,0,1]) = 0
        result_median = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["median"]
        )
        np.testing.assert_array_almost_equal(result_median, [0.0, 0.0], decimal=6)

        # Test RMS: sqrt((4+0+4)/3) = sqrt(8/3), sqrt((1+1+0)/3) = sqrt(2/3)
        result_rms = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["rms"]
        )
        expected_rms = [np.sqrt(8/3), np.sqrt(2/3)]
        np.testing.assert_array_almost_equal(result_rms, expected_rms, decimal=6)

    def test_error_empty_residue_group(self):
        """
        Test error handling with truly empty residue group.

        This should cause a division by zero or similar error.
        """
        atom_values = np.array([1.0, 2.0])
        group_indices = np.array([0, 1])
        # boundaries [0, 0, 2] means residue 0 has no atoms (start=0, end=0)
        group_boundaries = np.array([0, 0, 2])

        # This should raise an error due to empty group (division by zero)
        with pytest.raises((ZeroDivisionError, ValueError)):
            aggregate_residues_jit(
                atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["mean"]
            )

    def test_error_mismatched_boundaries(self):
        """
        Test error handling with inconsistent boundary specifications.

        Group boundaries that don't match the actual group_indices should fail.
        """
        atom_values = np.array([1.0, 2.0, 3.0, 4.0])
        group_indices = np.array([0, 1, 2, 3])
        # boundaries claim 3 residues but only provide indices for 4 atoms
        group_boundaries = np.array([0, 2, 4, 8])  # Invalid: points beyond available atoms

        # This can raise different errors depending on implementation details
        with pytest.raises((IndexError, ZeroDivisionError)):
            aggregate_residues_jit(
                atom_values, group_indices, group_boundaries, AGGREGATOR_CODES["mean"]
            )

    def test_error_invalid_aggregator_code(self):
        """
        Test behavior with invalid aggregator code.

        Aggregator codes outside the valid range [0-3] may produce undefined results.
        """
        atom_values = np.array([1.0, 2.0])
        group_indices = np.array([0, 1])
        group_boundaries = np.array([0, 2])

        # Test invalid aggregator code (not in 0-3 range)
        # The JIT function may not validate this and could return undefined results
        result = aggregate_residues_jit(
            atom_values, group_indices, group_boundaries, 99  # Invalid code
        )

        # The result is undefined but should at least have the correct shape
        assert result.shape == (1,)
        # Could be NaN, 0, or some other value - implementation dependent