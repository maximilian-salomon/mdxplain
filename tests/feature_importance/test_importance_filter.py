# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Opus 4.8).
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

"""Unit tests for the feature importance redundancy filter."""

import numpy as np

from mdxplain.feature_importance.helper.importance_filter_helper import (
    ImportanceFilterHelper,
)


def _feat(anchors, ftype="distances"):
    """Build a feature metadata entry from (index, seqid) anchors."""
    partners = np.array(
        [{"residue": {"index": i, "seqid": s}} for (i, s) in anchors],
        dtype=object,
    )
    return {"features": partners, "type": ftype}


def _contiguous_chain(start, stop):
    """Residue labels for one chain with contiguous seqids."""
    return [{"index": i, "seqid": i} for i in range(start, stop)]


def test_build_chain_map_starts_new_segment_at_seqid_break():
    """A sequence-id jump starts a new chain segment."""
    labels = [
        {"index": 0, "seqid": 1},
        {"index": 1, "seqid": 2},
        {"index": 2, "seqid": 100},
        {"index": 3, "seqid": 101},
    ]
    assert ImportanceFilterHelper.build_chain_map(labels) == {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    }


def test_long_range_pairs_kept_short_range_dropped():
    """Within-chain short-range pairs are zeroed, long-range pairs kept."""
    chain_of = ImportanceFilterHelper.build_chain_map(
        _contiguous_chain(10, 61)
    )
    metadata = [
        _feat([(10, 10), (50, 50)]),
        _feat([(10, 10), (15, 15)]),
    ]
    importances = np.array([0.9, 0.8])

    filtered, counts = ImportanceFilterHelper.filter_comparison(
        importances, metadata, chain_of, 20, 5
    )

    assert filtered[0] == 0.9
    assert filtered[1] == 0.0
    assert counts == {}


def test_near_neighbours_merge_into_strongest_pair():
    """Near-identical neighbour pairs collapse into the strongest one."""
    chain_of = ImportanceFilterHelper.build_chain_map(
        _contiguous_chain(10, 61)
    )
    metadata = [
        _feat([(10, 10), (50, 50)]),
        _feat([(11, 11), (51, 51)]),
        _feat([(12, 12), (49, 49)]),
    ]
    importances = np.array([0.9, 0.8, 0.7])

    filtered, counts = ImportanceFilterHelper.filter_comparison(
        importances, metadata, chain_of, 20, 5
    )

    assert filtered[0] == 0.9
    assert filtered[1] == 0.0
    assert filtered[2] == 0.0
    assert counts == {0: 2}


def test_single_features_merge_within_chain():
    """Adjacent single-residue features collapse into the strongest one."""
    chain_of = ImportanceFilterHelper.build_chain_map(
        _contiguous_chain(0, 20)
    )
    metadata = [
        _feat([(5, 5)], "torsions"),
        _feat([(7, 7)], "torsions"),
    ]
    importances = np.array([0.9, 0.4])

    filtered, counts = ImportanceFilterHelper.filter_comparison(
        importances, metadata, chain_of, 20, 5
    )

    assert filtered[0] == 0.9
    assert filtered[1] == 0.0
    assert counts == {0: 1}


def test_filter_never_merges_across_chain_offset_collision():
    """Residues in different chains that share the seqid/index offset stay
    separate (the segment scan is not fooled by a matching offset).
    """
    labels = [
        {"index": 0, "seqid": 1},
        {"index": 1, "seqid": 2},
        {"index": 2, "seqid": 100},
        {"index": 3, "seqid": 101},
        {"index": 4, "seqid": 5},
        {"index": 5, "seqid": 6},
    ]
    chain_of = ImportanceFilterHelper.build_chain_map(labels)
    assert chain_of[0] != chain_of[5]

    metadata = [
        _feat([(0, 1)], "torsions"),
        _feat([(5, 6)], "torsions"),
    ]
    importances = np.array([0.9, 0.5])

    filtered, counts = ImportanceFilterHelper.filter_comparison(
        importances, metadata, chain_of, 20, 5
    )

    assert filtered[0] == 0.9
    assert filtered[1] == 0.5
    assert counts == {}


def test_cross_chain_pairs_are_kept_as_long_range():
    """Pairs whose residues are on different chains are kept (inter-chain)."""
    labels = _contiguous_chain(0, 10) + [
        {"index": i, "seqid": i - 9} for i in range(10, 20)
    ]
    chain_of = ImportanceFilterHelper.build_chain_map(labels)
    metadata = [_feat([(2, 2), (12, 3)])]  # index 2 chain 0, index 12 chain 1
    importances = np.array([0.7])

    filtered, counts = ImportanceFilterHelper.filter_comparison(
        importances, metadata, chain_of, 20, 5
    )

    assert filtered[0] == 0.7
    assert counts == {}
