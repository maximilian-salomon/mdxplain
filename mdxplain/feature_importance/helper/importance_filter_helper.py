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

"""
Redundancy filter for feature importance results.

Collapses near-identical neighbour features into one representative per
coupling. A single physical coupling appears in the raw features as many
almost-identical neighbour pairs; this helper keeps only the strongest and
records how many neighbours were merged into it.

Sequence logic is chain-aware. Chains are derived once from the full,
index-ordered residue list by walking it and starting a new chain segment at
every sequence-id break (``|delta seqid| != 1``), the same convention used when
the residue pairs are generated. Two residues are the same chain only when they
share that segment id, so neither the long-range filter nor the merge ever
works across a chain break.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.feature_metadata_utils import FeatureMetadataUtils

# A residue anchor is (chain_segment_id, sequence_id).
Anchor = Tuple[int, int]


class ImportanceFilterHelper:
    """
    Collapse redundant neighbour features into one representative each.

    Works for pair features (two residues, e.g. contacts/distances) and for
    single-residue features (e.g. torsions). For pairs a long-range filter
    first discards trivial within-chain short-range couplings; then a greedy,
    strength-ordered pass merges near-identical neighbours into the strongest
    representative and counts them. Filtered features are set to zero rather
    than removed, so the feature index space stays aligned.

    Examples
    --------
    >>> chain_of = ImportanceFilterHelper.build_chain_map(residue_labels)
    >>> filtered, counts = ImportanceFilterHelper.filter_comparison(
    ...     importances, feature_metadata, chain_of,
    ...     min_sequence_separation=20, merge_radius=5,
    ... )
    >>> counts  # {representative_index: number_of_merged_neighbours}
    """

    @staticmethod
    def build_chain_map(residue_labels: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Assign a chain-segment id to every residue index.

        Walks the residues ordered by topology index and starts a new segment
        whenever consecutive sequence ids do not differ by exactly one, which
        marks a chain break (or a sequence gap).

        Parameters
        ----------
        residue_labels : List[Dict[str, Any]]
            Residue label dicts (from ``res_label_data``), each with an
            ``index`` and a ``seqid``

        Returns
        -------
        Dict[int, int]
            Mapping from residue index to its chain-segment id

        Examples
        --------
        >>> chain_of = ImportanceFilterHelper.build_chain_map(residue_labels)
        >>> chain_of[0]
        0
        """
        ordered = sorted(residue_labels, key=lambda entry: entry["index"])
        chain_of: Dict[int, int] = {}
        segment = 0
        previous_seqid: Optional[int] = None
        for entry in ordered:
            seqid = int(entry["seqid"])
            if previous_seqid is not None and abs(seqid - previous_seqid) != 1:
                segment += 1
            chain_of[int(entry["index"])] = segment
            previous_seqid = seqid
        return chain_of

    @staticmethod
    def filter_comparison(
        importances: np.ndarray,
        feature_metadata: Optional[List[Any]],
        chain_of: Dict[int, int],
        min_sequence_separation: int,
        merge_radius: int,
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Filter one sub-comparison's importances.

        Parameters
        ----------
        importances : np.ndarray
            Feature importance scores, shape (n_features,)
        feature_metadata : list or None
            Feature metadata (from ``get_selected_metadata``) used to map each
            feature index to its residue anchors
        chain_of : Dict[int, int]
            Mapping from residue index to chain-segment id
        min_sequence_separation : int
            Minimum within-chain sequence separation to keep a pair feature
        merge_radius : int
            Maximum within-chain sequence distance for two features to count
            as the same event

        Returns
        -------
        Tuple[np.ndarray, Dict[int, int]]
            (filtered_importances, merged_counts). filtered_importances has the
            same shape as the input with non-representative features set to
            zero; merged_counts maps each representative index to the number of
            merged neighbours.

        Examples
        --------
        >>> filtered, counts = ImportanceFilterHelper.filter_comparison(
        ...     imp, metadata, chain_of, 20, 5
        ... )
        """
        anchors = ImportanceFilterHelper._collect_anchors(
            feature_metadata, len(importances), chain_of
        )
        candidates = ImportanceFilterHelper._long_range_candidates(
            anchors, importances, min_sequence_separation
        )
        representatives, merged_counts = ImportanceFilterHelper._greedy_dedup(
            candidates, anchors, importances, merge_radius
        )

        filtered = np.zeros_like(importances)
        for idx in representatives:
            filtered[idx] = importances[idx]
        return filtered, merged_counts

    @staticmethod
    def _collect_anchors(
        feature_metadata: Optional[List[Any]],
        n_features: int,
        chain_of: Dict[int, int],
    ) -> Dict[int, List[Anchor]]:
        """
        Map each feature index to its residue anchors.

        Parameters
        ----------
        feature_metadata : list or None
            Feature metadata list
        n_features : int
            Number of features
        chain_of : Dict[int, int]
            Mapping from residue index to chain-segment id

        Returns
        -------
        Dict[int, List[Anchor]]
            Mapping from feature index to a list of (segment, seqid) anchors
        """
        anchors: Dict[int, List[Anchor]] = {}
        for idx in range(n_features):
            residues = FeatureMetadataUtils.get_feature_residues(
                feature_metadata, idx
            )
            anchors[idx] = [
                ImportanceFilterHelper._anchor(res, chain_of)
                for res in residues
            ]
        return anchors

    @staticmethod
    def _anchor(residue: Dict[str, Any], chain_of: Dict[int, int]) -> Anchor:
        """
        Build a (segment, seqid) anchor for one residue.

        Residues without a known chain segment get a unique negative segment so
        they never match any other residue.

        Parameters
        ----------
        residue : Dict[str, Any]
            Residue dict with an ``index`` and a ``seqid``
        chain_of : Dict[int, int]
            Mapping from residue index to chain-segment id

        Returns
        -------
        Anchor
            (chain_segment_id, sequence_id)
        """
        index = int(residue["index"])
        segment = chain_of.get(index, -(index + 1))
        return (segment, int(residue["seqid"]))

    @staticmethod
    def _long_range_candidates(
        anchors: Dict[int, List[Anchor]],
        importances: np.ndarray,
        min_sequence_separation: int,
    ) -> List[int]:
        """
        Select non-zero features that pass the long-range filter.

        Parameters
        ----------
        anchors : Dict[int, List[Anchor]]
            Feature index to residue anchors
        importances : np.ndarray
            Feature importance scores
        min_sequence_separation : int
            Minimum within-chain sequence separation for pair features

        Returns
        -------
        List[int]
            Feature indices that survive the long-range filter
        """
        kept = []
        for idx, anchor_list in anchors.items():
            if importances[idx] <= 0:
                continue
            if ImportanceFilterHelper._passes_long_range(
                anchor_list, min_sequence_separation
            ):
                kept.append(idx)
        return kept

    @staticmethod
    def _passes_long_range(anchors: List[Anchor], min_sep: int) -> bool:
        """
        Check the long-range criterion for one feature.

        Single-residue features always pass. Pair features pass when the two
        residues are on different chains or are at least ``min_sep`` apart in
        sequence within the same chain.

        Parameters
        ----------
        anchors : List[Anchor]
            Residue anchors of the feature (one or two)
        min_sep : int
            Minimum within-chain sequence separation

        Returns
        -------
        bool
            True if the feature passes the long-range filter
        """
        if len(anchors) < 2:
            return True
        first, second = anchors[0], anchors[1]
        if not ImportanceFilterHelper._same_chain(first, second):
            return True
        return ImportanceFilterHelper._seq_distance(first, second) >= min_sep

    @staticmethod
    def _greedy_dedup(
        candidates: List[int],
        anchors: Dict[int, List[Anchor]],
        importances: np.ndarray,
        merge_radius: int,
    ) -> Tuple[List[int], Dict[int, int]]:
        """
        Merge near-identical neighbours into strongest representatives.

        Parameters
        ----------
        candidates : List[int]
            Candidate feature indices (already long-range filtered)
        anchors : Dict[int, List[Anchor]]
            Feature index to residue anchors
        importances : np.ndarray
            Feature importance scores
        merge_radius : int
            Maximum within-chain sequence distance to merge

        Returns
        -------
        Tuple[List[int], Dict[int, int]]
            (representatives, merged_counts)
        """
        order = sorted(
            candidates, key=lambda idx: importances[idx], reverse=True
        )
        representatives: List[int] = []
        merged_counts: Dict[int, int] = {}
        for idx in order:
            rep = ImportanceFilterHelper._find_close_representative(
                idx, representatives, anchors, merge_radius
            )
            if rep is None:
                representatives.append(idx)
            else:
                merged_counts[rep] = merged_counts.get(rep, 0) + 1
        return representatives, merged_counts

    @staticmethod
    def _find_close_representative(
        idx: int,
        representatives: List[int],
        anchors: Dict[int, List[Anchor]],
        merge_radius: int,
    ) -> Optional[int]:
        """
        Find an already-kept representative that this feature merges into.

        Parameters
        ----------
        idx : int
            Candidate feature index
        representatives : List[int]
            Indices already kept as representatives
        anchors : Dict[int, List[Anchor]]
            Feature index to residue anchors
        merge_radius : int
            Maximum within-chain sequence distance to merge

        Returns
        -------
        int or None
            The representative index to merge into, or None for a new one
        """
        for rep in representatives:
            if ImportanceFilterHelper._anchors_close(
                anchors[idx], anchors[rep], merge_radius
            ):
                return rep
        return None

    @staticmethod
    def _anchors_close(
        anchors_a: List[Anchor], anchors_b: List[Anchor], radius: int
    ) -> bool:
        """
        Check whether two features describe the same event.

        Parameters
        ----------
        anchors_a : List[Anchor]
            Residue anchors of the first feature
        anchors_b : List[Anchor]
            Residue anchors of the second feature
        radius : int
            Maximum within-chain sequence distance

        Returns
        -------
        bool
            True if the two features are close enough to be one event
        """
        if len(anchors_a) != len(anchors_b):
            return False
        if len(anchors_a) == 1:
            return ImportanceFilterHelper._residues_close(
                anchors_a[0], anchors_b[0], radius
            )
        return ImportanceFilterHelper._pairs_close(anchors_a, anchors_b, radius)

    @staticmethod
    def _pairs_close(
        pair_a: List[Anchor], pair_b: List[Anchor], radius: int
    ) -> bool:
        """
        Check pair closeness, accounting for the swappability of the ends.

        Parameters
        ----------
        pair_a : List[Anchor]
            Two anchors of the first pair
        pair_b : List[Anchor]
            Two anchors of the second pair
        radius : int
            Maximum within-chain sequence distance

        Returns
        -------
        bool
            True if the pairs match directly or with the ends swapped
        """
        close = ImportanceFilterHelper._residues_close
        direct = close(pair_a[0], pair_b[0], radius) and close(
            pair_a[1], pair_b[1], radius
        )
        swapped = close(pair_a[0], pair_b[1], radius) and close(
            pair_a[1], pair_b[0], radius
        )
        return direct or swapped

    @staticmethod
    def _residues_close(
        anchor_a: Anchor, anchor_b: Anchor, radius: int
    ) -> bool:
        """
        Check whether two residues are close within the same chain.

        Parameters
        ----------
        anchor_a : Anchor
            First residue anchor (segment, seqid)
        anchor_b : Anchor
            Second residue anchor (segment, seqid)
        radius : int
            Maximum within-chain sequence distance

        Returns
        -------
        bool
            True if same chain and within radius residues
        """
        if not ImportanceFilterHelper._same_chain(anchor_a, anchor_b):
            return False
        return (
            ImportanceFilterHelper._seq_distance(anchor_a, anchor_b) <= radius
        )

    @staticmethod
    def _same_chain(anchor_a: Anchor, anchor_b: Anchor) -> bool:
        """
        Check whether two residues lie on the same chain segment.

        Parameters
        ----------
        anchor_a : Anchor
            First residue anchor (segment, seqid)
        anchor_b : Anchor
            Second residue anchor (segment, seqid)

        Returns
        -------
        bool
            True if both residues share the same chain segment
        """
        return anchor_a[0] == anchor_b[0]

    @staticmethod
    def _seq_distance(anchor_a: Anchor, anchor_b: Anchor) -> int:
        """
        Sequence distance between two residues (sequence-id distance).

        Parameters
        ----------
        anchor_a : Anchor
            First residue anchor (segment, seqid)
        anchor_b : Anchor
            Second residue anchor (segment, seqid)

        Returns
        -------
        int
            Absolute sequence-id distance
        """
        return abs(anchor_a[1] - anchor_b[1])
