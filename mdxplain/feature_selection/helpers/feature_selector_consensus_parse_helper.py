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

"""Consensus nomenclature pattern parsing utilities."""

from typing import List, Optional, Dict, Any, Tuple, Set
import re


class FeatureSelectorConsensusParseHelper:
    """
    Helper class for parsing consensus nomenclature patterns.

    Provides static methods to parse consensus patterns like "7x50", "7x*", "7x-8x",
    and "*40-*50" to identify matching residue indices based on trajectory metadata.
    """

    @staticmethod
    def parse_consensus_category(param_tokens: List[str], features_list: List[list]) -> Tuple[List[int], Set[int]]:
        """
        Parse 'consensus' category and return matching feature indices plus matched residue indices.

        Uses unified interface like other parsers. Converts features_list to metadata
        format and delegates to parse_consensus_pattern.

        Parameters
        ----------
        param_tokens : List[str]
            List of parameter tokens (should be single pattern like ["7x50-8x50"])
        features_list : List[list]
            List of features from metadata (each feature is a list of partners)

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)

        Examples
        --------
        >>> # Single consensus position
        >>> indices, residue_indices = FeatureSelectorConsensusParseHelper.parse_consensus_category(
        ...     ["7x50"], features_list
        ... )

        >>> # Consensus range
        >>> indices, residue_indices = FeatureSelectorConsensusParseHelper.parse_consensus_category(
        ...     ["7x50-8x50"], features_list
        ... )
        """
        if not param_tokens:
            return [], set()

        # Join all tokens into single pattern
        pattern = " ".join(param_tokens)

        # Check for "all" prefix first (before splitting)
        # "all" is a special prefix for ranges, not a separate pattern
        has_all_prefix = pattern.strip().startswith("all ")
        if has_all_prefix:
            # Remove "all" prefix but remember it for later
            pattern = pattern.strip()[4:]  # Remove "all " (4 characters)

        # Split pattern by comma, semicolon, or whitespace for multi-value selections
        # This enables "1.50,3.50,5.50" or "7x50;8x50" or "1.50 3.50 5.50" style selections
        patterns = [p.strip() for p in re.split(r"[,;\s]+", pattern) if p.strip()]

        # Convert features_list to metadata format expected by parse_consensus_pattern
        metadata = {"features": features_list}

        # Process each pattern and combine results
        all_indices = []
        all_matched_residue_indices = set()

        for single_pattern in patterns:
            # Remove any remaining spaces in pattern (for ranges like "7x50 - 8x50" if present)
            single_pattern_cleaned = single_pattern.replace(" ", "")

            # Re-add "all" prefix if it was present in original pattern
            if has_all_prefix:
                single_pattern_cleaned = "all " + single_pattern_cleaned

            indices, matched_residue_indices = FeatureSelectorConsensusParseHelper.parse_consensus_pattern(
                metadata, single_pattern_cleaned
            )
            all_indices.extend(indices)
            all_matched_residue_indices.update(matched_residue_indices)

        return list(set(all_indices)), all_matched_residue_indices

    @staticmethod
    def parse_consensus_pattern(metadata: dict, pattern: str) -> Tuple[List[int], Set[int]]:
        """
        Parse consensus nomenclature pattern and return matching feature indices plus residue indices.

        Parameters
        ----------
        metadata : dict
            Trajectory metadata with residue information
        pattern : str
            Consensus pattern to parse:
            - Single: "7x50" → Find first entry containing substring
            - Wildcard: "7x*" → Find all containing "7x"
            - Range: "7x-8x" → From first "7x" to last "8x" (consensus != None)
            - Range All: "all 7x-8x" → Same but include None entries
            - Multi-Pattern: "*40-*50" → All blocks like 1x40-1x50, 2x40-2x50, etc.

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)

        Examples
        --------
        >>> # Single position
        >>> indices, residue_indices = FeatureSelectorConsensusParseHelper.parse_consensus_pattern(metadata, "7x50")
        >>> # ([145], {5})  # Feature 145, residue.index 5

        >>> # Wildcard
        >>> indices, residue_indices = FeatureSelectorConsensusParseHelper.parse_consensus_pattern(metadata, "7x*")
        >>> # ([145, 146, ...], {5, 6, ...})  # All features and residues with "7x" in consensus

        >>> # Range
        >>> indices, residue_indices = FeatureSelectorConsensusParseHelper.parse_consensus_pattern(metadata, "7x-8x")
        >>> # ([145, 146, ..., 180], {5, 6, ..., 20})  # From first "7x" to last "8x"
        """
        pattern = pattern.strip()

        # Check for "all" prefix
        include_none = pattern.startswith("all")
        if include_none:
            pattern = pattern[3:].strip()

        if "-" in pattern:
            return FeatureSelectorConsensusParseHelper._parse_range_pattern(
                metadata, pattern, include_none
            )
        else:
            return FeatureSelectorConsensusParseHelper._parse_single_pattern(
                metadata, pattern
            )

    @staticmethod
    def _find_first_substring_match(metadata: dict, substring: str) -> Optional[int]:
        """
        Find first feature index where consensus contains substring.

        Handles both partner-based features (distances) and simple features.

        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        substring : str
            Substring to search for (e.g., "7x", "40")

        Returns
        -------
        Optional[int]
            First matching feature index or None if not found
        """
        features = metadata.get("features", [])
        for i, feature in enumerate(features):
            if FeatureSelectorConsensusParseHelper._feature_matches_consensus(feature, substring, exact_match=False):
                return i
        return None
    
    @staticmethod
    def _find_last_substring_match(metadata: dict, substring: str) -> Optional[int]:
        """
        Find last feature index where consensus contains substring.

        Handles both partner-based features (distances) and simple features.

        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        substring : str
            Substring to search for (e.g., "8x", "50")

        Returns
        -------
        Optional[int]
            Last matching feature index or None if not found
        """
        features = metadata.get("features", [])
        last_idx = None
        for i, feature in enumerate(features):
            if FeatureSelectorConsensusParseHelper._feature_matches_consensus(feature, substring, exact_match=False):
                last_idx = i
        return last_idx
    
    @staticmethod
    def _find_all_substring_matches(metadata: dict, substring: str) -> List[int]:
        """
        Find all feature indices where consensus contains substring.

        Handles both partner-based features (distances) and simple features.

        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        substring : str
            Substring to search for

        Returns
        -------
        List[int]
            List of all matching feature indices
        """
        features = metadata.get("features", [])
        matches = []
        for i, feature in enumerate(features):
            if FeatureSelectorConsensusParseHelper._feature_matches_consensus(feature, substring, exact_match=False):
                matches.append(i)
        return matches

    @staticmethod
    def _feature_matches_consensus(feature: List[Dict[str, Any]], pattern: str, exact_match: bool = False) -> bool:
        """
        Check if a feature matches consensus pattern (substring or exact).

        Parameters
        ----------
        feature : list
            List of partners in the feature
        pattern : str
            Pattern to match (substring or exact)
        exact_match : bool, default=False
            If True, use exact matching; if False, use substring matching

        Returns
        -------
        bool
            True if at least one partner matches the consensus criteria
        """
        for partner in feature:
            consensus = partner.get("residue", {}).get("consensus")
            if exact_match:
                if consensus == pattern:
                    return True
            else:
                if consensus and pattern in str(consensus):
                    return True
        return False

    @staticmethod
    def _feature_has_consensus(feature: List[Dict[str, Any]]) -> bool:
        """
        Check if a feature has any consensus information.
        
        Feature structure is always: feature = [partner1, partner2, ...]
        Each partner has: {"residue": {"consensus": "value"}, ...}
        
        Parameters
        ----------
        feature : list
            List of partners in the feature
            
        Returns
        -------
        bool
            True if any partner has consensus information
        """
        for partner in feature:
            if isinstance(partner, dict):
                residue = partner.get("residue", {})
                consensus = residue.get("consensus")
                if consensus is not None:
                    return True
        return False

    @staticmethod
    def _parse_single_pattern(metadata: dict, pattern: str) -> Tuple[List[int], Set[int]]:
        """
        Parse single consensus position or wildcard pattern.

        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        pattern : str
            Single pattern (e.g., "7x50", "7x*")

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
        """
        search_substring = pattern.replace("*","") if "*" in pattern else pattern
        search_substring = search_substring.strip()

        features = metadata.get("features", [])
        matching_indices = []
        matched_residue_indices = set()

        for i, feature in enumerate(features):
            for partner in feature:
                consensus = partner.get("residue", {}).get("consensus")
                if consensus and search_substring in str(consensus):
                    matching_indices.append(i)
                    # Track this residue index
                    residue_index = partner.get("residue", {}).get("index")
                    if residue_index is not None:
                        matched_residue_indices.add(residue_index)
                    break  # Found match in this feature

        return matching_indices, matched_residue_indices

    @staticmethod
    def _parse_range_pattern(metadata: dict, pattern: str, include_none: bool) -> Tuple[List[int], Set[int]]:
        """
        Parse consensus range pattern with support for wildcards and multiple blocks.

        Parameters
        ----------
        metadata : dict
            Trajectory metadata with residue information
        pattern : str
            Range pattern (e.g., "7x-8x", "*40-*50")
        include_none : bool
            Whether to include None consensus entries

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
        """
        parts = pattern.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid consensus range pattern: {pattern}")

        start_pattern, end_pattern = parts

        # Handle wildcard ranges like "*40-*50"
        if "*" in start_pattern or "*" in end_pattern:
            return FeatureSelectorConsensusParseHelper._parse_consensus_range(
                metadata, start_pattern, end_pattern, include_none, is_wildcard=True
            )
        else:
            # Simple range like "3.50-4.50"
            return FeatureSelectorConsensusParseHelper._parse_consensus_range(
                metadata, start_pattern, end_pattern, include_none, is_wildcard=False
            )

    
    @staticmethod
    def _parse_consensus_range(
        metadata: dict, start_pattern: str, end_pattern: str, include_none: bool, is_wildcard: bool = True
    ) -> Tuple[List[int], Set[int]]:
        """
        Parse wildcard range like "*40-*50" with multiple blocks.

        Finds blocks like 1x40-1x50, 2x40-2x50, etc.
        Handles edge cases where start or end is missing.

        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        start_pattern : str
            Start pattern with wildcard (e.g., "*40")
        end_pattern : str
            End pattern with wildcard (e.g., "*50")
        include_none : bool
            Whether to include None consensus entries
        is_wildcard : bool, default=True
            If True, use substring matching; if False, use exact matching

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
        """
        features = metadata.get("features", [])
        partners = FeatureSelectorConsensusParseHelper._extract_unique_partners(features)

        start_search = start_pattern.replace("*", "")
        end_search = end_pattern.replace("*", "")

        match_func = FeatureSelectorConsensusParseHelper._create_match_function(is_wildcard)
        seqid_ranges = FeatureSelectorConsensusParseHelper._find_seqid_ranges(
            partners, start_search, end_search, match_func
        )

        return FeatureSelectorConsensusParseHelper._collect_features_in_seqid_ranges(
            features, seqid_ranges, include_none
        )

    @staticmethod
    def _create_match_function(is_wildcard: bool) -> callable:
        """
        Create pattern matching function based on wildcard mode.

        Parameters
        ----------
        is_wildcard : bool
            If True, use substring matching; if False, use exact matching

        Returns
        -------
        callable
            Function that takes (consensus, pattern) and returns bool
        """
        if is_wildcard:
            return lambda consensus, pattern: pattern in str(consensus) if consensus else False
        else:
            return lambda consensus, pattern: consensus == pattern

    @staticmethod
    def _collect_features_in_seqid_ranges(
        features: List[List[Dict[str, Any]]], seqid_ranges: List[Tuple[int, int]], include_none: bool
    ) -> Tuple[List[int], Set[int]]:
        """
        Collect features and residue indices for seqid ranges.

        Parameters
        ----------
        features : list
            List of features
        seqid_ranges : list
            List of (min_seqid, max_seqid) tuples
        include_none : bool
            Whether to include partners with consensus=None

        Returns
        -------
        Tuple[List[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
        """
        result_indices = set()
        matched_residue_indices = set()

        for min_seqid, max_seqid in seqid_ranges:
            indices, residue_indices = FeatureSelectorConsensusParseHelper._collect_features_in_single_range(
                features, min_seqid, max_seqid, include_none
            )
            result_indices.update(indices)
            matched_residue_indices.update(residue_indices)

        return sorted(result_indices), matched_residue_indices

    @staticmethod
    def _collect_features_in_single_range(
        features: List[List[Dict[str, Any]]], min_seqid: int, max_seqid: int, include_none: bool
    ) -> Tuple[Set[int], Set[int]]:
        """
        Collect features and residue indices for a single seqid range.

        Parameters
        ----------
        features : list
            List of features
        min_seqid : int
            Minimum seqid in range
        max_seqid : int
            Maximum seqid in range
        include_none : bool
            Whether to include partners with consensus=None

        Returns
        -------
        Tuple[Set[int], Set[int]]
            Tuple of (feature_indices, matched_residue_indices)
        """
        feature_indices = set()
        residue_indices = set()

        for i, feature in enumerate(features):
            for partner in feature:
                seqid = partner.get("residue", {}).get("seqid")
                consensus = partner.get("residue", {}).get("consensus")

                if seqid and min_seqid <= seqid <= max_seqid:
                    if include_none or consensus:
                        feature_indices.add(i)
                        residue_index = partner.get("residue", {}).get("index")
                        if residue_index is not None:
                            residue_indices.add(residue_index)

        return feature_indices, residue_indices






    @staticmethod
    def _extract_unique_partners(features: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract all unique partners (residues) from features.
        
        Parameters
        ----------
        features : list
            List of features
            
        Returns
        -------
        list
            List of unique partners with seqid and consensus
        """
        partners_map = {}
        for feature in features:
            for partner in feature:
                seqid = partner.get("residue", {}).get("seqid")
                if seqid and seqid not in partners_map:
                    partners_map[seqid] = {
                        "seqid": seqid,
                        "consensus": partner.get("residue", {}).get("consensus")
                    }
        return list(partners_map.values())
    
    @staticmethod
    def _find_seqid_ranges(partners: List[Dict[str, Any]], start_pattern: str, end_pattern: str, match_func: callable) -> List[Tuple[int, int]]:
        """
        Find all seqid ranges between start and end patterns.

        Parameters
        ----------
        partners : list
            List of partners with seqid and consensus
        start_pattern : str
            Start pattern to match
        end_pattern : str
            End pattern to match
        match_func : callable
            Function to match consensus against pattern

        Returns
        -------
        list
            List of (min_seqid, max_seqid) tuples
        """
        start_seqids, end_seqids = FeatureSelectorConsensusParseHelper._collect_matching_seqids(
            partners, start_pattern, end_pattern, match_func
        )

        start_seqids, end_seqids = FeatureSelectorConsensusParseHelper._handle_missing_boundaries(
            partners, start_seqids, end_seqids
        )

        return FeatureSelectorConsensusParseHelper._create_seqid_ranges(start_seqids, end_seqids)

    @staticmethod
    def _collect_matching_seqids(
        partners: List[Dict[str, Any]], start_pattern: str, end_pattern: str, match_func: callable
    ) -> Tuple[List[int], List[int]]:
        """
        Collect seqids matching start and end patterns.

        Parameters
        ----------
        partners : list
            List of partners with seqid and consensus
        start_pattern : str
            Start pattern to match
        end_pattern : str
            End pattern to match
        match_func : callable
            Function to match consensus against pattern

        Returns
        -------
        Tuple[List[int], List[int]]
            Tuple of (start_seqids, end_seqids)
        """
        start_seqids = []
        end_seqids = []

        for partner in partners:
            if match_func(partner["consensus"], start_pattern):
                start_seqids.append(partner["seqid"])
            if match_func(partner["consensus"], end_pattern):
                end_seqids.append(partner["seqid"])

        return start_seqids, end_seqids

    @staticmethod
    def _handle_missing_boundaries(
        partners: List[Dict[str, Any]], start_seqids: List[int], end_seqids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Handle missing start or end boundaries with fallback.

        Parameters
        ----------
        partners : list
            List of partners with seqid and consensus
        start_seqids : list
            List of start seqids
        end_seqids : list
            List of end seqids

        Returns
        -------
        Tuple[List[int], List[int]]
            Tuple of (start_seqids, end_seqids) with fallbacks applied
        """
        if not start_seqids and end_seqids:
            start_seqids = FeatureSelectorConsensusParseHelper._find_fallback_start(
                partners, end_seqids
            )

        if not end_seqids and start_seqids:
            end_seqids = FeatureSelectorConsensusParseHelper._find_fallback_end(
                partners, start_seqids
            )

        return start_seqids, end_seqids

    @staticmethod
    def _find_fallback_start(partners: List[Dict[str, Any]], end_seqids: List[int]) -> List[int]:
        """
        Find fallback start seqid when start pattern is missing.

        Parameters
        ----------
        partners : list
            List of partners with seqid and consensus
        end_seqids : list
            List of end seqids

        Returns
        -------
        List[int]
            List with fallback start seqid or empty list
        """
        min_end_seqid = min(end_seqids)
        fallback_start = FeatureSelectorConsensusParseHelper._find_fallback_seqid(
            partners, min_end_seqid, direction="backward"
        )
        return [fallback_start] if fallback_start else []

    @staticmethod
    def _find_fallback_end(partners: List[Dict[str, Any]], start_seqids: List[int]) -> List[int]:
        """
        Find fallback end seqid when end pattern is missing.

        Parameters
        ----------
        partners : list
            List of partners with seqid and consensus
        start_seqids : list
            List of start seqids

        Returns
        -------
        List[int]
            List with fallback end seqid or empty list
        """
        max_start_seqid = max(start_seqids)
        fallback_end = FeatureSelectorConsensusParseHelper._find_fallback_seqid(
            partners, max_start_seqid, direction="forward"
        )
        return [fallback_end] if fallback_end else []

    @staticmethod
    def _create_seqid_ranges(start_seqids: List[int], end_seqids: List[int]) -> List[Tuple[int, int]]:
        """
        Create seqid ranges from start and end seqids.

        Parameters
        ----------
        start_seqids : list
            List of start seqids
        end_seqids : list
            List of end seqids

        Returns
        -------
        List[Tuple[int, int]]
            List of (start_seqid, end_seqid) tuples
        """
        ranges = []
        for start_seqid in start_seqids:
            for end_seqid in end_seqids:
                if start_seqid <= end_seqid:
                    ranges.append((start_seqid, end_seqid))
        return ranges

    @staticmethod
    def _find_fallback_seqid(partners: List[Dict[str, Any]], from_seqid: int, direction: str) -> Optional[int]:
        """
        Find fallback seqid in given direction.
        
        Parameters
        ----------
        partners : list
            List of partners with seqid and consensus
        from_seqid : int
            Starting seqid
        direction : str
            "forward" or "backward"
            
        Returns
        -------
        int or None
            Fallback seqid or None
        """
        sorted_partners = sorted(partners, key=lambda p: p["seqid"])
        
        if direction == "backward":
            for partner in reversed(sorted_partners):
                if partner["seqid"] < from_seqid and partner["consensus"]:
                    return partner["seqid"]
        else:  # forward
            for partner in sorted_partners:
                if partner["seqid"] > from_seqid and partner["consensus"]:
                    return partner["seqid"]
        
        return None
    
    @staticmethod
    def _feature_matches_seqid_range(feature: List[Dict[str, Any]], min_seqid: int, max_seqid: int, include_none: bool) -> bool:
        """
        Check if feature matches seqid range criteria.

        Parameters
        ----------
        feature : list
            List of partners in feature
        min_seqid : int
            Minimum seqid in range
        max_seqid : int
            Maximum seqid in range
        include_none : bool
            Whether to include partners with consensus=None

        Returns
        -------
        bool
            True if at least one partner matches the criteria
        """
        # ANY partner in range is sufficient
        for partner in feature:
            seqid = partner.get("residue", {}).get("seqid")
            consensus = partner.get("residue", {}).get("consensus")

            if seqid and min_seqid <= seqid <= max_seqid:
                if include_none or consensus:
                    return True  # Early exit: found one matching partner

        return False

    @staticmethod
    def _extract_residue_indices(features_list: List[list], feature_indices: List[int]) -> Set[int]:
        """
        Extract all unique residue.index values from specified features.

        Parameters
        ----------
        features_list : List[list]
            Complete list of features
        feature_indices : List[int]
            Indices of features to extract residue indices from

        Returns
        -------
        Set[int]
            Set of unique residue.index values
        """
        residue_indices = set()
        for idx in feature_indices:
            feature = features_list[idx]
            for partner in feature:
                residue_index = partner.get("residue", {}).get("index")
                if residue_index is not None:
                    residue_indices.add(residue_index)
        return residue_indices
