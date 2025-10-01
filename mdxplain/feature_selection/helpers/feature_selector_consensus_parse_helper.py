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

from typing import List, Optional, Dict, Any, Tuple


class FeatureSelectorConsensusParseHelper:
    """
    Helper class for parsing consensus nomenclature patterns.

    Provides static methods to parse consensus patterns like "7x50", "7x*", "7x-8x",
    and "*40-*50" to identify matching residue indices based on trajectory metadata.
    """

    @staticmethod
    def parse_consensus_category(
        param_tokens: List[str], features_list: List[list], require_all_partners: bool = False
    ) -> List[int]:
        """
        Parse 'consensus' category and return matching feature indices.
        
        Uses unified interface like other parsers. Converts features_list to metadata
        format and delegates to parse_consensus_pattern.
        
        Parameters
        ----------
        param_tokens : List[str]
            List of parameter tokens (should be single pattern like ["7x50-8x50"])
        features_list : List[list]
            List of features from metadata (each feature is a list of partners)
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns
        -------
        List[int]
            List of feature indices matching the consensus criteria
            
        Examples
        --------
        >>> # Single consensus position
        >>> indices = FeatureSelectorConsensusParseHelper.parse_consensus_category(
        ...     ["7x50"], features_list
        ... )
        
        >>> # Consensus range
        >>> indices = FeatureSelectorConsensusParseHelper.parse_consensus_category(
        ...     ["7x50-8x50"], features_list
        ... )
        """
        if not param_tokens:
            return []
            
        # Join all tokens into single pattern (for ranges like "7x50 - 8x50")
        pattern = " ".join(param_tokens).replace(" ", "")
        
        # Convert features_list to metadata format expected by parse_consensus_pattern
        metadata = {"features": features_list}
        
        return FeatureSelectorConsensusParseHelper.parse_consensus_pattern(metadata, pattern, require_all_partners)

    @staticmethod
    def parse_consensus_pattern(metadata: dict, pattern: str, require_all_partners: bool = False) -> List[int]:
        """
        Parse consensus nomenclature pattern and return matching residue indices.

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
        require_all_partners : bool, default=False
            For pairwise features, require all partners to be present in selection

        Returns
        -------
        List[int]
            List of residue indices matching the pattern

        Examples
        --------
        >>> # Single position
        >>> indices = FeatureSelectorConsensusParseHelper.parse_consensus_pattern(metadata, "7x50")
        >>> # [145]  # Index of residue with consensus "7x50"

        >>> # Wildcard
        >>> indices = FeatureSelectorConsensusParseHelper.parse_consensus_pattern(metadata, "7x*")
        >>> # [145, 146, 147, ...]  # All residues with "7x" in consensus
        
        >>> # Range
        >>> indices = FeatureSelectorConsensusParseHelper.parse_consensus_pattern(metadata, "7x-8x")
        >>> # [145, 146, ..., 180]  # From first "7x" to last "8x"
        """
        pattern = pattern.strip()
        
        # Check for "all" prefix
        include_none = pattern.startswith("all")
        if include_none:
            pattern = pattern[3:].strip()
        
        if "-" in pattern:
            return FeatureSelectorConsensusParseHelper._parse_range_pattern(
                metadata, pattern, include_none, require_all_partners
            )
        else:
            return FeatureSelectorConsensusParseHelper._parse_single_pattern(
                metadata, pattern, require_all_partners
            )

    @staticmethod
    def _find_first_substring_match(metadata: dict, substring: str, require_all_partners: bool = False) -> Optional[int]:
        """
        Find first feature index where consensus contains substring.
        
        Handles both partner-based features (distances) and simple features.
        
        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        substring : str
            Substring to search for (e.g., "7x", "40")
        require_all_partners : bool, default=False
            If True, ALL partners must contain the substring

        Returns
        -------
        Optional[int]
            First matching feature index or None if not found
        """
        features = metadata.get("features", [])
        for i, feature in enumerate(features):
            if FeatureSelectorConsensusParseHelper._feature_matches_consensus(feature, substring, require_all_partners, exact_match=False):
                return i
        return None
    
    @staticmethod
    def _find_last_substring_match(metadata: dict, substring: str, require_all_partners: bool = False) -> Optional[int]:
        """
        Find last feature index where consensus contains substring.
        
        Handles both partner-based features (distances) and simple features.
        
        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        substring : str
            Substring to search for (e.g., "8x", "50")
        require_all_partners : bool, default=False
            If True, ALL partners must contain the substring
            
        Returns
        -------
        Optional[int]
            Last matching feature index or None if not found
        """
        features = metadata.get("features", [])
        last_idx = None
        for i, feature in enumerate(features):
            if FeatureSelectorConsensusParseHelper._feature_matches_consensus(feature, substring, require_all_partners, exact_match=False):
                last_idx = i
        return last_idx
    
    @staticmethod
    def _find_all_substring_matches(metadata: dict, substring: str, require_all_partners: bool = False) -> List[int]:
        """
        Find all feature indices where consensus contains substring.
        
        Handles both partner-based features (distances) and simple features.
        
        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        substring : str
            Substring to search for
        require_all_partners : bool, default=False
            If True, ALL partners must contain the substring
            
        Returns
        -------
        List[int]
            List of all matching feature indices
        """
        features = metadata.get("features", [])
        matches = []
        for i, feature in enumerate(features):
            if FeatureSelectorConsensusParseHelper._feature_matches_consensus(feature, substring, require_all_partners, exact_match=False):
                matches.append(i)
        return matches

    @staticmethod
    def _feature_matches_consensus(feature: List[Dict[str, Any]], pattern: str, require_all_partners: bool = False, exact_match: bool = False) -> bool:
        """
        Check if a feature matches consensus pattern (substring or exact).
        
        Parameters
        ----------
        feature : list
            List of partners in the feature
        pattern : str
            Pattern to match (substring or exact)
        require_all_partners : bool, default=False
            If True, ALL partners must match criteria
        exact_match : bool, default=False
            If True, use exact matching; if False, use substring matching
            
        Returns
        -------
        bool
            True if consensus criteria are met
        """
        if require_all_partners:
            if len(feature) == 0:
                return False
            for partner in feature:
                consensus = partner.get("residue", {}).get("consensus")
                if exact_match:
                    if consensus != pattern:
                        return False
                else:
                    if not (consensus and pattern in str(consensus)):
                        return False
            return True
        else:
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
    def _parse_single_pattern(metadata: dict, pattern: str, require_all_partners: bool = False) -> List[int]:
        """
        Parse single consensus position or wildcard pattern.

        Parameters
        ----------
        metadata : dict
            Feature metadata with features information
        pattern : str
            Single pattern (e.g., "7x50", "7x*")
        require_all_partners : bool, default=False
            If True, ALL partners must contain the substring

        Returns
        -------
        List[int]
            List of matching feature indices
        """
        search_substring = pattern.replace("*","") if "*" in pattern else pattern
        search_substring = search_substring.strip()
        
        return FeatureSelectorConsensusParseHelper._find_all_substring_matches(
            metadata, search_substring, require_all_partners
        )

    @staticmethod
    def _parse_range_pattern(metadata: dict, pattern: str, include_none: bool, require_all_partners: bool = False) -> List[int]:
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
        require_all_partners : bool, default=False
            If True, ALL partners must contain the substring

        Returns
        -------
        List[int]
            List of matching residue indices
        """
        parts = pattern.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid consensus range pattern: {pattern}")
            
        start_pattern, end_pattern = parts
        
        # Handle wildcard ranges like "*40-*50"
        if "*" in start_pattern or "*" in end_pattern:
            return FeatureSelectorConsensusParseHelper._parse_consensus_range(
                metadata, start_pattern, end_pattern, include_none, require_all_partners, is_wildcard=True
            )
        else:
            # Simple range like "3.50-4.50"
            return FeatureSelectorConsensusParseHelper._parse_consensus_range(
                metadata, start_pattern, end_pattern, include_none, require_all_partners, is_wildcard=False
            )

    
    @staticmethod
    def _parse_consensus_range(
        metadata: dict, start_pattern: str, end_pattern: str, include_none: bool, require_all_partners: bool = False, is_wildcard: bool = True
    ) -> List[int]:
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
        require_all_partners : bool, default=False
            If True, ALL partners must contain the substring
        is_wildcard : bool, default=True
            If True, use substring matching; if False, use exact matching
            
        Returns
        -------
        List[int]
            List of all feature indices in all matching blocks
        """
        features = metadata.get("features", [])
        
        # Extract all unique partners (residues) with their seqid and consensus
        partners = FeatureSelectorConsensusParseHelper._extract_unique_partners(features)
        start_search = start_pattern.replace("*", "")
        end_search = end_pattern.replace("*", "")
        # Pattern matching based on wildcard vs exact
        if is_wildcard:
            match_func = lambda consensus, pattern: pattern in str(consensus) if consensus else False
        else:
            match_func = lambda consensus, pattern: consensus == pattern
        
        # Find all seqid ranges
        seqid_ranges = FeatureSelectorConsensusParseHelper._find_seqid_ranges(
            partners, start_search, end_search, match_func
        )
        
        # Apply each range to features
        result_indices = set()
        for min_seqid, max_seqid in seqid_ranges:
            for i, feature in enumerate(features):
                if FeatureSelectorConsensusParseHelper._feature_matches_seqid_range(
                    feature, min_seqid, max_seqid, include_none, require_all_partners
                ):
                    result_indices.add(i)
        
        return sorted(result_indices)
    
    






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
        # Find all start and end partners
        start_seqids = []
        end_seqids = []
        
        for partner in partners:
            if match_func(partner["consensus"], start_pattern):
                start_seqids.append(partner["seqid"])
            if match_func(partner["consensus"], end_pattern):
                end_seqids.append(partner["seqid"])
        
        # Handle missing patterns with fallback
        if not start_seqids and end_seqids:
            # Orphaned end - find previous non-None before first end
            min_end_seqid = min(end_seqids)
            fallback_start = FeatureSelectorConsensusParseHelper._find_fallback_seqid(
                partners, min_end_seqid, direction="backward"
            )
            if fallback_start:
                start_seqids = [fallback_start]
        
        if not end_seqids and start_seqids:
            # Missing end - find next non-None after last start
            max_start_seqid = max(start_seqids)
            fallback_end = FeatureSelectorConsensusParseHelper._find_fallback_seqid(
                partners, max_start_seqid, direction="forward"
            )
            if fallback_end:
                end_seqids = [fallback_end]
        
        # Create ranges
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
    def _feature_matches_seqid_range(feature: List[Dict[str, Any]], min_seqid: int, max_seqid: int, include_none: bool, require_all_partners: bool = False) -> bool:
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
        require_all_partners : bool, default=False
            If True, ALL partners must be in the seqid range
            If False, ANY partner in range is sufficient
            
        Returns
        -------
        bool
            True if feature matches criteria
        """
        if require_all_partners:
            # ALL valid partners must be in the seqid range
            has_valid_partner = False
            for partner in feature:
                seqid = partner.get("residue", {}).get("seqid")
                consensus = partner.get("residue", {}).get("consensus")
                
                # Check if partner is valid (has seqid and optionally consensus)
                if seqid and (include_none or consensus):
                    has_valid_partner = True
                    # This valid partner must be in range
                    if not (min_seqid <= seqid <= max_seqid):
                        return False  # Early exit: valid partner not in range
            
            return has_valid_partner  # True if we had valid partners and all were in range
        
        # ANY partner in range is sufficient (original behavior)
        for partner in feature:
            seqid = partner.get("residue", {}).get("seqid")
            consensus = partner.get("residue", {}).get("consensus")
            
            if seqid and min_seqid <= seqid <= max_seqid:
                if include_none or consensus:
                    return True  # Early exit: found one matching partner
        
        return False
