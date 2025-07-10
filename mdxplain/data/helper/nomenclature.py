# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude-4-Sonnet and Cursor AI.
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
Nomenclature utilities for molecular dynamics trajectory labeling.

This module provides utilities to create consensus labels for MD trajectories
using different nomenclature systems (GPCR, CGN, KLIFS) via mdciao labelers.

Notes
-----
This module uses mdciao consensus nomenclature systems:
        https://proteinformatics.uni-leipzig.de/mdciao/api/generated/mdciao.nomenclature.html  # noqa: E501

        Supported fragment types:
        - gpcr: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerGPCR.html#mdciao.nomenclature.LabelerGPCR  # noqa: E501
        - cgn: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerCGN.html#mdciao.nomenclature.LabelerCGN  # noqa: E501
        - klifs: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS  # noqa: E501
"""

from typing import Dict, List, Optional, Tuple, Union

import mdciao
import mdtraj as md


class Nomenclature:
    """
    A utility class for creating consensus labels from MD trajectories.

    This class provides a unified interface to use different mdciao labelers
    (GPCR, CGN, KLIFS) and generate label lists for molecular dynamics analysis.
    """

    def __init__(
        self,
        topology: md.Topology,
        fragment_definition: Optional[Union[str, Dict[str, Tuple[int, int]]]] = None,
        fragment_type: Optional[Union[str, Dict[str, str]]] = None,
        fragment_molecule_name: Optional[Union[str, Dict[str, str]]] = None,
        consensus: bool = False,
        aa_short: bool = False,
        verbose: bool = False,
        try_web_lookup: bool = True,
        write_to_disk: bool = False,
        cache_folder: str = "./cache",
        **labeler_kwargs,
    ):
        """
        Initialize the Nomenclature labeler.

        This class provides a unified interface to create consensus labels for MD trajectories
        using different mdciao nomenclature systems.

        Parameters
        ----------
        topology : md.Topology
            MDTraj topology object to label
        fragment_definition : str or dict, default None
            If string, uses that as fragment name for entire topology.
            If dict, maps fragment names to residue ranges: {"cgn_a": (0, 348), "beta2": (400, 684)}
            Only required when consensus=True.
        fragment_type : str or dict, default None
            If string, uses that nomenclature type for all fragments.
            If dict, maps fragment names to nomenclature types: {"cgn_a": "cgn", "beta2": "gpcr"}
            Use mdciao nomenclature types.
            Allowed types: gpcr, cgn, klifs
            Only required when consensus=True.
        fragment_molecule_name : str or dict, default None
            If string, uses that molecule name for all fragments.
            If dict, maps fragment names to molecule names:
            {"cgn_a": "gnas2_bovin", "beta2": "adrb2_human"}
            Use the UniProt entry name (not accession ID) for GPCR/CGN labelers,
            or KLIFS string for KLIFS labelers.
            See https://www.uniprot.org/help/difference_accession_entryname for UniProt naming conventions.  # noqa: E501
            See https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS for KLIFS naming conventions.  # noqa: E501
            Only required when consensus=True.
        consensus : bool, default False
            Whether to use consensus labeling (combines AA codes with nomenclature labels).
            If False, only returns amino acid labels without nomenclature.
        aa_short : bool, default False
            Whether to use short amino acid names (T vs THR)
        verbose : bool, default False
            Whether to enable verbose output from labelers
        try_web_lookup : bool, default True
            Whether to try web lookup for molecule data
        write_to_disk : bool, default False
            Whether to write cache files to disk
        cache_folder : str, default "./cache"
            Folder for cache files
        **labeler_kwargs
            Additional keyword arguments passed to the mdciao labelers

        Returns:
        --------
        None
            Initializes the Nomenclature object

        Raises:
        -------
        ValueError
            If fragment_definition is required when consensus=True
        ValueError
            If fragment_type is required when consensus=True
        ValueError
            If fragment_molecule_name is required when consensus=True
        Notes
        -----
        This class uses mdciao consensus nomenclature systems:
        https://proteinformatics.uni-leipzig.de/mdciao/api/generated/mdciao.nomenclature.html

        Supported fragment types:
        - gpcr: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerGPCR.html#mdciao.nomenclature.LabelerGPCR
        - cgn: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerCGN.html#mdciao.nomenclature.LabelerCGN
        - klifs: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS

        Examples
        --------
        Amino acid labels only (no nomenclature):

        >>> nomenclature = Nomenclature(
        ...     topology=traj.top,
        ...     consensus=False,
        ...     aa_short=True  # Use single-letter codes: A, C, D, ...
        ... )
        >>> labels = nomenclature.create_labels()

        Simple single fragment labeling:

        >>> nomenclature = Nomenclature(
        ...     topology=traj.top,
        ...     fragment_definition="receptor",
        ...     fragment_type="gpcr",
        ...     fragment_molecule_name="adrb2_human",
        ...     consensus=True
        ... )
        >>> labels = nomenclature.create_labels()

        Complex multi-fragment labeling:

        >>> nomenclature = Nomenclature(
        ...     topology=traj.top,
        ...     fragment_definition={"gpcr": (0, 300), "g_protein": (300, 600)},
        ...     fragment_type={"gpcr": "gpcr", "g_protein": "cgn"},
        ...     fragment_molecule_name={"gpcr": "adrb2_human", "g_protein": "gnas_human"},
        ...     consensus=True
        ... )
        >>> labels = nomenclature.create_labels()
        """
        self.topology = topology
        self.consensus = consensus
        self.aa_short = aa_short
        self.verbose = verbose
        self.try_web_lookup = try_web_lookup
        self.write_to_disk = write_to_disk
        self.cache_folder = cache_folder
        self.labeler_kwargs = labeler_kwargs

        # Only parse fragments and init labelers if consensus mode is enabled
        if self.consensus:
            # Validate required parameters for consensus mode
            if fragment_definition is None:
                raise ValueError("fragment_definition is required when consensus=True")
            if fragment_type is None:
                raise ValueError("fragment_type is required when consensus=True")
            if fragment_molecule_name is None:
                raise ValueError(
                    "fragment_molecule_name is required when consensus=True"
                )

            self.fragments = self._parse_fragment_config(
                fragment_definition, fragment_type, fragment_molecule_name
            )
            self.labelers: Dict[str, object] = {}
            self._init_labelers()
            # Cache consensus labels for all residues once
            self.cached_consensus_labels = self._compute_all_consensus_labels()
        else:
            self.fragments = {}
            self.labelers = {}
            self.cached_consensus_labels = []

    def _parse_fragment_config(
        self,
        fragment_definition: Union[str, Dict[str, Tuple[int, int]]],
        fragment_type: Union[str, Dict[str, str]],
        fragment_molecule_name: Union[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, Union[Tuple[int, int], str]]]:
        """
        Parse fragment configuration into unified format.

        Parameters:
        -----------
        fragment_definition : Union[str, Dict[str, Tuple[int, int]]]
            Fragment definition input (string or dictionary)
        fragment_type : Union[str, Dict[str, str]]
            Fragment type input (string or dictionary)
        fragment_molecule_name : Union[str, Dict[str, str]]
            Fragment molecule name input (string or dictionary)

        Returns:
        --------
        Dict[str, Dict[str, Union[Tuple[int, int], str]]]
            Unified fragment configuration dictionary
        """
        fragment_names = self._extract_fragment_names(fragment_definition)
        ranges = self._normalize_ranges(fragment_definition)
        types = self._normalize_types(fragment_type, fragment_names)
        molecule_names = self._normalize_molecule_names(
            fragment_molecule_name, fragment_names
        )

        return self._build_and_validate_fragments(
            fragment_names, ranges, types, molecule_names
        )

    def _extract_fragment_names(
        self, fragment_definition: Union[str, Dict[str, Tuple[int, int]]]
    ) -> List[str]:
        """
        Extract fragment names from fragment definition.

        Parameters
        ----------
        fragment_definition : str or dict
            Fragment definition input (string or dictionary)

        Returns
        -------
        List[str]
            List of fragment names
        """
        if isinstance(fragment_definition, str):
            return [fragment_definition]
        return list(fragment_definition.keys())

    def _normalize_ranges(
        self, fragment_definition: Union[str, Dict[str, Tuple[int, int]]]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Normalize fragment ranges to dictionary format.

        Parameters
        ----------
        fragment_definition : str or dict
            Fragment definition with ranges or single string

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Dictionary mapping fragment names to (start, end) residue ranges
        """
        if isinstance(fragment_definition, str):
            return {fragment_definition: (0, len(list(self.topology.residues)))}
        return fragment_definition

    def _normalize_types(
        self, fragment_type: Union[str, Dict[str, str]], fragment_names: List[str]
    ) -> Dict[str, str]:
        """
        Normalize fragment types to dictionary format.

        Parameters
        ----------
        fragment_type : str or dict
            Nomenclature type(s) - single string or dict mapping
        fragment_names : List[str]
            List of fragment names for mapping

        Returns
        -------
        Dict[str, str]
            Dictionary mapping fragment names to nomenclature types
        """
        if isinstance(fragment_type, str):
            return {name: fragment_type for name in fragment_names}
        return fragment_type

    def _normalize_molecule_names(
        self,
        fragment_molecule_name: Union[str, Dict[str, str]],
        fragment_names: List[str],
    ) -> Dict[str, str]:
        """
        Normalize molecule names to dictionary format.

        Parameters
        ----------
        fragment_molecule_name : str or dict
            Molecule identifier(s) - single string or dict mapping
        fragment_names : List[str]
            List of fragment names for mapping

        Returns
        -------
        Dict[str, str]
            Dictionary mapping fragment names to molecule identifiers
        """
        if isinstance(fragment_molecule_name, str):
            return {name: fragment_molecule_name for name in fragment_names}
        return fragment_molecule_name

    def _build_and_validate_fragments(
        self,
        fragment_names: List[str],
        ranges: Dict[str, Tuple[int, int]],
        types: Dict[str, str],
        molecule_names: Dict[str, str],
    ) -> Dict[str, Dict[str, Union[Tuple[int, int], str]]]:
        """
        Build unified configuration and validate completeness.

        Parameters
        ----------
        fragment_names : List[str]
            List of fragment names
        ranges : Dict[str, Tuple[int, int]]
            Dictionary mapping fragment names to residue ranges
        types : Dict[str, str]
            Dictionary mapping fragment names to nomenclature types
        molecule_names : Dict[str, str]
            Dictionary mapping fragment names to molecule identifiers

        Returns
        -------
        Dict[str, Dict[str, Union[Tuple[int, int], str]]]
            Unified fragment configuration dictionary

        Raises
        ------
        ValueError
            If any fragment has incomplete configuration
        """
        fragments: Dict[str, Dict[str, Union[Tuple[int, int], str]]] = {}
        for fragment_name in fragment_names:
            range_val = ranges.get(fragment_name)
            type_val = types.get(fragment_name)
            molecule_val = molecule_names.get(fragment_name)

            # Validate configuration
            if range_val is None or type_val is None or molecule_val is None:
                raise ValueError(
                    f"Incomplete configuration for fragment '{fragment_name}'"
                )

            fragments[fragment_name] = {
                "range": range_val,
                "type": type_val,
                "molecule_name": molecule_val,
            }

        return fragments

    def _init_labelers(self) -> None:
        """
        Initialize the appropriate mdciao labelers for each fragment.

        Creates the correct labeler type (GPCR, CGN, or KLIFS) for each fragment
        based on its nomenclature type. Each labeler is configured with the
        molecule identifier and the common parameters (verbose, web lookup, etc.).

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            Initializes the labelers
        Notes
        -----
        - GPCR and CGN labelers use UniProt_name parameter
        - KLIFS labeler uses KLIFS_string parameter
        - All labelers receive the same common configuration parameters

        Raises
        ------
        ValueError
            If an unknown nomenclature type is encountered
        """
        for fragment_name, fragment_config in self.fragments.items():
            nomenclature_type = str(fragment_config["type"])
            molecule_name = str(fragment_config["molecule_name"])

            common_params = self._build_common_params()
            labeler = self._create_labeler_for_type(
                nomenclature_type, molecule_name, common_params, fragment_name
            )
            self.labelers[fragment_name] = labeler

    def _build_common_params(self) -> dict:
        """
        Build common parameters for all labelers.

        Parameters:
        -----------
        None

        Returns:
        --------
        dict
            Dictionary of common parameters for mdciao labelers
        """
        common_params = {
            "verbose": self.verbose,
            "try_web_lookup": self.try_web_lookup,
            "write_to_disk": self.write_to_disk,
            **self.labeler_kwargs,
        }
        if self.cache_folder is not None:
            common_params["local_path"] = self.cache_folder
        return common_params

    def _create_labeler_for_type(
        self,
        nomenclature_type: str,
        molecule_name: str,
        common_params: dict,
        fragment_name: str,
    ):
        """
        Create appropriate labeler based on nomenclature type.

        Parameters:
        -----------
        nomenclature_type : str
            Type of nomenclature labeler to create
        molecule_name : str
            Molecule identifier for the labeler
        common_params : dict
            Common parameters for labeler initialization
        fragment_name : str
            Fragment name for error messages

        Returns:
        --------
        object
            Initialized mdciao labeler instance

        Raises:
        -------
        ValueError
            If unknown nomenclature type is provided
        """
        labeler_creators = {
            "gpcr": lambda: mdciao.nomenclature.LabelerGPCR(
                UniProt_name=molecule_name, **common_params
            ),
            "cgn": lambda: mdciao.nomenclature.LabelerCGN(
                UniProt_name=molecule_name, **common_params
            ),
            "klifs": lambda: mdciao.nomenclature.LabelerKLIFS(
                KLIFS_string=molecule_name, **common_params
            ),
        }

        nomenclature_key = nomenclature_type.lower()
        if nomenclature_key not in labeler_creators:
            raise ValueError(
                f"Unknown nomenclature type for fragment '{fragment_name}': {nomenclature_type}"
            )

        return labeler_creators[nomenclature_key]()

    def create_trajectory_label_dicts(self) -> List[Dict]:
        """
        Create structured label dictionaries for the trajectory.

        Parameters:
        -----------
        None

        Returns
        -------
        List[Dict]
            List of label dictionaries for each residue in the topology.
            Each dict contains: aaa_code, a_code, index, consensus, full_name
        """
        return self._create_structured_labels()

    def _create_structured_labels(self) -> List[Dict]:
        """
        Create structured label dictionaries from topology residues.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[Dict]
            List of label dictionaries for each residue in the topology.
            Each dict contains: aaa_code, a_code, index, consensus, full_name
        """
        labels = []

        for idx, residue in enumerate(self.topology.residues):
            aaa_code = residue.name
            a_code = self._get_aa_short_code(aaa_code)
            consensus = self.cached_consensus_labels[idx] if self.consensus else None
            full_name = self._get_full_name(aaa_code, a_code, residue.index, consensus)

            label_dict = {
                "aaa_code": aaa_code,
                "a_code": a_code,
                "index": residue.index,
                "consensus": consensus,
                "full_name": full_name,
            }
            labels.append(label_dict)

        return labels

    def _compute_all_consensus_labels(self) -> List[Optional[str]]:
        """
        Compute and cache consensus labels for all residues.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[Optional[str]]
            List of consensus labels for all residues
        """
        # Initialize array with None for all residues
        cached_labels: List[Optional[str]] = [None] * self.topology.n_residues

        for fragment_name, fragment_config in self.fragments.items():
            self._process_fragment_labels(cached_labels, fragment_name, fragment_config)

        return cached_labels

    def _process_fragment_labels(
        self,
        cached_labels: List[Optional[str]],
        fragment_name: str,
        fragment_config: dict,
    ):
        """
        Process labels for a single fragment.

        Parameters:
        -----------
        cached_labels : List[Optional[str]]
            List of cached consensus labels to update
        fragment_name : str
            Name of the fragment for error messages
        fragment_config : dict
            Fragment configuration dictionary

        Returns:
        --------
        None
            Updates cached_labels in-place
        """
        start_idx, end_idx = self._validate_and_extract_range(
            fragment_config, fragment_name
        )
        labeler = self._get_fragment_labeler(fragment_name)
        fragment_labels = labeler.top2labels(self.topology)  # type: ignore

        self._apply_fragment_labels_to_cache(
            cached_labels, fragment_labels, start_idx, end_idx, fragment_name
        )

    def _apply_fragment_labels_to_cache(
        self,
        cached_labels: List[Optional[str]],
        fragment_labels: List[str],
        start_idx: int,
        end_idx: int,
        fragment_name: str,
    ):
        """
        Apply fragment labels to cache with overlap detection.

        Parameters:
        -----------
        cached_labels : List[Optional[str]]
            List of cached consensus labels to update
        fragment_labels : List[str]
            Fragment-specific labels from mdciao labeler
        start_idx : int
            Starting index for fragment range
        end_idx : int
            Ending index for fragment range
        fragment_name : str
            Name of the fragment for error messages

        Returns:
        --------
        None
            Updates cached_labels in-place
        """
        for idx in range(start_idx, end_idx):
            if idx < len(fragment_labels) and fragment_labels[idx] is not None:
                self._check_and_set_label(
                    cached_labels, idx, fragment_labels[idx], fragment_name
                )

    def _check_and_set_label(
        self,
        cached_labels: List[Optional[str]],
        idx: int,
        new_label: str,
        fragment_name: str,
    ):
        """
        Check for overlap and set label if valid.

        Parameters:
        -----------
        cached_labels : List[Optional[str]]
            List of cached consensus labels
        idx : int
            Index position to set
        new_label : str
            New label to set
        fragment_name : str
            Fragment name for error messages

        Returns:
        --------
        None
            Updates cached_labels[idx] in-place

        Raises:
        -------
        ValueError
            If overlap is detected at the same index
        """
        if cached_labels[idx] is not None:
            raise ValueError(
                f"Fragment overlap detected at residue {idx}: "
                f"existing label '{cached_labels[idx]}' vs new label '{new_label}' "
                f"from fragment '{fragment_name}'"
            )
        cached_labels[idx] = new_label

    def _get_full_name(
        self, aaa_code: str, a_code: str, residue_index: int, consensus: Optional[str]
    ) -> str:
        """
        Generate full name based on aa_short setting and consensus label.

        Parameters:
        -----------
        aaa_code : str
            Three-letter amino acid code
        a_code : str
            One-letter amino acid code
        residue_index : int
            Residue index from topology
        consensus : Optional[str]
            Consensus label from nomenclature system

        Returns:
        --------
        str
            Full residue name with optional consensus label
        """
        base_code = a_code if self.aa_short else aaa_code
        base_name = f"{base_code}{residue_index}"

        if consensus is not None:
            return f"{base_name}x{consensus}"
        return base_name

    def _validate_and_extract_range(
        self, fragment_config: dict, fragment_name: str
    ) -> Tuple[int, int]:
        """
        Validate range tuple and extract start/end indices.

        Parameters:
        -----------
        fragment_config : dict
            Fragment configuration dictionary
        fragment_name : str
            Fragment name for error messages

        Returns:
        --------
        Tuple[int, int]
            Tuple of (start_idx, end_idx) for the fragment range

        Raises:
        -------
        ValueError
            If range is not a tuple
        """
        range_tuple = fragment_config["range"]
        if not isinstance(range_tuple, tuple):
            raise ValueError(f"Invalid range type for fragment '{fragment_name}'")
        return range_tuple

    def _get_fragment_labeler(self, fragment_name: str):
        """
        Get labeler for fragment, ensuring it exists.

        Parameters:
        -----------
        fragment_name : str
            Name of the fragment to get labeler for

        Returns:
        --------
        object
            mdciao labeler instance for the fragment

        Raises:
        -------
        ValueError
            If no labeler is configured for the fragment
        """
        if fragment_name not in self.labelers:
            raise ValueError(f"No labeler configured for fragment: {fragment_name}")
        return self.labelers[fragment_name]

    def _get_aa_short_code(self, three_letter: str) -> str:
        """
        Convert three-letter amino acid code to one-letter code.

        Includes standard amino acids and common special residues.

        Parameters
        ----------
        three_letter : str
            Three-letter amino acid/residue code

        Returns
        -------
        str
            One-letter amino acid code or special designation
        """
        aa_map = {
            # Standard amino acids
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
            # Special residues and modifications
            "HSP": "H",  # Histidine phosphate
            "SEP": "S",  # Phosphoserine
            "TPO": "T",  # Phosphothreonine
            "PTR": "Y",  # Phosphotyrosine
            "MSE": "M",  # Selenomethionine
            "CSO": "C",  # S-hydroxycysteine
            "HIP": "H",  # Histidine protonated at ND1
            "HIE": "H",  # Histidine protonated at NE2
            "HID": "H",  # Histidine neutral
            "CYX": "C",  # Cysteine in disulfide bond
            "GLH": "E",  # Protonated glutamic acid
            "ASH": "D",  # Protonated aspartic acid
            "LYN": "K",  # Neutral lysine
            "ARN": "R",  # Neutral arginine
        }

        return aa_map.get(three_letter.upper(), three_letter[:3])
