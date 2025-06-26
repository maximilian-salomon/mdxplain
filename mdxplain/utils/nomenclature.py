"""
Nomenclature utilities for molecular dynamics trajectory labeling.

This module provides utilities to create consensus labels for MD trajectories
using different nomenclature systems (GPCR, CGN, KLIFS) via mdciao labelers.

Notes
-----
This module uses mdciao consensus nomenclature systems:
https://proteinformatics.uni-leipzig.de/mdciao/api/generated/mdciao.nomenclature.html

Supported fragment types:
- gpcr: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerGPCR.html#mdciao.nomenclature.LabelerGPCR
- cgn: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerCGN.html#mdciao.nomenclature.LabelerCGN  
- klifs: https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS
"""

from typing import Dict, List, Tuple, Union
import mdtraj as md
import mdciao


class Nomenclature:
    """
    A utility class for creating consensus labels from MD trajectories.
    
    This class provides a unified interface to use different mdciao labelers
    (GPCR, CGN, KLIFS) and generate label lists for molecular dynamics analysis.
    """
    
    def __init__(
        self,
        topology: md.Topology,
        fragment_definition: Union[str, Dict[str, Tuple[int, int]]],
        fragment_type: Union[str, Dict[str, str]],
        fragment_molecule_name: Union[str, Dict[str, str]],
        consensus: bool = True,
        aa_short: bool = True,
        verbose: bool = False,
        try_web_lookup: bool = True,
        write_to_disk: bool = False,
        cache_folder: str = "./cache",
        **labeler_kwargs
    ):
        """
        Initialize the Nomenclature labeler.
        
        This class provides a unified interface to create consensus labels for MD trajectories
        using different mdciao nomenclature systems.
        
        Parameters
        ----------
        topology : md.Topology
            MDTraj topology object to label
        fragment_definition : str or dict
            If string, uses that as fragment name for entire topology.
            If dict, maps fragment names to residue ranges: {"cgn_a": (0, 348), "beta2": (400, 684)}
        fragment_type : str or dict
            If string, uses that nomenclature type for all fragments.
            If dict, maps fragment names to nomenclature types: {"cgn_a": "cgn", "beta2": "gpcr"}
            Use mdciao nomenclature types.
            Allowed types: gpcr, cgn, klifs
        fragment_molecule_name : str or dict
            If string, uses that molecule name for all fragments.
            If dict, maps fragment names to molecule names: {"cgn_a": "gnas2_bovin", "beta2": "adrb2_human"}
            Use the UniProt entry name (not accession ID) for GPCR/CGN labelers, or KLIFS string for KLIFS labelers.
            See https://www.uniprot.org/help/difference_accession_entryname for UniProt naming conventions.
            See https://proteinformatics.uni-leipzig.de/mdciao/api/generated/generated/mdciao.nomenclature.LabelerKLIFS.html#mdciao.nomenclature.LabelerKLIFS for KLIFS naming conventions.
        consensus : bool, default True
            Whether to use consensus labeling (combines AA codes with nomenclature labels)
        aa_short : bool, default True
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
        Simple single fragment labeling:
        
        >>> nomenclature = Nomenclature(
        ...     topology=traj.top,
        ...     fragment_definition="receptor",
        ...     fragment_type="gpcr", 
        ...     fragment_molecule_name="adrb2_human"
        ... )
        >>> labels = nomenclature.create_labels()
        
        Complex multi-fragment labeling:
        
        >>> nomenclature = Nomenclature(
        ...     topology=traj.top,
        ...     fragment_definition={"gpcr": (0, 300), "g_protein": (300, 600)},
        ...     fragment_type={"gpcr": "gpcr", "g_protein": "cgn"},
        ...     fragment_molecule_name={"gpcr": "adrb2_human", "g_protein": "gnas_human"}
        ... )
        >>> labels = nomenclature.create_labels()
        """
        self.topology = topology
        self.consensus = consensus
        self.aa_short = aa_short
        self.labeler_kwargs = labeler_kwargs
        self.verbose = verbose
        self.try_web_lookup = try_web_lookup
        self.write_to_disk = write_to_disk
        self.cache_folder = cache_folder
        self.labeler_kwargs = labeler_kwargs
        
        self.fragments = self._parse_fragment_config(
            fragment_definition, fragment_type, fragment_molecule_name
        )
        self.labelers = {}
        self._init_labelers()
    
    def _parse_fragment_config(
        self,
        fragment_definition: Union[str, Dict[str, Tuple[int, int]]],
        fragment_type: Union[str, Dict[str, str]],
        fragment_molecule_name: Union[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, Union[Tuple[int, int], str]]]:
        """Parse fragment configuration into unified format."""
        fragment_names = self._extract_fragment_names(fragment_definition)
        ranges = self._normalize_ranges(fragment_definition)
        types = self._normalize_types(fragment_type, fragment_names)
        molecule_names = self._normalize_molecule_names(fragment_molecule_name, fragment_names)
        
        return self._build_and_validate_fragments(fragment_names, ranges, types, molecule_names)
    
    def _extract_fragment_names(self, fragment_definition: Union[str, Dict[str, Tuple[int, int]]]) -> List[str]:
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
        self, 
        fragment_definition: Union[str, Dict[str, Tuple[int, int]]]
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
        self, 
        fragment_type: Union[str, Dict[str, str]], 
        fragment_names: List[str]
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
        fragment_names: List[str]
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
        molecule_names: Dict[str, str]
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
        fragments = {}
        for fragment_name in fragment_names:
            fragments[fragment_name] = {
                'range': ranges.get(fragment_name),
                'type': types.get(fragment_name),
                'molecule_name': molecule_names.get(fragment_name)
            }
            
            # Validate configuration
            if not all(fragments[fragment_name].values()):
                raise ValueError(f"Incomplete configuration for fragment '{fragment_name}'")
        
        return fragments
    
    def _init_labelers(self) -> None:
        """
        Initialize the appropriate mdciao labelers for each fragment.
        
        Creates the correct labeler type (GPCR, CGN, or KLIFS) for each fragment
        based on its nomenclature type. Each labeler is configured with the
        molecule identifier and the common parameters (verbose, web lookup, etc.).
        
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
            nomenclature_type = fragment_config['type']
            molecule_name = fragment_config['molecule_name']
            
            # Common parameters for all labelers
            common_params = {
                'verbose': self.verbose,
                'try_web_lookup': self.try_web_lookup,
                'write_to_disk': self.write_to_disk,
                'local_path': self.cache_folder,
                **self.labeler_kwargs
            }
            
            if nomenclature_type.lower() == "gpcr":
                self.labelers[fragment_name] = mdciao.nomenclature.LabelerGPCR(
                    UniProt_name=molecule_name, **common_params
                )
            elif nomenclature_type.lower() == "cgn":
                self.labelers[fragment_name] = mdciao.nomenclature.LabelerCGN(
                    UniProt_name=molecule_name, **common_params
                )
            elif nomenclature_type.lower() == "klifs":
                self.labelers[fragment_name] = mdciao.nomenclature.LabelerKLIFS(
                    KLIFS_string=molecule_name, **common_params
                )
            else:
                raise ValueError(f"Unknown nomenclature type for fragment '{fragment_name}': {nomenclature_type}")
    
    def create_labels(self) -> List[str]:
        """
        Create label list for the trajectory.
        
        Returns
        -------
        List[str]
            List of labels for each residue in the topology
        """
        labels = list(self.topology.residues)
        if self.aa_short:
            labels = [self._get_aa_short_code(res.name) for res in labels]
        
        for fragment_name, fragment_config in self.fragments.items():
            start_idx, end_idx = fragment_config['range']
            
            if fragment_name not in self.labelers:
                raise ValueError(f"No labeler configured for fragment: {fragment_name}")
                
            labeler = self.labelers[fragment_name]
            
            # Get labels from the labeler
            fragment_labels = labeler.top2labels(self.topology)
            
            # Apply labels to the correct range
            for i, label in enumerate(fragment_labels[start_idx:end_idx], start=start_idx):
                if label is not None:
                    if self.consensus:
                        labels[i] = f"{labels[i]}x{label}"
                    else:
                        labels[i] = label
        
        return labels
    
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
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            # Special residues and modifications
            'HSP': 'H',   # Histidine phosphate
            'SEP': 'S',   # Phosphoserine
            'TPO': 'T',   # Phosphothreonine
            'PTR': 'Y',   # Phosphotyrosine
            'MSE': 'M',   # Selenomethionine
            'CSO': 'C',   # S-hydroxycysteine
            'HIP': 'H',   # Histidine protonated at ND1
            'HIE': 'H',   # Histidine protonated at NE2
            'HID': 'H',   # Histidine neutral
            'CYX': 'C',   # Cysteine in disulfide bond
            'GLH': 'E',   # Protonated glutamic acid
            'ASH': 'D',   # Protonated aspartic acid
            'LYN': 'K',   # Neutral lysine
            'ARN': 'R',   # Neutral arginine
        }
        
        return aa_map.get(three_letter.upper(), three_letter[:3])
    