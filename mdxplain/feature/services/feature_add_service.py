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

"""Factory for adding features with simplified syntax."""

from __future__ import annotations
from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.feature_manager import FeatureManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..feature_type.distances import Distances
from ..feature_type.contacts import Contacts
from ..feature_type.torsions import Torsions
from ..feature_type.dssp import DSSP
from ..feature_type.sasa import SASA
from ..feature_type.coordinates import Coordinates


class FeatureAddService:
    """
    Service for adding features without explicit type instantiation.
    
    This service provides an intuitive interface for adding features to the
    pipeline without requiring users to import and instantiate feature types
    directly. All feature type parameters are combined with add_feature parameters.
    
    Examples
    --------
    >>> pipeline.feature.add.distances(excluded_neighbors=2)
    >>> pipeline.feature.add.contacts(threshold=5.0, traj_selection=[0,1,2])
    >>> pipeline.feature.add.torsions(calculate_chi=False, force=True)
    """
    
    def __init__(self, manager: FeatureManager, pipeline_data: PipelineData) -> None:
        """
        Initialize factory with manager and pipeline data.
        
        Parameters
        ----------
        manager : FeatureManager
            Feature manager instance
        pipeline_data : PipelineData
            Pipeline data container (injected by AutoInjectProxy)
            
        Returns
        -------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    def distances(
        self,
        excluded_neighbors: int = 1,
        use_pbc: bool = True,
        traj_selection: Union[str, int, List] = "all",
        force: bool = False,
        force_original: bool = True,
    ) -> None:
        """
        Add distances feature type.

        Computes all pairwise distances from molecular dynamics trajectories.
        This is a base feature type with no dependencies.

        Parameters
        ----------
        excluded_neighbors : int, default=1
            Number of nearest neighbors to exclude from distance calculation.
            Chain breaks are automatically excluded based on sequence ID jumps.
            0 = all pairs, 1 = exclude direct neighbors, 2 = exclude up to 2nd neighbors, etc.
        use_pbc : bool, default=True
            Use periodic boundary conditions when computing distances
        traj_selection : str, int, list, default="all"
            Which trajectories to compute features for ("all", index, list of indices, or trajectory names)
        force : bool, default=False
            Force recalculation even if feature already exists
        force_original : bool, default=True
            Whether to force using original trajectory data instead of reduced data

        Returns
        -------
        None
            Adds distance features to pipeline data

        Examples
        --------
        >>> # Basic distance calculation
        >>> pipeline.feature.add.distances()

        >>> # With custom neighbor exclusion
        >>> pipeline.feature.add.distances(excluded_neighbors=2)

        >>> # Without periodic boundary conditions
        >>> pipeline.feature.add.distances(use_pbc=False)

        >>> # For specific trajectories only
        >>> pipeline.feature.add.distances(traj_selection=[0,1,2], force=True)

        Notes
        -----
        Distance features are computed using MDTraj and returned in Angstroms.
        Missing pairs (due to chain breaks) are handled automatically.
        """
        feature_type = Distances(excluded_neighbors=excluded_neighbors, use_pbc=use_pbc)
        return self._manager.add_feature(
            self._pipeline_data,
            feature_type,
            traj_selection=traj_selection,
            force=force,
            force_original=force_original,
        )
    
    def contacts(
        self,
        cutoff: float = 4.5,
        traj_selection: Union[str, int, List] = "all",
        force: bool = False,
        force_original: bool = True,
    ) -> None:
        """
        Add contacts feature type.
        
        Computes binary contact matrices from distance data using a distance threshold.
        Requires distances feature to be computed first as input dependency.
        
        Parameters
        ----------
        cutoff : float, default=4.5
            Distance cutoff in Angstroms for contact determination
        traj_selection : str, int, list, default="all"
            Which trajectories to compute features for
        force : bool, default=False
            Force recalculation even if feature already exists
        force_original : bool, default=True
            Whether to force using original trajectory data
            
        Returns
        -------
        None
            Adds contact features to pipeline data
            
        Examples
        --------
        >>> # Standard contacts with 4.5Å threshold
        >>> pipeline.feature.add.contacts()
        
        >>> # Longer-range contacts
        >>> pipeline.feature.add.contacts(cutoff=6.0)
        
        >>> # Force recalculation for specific trajectories
        >>> pipeline.feature.add.contacts(cutoff=5.0, traj_selection="all", force=True)
        
        Notes
        -----
        Contacts are binary (0 or 1) indicating whether atom pairs are within threshold distance.
        This feature depends on distances - ensure distances are computed first.
        """
        feature_type = Contacts(cutoff=cutoff)
        return self._manager.add_feature(
            self._pipeline_data,
            feature_type,
            traj_selection=traj_selection,
            force=force,
            force_original=force_original,
        )
    
    def torsions(
        self,
        calculate_phi: bool = True,
        calculate_psi: bool = True,
        calculate_omega: bool = True,
        calculate_chi: bool = True,
        use_pbc: bool = True,
        traj_selection: Union[str, int, List] = "all",
        force: bool = False,
        force_original: bool = True,
    ) -> None:
        """
        Add torsions feature type.

        Computes dihedral torsion angles including backbone (phi, psi, omega)
        and side chain angles (chi1-4). All angles are returned in degrees (-180 to +180).

        Parameters
        ----------
        calculate_phi : bool, default=True
            Whether to compute phi backbone angles
        calculate_psi : bool, default=True
            Whether to compute psi backbone angles
        calculate_omega : bool, default=True
            Whether to compute omega backbone angles
        calculate_chi : bool, default=True
            Whether to compute side chain chi angles (chi1, chi2, chi3, chi4)
        use_pbc : bool, default=True
            Use periodic boundary conditions when computing torsion angles
        traj_selection : str, int, list, default="all"
            Which trajectories to compute features for
        force : bool, default=False
            Force recalculation even if feature already exists
        force_original : bool, default=True
            Whether to force using original trajectory data

        Returns
        -------
        None
            Adds torsion features to pipeline data

        Examples
        --------
        >>> # All angles (default)
        >>> pipeline.feature.add.torsions()

        >>> # Only backbone angles
        >>> pipeline.feature.add.torsions(calculate_chi=False)

        >>> # Without periodic boundary conditions
        >>> pipeline.feature.add.torsions(use_pbc=False)

        >>> # Only phi and psi
        >>> pipeline.feature.add.torsions(
        ...     calculate_phi=True,
        ...     calculate_psi=True,
        ...     calculate_omega=False,
        ...     calculate_chi=False
        ... )

        >>> # Only side chain chi angles
        >>> pipeline.feature.add.torsions(
        ...     calculate_phi=False,
        ...     calculate_psi=False,
        ...     calculate_omega=False,
        ...     calculate_chi=True
        ... )

        Notes
        -----
        All angles are computed and returned in degrees.
        Uses the MDTraj library for angle calculations.
        Circular statistics should be used for analysis of torsion angles.
        """
        feature_type = Torsions(
            calculate_phi=calculate_phi,
            calculate_psi=calculate_psi,
            calculate_omega=calculate_omega,
            calculate_chi=calculate_chi,
            use_pbc=use_pbc,
        )
        return self._manager.add_feature(
            self._pipeline_data,
            feature_type,
            traj_selection=traj_selection,
            force=force,
            force_original=force_original,
        )
    
    def dssp(
        self,
        simplified: bool = False,
        encoding: str = 'char',
        traj_selection: Union[str, int, List] = "all",
        force: bool = False,
        force_original: bool = True,
    ) -> None:
        """
        Add DSSP secondary structure feature type.
        
        Computes secondary structure classification using DSSP algorithm.
        Provides either 8-class (H, B, E, G, I, T, S, C) or simplified 3-class (H, E, C) output.
        
        Parameters
        ----------
        simplified : bool, default=False
            Use simplified 3-state classification (H=helix, E=sheet, C=coil)
            If False, uses full 8-state DSSP classification
        encoding : str, default='char'
            Output encoding format:

            - 'char': Character codes ('H', 'E', 'C', etc.)
            - 'onehot': One-hot encoded binary vectors
            - 'integer': Integer class indices (0, 1, 2, ...)
        traj_selection : str, int, list, default="all"
            Which trajectories to compute features for
        force : bool, default=False
            Force recalculation even if feature already exists
        force_original : bool, default=True
            Whether to force using original trajectory data
            
        Returns
        -------
        None
            Adds DSSP features to pipeline data
            
        Examples
        --------
        >>> # Full 8-state DSSP classification
        >>> pipeline.feature.add.dssp()
        
        >>> # Simplified 3-state classification
        >>> pipeline.feature.add.dssp(simplified=True)
        
        >>> # One-hot encoded output
        >>> pipeline.feature.add.dssp(encoding='onehot')
        
        >>> # Integer encoded with simplified classification
        >>> pipeline.feature.add.dssp(simplified=True, encoding='integer')
        
        >>> # Force recalculation for specific trajectories
        >>> pipeline.feature.add.dssp(simplified=False, traj_selection=[0,1], force=True)
        
        Notes
        -----
        Uses MDTraj's DSSP implementation. The full 8-state codes are:
        DSSP classification: H (α-helix), B (β-bridge), E (β-sheet), 
        G (3-10 helix), I (π-helix), T (turn), S (bend), C (coil).
        Simplified mode maps: H,G,I → H; B,E → E; T,S,C → C.
        """
        feature_type = DSSP(simplified=simplified, encoding=encoding)
        return self._manager.add_feature(
            self._pipeline_data,
            feature_type,
            traj_selection=traj_selection,
            force=force,
            force_original=force_original,
        )
    
    def sasa(
        self,
        mode: str = 'residue',
        probe_radius: float = 0.14,
        traj_selection: Union[str, int, List] = "all",
        force: bool = False,
        force_original: bool = True,
    ) -> None:
        """
        Add SASA (Solvent Accessible Surface Area) feature type.
        
        Computes solvent accessible surface area for each residue using 
        the Shrake-Rupley algorithm implemented in MDTraj.
        
        Parameters
        ----------
        mode : str, default='residue'
            Level of SASA calculation:
            
            - 'residue': SASA per residue (sum of constituent atoms)
            - 'atom': SASA per individual atom
        probe_radius : float, default=0.14
            Probe radius in nanometers (water molecule radius)
            Standard values: 0.14 nm (water), 0.12 nm (smaller probe)
        traj_selection : str, int, list, default="all"
            Which trajectories to compute features for
        force : bool, default=False
            Force recalculation even if feature already exists
        force_original : bool, default=True
            Whether to force using original trajectory data
            
        Returns
        -------
        None
            Adds SASA features to pipeline data
            
        Examples
        --------
        >>> # Standard SASA with water probe
        >>> pipeline.feature.add.sasa()
        
        >>> # Custom probe radius
        >>> pipeline.feature.add.sasa(probe_radius=0.12)
        
        >>> # Atom-level SASA calculation
        >>> pipeline.feature.add.sasa(mode='atom')
        
        >>> # Atom-level with custom probe
        >>> pipeline.feature.add.sasa(mode='atom', probe_radius=0.12)
        
        >>> # For specific trajectories
        >>> pipeline.feature.add.sasa(probe_radius=0.14, traj_selection="all", force=True)
        
        Notes
        -----
        SASA values are returned in nm². Higher values indicate more solvent exposure.
        Mode 'residue' provides per-residue values, 'atom' provides per-atom values.
        Useful for identifying buried vs. exposed residues and conformational changes
        affecting protein-solvent interactions.
        """
        feature_type = SASA(mode=mode, probe_radius=probe_radius)
        return self._manager.add_feature(
            self._pipeline_data,
            feature_type,
            traj_selection=traj_selection,
            force=force,
            force_original=force_original,
        )
    
    def coordinates(
        self,
        atom_selection: str = "name CA",
        traj_selection: Union[str, int, List] = "all",
        force: bool = False,
        force_original: bool = True,
    ) -> None:
        """
        Add coordinates feature type.
        
        Extracts 3D coordinates (x, y, z) for selected atoms from trajectories.
        Useful for structural analysis and dimensionality reduction.
        
        Parameters
        ----------
        atom_selection : str, default="name CA"
            MDTraj atom selection string specifying which atoms to extract
            Examples: "name CA", "backbone", "protein", "resid 10 to 50 and name CA"
        traj_selection : str, int, list, default="all"
            Which trajectories to compute features for
        force : bool, default=False
            Force recalculation even if feature already exists
        force_original : bool, default=True
            Whether to force using original trajectory data
            
        Returns
        -------
        None
            Adds coordinate features to pipeline data
            
        Examples
        --------
        >>> # CA atoms only (default)
        >>> pipeline.feature.add.coordinates()
        
        >>> # All backbone atoms
        >>> pipeline.feature.add.coordinates(atom_selection="backbone")
        
        >>> # Specific residue range
        >>> pipeline.feature.add.coordinates(
        ...     atom_selection="resid 1 to 100 and name CA",
        ...     traj_selection=[0,1,2]
        ... )
        
        >>> # All protein atoms
        >>> pipeline.feature.add.coordinates(atom_selection="protein")
        
        Notes
        -----
        Coordinates are returned in nanometers as (x, y, z) triplets for each selected atom.
        The resulting feature matrix has shape (n_frames, n_atoms * 3).
        Consider alignment/centering for meaningful coordinate-based analysis.
        """
        feature_type = Coordinates(selection=atom_selection)
        return self._manager.add_feature(
            self._pipeline_data,
            feature_type,
            traj_selection=traj_selection,
            force=force,
            force_original=force_original,
        )
