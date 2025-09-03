# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0).
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
Diffusion Maps decomposition type for nonlinear dimensionality reduction.

Diffusion Maps decomposition type implementation with nonlinear dimensionality
reduction using diffusion processes and spectral analysis of transition matrices.
"""

from typing import Dict, Tuple, Optional, Any

import numpy as np

from ..interfaces.decomposition_type_base import DecompositionTypeBase
from .diffusion_maps_calculator import DiffusionMapsCalculator


class DiffusionMaps(DecompositionTypeBase):
    """
    Performs Diffusion Map analysis for nonlinear dimensionality reduction.

    This class provides an implementation of the Diffusion Map algorithm,
    inspired by the MDAnalysis toolkit [2]_, and adapted for the specific
    data structures and workflows of this software package.

    The method analyzes the intrinsic geometry of high-dimensional data by
    constructing a diffusion process on the data manifold, as originally
    proposed by Coifman & Lafon [1]_. 
    
    As a distance metric between configurations, this implementation uses the Root Mean Square Deviation
    (RMSD). The resulting diffusion coordinates correspond to the slowest
    collective motions in the system's dynamics, making them powerful for
    analyzing molecular simulations. [4]_

    A clear, less mathematical introduction to the method can be found in the
    work by de la Porte et al. [3]_.

    ## Mathematical Workflow

    The algorithm follows the "Random Walk" normalization approach:

    1.  **Kernel Matrix Construction:** A Gaussian kernel is constructed from the
        pairwise RMSD distance matrix `d(i, j)`. The parameter `epsilon`
        controls the kernel's bandwidth.
        `K_ij = exp(-d(i, j)**2 / epsilon)`

    2.  **Transition Matrix Normalization:** The kernel matrix `K` is
        row-normalized to create a row-stochastic transition matrix `M`, where
        each row sums to 1. `D` is a diagonal matrix of the row sums of `K`.
        `M = D⁻¹ * K`

    3.  **Eigendecomposition:** The top eigenvectors of the transition matrix `M`
        are computed. The first eigenvector, corresponding to the eigenvalue λ₀=1,
        is a trivial constant vector representing the stationary distribution of
        the diffusion process. It contains no geometric information and is
        therefore discarded.[1]_ The subsequent eigenvectors form the final
        diffusion coordinates.
        `M * v_k = λ_k * v_k`

    References
    ----------
    .. [1] Coifman, R. R.; Lafon, S. Diffusion maps.
           Appl. Comput. Harmon. Anal. 2006, 21 (1), 5–30.
           (See Section 3, "The Diffusion Map," for the reasoning on
           discarding the first eigenvector).
    .. [2] Michaud-Agrawal, N.; Denning, E. J.; Woolf, T. B.; Beckstein, O.
           MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics
           Simulations. J. Comput. Chem. 2011, 32, 2319–2327.
    .. [3] de la Porte, J.; Herbst, B. M.; Hereman, W.; van der Walt, S. J.
           An introduction to diffusion maps. In The 19th Symposium of the
           Pattern Recognition Association of South Africa. 2008.
    .. [4] Ferguson, A. L.; Panagiotopoulos, A. Z.; Debenedetti, P. G.;
           Kevrekidis, I. G. Nonlinear dimensionality reduction in molecular
           simulation: The diffusion map approach. Chem. Phys. Lett. 2011,
           509 (1-3), 1–11.

    Examples:
    ---------
    >>> # Basic Diffusion Maps decomposition via DecompositionManager
    >>> from mdxplain.decomposition import decomposition_type
    >>> decomp_manager = DecompositionManager()
    >>> decomp_manager.add(
    ...     traj_data, "feature_selection", decomposition_type.DiffusionMaps(n_components=10, epsilon=0.05)
    ... )

    >>> # Diffusion Maps with incremental computation for large datasets
    >>> diffmaps = decomposition_type.DiffusionMaps(n_components=20)
    >>> diffmaps.init_calculator(use_memmap=True, chunk_size=1000)
    >>> transformed, metadata = diffmaps.compute(large_data)

    >>> # Diffusion Maps with Nyström approximation for very large datasets
    >>> diffmaps = decomposition_type.DiffusionMaps(use_nystrom=True, n_landmarks=2000)
    >>> diffmaps.init_calculator()
    >>> transformed, metadata = diffmaps.compute(very_large_data, n_components=50)
    """

    def __init__(
        self,
        n_components: int,
        epsilon: Optional[float] = None,
        use_nystrom: bool = False,
        n_landmarks: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Diffusion Maps decomposition type.

        Creates a Diffusion Maps instance with specified parameters for constructing
        the diffusion process and extracting diffusion coordinates.

        Parameters:
        -----------
        n_components : int, required
            Number of diffusion coordinates to keep
        epsilon : float, optional, default=1.0
            Kernel bandwidth parameter for Gaussian kernel. Controls diffusion scale.
            If None, automatically estimated using k-nearest neighbors heuristic
        use_nystrom : bool, default=False
            Whether to use Nyström approximation for very large datasets
        n_landmarks : int, default=1000
            Number of landmarks for Nyström approximation
        random_state : int, optional
            Random state for reproducible results

        Returned Metadata:
        ------------------
        hyperparameters : dict
            Dictionary containing all Diffusion Maps parameters used
        original_shape : tuple
            Shape of the input data (n_samples, n_features)
        use_memmap : bool
            Whether memory mapping was used
        chunk_size : int
            Chunk size used for processing
        cache_path : str
            Path used for caching results
        method : str
            Method used ('standard_diffusion_maps', 'nystrom_diffusion_maps', or 'iterative_diffusion_maps')
        approximation : str
            Approximation method used ('nystrom' when Nyström approximation is enabled)
        n_landmarks : int
            Number of landmarks used for Nyström approximation (when applicable)
        eigenvalues : numpy.ndarray
            Eigenvalues of the diffusion operator (diffusion timescales)

        Examples:
        ---------
        >>> # Create Diffusion Maps instance
        >>> diffmaps = DiffusionMaps(n_components=10, epsilon=0.1)
        >>> print(f"Type: {diffmaps.get_type_name()}")
        'diffusion_maps'

        >>> # Create Diffusion Maps with Nyström approximation
        >>> diffmaps = DiffusionMaps(n_components=20, use_nystrom=True, n_landmarks=2000)
        """
        super().__init__()
        self.n_components = n_components
        self.epsilon = epsilon
        self.use_nystrom = use_nystrom
        self.n_landmarks = n_landmarks
        self.random_state = random_state

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the type name for Diffusion Maps decomposition.

        Returns the unique string identifier for Diffusion Maps decomposition type
        used for storing results and type identification.

        Parameters:
        -----------
        cls : type
            The DiffusionMaps class

        Returns:
        --------
        str
            String identifier 'diffusion_maps'

        Examples:
        ---------
        >>> print(DiffusionMaps.get_type_name())
        'diffusion_maps'
        >>> # Can also be used via class directly
        >>> print(decomposition_type.DiffusionMaps.get_type_name())
        'diffusion_maps'
        """
        return "diffusion_maps"

    def get_required_feature_type(self) -> str:
        """
        DiffusionMaps requires coordinate features for RMSD-based distance computation.

        Returns:
        --------
        str
            'coordinates' - indicating this method requires coordinate features
        """
        return "coordinates"

    def init_calculator(
        self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000
    ) -> None:
        """
        Initialize the Diffusion Maps calculator with specified configuration.

        Sets up the Diffusion Maps calculator with options for memory mapping and
        iterative kernel computation for large datasets.

        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use iterative computation with memory mapping for large datasets
        cache_path : str, optional
            Path for cache files when using memory mapping
        chunk_size : int, optional
            Number of samples to process per chunk for iterative computation

        Returns:
        --------
        None
            Sets self.calculator to initialized DiffusionMapsCalculator instance

        Examples:
        ---------
        >>> # Basic initialization
        >>> diffmaps = DiffusionMaps()
        >>> diffmaps.init_calculator()

        >>> # With iterative computation for large datasets
        >>> diffmaps.init_calculator(use_memmap=True, chunk_size=500)

        >>> # With custom cache path
        >>> diffmaps.init_calculator(
        ...     use_memmap=True,
        ...     cache_path="./cache/diffusion_maps.dat"
        ... )
        """
        self.calculator = DiffusionMapsCalculator(
            use_memmap=use_memmap, cache_path=cache_path, chunk_size=chunk_size
        )

    def compute(self, data, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Diffusion Maps decomposition of input trajectory.

        Performs Diffusion Maps analysis on the input MDTraj trajectory using the
        initialized calculator with the parameters provided during initialization.
        Constructs diffusion coordinates that capture the intrinsic geometry.

        Parameters:
        -----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, n_features) where n_features = n_atoms * 3
        **kwargs : dict
            Additional optional parameters (currently none supported)

        Returns:
        --------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            - diffusion_coords: Diffusion coordinates (n_frames, n_components)
            - metadata: Dictionary with Diffusion Maps information including:
              * hyperparameters: Used parameters
              * method: computation method used
              * eigenvalues: Eigenvalues of diffusion operator

        Examples:
        ---------
        >>> # Load trajectory and compute Diffusion Maps
        >>> import mdtraj as md
        >>> traj = md.load('trajectory.xtc', top='topology.pdb')
        >>> diffmaps = DiffusionMaps(n_components=10, epsilon=0.5, atom_selection="backbone")
        >>> diffmaps.init_calculator()
        >>> coords, metadata = diffmaps.compute(traj)
        >>> print(f"Coordinates shape: {coords.shape}")
        >>> print(f"Eigenvalues: {metadata['eigenvalues'][:3]}")

        >>> # Iterative Diffusion Maps for large trajectories
        >>> diffmaps = DiffusionMaps(n_components=20, epsilon=1.0)
        >>> diffmaps.init_calculator(use_memmap=True, chunk_size=200)
        >>> coords, metadata = diffmaps.compute(large_trajectory)

        Raises:
        -------
        ValueError
            If calculator is not initialized, input data is invalid,
            or n_components is too large
        """
        if self.calculator is None:
            raise ValueError(
                "Calculator not initialized. Call init_calculator() first."
            )

        return self.calculator.compute(
            data,
            n_components=self.n_components,
            epsilon=self.epsilon,
            use_nystrom=self.use_nystrom,
            n_landmarks=self.n_landmarks,
            random_state=self.random_state,
            **kwargs,
        )
