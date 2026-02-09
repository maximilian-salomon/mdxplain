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
Diffusion Maps calculator for nonlinear dimensionality reduction.

Implements Diffusion Maps computation with support for standard in-memory
computation and iterative memory-mapped computation for large datasets.
Uses MDTraj trajectories and RMSD-based distance computation.
"""

import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from mdxplain.utils.progress_utils import ProgressUtils
from mdxplain.utils.resource_utils import ResourceUtils
from mdxplain.utils.cleanup_utils import CleanupUtils
from mdxplain.utils.memmap_utils import MemmapUtils
from mdxplain.utils.path_utils import PathUtils
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

from ..interfaces.calculator_base import CalculatorBase


class DiffusionMapsCalculator(CalculatorBase):
    """
    Calculator for Diffusion Maps decomposition using MDTraj trajectories.

    Implements Diffusion Maps computation with support for standard in-memory
    computation and iterative memory-mapped computation for large datasets.
    Uses RMSD distances and follows the mathematical framework from
    Coifman & Lafon (2006).

    The algorithm consists of three main steps:
    
    1. Construct Gaussian kernel from RMSD distances: K_ij = exp(-d_ij^2 / epsilon)
    2. Normalize to transition matrix: M = D^(-1) * K (Random Walk normalization)
    3. Compute eigenvectors of M, skip first (stationary distribution)

    It also supports Nyström approximation for very large datasets.
    This method approximates the kernel matrix using a subset of the data,
    significantly reducing memory usage and computation time.
    See Fowlkes et al. (2004) for details.

    References
    ----------

    [1] Coifman, R. R.; Lafon, S. Diffusion maps.
    Appl. Comput. Harmon. Anal. 2006, 21 (1), 5–30.
    (See Section 3, "The Diffusion Map," for the reasoning on
    discarding the first eigenvector).

    [2] Michaud-Agrawal, N.; Denning, E. J.; Woolf, T. B.; Beckstein, O.
    MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics
    Simulations. J. Comput. Chem. 2011, 32, 2319–2327.

    [3] de la Porte, J.; Herbst, B. M.; Hereman, W.; van der Walt, S. J.
    An introduction to diffusion maps. In The 19th Symposium of the
    Pattern Recognition Association of South Africa. 2008.

    [4] Ferguson, A. L.; Panagiotopoulos, A. Z.; Debenedetti, P. G.;
    Kevrekidis, I. G. Nonlinear dimensionality reduction in molecular
    simulation: The diffusion map approach. Chem. Phys. Lett. 2011,
    509 (1-3), 1–11.
    
    [5] Fowlkes, C., Belongie, S., Chung, F., & Malik, J. (2004). 
    Spectral grouping using the nystrom method. 
    IEEE transactions on pattern analysis and 
    machine intelligence, 26(2), 214-225.
           
    Examples
    --------
    >>> # Standard Diffusion Maps for small trajectories
    >>> import mdtraj as md
    >>> calc = DiffusionMapsCalculator()
    >>> traj = md.load('small_traj.xtc', top='topology.pdb')
    >>> coords, metadata = calc.compute(traj, n_components=10, epsilon=0.5)

    >>> # Iterative Diffusion Maps for large trajectories  
    >>> calc = DiffusionMapsCalculator(use_memmap=True, chunk_size=500)
    >>> large_traj = md.load('large_traj.xtc', top='topology.pdb')
    >>> coords, metadata = calc.compute(large_traj, n_components=20, epsilon=1.0)
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
        """
        Initialize Diffusion Maps calculator.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping and iterative computation for large datasets
        cache_path : str, optional
            Path for memory-mapped cache files  
        chunk_size : int, optional
            Size of chunks for iterative computation (number of frames per chunk)

        Returns
        -------
        None
            Initializes Diffusion Maps calculator with specified configuration

        Examples
        --------
        >>> # Standard Diffusion Maps (small trajectories)
        >>> calc = DiffusionMapsCalculator()

        >>> # Iterative Diffusion Maps (large trajectories)
        >>> calc = DiffusionMapsCalculator(use_memmap=True, chunk_size=1000)
        """
        super().__init__(use_memmap, cache_path, chunk_size)
        self._cache_prefix = "diffusion_maps"
        self._temp_memmap_paths = []

    def compute(self, data, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Diffusion Maps decomposition of coordinate matrix.

        Performs Diffusion Maps analysis on the input coordinate matrix using either
        standard in-memory computation or iterative memory-mapped computation
        based on configuration settings.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        kwargs : dict
            Diffusion Maps parameters:

            - n_components : int, required
                Number of diffusion coordinates to keep
            - epsilon : float, optional
                Kernel bandwidth parameter. If None, estimated using k-NN heuristic
            - use_nystrom : bool, optional
                Whether to use Nyström approximation (default: False)
            - n_landmarks : int, optional
                Number of landmarks for Nyström approximation (default: 1000)
            - landmark_selection_mode : str, optional
                Landmark selection mode for Nyström ("kmeans" or "random")
            - alpha : float, optional
                Diffusion maps alpha normalization parameter (default: 0.0)
            - random_state : int, optional
                Random state for reproducible results
            - epsilon_k : int, optional
                k for k-NN epsilon estimation when epsilon is None
            - epsilon_n_samples : int, optional
                Number of samples used for epsilon estimation
            - epsilon_ref_size : int, optional
                Reference pool size used for epsilon estimation

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:

            - diffusion_coords: Diffusion coordinates (n_frames, n_components)
            - metadata: Dictionary with computation information and eigenvalues

        Examples
        --------
        >>> # Compute Diffusion Maps
        >>> calc = DiffusionMapsCalculator()
        >>> coords, metadata = calc.compute(
        ...     coord_matrix, n_components=10, epsilon=0.5
        ... )
        >>> print(f"Method: {metadata['method']}")
        >>> print(f"Eigenvalues: {metadata['eigenvalues']}")

        Raises
        ------
        ValueError
            If input is not numpy array or parameters are invalid
        """
        self._validate_input_data(data)
        hyperparameters, epsilon_diagnostics = self._extract_hyperparameters(data, kwargs)
        self._temp_memmap_paths = []

        if hyperparameters["use_nystrom"]:
            coords, metadata = self._compute_nystrom_diffusion_maps(data, hyperparameters)
        elif self.use_memmap:
            coords, metadata = self._compute_iterative_diffusion_maps(data, hyperparameters)
        else:
            coords, metadata = self._compute_standard_diffusion_maps(data, hyperparameters)

        if epsilon_diagnostics is not None:
            metadata["epsilon_diagnostics"] = epsilon_diagnostics

        self._cleanup_tracked_temp_memmaps()
        return coords, metadata

    def _validate_input_data(self, data) -> None:
        """
        Validate input coordinate matrix.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix to validate

        Returns
        -------
        None
            Validates input, raises ValueError if invalid

        Raises
        ------
        ValueError
            If input is not numpy array or has invalid shape
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if data.ndim != 2:
            raise ValueError("Input must be 2D array (n_frames, n_features)")

        if data.shape[0] < 2:
            raise ValueError("Data must have at least 2 frames")
        
        if data.shape[1] % 3 != 0:
            raise ValueError("n_features must be divisible by 3 (n_atoms * 3)")

    def _extract_hyperparameters(
        self, data, kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Extract and validate Diffusion Maps hyperparameters.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix for parameter validation
        kwargs : dict
            Input parameters to extract and validate

        Returns
        -------
        tuple
            (validated hyperparameters, epsilon diagnostics if computed)

        Raises
        ------
        ValueError
            If required parameters are missing or invalid
        """
        n_components = kwargs.get("n_components")
        if n_components is None:
            raise ValueError("n_components must be specified")

        # Validate n_components against n_frames first
        n_frames = data.shape[0]
        max_components = n_frames - 1  # Skip first (trivial) eigenvector
        if n_components > max_components:
            raise ValueError(
                f"n_components ({n_components}) cannot be larger than {max_components}"
            )

        use_nystrom = kwargs.get("use_nystrom", False)
        n_landmarks = kwargs.get("n_landmarks", 1000)
        landmark_selection_mode = kwargs.get("landmark_selection_mode", "kmeans")
        alpha = kwargs.get("alpha", 0.0)
        random_state = kwargs.get("random_state", None)

        epsilon = kwargs.get("epsilon")
        epsilon_k = kwargs.get("epsilon_k", None)
        epsilon_n_samples = kwargs.get("epsilon_n_samples", None)
        epsilon_ref_size = kwargs.get("epsilon_ref_size", None)
        epsilon_bootstrap_samples = 200
        epsilon_diagnostics = None

        if epsilon is None and not use_nystrom:
            epsilon, epsilon_diagnostics = self._estimate_epsilon_knn(
                data,
                random_state,
                k=epsilon_k,
                n_samples=epsilon_n_samples,
                ref_size=epsilon_ref_size,
                bootstrap_samples=epsilon_bootstrap_samples,
                return_diagnostics=True,
            )
        if epsilon is not None and epsilon <= 0:
            raise ValueError("epsilon must be positive")

        # Validate n_landmarks for Nyström
        if use_nystrom and n_landmarks >= n_frames:
            n_landmarks = min(n_landmarks, n_frames - 1)
            if n_landmarks < n_components:
                raise ValueError(
                    f"n_landmarks ({n_landmarks}) must be >= n_components ({n_components})"
                )
        if use_nystrom and n_landmarks > self.chunk_size:
            print(
                f"Warning: n_landmarks ({n_landmarks}) is larger than chunk_size ({self.chunk_size}). "
                "Nyström uses matrices of shape (n_landmarks x features). "
                "In other words: We use n_landmark as chunk_size here. "
                "Consider increasing chunk_size (>= n_landmarks) or reducing n_landmarks "
                "if this causes memory issues."
            )
        if use_nystrom and n_landmarks > 5000:
            # Empirically, 1k–5k landmarks are usually sufficient; more brings diminishing returns.
            warnings.warn(
                f"n_landmarks ({n_landmarks}) is > 5000. Values between 1000 and 5000 are "
                "usually sufficient; larger values often provide little benefit but add cost.",
                UserWarning,
            )
        if use_nystrom and landmark_selection_mode not in ["kmeans", "random"]:
            raise ValueError(
                f"Invalid landmark_selection_mode: '{landmark_selection_mode}'. Must be 'kmeans' or 'random'."
            )
        if not np.isfinite(alpha):
            raise ValueError("alpha must be a finite number")

        return {
            "n_components": n_components,
            "epsilon": epsilon,
            "use_nystrom": use_nystrom,
            "n_landmarks": n_landmarks,
            "landmark_selection_mode": landmark_selection_mode,
            "alpha": float(alpha),
            "random_state": random_state,
            "epsilon_k": epsilon_k,
            "epsilon_n_samples": epsilon_n_samples,
            "epsilon_ref_size": epsilon_ref_size,
        }, epsilon_diagnostics

    def _compute_standard_diffusion_maps(
        self, data: np.ndarray, hyperparameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute standard Diffusion Maps using in-memory computation.

        Recommended for < 10000 frames. Follows Coifman & Lafon (2006) approach:
        1. Compute RMSD distance matrix
        2. Build Gaussian kernel: K_ij = exp(-d_ij^2 / epsilon)
        3. Random Walk normalization: M = D^(-1) * K
        4. Eigendecomposition of M
        5. Skip first eigenvector (stationary distribution)

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        hyperparameters : dict
            Diffusion Maps hyperparameters

        Returns
        -------
        tuple
            Tuple of (diffusion_coordinates, metadata)
        """
        n_frames, n_features = data.shape
        n_atoms = n_features // 3
        
        # Step 1: Compute RMSD distance matrix
        # Reference: Coifman & Lafon (2006)
        rmsd_matrix = self._compute_rmsd_distance_matrix(data, n_atoms, "standard_rmsd_matrix.dat")

        # Step 2: Compute Gaussian kernel
        # Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 1):
        # Choose a rotation-invariant kernel  k_ε(x,y) = h(||x-y||^2 / ε).
        # Discrete analogue: K_ij = exp(- d_ij^2 / ε ), here using RMSD distances d_ij.
        kernel = np.exp(-(rmsd_matrix ** 2) / hyperparameters["epsilon"])
        # Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 2):
        # Define the kernel density estimate
        #     q_ε(x) = ∫ k_ε(x,y) q(y) dy
        # and the α-normalized (anisotropic) kernel
        #     k_ε^(α)(x,y) = k_ε(x,y) / ( q_ε(x)^α q_ε(y)^α ).
        # Discrete analogue:
        #     q_i ≈ Σ_j K_ij,
        #     K_ij^(α) = K_ij / (q_i^α q_j^α).
        self._apply_alpha_normalization(
            kernel,
            hyperparameters["alpha"],
            q_row=None,
            q_col=None,
            compute_row_sums=False,
            desc="Applying alpha normalization (dense)",
        )

        # Step 3: Random Walk normalization to create transition matrix
        # Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 3):
        # Define
        #     d_ε^(α)(x) = ∫ k_ε^(α)(x,y) q(y) dy
        # and the anisotropic transition kernel
        #     p_{ε,α}(x,y) = k_ε^(α)(x,y) / d_ε^(α)(x).
        # Discrete analogue:
        #     d_i^(α) ≈ Σ_j K_ij^(α),
        #     P_ij = K_ij^(α) / d_i^(α)  (row-stochastic Markov matrix).
        # Or in other words to map to formula in paper => Formula: M = D^(-1) * K where D_ii = sum_j K_ij
        # Reference: Coifman & Lafon (2006)
        transition_matrix, inv_row_sums = self._normalize_to_transition_matrix(kernel)

        # Step 4: Symmetric eigendecomposition (see helper for full derivation)
        # Perron-Frobenius guarantees the leading eigenvalue λ₀ = 1 (simple) and
        # a positive eigenvector for irreducible row-stochastic M. With symmetric
        # K^(α), M = D^(-1) K^(α) is similar to the symmetric operator
        # S = D^(1/2) M D^(-1/2) = D^(-1/2) K^(α) D^(-1/2), so eigenvalues are real.
        # Source: von Luxburg (2007), "A Tutorial on Spectral Clustering", Sec. 3.2, Prop. 3.
        eigenvals, eigenvecs = self._solve_markov_eigenvalue_problem(
            transition_matrix,
            inv_row_sums,
            n_components=hyperparameters["n_components"],
        )

        # Step 5: Extract diffusion coordinates (skip first eigenvector)
        # Reference: Coifman & Lafon (2006)
        diff_eigenvals, diff_coords = self._extract_diffusion_coordinates(
            eigenvals, eigenvecs, hyperparameters["n_components"]
        )

        metadata = self._prepare_metadata(hyperparameters, (n_frames, n_features))
        metadata.update({
            "method": "standard_diffusion_maps",
            "epsilon": hyperparameters["epsilon"],
            "eigenvalues": diff_eigenvals,
        })

        return diff_coords, metadata

    def _compute_iterative_diffusion_maps(
        self, data: np.ndarray, hyperparameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute iterative Diffusion Maps using memory mapping for large datasets.

        Uses memory-mapped arrays for chunk-wise RMSD computation and LinearOperator 
        for iterative eigenvalue computation. Follows Coifman & Lafon (2006) but 
        with memory-efficient implementation for large datasets.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        hyperparameters : dict
            Diffusion Maps hyperparameters

        Returns
        -------
        tuple
            Tuple of (diffusion_coordinates, metadata)
        """
        n_frames, n_features = data.shape
        n_atoms = n_features // 3
        epsilon = hyperparameters["epsilon"]

        # Step 1: Compute RMSD matrix as memmap
        # Reference: Coifman & Lafon (2006), memory-efficient approach
        rmsd_matrix = self._compute_rmsd_distance_matrix(data, n_atoms, "iterative_rmsd_matrix.dat")
        self._track_temp_memmap(rmsd_matrix)

        # Step 2: Compute kernel matrix as memmap and collect row sums
        kernel_matrix, inv_row_sums = self._compute_kernel_matrix(
            rmsd_matrix,
            epsilon,
            hyperparameters["alpha"],
            "iterative_kernel_matrix.dat",
        )
        self._track_temp_memmap(kernel_matrix)

        # Step 3: Symmetric eigendecomposition using sparse methods
        # This avoids materializing the full transition matrix
        eigenvals, eigenvecs = self._solve_operator_markov_eigenvalue_problem(
            kernel_matrix,
            inv_row_sums,
            n_components=hyperparameters["n_components"],
        )

        # Step 4: Extract diffusion coordinates
        diff_eigenvals, diff_coords = self._extract_diffusion_coordinates(
            eigenvals, eigenvecs, hyperparameters["n_components"]
        )

        metadata = self._prepare_metadata(hyperparameters, (n_frames, n_features))
        metadata.update({
            "method": "iterative_diffusion_maps",
            "epsilon": hyperparameters["epsilon"],
            "eigenvalues": diff_eigenvals,
            "n_chunks": int(np.ceil(n_frames / self.chunk_size)),
        })

        return diff_coords, metadata

    def _compute_nystrom_diffusion_maps(
        self, data: np.ndarray, hyperparameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Nyström approximation for Diffusion Maps using asymmetric normalization.
        
        Combines:
        
        - Coifman & Lafon (2006): Diffusion Maps framework with Markov matrices
        - Fowlkes et al. (2004): Nyström method for spectral decomposition
        
        This implementation uses asymmetric normalization to avoid the d_hat 
        approximation problem that arises with symmetric normalization.
        The resulting Markov operator is row-stochastic: M = D^(-1) K (with optional
        α-normalization of K before constructing M).

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        hyperparameters : dict
            Diffusion Maps hyperparameters

        Returns
        -------
        tuple
            Tuple of (diffusion_coordinates, metadata)
        """
        n_frames, n_features = data.shape
        n_atoms = n_features // 3
        n_landmarks = hyperparameters["n_landmarks"]
        landmark_selection_mode = hyperparameters.get("landmark_selection_mode", "kmeans")
        alpha = hyperparameters.get("alpha", 0.0)
        epsilon = hyperparameters["epsilon"]
        random_state = hyperparameters["random_state"]

        # STEP 1: Select Landmarks
        # Fowlkes et al. (2004): "randomly chosen samples"
        # improvement: Use KMeans for better coverage instead of random
        if landmark_selection_mode == "kmeans":
            landmark_idx = self._select_landmarks_kmeans(data, n_landmarks, random_state)
        elif landmark_selection_mode == "random":
            rng = np.random.RandomState(random_state)
            landmark_idx = rng.choice(n_frames, n_landmarks, replace=False)
            landmark_idx = np.sort(landmark_idx)
        else:
            raise ValueError("The parameter landmark_selection_mode only knows 'random' or 'kmeans'.")

        if epsilon is None:
            epsilon, epsilon_diagnostics = self._estimate_epsilon_from_landmarks(
                data,
                landmark_idx,
                hyperparameters["epsilon_k"],
                hyperparameters["epsilon_n_samples"],
                hyperparameters["epsilon_ref_size"],
                hyperparameters["random_state"],
                200,
                return_diagnostics=True,
            )
            hyperparameters["epsilon"] = epsilon
        else:
            epsilon_diagnostics = None

        # STEP 2: Compute Landmark Kernel Matrix A
        # Fowlkes et al. (2004): "partition the affinity matrix W = [A B; B^T C]"
        # Coifman & Lafon (2006), "Diffusion maps", 
        # Sec. 3.1 (Step 1): "κ(x,y) = exp(-||x-y||²/ε)"
        # Landmark kernel K_LL with entries k_ε(x,y) = exp(-||x-y||^2 / ε).
        K_landmarks = self._compute_landmarks_kernel(data, landmark_idx, epsilon, n_atoms)
        q_landmarks = None
        if alpha != 0.0:
            # Coifman & Lafon (2006), Section 3.1: k^α(x,y) = k(x,y) / (q(x)^α q(y)^α).
            # q(ℓ) ≈ Σ_{ℓ'} K(ℓ,ℓ')  (landmark density estimate).
            q_landmarks = K_landmarks.sum(axis=1)
            q_landmarks = np.maximum(q_landmarks, 1e-12)
            # q(ℓ)^α.
            q_landmarks_alpha = q_landmarks ** alpha
            # K_LL^(α) = K_LL / (q_L^α q_L^α).
            K_landmarks /= q_landmarks_alpha[:, np.newaxis]
            K_landmarks /= q_landmarks_alpha[np.newaxis, :]

        # STEP 3: Normalize to Markov Matrix (Asymmetric)
        # Coifman & Lafon (2006): "M_ij = K_ij/d_i where d_i = Σ_j K_ij"
        # This creates a row-stochastic matrix (rows sum to 1)
        M_small, inv_row_sums = self._nystrom_normalize_to_markov(K_landmarks)

        # STEP 4: Solve Eigenvalue Problem for Small Matrix
        # Standard eigendecomposition: P·v = λ·v
        # With symmetric K_LL^(α), M is similar to the symmetric operator
        # S = D^(1/2) M D^(-1/2) = D^(-1/2) K_LL^(α) D^(-1/2), so eigenvalues are real.
        # Source: von Luxburg (2007), "A Tutorial on Spectral Clustering", Sec. 3.2, Prop. 3.
        eigvals_small, eigvecs_small = self._solve_markov_eigenvalue_problem(
            M_small, inv_row_sums, hyperparameters["n_components"]
        )

        # STEP 5: Compute Kernel from All Points to Landmarks
        # This is B^T from Fowlkes et al. (2004), includes all n points
        # Shape: (n_frames × n_landmarks)
        # Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 1):
        # All-to-landmarks kernel K_XL with entries k_ε(x,ℓ).
        K_all_to_landmarks = self._compute_all_to_landmarks_kernel(
            data, landmark_idx, epsilon, n_atoms, n_frames, n_landmarks
        )
        self._track_temp_memmap(K_all_to_landmarks)
        if alpha != 0.0:
            # Coifman & Lafon (2006), Section 3.1: k^α(x,y) = k(x,y) / (q(x)^α q(y)^α).
            # K^(α)(x,ℓ) = K(x,ℓ) / (q(x)^α q(ℓ)^α).
            # Here q(ℓ) uses landmark degrees; q(x) is approximated from Σ_ℓ K(x,ℓ).
            self._apply_alpha_normalization(
                K_all_to_landmarks,
                alpha,
                q_row=None,
                q_col=q_landmarks,
                compute_row_sums=False,
                desc="Applying alpha normalization (all-to-landmarks)",
            )

        # STEP 6: Nyström Extension of Eigenvectors
        # Fowlkes et al. (2004): "ψ̂_i(x) = (1/nλ_i) Σ_j W(x,ξ_j) ψ̂_i(ξ_j)"
        # For Markov matrices: ψ(x) = (1/λ) Σ_j P(x,ξ_j) v(ξ_j)
        eigenvectors_full = self._nystrom_extend_eigenvectors(
            K_all_to_landmarks, eigvecs_small, eigvals_small, n_frames
        )
        self._track_temp_memmap(eigenvectors_full)

        # STEP 7: Extract Diffusion Coordinates
        # Coifman & Lafon (2006): Skip first eigenvector (stationary distribution)
        # First eigenvalue λ₁ = 1 with constant eigenvector for connected graphs
        diff_coords, diff_eigenvals = self._nystrom_extract_coordinates(
            eigenvectors_full, eigvals_small, hyperparameters["n_components"]
        )
        if MemmapUtils.is_memmap_view(diff_coords):
            diff_coords = np.array(diff_coords, copy=True)

        metadata = self._prepare_metadata(hyperparameters, (n_frames, n_features))
        metadata.update({
            "method": "nystrom_diffusion_maps",
            "epsilon": hyperparameters["epsilon"],
            "eigenvalues": diff_eigenvals,
            "n_landmarks": n_landmarks,
            "approximation": "asymmetric_nystrom",
            "landmark_selection_mode": landmark_selection_mode,
        })
        if epsilon_diagnostics is not None:
            metadata["epsilon_diagnostics"] = epsilon_diagnostics

        return diff_coords, metadata

    def _compute_rmsd_distance_matrix(self, data: np.ndarray, n_atoms: int, 
                                    filename: str = "rmsd_matrix.dat") -> np.ndarray:
        """
        Compute RMSD distance matrix with automatic memmap/array selection.
        
        Uses memmap if self.use_memmap is True, otherwise regular numpy array.
        Automatically handles cache path combination with cache_prefix.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        n_atoms : int
            Number of atoms (n_features // 3)
        filename : str, default="rmsd_matrix.dat"
            Filename for memmap (automatically combined with cache_path and prefix)

        Returns
        -------
        numpy.ndarray
            RMSD distance matrix (n_frames, n_frames) - memmap or regular array

        Notes
        -----
        The RMSD here is translation-invariant (structures are centered) but not
        rotation-invariant (no optimal rotational alignment is performed).
        """
        n_frames = data.shape[0]
        
        # Use helper method from CalculatorBase
        rmsd_matrix = self._create_array_or_memmap(
            shape=(n_frames, n_frames),
            dtype=np.float32,
            filename=filename
        )
        if self.use_memmap:
            ResourceUtils.tune_memmap(rmsd_matrix, "sequential")
        is_memmap_data = MemmapUtils.is_memmap_view(data)
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "sequential")
        
        # Compute RMSD matrix using symmetry and per-row chunking.
        for i in ProgressUtils.iterate(
            range(n_frames), desc="Computing RMSD matrix", unit="frames"
        ):
            rmsd_matrix[i, i] = 0.0
            ref = data[i]
            single_xyz = ref.reshape(n_atoms, 3)
            single_centered = single_xyz - single_xyz.mean(axis=0, dtype=np.float64)
            for j_start in range(i + 1, n_frames, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_frames)
                ref_chunk = data[j_start:j_end]
                distances = self._compute_rmsd_chunk_to_single_centered(
                    ref_chunk, single_centered, n_atoms
                )
                rmsd_matrix[i, j_start:j_end] = distances
                rmsd_matrix[j_start:j_end, i] = distances
            if self.use_memmap:
                rmsd_matrix.flush()

        if self.use_memmap:
            ResourceUtils.tune_memmap(rmsd_matrix, "random")
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "random")
        
        return rmsd_matrix

    def _compute_kernel_matrix(
        self,
        rmsd_matrix: np.ndarray,
        epsilon: float,
        alpha: float = 0.0,
        filename: str = "kernel_matrix.dat",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gaussian kernel matrix with automatic memmap/array selection.
        
        Uses memmap if self.use_memmap is True, otherwise regular numpy array.
        Automatically handles cache path combination with cache_prefix.

        Parameters
        ----------
        rmsd_matrix : numpy.ndarray
            RMSD distance matrix (n_frames, n_frames)
        epsilon : float
            Kernel bandwidth parameter
        alpha : float, default=0.0
            Diffusion maps alpha normalization parameter
        filename : str, default="kernel_matrix.dat"
            Filename for memmap (automatically combined with cache_path and prefix)

        Returns
        -------
        tuple
            (kernel_matrix, inv_row_sums) where kernel_matrix has shape
            (n_frames, n_frames) and inv_row_sums has shape (n_frames,)

        Notes
        -----
        The kernel uses RMSD distances that are translation-invariant but not
        rotation-invariant (no optimal rotational alignment is performed).
        """
        n_frames = rmsd_matrix.shape[0]
        
        # Use helper method from CalculatorBase
        kernel_matrix = self._create_array_or_memmap(
            shape=(n_frames, n_frames),
            dtype=np.float32,
            filename=filename
        )
        if self.use_memmap:
            ResourceUtils.tune_memmap(kernel_matrix, "sequential")
            ResourceUtils.tune_memmap(rmsd_matrix, "sequential")
        row_sums = np.zeros(n_frames)

        for i in ProgressUtils.iterate(
            range(0, n_frames, self.chunk_size),
            desc="Computing kernel matrix",
            unit="chunks",
        ):
            end_i = min(i + self.chunk_size, n_frames)
            # Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 1):
            # K_ij = k_ε(x_i, x_j) = exp(- d_ij^2 / ε).
            chunk_kernel = kernel_matrix[i:end_i]
            np.square(rmsd_matrix[i:end_i], out=chunk_kernel)
            chunk_kernel *= -(1.0 / epsilon)
            np.exp(chunk_kernel, out=chunk_kernel)
            # q_i ≈ Σ_j K_ij  (discrete estimate of q_ε(x_i)).
            row_sums[i:end_i] = chunk_kernel.sum(axis=1, dtype=np.float64)
            if self.use_memmap:
                kernel_matrix.flush()

        # Enforce exact diagonal (RMSD=0 => K_ii=1).
        np.fill_diagonal(kernel_matrix, 1.0)

        row_sums[row_sums < 1e-12] = 1e-12  # Numerical stability

        if alpha != 0.0:
            # Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 2):
            # K^(α) = D_q^{-α} K D_q^{-α}, with (D_q)_ii = q_i.
            # This modifies kernel_matrix in-place and recomputes
            # d_i^(α) = Σ_j K_ij^(α)  (needed for Step 3).
            row_sums = self._apply_alpha_normalization(
                kernel_matrix,
                alpha,
                q_row=row_sums,
                q_col=None,
                compute_row_sums=True,
                desc="Applying alpha normalization",
            )

        # Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 3):
        # P_ij = K_ij^(α) / d_i^(α)  => inv_row_sums[i] = 1 / d_i^(α).
        inv_row_sums = 1.0 / row_sums

        if self.use_memmap:
            ResourceUtils.tune_memmap(kernel_matrix, "random")
            ResourceUtils.tune_memmap(rmsd_matrix, "random")

        return kernel_matrix, inv_row_sums

    def _create_symmetric_operator(
        self,
        kernel_matrix: np.ndarray,
        inv_d_sqrt: np.ndarray,
    ) -> LinearOperator:
        """
        Create LinearOperator for the symmetric Markov operator S = D^(-1/2) K D^(-1/2).

        This operator is similar to the row-stochastic Markov matrix
        M = D^(-1) K but is symmetric (for symmetric K), enabling
        stable eigendecomposition with symmetric solvers.

        Parameters
        ----------
        kernel_matrix : numpy.ndarray
            Gaussian kernel matrix (can be memmap)
        inv_d_sqrt : numpy.ndarray
            D^(-1/2) vector for similarity transform

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            LinearOperator that computes S*v without materializing S
        """
        n_frames = kernel_matrix.shape[0]
        is_memmap_kernel = MemmapUtils.is_memmap_view(kernel_matrix)

        def matvec_mult(v):
            result = np.empty_like(v)
            if is_memmap_kernel:
                ResourceUtils.tune_memmap(kernel_matrix, "sequential")
            # Markov operator: M = D^(-1) K with D_ii = Σ_j K_ij.
            # Symmetric operator (similar to M):
            #     S = D^(-1/2) K D^(-1/2),
            # so S and M share eigenvalues.
            # Source: von Luxburg (2007), "A Tutorial on Spectral Clustering", Sec. 3.2, Prop. 3.
            #
            # Iterative eigensolvers (e.g., ARPACK) do not need S explicitly.
            # They only require a routine that returns S v for arbitrary input v.
            # We compute S v in three steps:
            #   1) w = D^(-1/2) v           (diagonal scaling)
            #   2) u = K w                 (kernel matrix-vector product)
            #   3) S v = D^(-1/2) u        (diagonal scaling again)
            # Since D^(-1/2) is diagonal, step (1) is elementwise:
            #   w_i = (1 / sqrt(d_i)) * v_i.
            weighted_v = inv_d_sqrt * v
            for i in range(0, n_frames, self.chunk_size):
                end_i = min(i + self.chunk_size, n_frames)
                kernel_chunk = kernel_matrix[i:end_i, :]
                result[i:end_i] = inv_d_sqrt[i:end_i] * (kernel_chunk @ weighted_v)
            if is_memmap_kernel:
                ResourceUtils.tune_memmap(kernel_matrix, "random")
            return result

        return LinearOperator((n_frames, n_frames), matvec=matvec_mult)

    def _normalize_to_transition_matrix(self, kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize kernel matrix to transition matrix using Random Walk normalization.

        Implements M = D^(-1) * K where D_ii = sum_j K_ij.

        Parameters
        ----------
        kernel : numpy.ndarray
            Gaussian kernel matrix

        Returns
        -------
        tuple
            (transition_matrix, inv_row_sums)
        """
        row_sums = kernel.sum(axis=1)
        row_sums = np.maximum(row_sums, 1e-12)  # Numerical stability
        inv_row_sums = 1.0 / row_sums
        return inv_row_sums[:, np.newaxis] * kernel, inv_row_sums

    def _apply_alpha_normalization(
        self,
        kernel_matrix: np.ndarray,
        alpha: float,
        q_row: Optional[np.ndarray] = None,
        q_col: Optional[np.ndarray] = None,
        compute_row_sums: bool = False,
        desc: str = "Applying alpha normalization",
    ) -> Optional[np.ndarray]:
        """
        Apply α-normalization to a kernel matrix (dense or iterative path).

        Parameters
        ----------
        kernel_matrix : numpy.ndarray
            Kernel matrix (n_rows, n_cols) from the dense or iterative path
        alpha : float
            Diffusion maps alpha normalization parameter
        q_row : numpy.ndarray, optional
            Row-side density estimates q(x). If None, computed from row sums.
        q_col : numpy.ndarray, optional
            Column-side density estimates q(y). If None, computed from column sums
            of the same matrix (only valid for square kernels).
        compute_row_sums : bool, default=False
            Whether to compute and return row sums after normalization
        desc : str, default="Applying alpha normalization"
            Progress description

        Returns
        -------
        numpy.ndarray or None
            Row sums after normalization if requested

        Notes
        -----
        Mapping to Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (Step 2):
            q_ε(x) = ∫ k_ε(x,y) q(y) dy
            k_ε^(α)(x,y) = k_ε(x,y) / ( q_ε(x)^α q_ε(y)^α )

        Discrete analogue:
            q_i ≈ Σ_j K_ij
            K_ij^(α) = K_ij / (q_i^α q_j^α)
        """
        # α = 0 means no density correction; optionally return current row sums if requested.
        if alpha == 0.0:
            return None if not compute_row_sums else kernel_matrix.sum(axis=1)

        # Cache matrix shape for square vs rectangular logic.
        n_rows = kernel_matrix.shape[0]
        n_cols = kernel_matrix.shape[1]

        # If row-side q(x) is not provided, estimate it from row sums (dense or iterative path).
        if q_row is None:
            q_row = kernel_matrix.sum(axis=1)
        # Clamp and compute q_row^α.
        q_row, q_row_alpha = self._compute_q_alpha(q_row, alpha)

        # If column-side q(y) is not provided, reuse row-side values for square kernels.
        if q_col is None:
            # For non-square kernels (e.g., all-to-landmarks), q_col must be supplied explicitly.
            if n_rows != n_cols:
                raise ValueError("q_col must be provided for non-square kernels")
            # Square kernel: q_col equals q_row by symmetry.
            q_col = q_row
            q_col_alpha = q_row_alpha
        else:
            # Clamp and compute q_col^α for provided column densities.
            q_col, q_col_alpha = self._compute_q_alpha(q_col, alpha)

        # Apply K_ij^(α) = K_ij / (q_i^α q_j^α) in chunks; optionally recompute d_i^(α).
        new_row_sums = self._apply_alpha_normalization_chunked(
            kernel_matrix,
            q_row_alpha,
            q_col_alpha,
            desc=desc,
            compute_row_sums=compute_row_sums,
        )
        return new_row_sums

    def _compute_q_alpha(
        self,
        row_sums: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute q and q^alpha with numerical safeguards.

        Parameters
        ----------
        row_sums : numpy.ndarray
            Row sums (q_i ≈ Σ_j K_ij)
        alpha : float
            Diffusion maps alpha normalization parameter

        Returns
        -------
        tuple
            (row_sums_clamped, q_alpha)
        """
        # Clamp to avoid division by zero.
        row_sums = np.maximum(row_sums, 1e-12)
        # q_i^α.
        q_alpha = row_sums ** alpha
        return row_sums, q_alpha

    def _apply_alpha_normalization_chunked(
        self,
        kernel_matrix: np.ndarray,
        q_row_alpha: np.ndarray,
        q_col_alpha: np.ndarray,
        desc: str,
        compute_row_sums: bool,
    ) -> Optional[np.ndarray]:
        """
        Apply alpha normalization in chunks and optionally return row sums.

        Parameters
        ----------
        kernel_matrix : numpy.ndarray
            Kernel matrix (dense or iterative path)
        q_row_alpha : numpy.ndarray
            q(x)^α for row indices
        q_col_alpha : numpy.ndarray
            q(y)^α for column indices
        desc : str
            Progress description
        compute_row_sums : bool
            Whether to compute and return row sums after normalization

        Returns
        -------
        numpy.ndarray or None
            Row sums after normalization if requested
        """
        # Total number of rows to process.
        n_rows = kernel_matrix.shape[0]
        # Switch memmap to sequential for efficient chunked access.
        if self.use_memmap:
            ResourceUtils.tune_memmap(kernel_matrix, "sequential")

        # Allocate output row sums only when requested (d_i^(α)).
        new_row_sums = None
        if compute_row_sums:
            new_row_sums = np.zeros(n_rows, dtype=np.float64)

        for i in ProgressUtils.iterate(
            range(0, n_rows, self.chunk_size),
            desc=desc,
            unit="chunks",
        ):
            end_i = min(i + self.chunk_size, n_rows)
            # K_ij^(α) = K_ij / (q_i^α q_j^α).
            chunk = kernel_matrix[i:end_i]
            chunk = chunk / q_row_alpha[i:end_i, np.newaxis]
            chunk = chunk / q_col_alpha[np.newaxis, :]
            kernel_matrix[i:end_i] = chunk
            if compute_row_sums:
                # d_i^(α) = Σ_j K_ij^(α).
                new_row_sums[i:end_i] = chunk.sum(axis=1)
            if self.use_memmap:
                kernel_matrix.flush()

        # Restore memmap access pattern.
        if self.use_memmap:
            ResourceUtils.tune_memmap(kernel_matrix, "random")

        # Clamp d_i^(α) to avoid division by zero downstream.
        if compute_row_sums and new_row_sums is not None:
            new_row_sums[new_row_sums < 1e-12] = 1e-12
        return new_row_sums

    def _extract_diffusion_coordinates(self, eigenvals: np.ndarray, 
                                     eigenvecs: np.ndarray, 
                                     n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract diffusion coordinates from eigenvalues and eigenvectors.

        Skips the first (trivial) eigenvector and returns the requested components.

        Parameters
        ----------
        eigenvals : numpy.ndarray
            Eigenvalues from transition matrix
        eigenvecs : numpy.ndarray
            Eigenvectors from transition matrix
        n_components : int
            Number of diffusion coordinates to return

        Returns
        -------
        tuple
            (diffusion_eigenvalues, diffusion_coordinates)
        """
        if eigenvals.dtype == complex:
            eigenvals = eigenvals.real
            eigenvecs = eigenvecs.real

        order = np.argsort(eigenvals)[::-1]
        sorted_eigenvals = eigenvals[order]
        sorted_eigenvecs = eigenvecs[:, order]

        diff_eigenvals = sorted_eigenvals[1:n_components+1]
        diff_coords = sorted_eigenvecs[:, 1:n_components+1]

        return diff_eigenvals, diff_coords

    def _cleanup_memmaps(self, memmap_paths: list) -> None:
        """
        Clean up temporary memory-mapped files.

        Parameters
        ----------
        memmap_paths : list
            List of paths to memory-mapped files to remove
        """
        for path in memmap_paths:
            CleanupUtils.remove_file(
                path,
                missing_ok=True,
                ignore_errors=True,
                purpose="temporary memmap path",
            )

    def _track_temp_memmap(self, array: np.ndarray) -> None:
        """
        Track a temporary memmap-backed array for cleanup after compute().

        Parameters
        ----------
        array : numpy.ndarray
            Array that may be backed by a temporary memmap file.

        Returns
        -------
        None
        """
        if not self.use_memmap:
            return
        if not MemmapUtils.is_memmap_view(array):
            return
        path = getattr(array, "filename", None)
        if isinstance(path, str) and path and path not in self._temp_memmap_paths:
            self._temp_memmap_paths.append(path)

    def _cleanup_tracked_temp_memmaps(self) -> None:
        """
        Cleanup all tracked temporary memmap files from the current compute run.

        Returns
        -------
        None
        """
        if not self.use_memmap:
            self._temp_memmap_paths = []
            return
        self._cleanup_memmaps(self._temp_memmap_paths)
        self._temp_memmap_paths = []

    def _estimate_epsilon_knn(
        self,
        data: np.ndarray,
        random_state: Optional[int],
        k: Optional[int] = None,
        n_samples: Optional[int] = None,
        ref_size: Optional[int] = None,
        index_pool: Optional[np.ndarray] = None,
        bootstrap_samples: int = 200,
        return_diagnostics: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Estimate epsilon using a k-nearest neighbors heuristic on a reference subset.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        random_state : int, optional
            Random state for reproducible sampling
        k : int, optional
            k-th neighbor used for distance scale estimation (0-based index).
        n_samples : int, optional
            Number of samples to estimate the k-NN distance scale
        ref_size : int, optional
            Size of reference pool to compute distances against
        index_pool : numpy.ndarray, optional
            Index pool to sample from (e.g., landmarks). If None, sample from all frames.
        bootstrap_samples : int, default=200
            Number of bootstrap resamples for median confidence interval. Use 0 to disable.
        return_diagnostics : bool, default=False
            Whether to return diagnostics alongside epsilon.

        Returns
        -------
        float or tuple
            Estimated epsilon value or (epsilon, diagnostics) when requested

        Notes
        -----
        The k-th neighbor here is 0-based: k=0 corresponds to the nearest neighbor,
        consistent with the use of np.partition(distances, k)[k].
        """
        n_frames, n_features = data.shape
        n_atoms = n_features // 3
        n_points = n_frames if index_pool is None else len(index_pool)

        k, n_samples, ref_size, sample_indices, ref_indices = self._resolve_epsilon_sampling(
            n_points, n_frames, random_state, k, n_samples, ref_size, index_pool
        )

        k_distances = self._compute_knn_distances(
            data, sample_indices, ref_indices, n_atoms, k
        )
        median_distance = float(np.median(k_distances))
        epsilon = median_distance ** 2
        
        diagnostics = self._compute_epsilon_diagnostics(
            k_distances, median_distance, bootstrap_samples, random_state
        )

        if epsilon < 1e-12:  # If epsilon is zero or very close to it
            epsilon = 1e-5

        if return_diagnostics:
            diagnostics["epsilon"] = epsilon
            return epsilon, diagnostics
        return epsilon

    def _compute_epsilon_diagnostics(
        self,
        k_distances: np.ndarray,
        median_distance: float,
        bootstrap_samples: int,
        random_state: Optional[int],
    ) -> Dict[str, Any]:
        """
        Compute diagnostics for epsilon estimation from k-NN distances.

        Parameters
        ----------
        k_distances : numpy.ndarray
            k-NN distance values used for epsilon estimation
        median_distance : float
            Median distance used to compute epsilon
        bootstrap_samples : int
            Number of bootstrap resamples for median confidence interval
        random_state : int, optional
            Random state for reproducible bootstrap sampling

        Returns
        -------
        dict
            Diagnostics including median, quantiles, and optional bootstrap CI
        """
        quantiles = np.quantile(k_distances, [0.05, 0.95])
        diagnostics = {
            "median_distance": float(median_distance),
            "quantiles": {
                "q05": float(quantiles[0]),
                "q50": float(median_distance),
                "q95": float(quantiles[1]),
            },
        }

        if bootstrap_samples <= 0:
            return diagnostics

        rng = np.random.RandomState(random_state)
        n_values = k_distances.shape[0]
        bootstrap_medians = np.zeros(bootstrap_samples, dtype=np.float32)
        for i in range(bootstrap_samples):
            sample_idx = rng.randint(0, n_values, size=n_values)
            bootstrap_medians[i] = np.median(k_distances[sample_idx])

        ci_low, ci_high = np.quantile(bootstrap_medians, [0.025, 0.975])
        diagnostics["bootstrap_ci"] = {
            "q025": float(ci_low),
            "q975": float(ci_high),
            "n_bootstrap": int(bootstrap_samples),
        }
        return diagnostics

    def _resolve_epsilon_sampling(
        self,
        n_points: int,
        n_frames_cap: int,
        random_state: Optional[int],
        k: Optional[int],
        n_samples: Optional[int],
        ref_size: Optional[int],
        index_pool: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
        """
        Resolve sampling parameters for epsilon estimation.

        Parameters
        ----------
        n_points : int
            Number of points in the sampling pool
        n_frames_cap : int
            Frame count used for default cap calculations
        random_state : int, optional
            Random state for reproducible sampling
        k : int, optional
            Requested k for k-NN estimation
        n_samples : int, optional
            Requested number of samples
        ref_size : int, optional
            Requested reference pool size
        index_pool : numpy.ndarray, optional
            Index pool to sample from (e.g., landmarks)

        Returns
        -------
        tuple
            (k, n_samples, ref_size, sample_indices, ref_indices)
        """
        k = self._resolve_epsilon_k(n_points, k)
        ref_size = self._resolve_epsilon_ref_size(n_points, n_frames_cap, k, ref_size)
        n_samples = self._resolve_epsilon_n_samples(n_points, n_frames_cap, ref_size, n_samples)

        rng = np.random.RandomState(random_state)
        if index_pool is None:
            ref_indices = rng.choice(n_points, ref_size, replace=False)
        else:
            ref_indices = rng.choice(index_pool, ref_size, replace=False)
        sample_indices = rng.choice(ref_indices, n_samples, replace=False)

        return k, n_samples, ref_size, sample_indices, ref_indices

    def _resolve_epsilon_k(self, n_frames: int, k: Optional[int]) -> int:
        """
        Resolve default k for epsilon estimation.

        Parameters
        ----------
        n_frames : int
            Number of frames in the dataset
        k : int, optional
            User-provided k for k-NN estimation

        Returns
        -------
        int
            Resolved k value
        """
        if k is None:
            k = int(5 * np.log(max(n_frames, 2)))
            k = max(20, min(k, 100))
        return max(0, min(k, n_frames - 1))

    def _resolve_epsilon_ref_size(
        self, n_points: int, n_frames_cap: int, k: int, ref_size: Optional[int]
    ) -> int:
        """
        Resolve default reference pool size for epsilon estimation.

        Parameters
        ----------
        n_points : int
            Number of points in the sampling pool
        n_frames_cap : int
            Frame count used for default cap calculations
        k : int
            Resolved k for k-NN estimation
        ref_size : int, optional
            User-provided reference pool size

        Returns
        -------
        int
            Resolved reference pool size
        """
        if ref_size is None:
            cap_frames = max(1, n_frames_cap)
            ref_size = min(50000, int(0.25 * cap_frames))
        ref_size = max(k + 1, min(ref_size, n_points))
        return max(1, ref_size)

    def _resolve_epsilon_n_samples(
        self, n_points: int, n_frames_cap: int, ref_size: int, n_samples: Optional[int]
    ) -> int:
        """
        Resolve default sample count for epsilon estimation.

        Parameters
        ----------
        n_points : int
            Number of points in the sampling pool
        n_frames_cap : int
            Frame count used for default cap calculations
        ref_size : int
            Resolved reference pool size
        n_samples : int, optional
            User-provided number of samples

        Returns
        -------
        int
            Resolved number of samples
        """
        if n_samples is None:
            cap_frames = max(1, n_frames_cap)
            n_samples = min(5000, int(0.05 * cap_frames))
        n_samples = max(1, min(n_samples, ref_size, n_points))
        return n_samples

    def _compute_knn_distances(
        self,
        data: np.ndarray,
        sample_indices: np.ndarray,
        ref_indices: np.ndarray,
        n_atoms: int,
        k: int,
    ) -> np.ndarray:
        """
        Compute k-NN distances for epsilon estimation.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        sample_indices : numpy.ndarray
            Indices of sample frames
        ref_indices : numpy.ndarray
            Indices of reference frames
        n_atoms : int
            Number of atoms (n_features // 3)
        k : int
            k-th neighbor to extract (0-based index)

        Returns
        -------
        numpy.ndarray
            Array of k-NN distances for each sample

        Notes
        -----
        Uses zero-based indexing: k=0 returns the nearest neighbor distance.
        """
        k_distances = np.zeros(len(sample_indices), dtype=np.float32)

        for i, sample_idx in enumerate(
            ProgressUtils.iterate(sample_indices, desc="Estimating epsilon", unit="samples")
        ):
            distances = self._distances_to_reference(
                data, ref_indices, data[sample_idx], n_atoms
            )
            k_distances[i] = np.partition(distances, k)[k]

        return k_distances

    def _distances_to_reference(
        self,
        data: np.ndarray,
        ref_indices: np.ndarray,
        sample_coords: np.ndarray,
        n_atoms: int,
    ) -> np.ndarray:
        """
        Compute RMSD distances from a sample frame to a reference pool.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        ref_indices : numpy.ndarray
            Indices of reference frames
        sample_coords : numpy.ndarray
            Sample coordinate vector (3 * n_atoms,)
        n_atoms : int
            Number of atoms (n_features // 3)

        Returns
        -------
        numpy.ndarray
            Distances from the sample to each reference frame
        """
        ref_size = ref_indices.shape[0]
        distances = np.zeros(ref_size, dtype=np.float32)
        sample_xyz = sample_coords.reshape(n_atoms, 3)
        sample_centered = sample_xyz - sample_xyz.mean(axis=0)

        is_memmap_data = MemmapUtils.is_memmap_view(data)
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "sequential")

        for chunk_start in range(0, ref_size, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, ref_size)
            ref_chunk_indices = ref_indices[chunk_start:chunk_end]
            ref_chunk = data[ref_chunk_indices]
            distances[chunk_start:chunk_end] = self._compute_rmsd_chunk_to_single_centered(
                ref_chunk, sample_centered, n_atoms
            )

        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "random")

        return distances

    def _estimate_epsilon_from_landmarks(
        self,
        data: np.ndarray,
        landmark_idx: np.ndarray,
        k: Optional[int],
        n_samples: Optional[int],
        ref_size: Optional[int],
        random_state: Optional[int],
        bootstrap_samples: int,
        return_diagnostics: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Estimate epsilon using k-NN distances within the landmark set.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        landmark_idx : numpy.ndarray
            Landmark indices
        k : int, optional
            k-th neighbor used for distance scale estimation
        n_samples : int, optional
            Number of samples used for epsilon estimation
        ref_size : int, optional
            Reference pool size used for epsilon estimation
        random_state : int, optional
            Random state for reproducible sampling
        bootstrap_samples : int
            Number of bootstrap resamples for median confidence interval
        return_diagnostics : bool, default=False
            Whether to return diagnostics alongside epsilon.

        Returns
        -------
        float or tuple
            Estimated epsilon value or (epsilon, diagnostics) when requested
        """
        if len(landmark_idx) < 2:
            if return_diagnostics:
                return 1e-5, {"epsilon": 1e-5}
            return 1e-5

        return self._estimate_epsilon_knn(
            data,
            random_state,
            k=k,
            n_samples=n_samples,
            ref_size=ref_size,
            index_pool=landmark_idx,
            bootstrap_samples=bootstrap_samples,
            return_diagnostics=return_diagnostics,
        )

    def _compute_rmsd_chunk_to_single_centered(
        self,
        chunk_coords: np.ndarray,
        single_centered: np.ndarray,
        n_atoms: int,
    ) -> np.ndarray:
        """
        Compute RMSD from chunk of frames to a pre-centered single structure.

        Parameters
        ----------
        chunk_coords : numpy.ndarray
            Chunk coordinate matrix (n_chunk_frames, 3 * n_atoms)
        single_centered : numpy.ndarray
            Centered single structure (n_atoms, 3)
        n_atoms : int
            Number of atoms

        Returns
        -------
        numpy.ndarray
            Array of RMSD values (n_chunk_frames,)
        """
        n_chunk_frames = chunk_coords.shape[0]

        # Reshape to (n_chunk_frames, n_atoms, 3)
        chunk_xyz = chunk_coords.reshape(n_chunk_frames, n_atoms, 3)

        # Center chunk structures (vectorized)
        chunk_centered = chunk_xyz - chunk_xyz.mean(axis=1, keepdims=True)

        # Compute RMSD for all frames in chunk (vectorized)
        diff = chunk_centered - single_centered[np.newaxis, :, :]
        rmsd_values = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))

        return rmsd_values

    def _compute_rmsd_sq_chunk_to_landmarks(
        self,
        chunk_coords: np.ndarray,
        landmark_flat: np.ndarray,
        landmark_ms: np.ndarray,
        n_atoms: int,
    ) -> np.ndarray:
        """
        Compute RMSD^2 from a chunk of frames to all landmarks (vectorized).

        Parameters
        ----------
        chunk_coords : numpy.ndarray
            Chunk coordinate matrix (n_chunk_frames, 3 * n_atoms)
        landmark_flat : numpy.ndarray
            Flattened centered landmark coordinates (n_landmarks, 3 * n_atoms)
        landmark_ms : numpy.ndarray
            Mean-square of landmarks (n_landmarks,)
        n_atoms : int
            Number of atoms

        Returns
        -------
        numpy.ndarray
            RMSD^2 values (n_chunk_frames, n_landmarks)

        Notes
        -----
        The RMSD here is translation-invariant (structures are centered) but not
        rotation-invariant (no optimal rotational alignment is performed).
        """
        n_chunk_frames = chunk_coords.shape[0]

        # Reshape to (n_chunk_frames, n_atoms, 3) and center each frame.
        chunk_xyz = chunk_coords.reshape(n_chunk_frames, n_atoms, 3)
        chunk_centered = chunk_xyz - chunk_xyz.mean(axis=1, keepdims=True)

        # Flatten for dot products: (n_chunk_frames, n_atoms*3).
        chunk_flat = chunk_centered.reshape(n_chunk_frames, -1)

        # RMSD^2 between centered structures x and y:
        #   RMSD^2(x,y) = mean(|x - y|^2)
        #              = mean(|x|^2) + mean(|y|^2) - 2 * mean(x·y).
        # This matches _compute_rmsd_chunk_to_single but vectorized over landmarks.
        # Source: Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1 (uses RMSD-based kernel).
        chunk_ms = np.mean(chunk_flat ** 2, axis=1, dtype=np.float32)
        dot = (chunk_flat @ landmark_flat.T) / (n_atoms * 3)
        rmsd_sq = chunk_ms[:, np.newaxis] + landmark_ms[np.newaxis, :] - 2.0 * dot
        rmsd_sq = np.maximum(rmsd_sq, 0.0)

        return rmsd_sq

    def _compute_landmarks_kernel(self, data: np.ndarray, landmark_idx: np.ndarray, 
                                 epsilon: float, n_atoms: int) -> np.ndarray:
        """
        Compute kernel matrix between landmark frames (n_landmarks × n_landmarks).

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        landmark_idx : numpy.ndarray
            Indices of landmark frames
        epsilon : float
            Kernel bandwidth parameter
        n_atoms : int
            Number of atoms (n_features // 3)

        Returns
        -------
        numpy.ndarray
            Kernel matrix between landmarks

        Notes
        -----
        The RMSD here is translation-invariant (structures are centered) but not
        rotation-invariant (no optimal rotational alignment is performed).
        """
        n_landmarks = len(landmark_idx)
        K_landmarks = np.zeros((n_landmarks, n_landmarks), dtype=np.float32)
        # This is allowed, cause we need to do it also at other places and warned the user. 
        # Usually this should not be a problem, if users choose a normal size.
        landmark_coords = data[landmark_idx]
        is_memmap_data = MemmapUtils.is_memmap_view(data)
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "sequential")
        
        for i in ProgressUtils.iterate(
            range(n_landmarks),
            desc="Computing landmark kernel matrix",
            unit="landmarks",
        ):
            K_landmarks[i, i] = 1.0
            ref = landmark_coords[i]
            single_xyz = ref.reshape(n_atoms, 3)
            single_centered = single_xyz - single_xyz.mean(axis=0)
            if i + 1 < n_landmarks:
                ref_chunk = landmark_coords[i + 1:]
                rmsd = self._compute_rmsd_chunk_to_single_centered(
                    ref_chunk, single_centered, n_atoms
                )
                kernel_vals = np.exp(-(rmsd ** 2) / epsilon)
                K_landmarks[i, i + 1:] = kernel_vals
                K_landmarks[i + 1:, i] = kernel_vals
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "random")
        
        return K_landmarks

    def _compute_all_to_landmarks_kernel(self, data: np.ndarray, landmark_idx: np.ndarray, 
                                        epsilon: float, n_atoms: int, 
                                        n_frames: int, n_landmarks: int) -> np.ndarray:
        """
        Compute kernel matrix from all frames to landmarks (n_frames × n_landmarks).

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, 3 * n_atoms)
        landmark_idx : numpy.ndarray
            Indices of landmark frames
        epsilon : float
            Kernel bandwidth parameter
        n_atoms : int
            Number of atoms (n_features // 3)
        n_frames : int
            Number of frames in trajectory
        n_landmarks : int
            Number of landmarks

        Returns
        -------
        numpy.ndarray
            Kernel matrix from all frames to landmarks

        Notes
        -----
        The RMSD here is translation-invariant (structures are centered) but not
        rotation-invariant (no optimal rotational alignment is performed).
        This matrix corresponds to the Nyström cross-kernel K_XL used for the
        approximation (see Fowlkes et al., 2004).
        """
        # Use helper method to create K_all_to_landmarks matrix
        K_all_to_landmarks = self._create_array_or_memmap(
            shape=(n_frames, n_landmarks),
            dtype=np.float32,
            filename="nystrom_K_all.dat"
        )
        if self.use_memmap:
            ResourceUtils.tune_memmap(K_all_to_landmarks, "sequential")
        is_memmap_data = MemmapUtils.is_memmap_view(data)
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "sequential")

        # Precompute centered landmark coordinates once.
        landmark_coords = data[landmark_idx]
        landmark_xyz = landmark_coords.reshape(n_landmarks, n_atoms, 3)
        landmark_centered = landmark_xyz - landmark_xyz.mean(axis=1, keepdims=True, dtype=np.float32)
        # Precompute flattened landmarks and mean-square once.
        landmark_flat = landmark_centered.reshape(n_landmarks, -1)
        landmark_ms = np.mean(landmark_flat ** 2, axis=1, dtype=np.float32)

        # Chunk-wise computation (outer loop over frames, vectorized over landmarks).
        for chunk_start in ProgressUtils.iterate(
            range(0, n_frames, self.chunk_size),
            desc="Nystroem all-to-landmarks kernel",
            unit="chunks",
        ):
            chunk_end = min(chunk_start + self.chunk_size, n_frames)
            chunk_data = data[chunk_start:chunk_end]

            # Vectorized RMSD^2 from chunk to all landmarks.
            rmsd_sq = self._compute_rmsd_sq_chunk_to_landmarks(
                chunk_data,
                landmark_flat,
                landmark_ms,
                n_atoms,
            )

            # K(x,ℓ) = exp(-RMSD(x,ℓ)^2 / ε)
            K_all_to_landmarks[chunk_start:chunk_end, :] = np.exp(-rmsd_sq / epsilon)
            if self.use_memmap:
                K_all_to_landmarks.flush()
        
        if self.use_memmap:
            ResourceUtils.tune_memmap(K_all_to_landmarks, "random")
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "random")

        return K_all_to_landmarks

    def _nystrom_normalize_to_markov(self, K_landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize kernel matrix to Markov matrix using asymmetric normalization.
        
        Coifman & Lafon (2006): "M_ij = K_ij/d_i where d_i = Σ_j K_ij"
        This creates a row-stochastic matrix (rows sum to 1).
        Avoids the d_hat problem from symmetric normalization.

        Parameters
        ----------
        K_landmarks : numpy.ndarray
            Kernel matrix between landmarks

        Returns
        -------
        tuple
            (M_small, inv_row_sums) where M_small is row-stochastic matrix
        """
        row_sums_landmarks = K_landmarks.sum(axis=1)
        row_sums_landmarks = np.maximum(row_sums_landmarks, 1e-12)  # Numerical stability
        inv_row_sums = 1.0 / row_sums_landmarks
        
        # Create row-stochastic matrix: M_small = D^(-1) * K_landmarks
        M_small = inv_row_sums[:, np.newaxis] * K_landmarks
        
        return M_small, inv_row_sums

    def _solve_markov_eigenvalue_problem(
        self,
        markov_matrix: np.ndarray,
        inv_row_sums: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for a Markov matrix using a symmetric formulation.

        Symmetrize the Markov operator:
        S = D^(1/2) * M * D^(-1/2), where D = diag(row_sums(K)).
        Solve S·u = λ·u (symmetric), then recover v = D^(-1/2) * u.
        Eigenvalues match the original Markov operator M.

        Parameters
        ----------
        markov_matrix : numpy.ndarray
            Row-stochastic matrix M = D^(-1) K
        inv_row_sums : numpy.ndarray
            Inverse row sums (1 / d_i) from kernel matrix
        n_components : int, optional
            If provided, compute only the top-(n_components + 1) eigenpairs

        Returns
        -------
        tuple
            (eigenvalues, eigenvectors) sorted by eigenvalue magnitude (descending)
        """
        # We solve the Markov eigenproblem:
        #     M v = λ v,  with M = D^(-1) K and D_ii = Σ_j K_ij.
        # For symmetric K, M is similar to the symmetric operator
        #     S = D^(1/2) M D^(-1/2) = D^(-1/2) K D^(-1/2),
        # so we solve the symmetric eigenproblem
        #     S u = λ u,
        # then map back
        #     v = D^(-1/2) u.
        # Source: von Luxburg (2007), "A Tutorial on Spectral Clustering", Sec. 3.2, Prop. 3.
        # The Markov operator M = D^(-1) K is the diffusion maps transition matrix.
        # Source: Coifman & Lafon (2006), "Diffusion maps", Sec. 3.1.
        n_rows = markov_matrix.shape[0]
        # We need (n_components + 1) eigenpairs to drop the trivial stationary one.
        k = None if n_components is None else int(n_components + 1)
        use_full = k is None or n_rows <= 2 or k >= n_rows - 1 or n_rows <= 200

        S, inv_d_sqrt = self._symmetrize_markov_operator(markov_matrix, inv_row_sums)

        if use_full:
            # Full symmetric eigendecomposition (dense path).
            eigvals, eigvecs = np.linalg.eigh(S)
        else:
            try:
                # Partial symmetric eigendecomposition (top eigenpairs only).
                k_eff = max(1, min(int(k), n_rows - 1))
                eigvals, eigvecs = eigsh(S, k=k_eff, which="LA")
            # In rare cases it can trough an error if it does not converge, then we need to swap to full calculation.
            except ArpackNoConvergence as err:
                eigvals = err.eigenvalues
                eigvecs = err.eigenvectors
                if eigvals is None or eigvecs is None or eigvals.shape[0] < k_eff:
                    # ARPACK may return partial results; fall back to full dense solve for robustness.
                    eigvals, eigvecs = np.linalg.eigh(S)

        return self._postprocess_markov_eigenpairs(
            eigvals,
            eigvecs,
            inv_row_sums,
            inv_d_sqrt,
        )

    def _solve_operator_markov_eigenvalue_problem(
        self,
        kernel_matrix: np.ndarray,
        inv_row_sums: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for a large Markov operator using a symmetric LinearOperator.

        Constructs the symmetric operator S = D^(-1/2) K D^(-1/2) as a LinearOperator,
        then computes the leading eigenpairs with symmetric sparse solvers.

        Perron-Frobenius guarantees the leading eigenvalue λ₀ = 1 (simple) and
        a positive eigenvector for irreducible row-stochastic M. With symmetric
        K^(α), M = D^(-1) K^(α) is similar to the symmetric operator
        S = D^(1/2) M D^(-1/2) = D^(-1/2) K^(α) D^(-1/2), so eigenvalues are real.
        Source: von Luxburg (2007), "A Tutorial on Spectral Clustering", Sec. 3.2, Prop. 3.
        And Coifman & Lafon (2006)

        Parameters
        ----------
        kernel_matrix : numpy.ndarray
            Gaussian kernel matrix (can be memmap)
        inv_row_sums : numpy.ndarray
            Inverse row sums (1 / d_i) from kernel matrix
        n_components : int
            Number of diffusion components (excluding the trivial eigenvector)

        Returns
        -------
        tuple
            (eigenvalues, eigenvectors) sorted by eigenvalue magnitude (descending)
        """
        n_rows = kernel_matrix.shape[0]
        k_eff = max(1, min(int(n_components + 1), n_rows - 1))

        # Stabilize row sums and build D^(-1/2) for symmetric operator.
        # d_i = Σ_j K_ij, inv_row_sums = 1 / d_i, inv_d_sqrt = 1 / sqrt(d_i).
        row_sums = 1.0 / inv_row_sums
        row_sums = np.maximum(row_sums, 1e-12)
        inv_row_sums = 1.0 / row_sums
        inv_d_sqrt = np.sqrt(inv_row_sums)

        # Symmetric operator:
        #     S = D^(-1/2) K D^(-1/2),
        # which is similar to M = D^(-1) K and shares eigenvalues with M.
        # Source: von Luxburg (2007), "A Tutorial on Spectral Clustering", Sec. 3.2, Prop. 3.
        symmetric_operator = self._create_symmetric_operator(kernel_matrix, inv_d_sqrt)

        try:
            # Symmetric sparse eigensolver for the leading eigenpairs.
            eigvals, eigvecs = eigsh(symmetric_operator, k=k_eff, which="LA")
        except ArpackNoConvergence as err:
            # Short retry with adjusted parameters (ncv/tol) and capped iterations.
            try:
                # Increase subspace size to help convergence while keeping it bounded.
                # ncv must be > k_eff, and should be <= n_rows.
                ncv_retry = min(n_rows, max(2 * k_eff + 1, k_eff + 2, 20))
                eigvals, eigvecs = eigsh(
                    symmetric_operator,
                    k=k_eff,
                    which="LA",
                    ncv=ncv_retry,
                    tol=1e-6,
                    maxiter=300,
                )
            except ArpackNoConvergence as retry_err:
                raise retry_err from err

        return self._postprocess_markov_eigenpairs(
            eigvals,
            eigvecs,
            inv_row_sums,
            inv_d_sqrt,
        )

    def _postprocess_markov_eigenpairs(
        self,
        eigenvals: np.ndarray,
        eigenvecs: np.ndarray,
        inv_row_sums: np.ndarray,
        inv_d_sqrt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map eigenpairs of the symmetric operator back to the Markov operator and normalize.

        This mirrors the Nyström path: eigenvectors of S are mapped to eigenvectors of
        M = D^(-1) K via v = D^(-1/2) u, then normalized in the π-weighted inner product.

        Parameters
        ----------
        eigenvals : numpy.ndarray
            Eigenvalues from symmetric operator S
        eigenvecs : numpy.ndarray
            Eigenvectors from symmetric operator S (columns)
        inv_row_sums : numpy.ndarray
            Inverse row sums (1 / d_i) from kernel matrix
        inv_d_sqrt : numpy.ndarray
            D^(-1/2) vector from the similarity transform

        Returns
        -------
        tuple
            (eigenvalues, eigenvectors) sorted by eigenvalue magnitude (descending)
        """
        # Eigenvalues are real because S is symmetric (similar to the Markov operator).
        # Source: von Luxburg (2007), A Tutorial on Spectral Clustering, Prop. 3.
        # Drop numerical imaginary parts from symmetric eigendecomposition.
        eigenvals = eigenvals.real
        # Drop numerical imaginary parts from symmetric eigendecomposition (numerical safeguard).
        eigenvecs = eigenvecs.real

        # Map eigenvectors u of S to eigenvectors v of P via v = D^{-1/2} u.
        # Source: von Luxburg (2007), Proposition .
        eigenvecs = (inv_d_sqrt[:, np.newaxis] * eigenvecs)

        # Recover degrees d_i = Σ_j K_ij from stored inverse degrees (definition of D).
        row_sums = 1.0 / inv_row_sums

        # Clamp degrees to avoid division by zero in normalization (numerical safeguard).
        row_sums = np.maximum(row_sums, 1e-12)

        # We compute the stationary distribution π of the Markov chain P := D^{-1}K.
        #
        # In Aldous & Fill, "Reversible Markov Chains and Random Walks on Graphs",
        # stationarity is defined by the equation (component-wise):
        #     π_j = Σ_i π_i p_ij  for all j.   (Eq. (2.1))
        #
        # Here p_ij = P_ij = K_ij / d_i with d_i := Σ_k K_ik.
        # Substituting into the stationarity equation gives:
        #     π_j = Σ_i π_i (K_ij / d_i).
        #
        # To construct a stationary solution, test the (unnormalized) degree vector q with q_i := d_i.
        # Then term-wise:
        #     q_i (K_ij / d_i) = d_i (K_ij / d_i) = K_ij,
        # and summing over i yields:
        #     Σ_i q_i (K_ij / d_i) = Σ_i K_ij.
        #
        # By symmetry of K (K_ij = K_ji) and definition of degrees:
        #     Σ_i K_ij = Σ_i K_ji = d_j = q_j.
        #
        # Hence q satisfies q = qP (stationary up to scale). Normalizing to sum to one gives:
        #     π_i = q_i / Σ_k q_k = d_i / Σ_k d_k.
        # In code, row_sums stores d_i, therefore:
        pi = row_sums / np.sum(row_sums)

        # ---------------------------------------------------------------------
        # Normalize eigenvectors of the random-walk matrix P = D^{-1}K in the π-weighted inner product.
        # ---------------------------------------------------------------------
        # Aldous & Fill use the π-weighted inner product for reversible chains:
        #     ⟨f,g⟩_π := Σ_i π_i f_i g_i.
        # (Aldous & Fill, "Reversible Markov Chains and Random Walks on Graphs", Lemma 4.39.)
        #
        # Reversibility (detailed balance) for P means:
        #     π_i P_ij = π_j P_ji  for all i,j.
        # (Aldous & Fill, "Reversible Markov Chains and Random Walks on Graphs", chapter 3.1.)
        # Using detailed balance and the definition of ⟨·,·⟩_π, for any vectors f,g:
        #
        #     ⟨f, Pg⟩_π
        #   = Σ_i π_i f_i (Pg)_i
        #   = Σ_i π_i f_i Σ_j P_ij g_j
        #   = Σ_{i,j} π_i P_ij f_i g_j
        #   = Σ_{i,j} π_j P_ji f_i g_j              (detailed balance)
        #   = Σ_j π_j g_j Σ_i P_ji f_i
        #   = Σ_j π_j g_j (Pf)_j
        #   = ⟨Pf, g⟩_π.
        #
        # Therefore, by definition of self-adjointness with respect to an inner product,
        # P is self-adjoint on the inner-product space (R^n, ⟨·,·⟩_π):
        #     ⟨f, Pg⟩_π = ⟨Pf, g⟩_π  for all f,g.
        #
        # By the spectral theorem for self-adjoint operators on finite-dimensional inner-product spaces,
        # eigenvectors can be chosen ⟨·,·⟩_π-orthonormal; in particular we normalize each eigenvector v_k so that
        #     ⟨v_k, v_k⟩_π = 1  ⇔  Σ_i π_i (v_{ik})^2 = 1.
        #
        # Implementation: for V = [v_1 ... v_m] stored column-wise in eigenvecs (shape n×m),
        # compute all π-norms:
        #     ||v_k||_π = sqrt( Σ_i π_i (v_{ik})^2 ),
        # and normalize each column.
        norms = np.sqrt(np.sum(pi[:, np.newaxis] * (eigenvecs ** 2), axis=0))

        # Prevent divide-by-zero in eigenvector normalization (numerical safeguard).
        norms = np.maximum(norms, 1e-12)

        # Enforce π-unit norm for each eigenvector: v_k ← v_k / ||v_k||_π.
        eigenvecs = eigenvecs / norms

        # Sort eigenvalues in descending order
        order = np.argsort(eigenvals)[::-1]
        eigvals_sorted = eigenvals[order]
        eigvecs_sorted = eigenvecs[:, order]

        return eigvals_sorted, eigvecs_sorted

    def _symmetrize_markov_operator(
        self,
        M_small: np.ndarray,
        inv_row_sums: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Symmetrize a Markov operator via a similarity transform.
        
        This maps the asymmetric random-walk operator M = D^{-1} K to the
        symmetric operator S = D^{1/2} M D^{-1/2} = D^{-1/2} K D^{-1/2}.
        Since K is symmetric, S is symmetric and has the same eigenvalues as M.
        Source: von Luxburg (2007), A Tutorial on Spectral Clustering, Section 3.2, Proposition 3.

        Parameters
        ----------
        M_small : numpy.ndarray
            Row-stochastic Markov matrix
        inv_row_sums : numpy.ndarray
            Inverse row sums (1 / d_i) from kernel matrix

        Returns
        -------
        tuple
            (S, inv_d_sqrt) where S is symmetric and inv_d_sqrt = D^(-1/2)
        """
        # We start with the asymmetric Markov operator M = D^{-1} K.
        # Because K is symmetric, the similarity transform
        # S = D^{1/2} M D^{-1/2} = D^{-1/2} K D^{-1/2}
        # yields a symmetric matrix with the same eigenvalues as M.
        #
        # Explicitly, with D_ii = d_i = Σ_j K_ij:
        #     M_ij = K_ij / d_i,
        #     S_ij = (1 / sqrt(d_i)) * K_ij * (1 / sqrt(d_j)).
        #     Source: von Luxburg (2007), A Tutorial on Spectral Clustering, Section 3.2, Proposition 3.

        # Recover D from inv_row_sums (D_ii = sum_j K_ij).
        row_sums = 1.0 / inv_row_sums
        # Clamp to avoid sqrt/division issues.
        row_sums = np.maximum(row_sums, 1e-12)
        # Recompute stable inverse after clamping.
        inv_row_sums = 1.0 / row_sums
        # D^{1/2} for similarity transform.
        d_sqrt = np.sqrt(row_sums)
        # D^{-1/2} for similarity transform.
        inv_d_sqrt = np.sqrt(inv_row_sums)
        # S = D^{1/2} M D^{-1/2}.
        S = (d_sqrt[:, np.newaxis] * M_small) * inv_d_sqrt[np.newaxis, :]
        # Enforce symmetry numerically (rounding-safe).
        S = 0.5 * (S + S.T)
        # Return S and D^{-1/2} for eigenvector back-transform.
        return S, inv_d_sqrt

    def _nystrom_extend_eigenvectors(self, K_all_to_landmarks: np.ndarray, 
                                    eigvecs_small: np.ndarray, eigvals_small: np.ndarray,
                                    n_frames: int) -> np.ndarray:
        """
        Extend eigenvectors from landmarks to all frames using Nyström method.
        
        Fowlkes et al. (2004): "ψ̂_i(x) = (1/nλ_i) Σ_j W(x,ξ_j) ψ̂_i(ξ_j)"
        For Markov matrices: ψ(x) = (1/λ) Σ_j P(x,ξ_j) v(ξ_j)
        The n factor disappears due to row normalization.
        
        Small eigenvalues (< 1e-10) correspond to numerically unreliable modes
        and are set to zero for physical consistency.

        Parameters
        ----------
        K_all_to_landmarks : numpy.ndarray
            Kernel matrix from all frames to landmarks
        eigvecs_small : numpy.ndarray
            Eigenvectors from landmark problem
        eigvals_small : numpy.ndarray
            Eigenvalues from landmark problem
        n_frames : int
            Number of frames

        Returns
        -------
        numpy.ndarray
            Extended eigenvectors for all frames
        """
        n_components = len(eigvals_small)
        
        # Create array for extended eigenvectors (initialized with zeros)
        eigenvectors_full = self._create_array_or_memmap(
            shape=(n_frames, n_components),
            dtype=np.float32,
            filename="nystrom_eigenvectors_full.dat"
        )
        if self.use_memmap:
            ResourceUtils.tune_memmap(eigenvectors_full, "sequential")
            ResourceUtils.tune_memmap(K_all_to_landmarks, "sequential")

        # Identify valid eigenvalues (avoid division by near-zero values)
        # For diffusion maps, we need at least n_components+1 eigenvalues (including stationary)
        # Use 1e-10 threshold like standard method to maintain consistency
        mask = np.abs(eigvals_small) > 1e-10
        valid_eigvals = np.where(mask, eigvals_small, 1e-10)
        
        # Extend eigenvectors chunk-wise for memory efficiency
        for chunk_start in ProgressUtils.iterate(
            range(0, n_frames, self.chunk_size),
            desc="Nystroem extension",
            unit="chunks",
        ):
            chunk_end = min(chunk_start + self.chunk_size, n_frames)
            K_chunk = K_all_to_landmarks[chunk_start:chunk_end, :]
            
            # Normalize chunk to get Markov transition probabilities
            # This implements P(x,ξ) = K(x,ξ)/d(x) for each row
            d_chunk = K_chunk.sum(axis=1)
            d_chunk = np.maximum(d_chunk, 1e-12)
            P_chunk = K_chunk / d_chunk[:, np.newaxis]
            
            # Apply vectorized Nyström extension formula
            # Only compute for valid eigenvalues, others remain 0
            eigenvectors_full[chunk_start:chunk_end, mask] = (
                (P_chunk @ eigvecs_small[:, mask]) / valid_eigvals[mask]
            )
            if self.use_memmap:
                eigenvectors_full.flush()

        if self.use_memmap:
            ResourceUtils.tune_memmap(eigenvectors_full, "random")
            ResourceUtils.tune_memmap(K_all_to_landmarks, "random")
        
        return eigenvectors_full

    def _nystrom_extract_coordinates(self, eigenvectors_full: np.ndarray, 
                                    eigvals_small: np.ndarray, 
                                    n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract diffusion coordinates by skipping the first eigenvector.
        
        Coifman & Lafon (2006): Skip first eigenvector (stationary distribution).
        First eigenvalue λ₁ = 1 with constant eigenvector for connected graphs.

        Parameters
        ----------
        eigenvectors_full : numpy.ndarray
            Extended eigenvectors for all frames
        eigvals_small : numpy.ndarray
            Eigenvalues from landmark problem
        n_components : int
            Number of diffusion coordinates to extract

        Returns
        -------
        tuple
            (diffusion_coordinates, diffusion_eigenvalues)
        """
        # Skip first eigenvector and eigenvalue (stationary distribution)
        diff_coords = eigenvectors_full[:, 1:n_components+1]
        diff_eigenvals = eigvals_small[1:n_components+1]
        
        return diff_coords, diff_eigenvals
    