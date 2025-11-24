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
from typing import Any, Dict, Optional, Tuple

import numpy as np
from mdxplain.utils.progress_utils import ProgressUtils
from scipy.linalg import eig
from scipy.sparse.linalg import LinearOperator, eigs
from sklearn.cluster import MiniBatchKMeans

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

    def compute(self, data, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Diffusion Maps decomposition of coordinate matrix.

        Performs Diffusion Maps analysis on the input coordinate matrix using either
        standard in-memory computation or iterative memory-mapped computation
        based on configuration settings.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, n_features) where n_features = n_atoms * 3
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
            - random_state : int, optional
                Random state for reproducible results

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
        hyperparameters = self._extract_hyperparameters(data, kwargs)

        if hyperparameters["use_nystrom"]:
            return self._compute_nystrom_diffusion_maps(data, hyperparameters)
        elif self.use_memmap:
            return self._compute_iterative_diffusion_maps(data, hyperparameters)
        else:
            return self._compute_standard_diffusion_maps(data, hyperparameters)

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

    def _extract_hyperparameters(self, data, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
        dict
            Validated hyperparameters

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

        random_state = kwargs.get("random_state", None)
        
        epsilon = kwargs.get("epsilon")
        if epsilon is None:
            epsilon = self._estimate_epsilon_knn(data, random_state)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        use_nystrom = kwargs.get("use_nystrom", False)
        n_landmarks = kwargs.get("n_landmarks", 1000)

        # Validate n_landmarks for Nyström
        if use_nystrom and n_landmarks >= n_frames:
            n_landmarks = min(n_landmarks, n_frames - 1)
            if n_landmarks < n_components:
                raise ValueError(
                    f"n_landmarks ({n_landmarks}) must be >= n_components ({n_components})"
                )

        return {
            "n_components": n_components,
            "epsilon": epsilon,
            "use_nystrom": use_nystrom,
            "n_landmarks": n_landmarks,
            "random_state": random_state,
        }

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
            Input coordinate matrix (n_frames, n_features)
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
        # Formula: K_ij = exp(-d_ij^2 / epsilon)  
        kernel = np.exp(-(rmsd_matrix ** 2) / hyperparameters["epsilon"])

        # Step 3: Random Walk normalization to create transition matrix
        # Formula: M = D^(-1) * K where D_ii = sum_j K_ij
        # Reference: Coifman & Lafon (2006)
        transition_matrix = self._normalize_to_transition_matrix(kernel)

        # Step 4: Eigendecomposition
        # Note: Real eigenvalues guaranteed by Perron-Frobenius theorem:
        # Row-stochastic matrix M has real eigenvalues with λ₀ = 1 (largest)
        eigenvals, eigenvecs = eig(transition_matrix)

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
            Input coordinate matrix (n_frames, n_features)
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

        # Step 2: Compute kernel matrix as memmap and collect row sums
        kernel_matrix, inv_row_sums = self._compute_kernel_matrix(rmsd_matrix, epsilon, "iterative_kernel_matrix.dat")

        # Step 3: Create LinearOperator for transition matrix M*v operations
        # This avoids materializing the full transition matrix
        transition_operator = self._create_transition_operator(kernel_matrix, inv_row_sums)

        # Step 4: Iterative eigendecomposition using sparse methods
        # Note: Real eigenvalues guaranteed by Perron-Frobenius theorem:
        # Row-stochastic matrix M has real eigenvalues with λ₀ = 1 (largest)        
        # Reference: Coifman & Lafon (2006)
        eigenvals, eigenvecs = eigs(
            transition_operator,
            k=hyperparameters["n_components"] + 1,  # +1 to skip trivial eigenvector
            which='LM'
        )

        # Step 5: Extract diffusion coordinates
        diff_eigenvals, diff_coords = self._extract_diffusion_coordinates(
            eigenvals, eigenvecs, hyperparameters["n_components"]
        )

        # Step 6: Cleanup temporary files (only if memmap was used)
        if self.use_memmap:
            memmap_paths = []
            if hasattr(rmsd_matrix, 'filename') and rmsd_matrix.filename:
                memmap_paths.append(rmsd_matrix.filename)
            if hasattr(kernel_matrix, 'filename') and kernel_matrix.filename:
                memmap_paths.append(kernel_matrix.filename)
            if memmap_paths:
                self._cleanup_memmaps(memmap_paths)

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

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, n_features)
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
        epsilon = hyperparameters["epsilon"]
        random_state = hyperparameters["random_state"]

        # STEP 1: Select Landmarks
        # Fowlkes et al. (2004): "randomly chosen samples"
        # improvement: Use KMeans for better coverage instead of random
        landmark_idx = self._select_landmarks_kmeans(data, n_landmarks, random_state)

        # STEP 2: Compute Landmark Kernel Matrix A
        # Fowlkes et al. (2004): "partition the affinity matrix W = [A B; B^T C]"
        # Coifman & Lafon (2006): "κ(x,y) = exp(-||x-y||²/ε)"
        K_landmarks = self._compute_landmarks_kernel(data, landmark_idx, epsilon, n_atoms)

        # STEP 3: Normalize to Markov Matrix (Asymmetric)
        # Coifman & Lafon (2006): "M_ij = K_ij/d_i where d_i = Σ_j K_ij"
        # This creates a row-stochastic matrix (rows sum to 1)
        M_small, inv_row_sums = self._nystrom_normalize_to_markov(K_landmarks)

        # STEP 4: Solve Eigenvalue Problem for Small Matrix
        # Standard eigendecomposition: P·v = λ·v
        # Perron-Frobenius theorem: eigenvalues are real for stochastic matrices
        eigvals_small, eigvecs_small = self._nystrom_solve_eigenvalue_problem(M_small)

        # STEP 5: Compute Kernel from All Points to Landmarks
        # This is B^T from Fowlkes et al. (2004), includes all n points
        # Shape: (n_frames × n_landmarks)
        K_all_to_landmarks = self._compute_all_to_landmarks_kernel(
            data, landmark_idx, epsilon, n_atoms, n_frames, n_landmarks
        )

        # STEP 6: Nyström Extension of Eigenvectors
        # Fowlkes et al. (2004): "ψ̂_i(x) = (1/nλ_i) Σ_j W(x,ξ_j) ψ̂_i(ξ_j)"
        # For Markov matrices: ψ(x) = (1/λ) Σ_j P(x,ξ_j) v(ξ_j)
        eigenvectors_full = self._nystrom_extend_eigenvectors(
            K_all_to_landmarks, eigvecs_small, eigvals_small, n_frames
        )

        # STEP 7: Extract Diffusion Coordinates
        # Coifman & Lafon (2006): Skip first eigenvector (stationary distribution)
        # First eigenvalue λ₁ = 1 with constant eigenvector for connected graphs
        diff_coords, diff_eigenvals = self._nystrom_extract_coordinates(
            eigenvectors_full, eigvals_small, hyperparameters["n_components"]
        )

        # Cleanup memmap files if used
        if hasattr(K_all_to_landmarks, 'filename') and K_all_to_landmarks.filename:
            memmap_paths = [K_all_to_landmarks.filename]
            del K_all_to_landmarks
            self._cleanup_memmaps(memmap_paths)

        metadata = self._prepare_metadata(hyperparameters, (n_frames, n_features))
        metadata.update({
            "method": "nystrom_diffusion_maps",
            "epsilon": hyperparameters["epsilon"],
            "eigenvalues": diff_eigenvals,
            "n_landmarks": n_landmarks,
            "approximation": "asymmetric_nystrom",
            "landmark_selection": "minibatch_kmeans",
        })

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
            Input coordinate matrix (n_frames, n_features)
        n_atoms : int
            Number of atoms (n_features // 3)
        filename : str, default="rmsd_matrix.dat"
            Filename for memmap (automatically combined with cache_path and prefix)

        Returns
        -------
        numpy.ndarray
            RMSD distance matrix (n_frames, n_frames) - memmap or regular array
        """
        n_frames = data.shape[0]
        
        # Use helper method from CalculatorBase
        rmsd_matrix = self._create_array_or_memmap(
            shape=(n_frames, n_frames),
            dtype=np.float32,
            filename=filename
        )
        
        # Compute RMSD matrix (same logic for both memmap and regular array)
        for i in ProgressUtils.iterate(
            range(n_frames), desc="Computing RMSD matrix", unit="frames"
        ):
            for j in range(n_frames):
                rmsd_matrix[i, j] = self._compute_rmsd_flattened(data[i], data[j], n_atoms)
        
        return rmsd_matrix

    def _compute_kernel_matrix(self, rmsd_matrix: np.ndarray, epsilon: float, 
                              filename: str = "kernel_matrix.dat") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gaussian kernel matrix with automatic memmap/array selection.
        
        Uses memmap if self.use_memmap is True, otherwise regular numpy array.
        Automatically handles cache path combination with cache_prefix.

        Parameters
        ----------
        rmsd_matrix : numpy.ndarray
            RMSD distance matrix (can be memmap or regular array)
        epsilon : float
            Kernel bandwidth parameter
        filename : str, default="kernel_matrix.dat"
            Filename for memmap (automatically combined with cache_path and prefix)

        Returns
        -------
        tuple
            (kernel_matrix, inv_row_sums) where kernel_matrix follows memmap setting
            and inv_row_sums is array of inverse row sums for normalization
        """
        n_frames = rmsd_matrix.shape[0]
        
        # Use helper method from CalculatorBase
        kernel_matrix = self._create_array_or_memmap(
            shape=(n_frames, n_frames),
            dtype=np.float32,
            filename=filename
        )
        row_sums = np.zeros(n_frames)

        for i in ProgressUtils.iterate(
            range(0, n_frames, self.chunk_size),
            desc="Computing kernel matrix",
            unit="chunks",
        ):
            end_i = min(i + self.chunk_size, n_frames)
            chunk_kernel = np.exp(-(rmsd_matrix[i:end_i] ** 2) / epsilon)
            kernel_matrix[i:end_i] = chunk_kernel
            row_sums[i:end_i] = chunk_kernel.sum(axis=1)

        row_sums[row_sums < 1e-12] = 1e-12  # Numerical stability
        inv_row_sums = 1.0 / row_sums

        return kernel_matrix, inv_row_sums

    def _create_transition_operator(self, kernel_matrix: np.ndarray, 
                                  inv_row_sums: np.ndarray) -> LinearOperator:
        """
        Create LinearOperator for transition matrix M*v operations.

        Implements M*v = D^(-1) * K * v without materializing M.

        Parameters
        ----------
        kernel_matrix : numpy.ndarray
            Gaussian kernel matrix (can be memmap)
        inv_row_sums : numpy.ndarray
            Inverse row sums of kernel matrix for normalization

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            LinearOperator that computes transition matrix operations
        """
        n_frames = kernel_matrix.shape[0]
        
        def matvec_mult(v):
            result = np.zeros_like(v)
            for i in range(0, n_frames, self.chunk_size):
                end_i = min(i + self.chunk_size, n_frames)
                kernel_chunk = kernel_matrix[i:end_i, :]
                result[i:end_i] = (kernel_chunk * inv_row_sums[i:end_i, np.newaxis]) @ v
            return result

        return LinearOperator((n_frames, n_frames), matvec=matvec_mult)

    def _normalize_to_transition_matrix(self, kernel: np.ndarray) -> np.ndarray:
        """
        Normalize kernel matrix to transition matrix using Random Walk normalization.

        Implements M = D^(-1) * K where D_ii = sum_j K_ij.

        Parameters
        ----------
        kernel : numpy.ndarray
            Gaussian kernel matrix

        Returns
        -------
        numpy.ndarray
            Transition matrix M
        """
        row_sums = kernel.sum(axis=1)
        row_sums = np.maximum(row_sums, 1e-12)  # Numerical stability
        return kernel / row_sums[:, np.newaxis]

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
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                print(f"Warning: Could not remove temporary file {path}")

    def _estimate_epsilon_knn(self, data: np.ndarray, random_state: Optional[int]) -> float:
        """
        Estimate epsilon using k-nearest neighbors heuristic.

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, n_features)
        random_state : int, optional
            Random state for reproducible sampling

        Returns
        -------
        float
            Estimated epsilon value
        """
        n_frames, n_features = data.shape
        n_atoms = n_features // 3
        k = max(20, int(0.01 * n_frames))
        n_samples = k

        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(n_frames, n_samples, replace=False)

        # Use helper method to create k_distances array
        k_distances = self._create_array_or_memmap(
            shape=(n_samples,),
            dtype=np.float32,
            filename="epsilon_estimator.dat"
        )
        for i, sample_idx in enumerate(sample_indices):
            distances = np.zeros(n_frames)
            for j in range(n_frames):
                distances[j] = self._compute_rmsd_flattened(data[sample_idx], data[j], n_atoms)
            k_distances[i] = np.partition(distances, k)[k]
        
        if self.use_memmap or n_samples > self.chunk_size:
            chunk_medians = []
            for i in ProgressUtils.iterate(
                range(0, n_samples, self.chunk_size),
                desc="Computing epsilon",
                unit="chunks",
            ):
                end = min(i + self.chunk_size, n_samples)
                chunk_vals = np.array(k_distances[i:end])
                chunk_medians.append(np.median(chunk_vals))
            epsilon = np.median(chunk_medians) ** 2
        
            # Cleanup memmap file if used
            if hasattr(k_distances, 'filename') and k_distances.filename:
                memmap_path = k_distances.filename
                del k_distances
                os.remove(memmap_path)
            else:
                del k_distances
        else:
            epsilon = np.median(k_distances) ** 2

        if epsilon < 1e-12: # If epsilon is zero or very close to it
            return 1e-5 # Return a small default value

        return epsilon

    def _compute_rmsd_flattened(self, coords1: np.ndarray, coords2: np.ndarray, n_atoms: int) -> float:
        """
        Compute RMSD between two flattened coordinate vectors.

        Parameters
        ----------
        coords1 : numpy.ndarray
            First flattened coordinate vector (n_atoms * 3,)
        coords2 : numpy.ndarray  
            Second flattened coordinate vector (n_atoms * 3,)
        n_atoms : int
            Number of atoms

        Returns
        -------
        float
            RMSD value between the two structures
        """
        # Reshape to (n_atoms, 3)
        xyz1 = coords1.reshape(n_atoms, 3)
        xyz2 = coords2.reshape(n_atoms, 3)
        
        # Center both structures
        xyz1_centered = xyz1 - xyz1.mean(axis=0)
        xyz2_centered = xyz2 - xyz2.mean(axis=0)
        
        # Compute RMSD (without rotation for simplicity)
        return np.sqrt(np.mean((xyz1_centered - xyz2_centered) ** 2))

    def _compute_rmsd_chunk_to_single(self, chunk_coords: np.ndarray, single_coord: np.ndarray, n_atoms: int) -> np.ndarray:
        """
        Compute RMSD from chunk of frames to single frame (vectorized).

        Parameters
        ----------
        chunk_coords : numpy.ndarray
            Chunk coordinate matrix (n_chunk_frames, n_features)
        single_coord : numpy.ndarray
            Single flattened coordinate vector (n_features,)
        n_atoms : int
            Number of atoms

        Returns
        -------
        numpy.ndarray
            Array of RMSD values (n_chunk_frames,)
        """
        n_chunk_frames = chunk_coords.shape[0]
        
        # Reshape to (n_chunk_frames, n_atoms, 3) and (n_atoms, 3)
        chunk_xyz = chunk_coords.reshape(n_chunk_frames, n_atoms, 3)
        single_xyz = single_coord.reshape(n_atoms, 3)
        
        # Center single structure
        single_centered = single_xyz - single_xyz.mean(axis=0)
        
        # Center chunk structures (vectorized)
        chunk_centered = chunk_xyz - chunk_xyz.mean(axis=1, keepdims=True)
        
        # Compute RMSD for all frames in chunk (vectorized)
        diff = chunk_centered - single_centered[np.newaxis, :, :]
        rmsd_values = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
        
        return rmsd_values

    def _select_landmarks_kmeans(self, data: np.ndarray, n_landmarks: int, random_state: Optional[int]) -> np.ndarray:
        """
        Select landmark frames using MiniBatchKMeans clustering (chunk-konform).

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, n_features)
        n_landmarks : int
            Number of landmarks to select
        random_state : int, optional
            Random state for reproducible results

        Returns
        -------
        numpy.ndarray
            Array of landmark frame indices
        """
        n_frames = data.shape[0]
        
        # Initialize MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks,
            batch_size=min(self.chunk_size, n_frames),
            random_state=random_state,
        )
        
        # Train MiniBatchKMeans chunk-wise
        for chunk_start in ProgressUtils.iterate(
            range(0, n_frames, self.chunk_size),
            desc="Training MiniBatch KMeans",
            unit="chunks",
        ):
            chunk_end = min(chunk_start + self.chunk_size, n_frames)
            chunk_coords = data[chunk_start:chunk_end]
            kmeans.partial_fit(chunk_coords)
        
        # Find frames closest to cluster centers - also chunk-wise!
        landmarks = []
        for center in kmeans.cluster_centers_:
            min_dist = float('inf')
            closest_frame = -1
            
            # Search chunk-wise for closest frame
            for chunk_start in ProgressUtils.iterate(
                range(0, n_frames, self.chunk_size),
                desc="Finding landmarks",
                unit="chunks",
                leave=False,
            ):
                chunk_end = min(chunk_start + self.chunk_size, n_frames)
                chunk_coords = data[chunk_start:chunk_end]
                
                # Euclidean distances to this center
                distances = np.sum((chunk_coords - center)**2, axis=1)
                chunk_min_idx = np.argmin(distances)
                chunk_min_dist = distances[chunk_min_idx]
                
                if chunk_min_dist < min_dist:
                    min_dist = chunk_min_dist
                    closest_frame = chunk_start + chunk_min_idx
            
            if closest_frame not in landmarks and closest_frame != -1:
                landmarks.append(closest_frame)
        
        # Fill remaining if needed
        rng = np.random.RandomState(random_state)
        while len(landmarks) < n_landmarks:
            frame = rng.randint(n_frames)
            if frame not in landmarks:
                landmarks.append(frame)
        
        return np.array(landmarks[:n_landmarks])

    def _compute_landmarks_kernel(self, data: np.ndarray, landmark_idx: np.ndarray, 
                                 epsilon: float, n_atoms: int) -> np.ndarray:
        """
        Compute kernel matrix between landmark frames (n_landmarks × n_landmarks).

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, n_features)
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
        """
        n_landmarks = len(landmark_idx)
        K_landmarks = np.zeros((n_landmarks, n_landmarks))
        
        for i in ProgressUtils.iterate(
            range(n_landmarks),
            desc="Computing landmark kernel matrix",
            unit="landmarks",
        ):
            for j in range(n_landmarks):
                rmsd = self._compute_rmsd_flattened(data[landmark_idx[i]], data[landmark_idx[j]], n_atoms)
                K_landmarks[i, j] = np.exp(-(rmsd ** 2) / epsilon)
        
        return K_landmarks

    def _compute_all_to_landmarks_kernel(self, data: np.ndarray, landmark_idx: np.ndarray, 
                                        epsilon: float, n_atoms: int, 
                                        n_frames: int, n_landmarks: int) -> np.ndarray:
        """
        Compute kernel matrix from all frames to landmarks (n_frames × n_landmarks).

        Parameters
        ----------
        data : numpy.ndarray
            Input coordinate matrix (n_frames, n_features)
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
        """
        # Use helper method to create K_all_to_landmarks matrix
        K_all_to_landmarks = self._create_array_or_memmap(
            shape=(n_frames, n_landmarks),
            dtype=np.float32,
            filename="nystrom_K_all.dat"
        )

        # Vectorized computation: outer loop over landmarks, inner chunk-wise
        for j in range(n_landmarks):
            landmark_coord = data[landmark_idx[j]]
            
            # Compute chunk-wise RMSD to this landmark (vectorized)
            for chunk_start in ProgressUtils.iterate(
                range(0, n_frames, self.chunk_size),
                desc=f"Landmark kernel {j+1}/{len(landmark_idx)}",
                unit="chunks",
                leave=False,
            ):
                chunk_end = min(chunk_start + self.chunk_size, n_frames)
                chunk_data = data[chunk_start:chunk_end]
                
                # Vectorized RMSD computation for entire chunk
                rmsd_values = self._compute_rmsd_chunk_to_single(chunk_data, landmark_coord, n_atoms)
                K_all_to_landmarks[chunk_start:chunk_end, j] = np.exp(-(rmsd_values ** 2) / epsilon)
        
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

    def _nystrom_solve_eigenvalue_problem(self, M_small: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for the small landmark matrix.
        
        Standard eigendecomposition: P·v = λ·v
        Perron-Frobenius theorem: eigenvalues are real for stochastic matrices.
        Use eig (not eigh) as M_small is not symmetric after row normalization.

        Parameters
        ----------
        M_small : numpy.ndarray
            Row-stochastic matrix from landmarks

        Returns
        -------
        tuple
            (eigenvalues, eigenvectors) sorted by eigenvalue magnitude (descending)
        """
        eigvals_small, eigvecs_small = np.linalg.eig(M_small)
        
        # For stochastic matrices, eigenvalues are real (Perron-Frobenius theorem)
        eigvals_small = eigvals_small.real
        eigvecs_small = eigvecs_small.real
        
        # Sort eigenvalues in descending order
        order = np.argsort(eigvals_small)[::-1]
        eigvals_sorted = eigvals_small[order]
        eigvecs_sorted = eigvecs_small[:, order]
        
        return eigvals_sorted, eigvecs_sorted

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
    