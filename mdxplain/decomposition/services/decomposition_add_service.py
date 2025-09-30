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

"""Factory for adding decomposition algorithms with simplified syntax."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..managers.decomposition_manager import DecompositionManager
    from ...pipeline.entities.pipeline_data import PipelineData

from ..decomposition_type import PCA, KernelPCA, ContactKernelPCA, DiffusionMaps


class DecompositionAddService:
    """
    Service for adding decomposition algorithms without explicit type instantiation.
    
    This service provides an intuitive interface for adding decomposition algorithms
    without requiring users to import and instantiate decomposition types directly.
    All decomposition type parameters are combined with manager.add parameters.
    
    Examples
    --------
    >>> pipeline.decomposition.add.pca("my_features", n_components=10)
    >>> pipeline.decomposition.add.kernel_pca("contact_features", kernel='rbf', n_components=20)
    >>> pipeline.decomposition.add.diffusion_maps("distance_features", n_components=15)
    """
    
    def __init__(self, manager: DecompositionManager, pipeline_data: PipelineData) -> None:
        """
        Initialize factory with manager and pipeline data.
        
        Parameters
        ----------
        manager : DecompositionManager
            Decomposition manager instance
        pipeline_data : PipelineData
            Pipeline data container (injected by AutoInjectProxy)
            
        Returns
        -------
        None
        """
        self._manager = manager
        self._pipeline_data = pipeline_data
    
    def pca(
        self,
        selection_name: str,
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
        decomposition_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Add PCA (Principal Component Analysis) decomposition.
        
        PCA reduces dimensionality by finding the directions of maximum variance
        in the data and projecting the data onto these principal components.
        
        Parameters
        ----------
        selection_name : str
            Name of feature selection to decompose
        n_components : int, optional
            Number of principal components to compute. If None, keeps min(n_samples, n_features)
        random_state : int, optional
            Random state for reproducible results
        decomposition_name : str, optional
            Name for the decomposition result. If None, uses algorithm-based name
        data_selector_name : str, optional
            Name of data selector to apply before decomposition
        force : bool, default=False
            Force recalculation even if decomposition already exists
            
        Returns
        -------
        None
            Adds PCA decomposition results to pipeline data
            
        Examples
        --------
        >>> # Basic PCA decomposition
        >>> pipeline.decomposition.add.pca("my_features", n_components=10)
        
        >>> # PCA with reproducible results
        >>> pipeline.decomposition.add.pca("my_features", n_components=15, random_state=42)
        
        >>> # PCA with custom name and data selector
        >>> pipeline.decomposition.add.pca(
        ...     "distance_features",
        ...     n_components=20,
        ...     decomposition_name="distance_pca",
        ...     data_selector_name="equilibrated_frames"
        ... )
        
        >>> # Force recalculation of existing PCA
        >>> pipeline.decomposition.add.pca(
        ...     "contact_features",
        ...     n_components=15,
        ...     force=True
        ... )
        
        Notes
        -----
        PCA is a linear dimensionality reduction technique that preserves
        the maximum amount of variance in the reduced representation.
        Components are ordered by explained variance ratio.
        """
        decomposition_type = PCA(n_components=n_components, random_state=random_state)
        return self._manager.add_decomposition(
            self._pipeline_data,
            selection_name,
            decomposition_type,
            decomposition_name=decomposition_name,
            data_selector_name=data_selector_name,
            force=force,
        )
    
    def kernel_pca(
        self,
        selection_name: str,
        n_components: Optional[int] = None,
        gamma: Optional[float] = None,
        use_nystrom: bool = False,
        n_landmarks: int = 10000,
        random_state: Optional[int] = None,
        use_parallel: bool = False,
        n_jobs: int = -1,
        min_chunk_size: int = 1000,
        decomposition_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Add Kernel PCA decomposition.
        
        Kernel PCA is a nonlinear extension of PCA that uses kernel functions
        to project data into higher-dimensional spaces where linear PCA can
        capture nonlinear relationships in the original space.
        
        Parameters
        ----------
        selection_name : str
            Name of feature selection to decompose
        n_components : int, optional
            Number of components to compute. If None, keeps min(n_samples, n_features)
        gamma : float, optional
            RBF kernel coefficient. If None, uses 1.0 / n_features
        use_nystrom : bool, default=False
            Whether to use Nyström approximation for large datasets
        n_landmarks : int, default=10000
            Number of landmarks for Nyström approximation
        random_state : int, optional
            Random state for reproducible results
        use_parallel : bool, default=False
            Whether to use parallel processing for matrix-vector multiplication
        n_jobs : int, default=-1
            Number of parallel jobs (-1 for all available CPU cores)
        min_chunk_size : int, default=1000
            Minimum chunk size per parallel process to avoid overhead
        decomposition_name : str, optional
            Name for the decomposition result
        data_selector_name : str, optional
            Name of data selector to apply before decomposition
        force : bool, default=False
            Force recalculation even if decomposition already exists
            
        Returns
        -------
        None
            Adds Kernel PCA decomposition results to pipeline data
            
        Examples
        --------
        >>> # Basic RBF Kernel PCA
        >>> pipeline.decomposition.add.kernel_pca("my_features", n_components=15)
        
        >>> # Kernel PCA with Nyström approximation for large datasets
        >>> pipeline.decomposition.add.kernel_pca(
        ...     "distance_features",
        ...     n_components=20,
        ...     use_nystrom=True,
        ...     n_landmarks=5000,
        ...     decomposition_name="nystrom_kpca"
        ... )
        
        >>> # Parallel Kernel PCA with custom parameters
        >>> pipeline.decomposition.add.kernel_pca(
        ...     "contact_features",
        ...     n_components=12,
        ...     gamma=0.1,
        ...     use_parallel=True,
        ...     n_jobs=8,
        ...     data_selector_name="folded_conformations"
        ... )
        
        Notes
        -----
        Kernel PCA uses RBF kernel to capture complex nonlinear patterns in data.
        Nyström approximation is recommended for datasets with >10,000 samples.
        """
        decomposition_type = KernelPCA(
            n_components=n_components,
            gamma=gamma,
            use_nystrom=use_nystrom,
            n_landmarks=n_landmarks,
            random_state=random_state,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
            min_chunk_size=min_chunk_size,
        )
        return self._manager.add_decomposition(
            self._pipeline_data,
            selection_name,
            decomposition_type,
            decomposition_name=decomposition_name,
            data_selector_name=data_selector_name,
            force=force,
        )
    
    def contact_kernel_pca(
        self,
        selection_name: str,
        n_components: Optional[int] = None,
        gamma: float = 1.0,
        use_nystrom: bool = False,
        n_landmarks: int = 2000,
        random_state: Optional[int] = None,
        use_parallel: bool = False,
        n_jobs: int = -1,
        min_chunk_size: int = 1000,
        decomposition_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Add Contact Kernel PCA decomposition.
        
        Specialized Kernel PCA implementation optimized for contact matrix data
        with contact-specific kernel functions and regularization.
        
        Parameters
        ----------
        selection_name : str
            Name of contact feature selection to decompose
        n_components : int, optional
            Number of components to compute. If None, keeps min(n_samples, n_features)
        gamma : float, default=1.0
            Kernel coefficient for Hamming/RBF kernel on binary contact data
        use_nystrom : bool, default=False
            Whether to use Nyström approximation for large datasets
        n_landmarks : int, default=2000
            Number of landmarks for Nyström approximation
        random_state : int, optional
            Random state for reproducible results
        use_parallel : bool, default=False
            Whether to use parallel processing for matrix-vector multiplication
        n_jobs : int, default=-1
            Number of parallel jobs (-1 for all available CPU cores)
        min_chunk_size : int, default=1000
            Minimum chunk size per parallel process to avoid overhead
        decomposition_name : str, optional
            Name for the decomposition result
        data_selector_name : str, optional
            Name of data selector to apply before decomposition
        force : bool, default=False
            Force recalculation even if decomposition already exists
            
        Returns
        -------
        None
            Adds Contact Kernel PCA decomposition results to pipeline data
            
        Examples
        --------
        >>> # Basic Contact Kernel PCA
        >>> pipeline.decomposition.add.contact_kernel_pca("contact_features", n_components=12)
        
        >>> # Contact Kernel PCA with custom gamma
        >>> pipeline.decomposition.add.contact_kernel_pca(
        ...     "persistent_contacts",
        ...     n_components=15,
        ...     gamma=0.5,
        ...     decomposition_name="contact_modes"
        ... )
        
        >>> # Contact Kernel PCA with Nyström approximation
        >>> pipeline.decomposition.add.contact_kernel_pca(
        ...     "native_contacts",
        ...     n_components=8,
        ...     gamma=2.0,
        ...     use_nystrom=True,
        ...     n_landmarks=1000,
        ...     data_selector_name="folded_states"
        ... )
        
        Notes
        -----
        Contact Kernel PCA uses specialized kernels that account for the
        binary nature of contact data and contact pattern correlations.
        """
        decomposition_type = ContactKernelPCA(
            n_components=n_components,
            gamma=gamma,
            use_nystrom=use_nystrom,
            n_landmarks=n_landmarks,
            random_state=random_state,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
            min_chunk_size=min_chunk_size,
        )
        return self._manager.add_decomposition(
            self._pipeline_data,
            selection_name,
            decomposition_type,
            decomposition_name=decomposition_name,
            data_selector_name=data_selector_name,
            force=force,
        )
    
    def diffusion_maps(
        self,
        selection_name: str,
        n_components: int,
        epsilon: Optional[float] = None,
        use_nystrom: bool = False,
        n_landmarks: int = 1000,
        random_state: Optional[int] = None,
        decomposition_name: Optional[str] = None,
        data_selector_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Add Diffusion Maps decomposition.
        
        Diffusion Maps is a nonlinear dimensionality reduction technique that
        captures the intrinsic geometry of data manifolds. It's particularly
        effective for analyzing conformational transitions in MD trajectories.
        
        Parameters
        ----------
        selection_name : str
            Name of feature selection to decompose
        n_components : int, required
            Number of diffusion components to compute
        epsilon : float, optional
            Kernel bandwidth parameter for Gaussian kernel. If None, estimated automatically
        use_nystrom : bool, default=False
            Whether to use Nyström approximation for very large datasets
        n_landmarks : int, default=1000
            Number of landmarks for Nyström approximation
        random_state : int, optional
            Random state for reproducible results
        decomposition_name : str, optional
            Name for the decomposition result
        data_selector_name : str, optional
            Name of data selector to apply before decomposition
        force : bool, default=False
            Force recalculation even if decomposition already exists
            
        Returns
        -------
        None
            Adds Diffusion Maps decomposition results to pipeline data
            
        Examples
        --------
        >>> # Basic Diffusion Maps
        >>> pipeline.decomposition.add.diffusion_maps("my_features", n_components=12)
        
        >>> # Diffusion Maps with custom epsilon
        >>> pipeline.decomposition.add.diffusion_maps(
        ...     "distance_features",
        ...     n_components=20,
        ...     epsilon=1.5,
        ...     decomposition_name="transition_coordinates"
        ... )
        
        >>> # Diffusion Maps with Nyström approximation
        >>> pipeline.decomposition.add.diffusion_maps(
        ...     "conformational_features",
        ...     n_components=15,
        ...     use_nystrom=True,
        ...     n_landmarks=2000,
        ...     data_selector_name="equilibrated_frames"
        ... )
        
        Notes
        -----
        Diffusion Maps preserves diffusion distances and is excellent for
        identifying slow conformational coordinates and transition pathways.
        Uses RMSD-based distances to construct the diffusion process.
        """
        decomposition_type = DiffusionMaps(
            n_components=n_components,
            epsilon=epsilon,
            use_nystrom=use_nystrom,
            n_landmarks=n_landmarks,
            random_state=random_state,
        )
        return self._manager.add_decomposition(
            self._pipeline_data,
            selection_name,
            decomposition_type,
            decomposition_name=decomposition_name,
            data_selector_name=data_selector_name,
            force=force,
        )