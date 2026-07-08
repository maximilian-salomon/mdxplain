# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Kiro AI (Claude Sonnet 4.0).
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
Abstract base class for decomposition calculators.

Defines the interface that all decomposition calculators must implement
for consistency across different dimensionality reduction methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ....utils.memmap_utils import MemmapUtils
from ....utils.memmap_reuse_helper import MemmapReuseHelper
from ....utils.path_utils import PathUtils
from ....utils.progress_utils import ProgressUtils
from ....utils.resource_utils import ResourceUtils


class CalculatorBase(ABC):
    """
    Abstract base class for decomposition calculators.

    Defines the interface that all decomposition calculators (PCA, KernelPCA,
    ContactKernelPCA) must implement for consistency across different
    dimensionality reduction methods.

    Examples
    --------
    >>> class MyCalculator(CalculatorBase):
    ...     def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000) -> None:
    ...         super().__init__(use_memmap, cache_path, chunk_size)
    ...     def compute(self, data, **kwargs):
    ...         # Implement computation logic
    ...         return transformed_data, metadata
    """

    def __init__(self, use_memmap: bool = False, cache_path: str = "./cache", chunk_size: int = 2000, reuse_memmap_cache: bool = False) -> None:
        """
        Initialize the decomposition calculator.

        Parameters
        ----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Path for memory-mapped cache files
        chunk_size : int, optional
            Size of chunks for incremental processing
        reuse_memmap_cache : bool, default=False
            Whether to reuse a matching cached result instead of recomputing
            it (only effective when use_memmap is True)

        Returns
        -------
        None
            Initializes calculator with specified configuration

        Examples
        --------
        >>> # Basic initialization
        >>> calc = MyCalculator()

        >>> # With memory mapping
        >>> calc = MyCalculator(
        ...     use_memmap=True,
        ...     cache_path="./cache/decomp.dat",
        ...     chunk_size=1000
        ... )
        """
        self.use_memmap = use_memmap
        self.cache_path = cache_path
        self.chunk_size = chunk_size
        self.reuse_memmap_cache = reuse_memmap_cache

    @abstractmethod
    def compute(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute decomposition of input data.

        This method performs the actual dimensionality reduction computation
        and returns the transformed data along with metadata about the
        transformation process.

        Parameters
        ----------
        data : numpy.ndarray
            Input data matrix to decompose, shape (n_samples, n_features)
        kwargs : dict
            Additional parameters specific to the decomposition method

        Returns
        -------
        Tuple[numpy.ndarray, Dict]
            Tuple containing:
            
            - transformed_data: Decomposed data matrix (n_samples, n_components)
            - metadata: Dictionary with transformation information including
              hyperparameters, explained variance, components, etc.

        Examples
        --------
        >>> # Compute decomposition
        >>> calc = MyCalculator()
        >>> data = np.random.rand(100, 50)
        >>> transformed, metadata = calc.compute(data, n_components=10)
        >>> print(f"Transformed shape: {transformed.shape}")
        >>> print(f"Explained variance: {metadata['explained_variance_ratio']}")
        """
        pass

    def _validate_input_data(self, data: np.ndarray) -> None:
        """
        Validate input data for decomposition.

        Parameters
        ----------
        data : numpy.ndarray
            Input data to validate

        Returns
        -------
        None
            Validates input data format and shape

        Raises
        ------
        ValueError
            If data format is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array")
        if data.shape[0] < 2:
            raise ValueError("Input data must have at least 2 samples")
        if data.shape[1] < 2:
            raise ValueError("Input data must have at least 2 feature")

    def _prepare_metadata(self, hyperparameters: Dict[str, Any], original_shape: Tuple) -> Dict[str, Any]:
        """
        Prepare base metadata dictionary.

        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters used for decomposition
        original_shape : tuple
            Shape of original input data

        Returns
        -------
        dict
            Base metadata dictionary with common information
        """
        return {
            "hyperparameters": hyperparameters,
            "original_shape": original_shape,
            "use_memmap": self.use_memmap,
            "chunk_size": self.chunk_size,
            "cache_path": self.cache_path,
        }

    def _create_array_or_memmap(self, shape: Tuple[int, ...], 
                               dtype: np.dtype = np.float32,
                               filename: Optional[str] = None) -> np.ndarray:
        """
        Create numpy array or memmap based on use_memmap setting.
        
        Automatically chooses between regular numpy array or memory-mapped array
        based on self.use_memmap. Combines cache_path with cache_prefix and filename.

        Parameters
        ----------
        shape : tuple
            Shape of the array to create
        dtype : numpy.dtype, default=np.float32
            Data type for the array
        filename : str, optional
            Filename for memmap. If None, uses "temp.dat"
            Will be combined with cache_path and cache_prefix

        Returns
        -------
        numpy.ndarray
            Either regular numpy array or memory-mapped array

        Examples
        --------
        >>> # Create distance matrix
        >>> matrix = self._create_array_or_memmap(
        ...     (n_frames, n_frames), 
        ...     filename="rmsd_matrix.dat"
        ... )
        
        >>> # Create temporary array
        >>> temp = self._create_array_or_memmap((1000, 50))
        """
        if self.use_memmap:
            if filename is None:
                filename = "temp.dat"
            memmap_path = self._memmap_result_path(filename)
            memmap_array = MemmapUtils.create_memmap(
                path=memmap_path,
                dtype=dtype,
                mode="w+",
                shape=shape,
            )
            return memmap_array
        else:
            return np.zeros(shape, dtype=dtype)

    def _memmap_result_path(self, filename: str) -> str:
        """
        Build the full cache path for a memmap filename.

        Combines the cache directory with the cache prefix (when present) and
        the filename, matching how persistent result memmaps are named.

        Parameters
        ----------
        filename : str
            Base filename for the memmap.

        Returns
        -------
        str
            Full path of the memmap file under the cache directory.
        """
        if hasattr(self, "_cache_prefix"):
            full_filename = f"{self._cache_prefix}_{filename}"
        else:
            full_filename = filename
        return PathUtils.get_cache_file_path(full_filename, self.cache_path)

    def _input_hash(self, data: np.ndarray) -> str:
        """
        Return a content hash of the decomposition input for reuse keying.

        Added to the cache parameters so a cached result is only reused when
        it was produced from the same input, not merely the same shape.

        Parameters
        ----------
        data : numpy.ndarray
            Input matrix passed to the decomposition.

        Returns
        -------
        str
            Hex digest identifying the input content.
        """
        return MemmapReuseHelper.hash_array(data, self.chunk_size)

    def _reuse_memmap_result(
        self, memmap_path: str, cache_params: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Return a cached (transformed_data, metadata) when reuse is valid.

        Parameters
        ----------
        memmap_path : str
            Path of the transformed-data memmap.
        cache_params : dict
            Parameters that define the result, matched against the sidecar.

        Returns
        -------
        Tuple[numpy.ndarray, dict] or None
            The reused result, or None when reuse is disabled or no matching
            cache is available.
        """
        if not (self.use_memmap and self.reuse_memmap_cache):
            return None
        result = MemmapReuseHelper.try_reuse_with_payload(
            memmap_path, cache_params
        )
        if result is None:
            print(f"No matching cache for {memmap_path}; recomputing.")
            return None
        data, metadata = result
        metadata["reused"] = True
        print(f"Reusing cached decomposition result: {memmap_path}")
        return data, metadata

    def _write_memmap_result_sidecar(
        self,
        memmap_path: str,
        result: np.ndarray,
        cache_params: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Write the reuse sidecar and metadata payload for a persistent result.

        Parameters
        ----------
        memmap_path : str
            Path of the fully written transformed-data memmap.
        result : numpy.ndarray
            The written transformed-data array.
        cache_params : dict
            Parameters that define the result, recorded in the sidecar.
        metadata : dict
            Result metadata restored on reuse.

        Returns
        -------
        None
            Writes the sidecar when memory mapping is in use.
        """
        if self.use_memmap:
            MemmapReuseHelper.write_sidecar(
                memmap_path,
                result.shape,
                str(result.dtype),
                cache_params,
                payload=metadata,
            )

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
        is_memmap_data = MemmapUtils.is_memmap_view(data)
        
        # Initialize MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks,
            batch_size=min(self.chunk_size, n_frames),
            random_state=random_state,
            n_init="auto",
        )
        
        # Train MiniBatchKMeans chunk-wise
        # Ensure the first batch is large enough to initialize all centers (n_landmarks)
        first_end = min(max(self.chunk_size, n_landmarks), n_frames)
        if first_end > self.chunk_size:
            print(f"Warning: Increasing first batch size to {first_end} for KMeans initialization. This is absolute necessary. "
                  "If this causes memory issues, consider reducing n_landmarks.")
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "sequential")
        kmeans.partial_fit(data[:first_end].astype(np.float32, copy=False))

        if n_frames > first_end:
            for start in ProgressUtils.iterate(range(first_end, n_frames, self.chunk_size), desc="Training MiniBatch KMeans", unit="chunks"):
                end = min(start + self.chunk_size, n_frames)
                kmeans.partial_fit(data[start:end].astype(np.float32, copy=False))
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "random")
        
        # Find frames closest to cluster centers - single pass
        centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
        best_dist = np.full(n_landmarks, np.inf, dtype=np.float64)
        best_idx = np.full(n_landmarks, -1, dtype=np.int64)
        
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "sequential")
        for start in ProgressUtils.iterate(range(0, n_frames, self.chunk_size), desc="Finding landmarks", unit="chunks"):
            end = min(start + self.chunk_size, n_frames)
            chunk = data[start:end].astype(np.float32, copy=False)
            labels = kmeans.predict(chunk)
            diff = chunk - centers[labels]
            d2 = np.sum(diff * diff, axis=1)

            for i, k in enumerate(labels):
                if d2[i] < best_dist[k]:
                    best_dist[k] = d2[i]
                    best_idx[k]  = start + i
        if is_memmap_data:
            ResourceUtils.tune_memmap(data, "random")

        landmarks, seen = [], set()
        for idx in best_idx:
            val = int(idx)
            if val != -1 and val not in seen:
                landmarks.append(val)
                seen.add(val)
        
        # Fill remaining if needed
        rng = np.random.RandomState(random_state)
        while len(landmarks) < n_landmarks:
            frame = int(rng.randint(n_frames))
            if frame not in seen:
                landmarks.append(frame)
                seen.add(frame)
        
        return np.array(landmarks[:n_landmarks])
