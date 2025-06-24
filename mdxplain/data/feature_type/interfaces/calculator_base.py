from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import numpy as np

class CalculatorBase(ABC):
    """
    Abstract base class for all calculator implementations.
    
    Defines the common interface that all calculator classes must implement.
    This ensures consistency across different feature calculators.
    """
    
    def __init__(self, 
                 use_memmap: bool = False, 
                 cache_path: Optional[str] = None, 
                 chunk_size: Optional[int] = None):
        """
        Initialize calculator with common parameters.
        
        Parameters:
        -----------
        use_memmap : bool, default=False
            Whether to use memory mapping for large datasets
        cache_path : str, optional
            Path for cache files when using memory mapping
        chunk_size : int, optional
            Chunk size for processing (None for automatic)
        squareform : bool, default=True
            Whether to use square form output
        k : int, default=0
            Diagonal offset parameter
        """
        self.use_memmap = use_memmap
        self.cache_path = cache_path
        self.chunk_size = chunk_size

        
        # Analysis object will be set by subclasses
        self.analysis = None
    
    @abstractmethod
    def compute(self, input_data: Any, **kwargs) -> tuple:
        """
        Compute feature data from input.
        
        Parameters:
        -----------
        input_data : Any
            Input data for computation (trajectories, distances, etc.)
        **kwargs : dict
            Additional parameters specific to the calculator
            
        Returns:
        --------
        tuple
            (computed_data, feature_names) tuple
        """
        pass
    
    @abstractmethod
    def compute_dynamic_values(self, 
                              input_data: np.ndarray, 
                              metric: str, 
                              threshold_min: Optional[float] = None, 
                              threshold_max: Optional[float] = None,
                              feature_names: Optional[list] = None,
                              output_path: Optional[str] = None,
                              transition_threshold: Optional[float] = None, 
                              window_size: Optional[int] = None,
                              transition_mode: Optional[str] = None, 
                              lag_time: Optional[int] = None
                              ) -> Dict[str, Any]:
        """
        Filter and select dynamic values based on specified criteria.
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input data array
        metric : str
            Metric to use for selection
        threshold_min : float, optional
            Minimum threshold for filtering
        threshold_max : float, optional
            Maximum threshold for filtering
        feature_names : list, optional
            Names for feature pairs
        output_path : str, optional
            Path for memory-mapped output
        transition_threshold : float, optional
            Distance threshold for detecting transitions
        window_size : int, optional
            Window size for computing transitions
        transition_mode : str, optional
            Mode for computing transitions ('window' or 'lagtime')
        lag_time : int, optional
            Lag time for computing transitions
        Returns:
        --------
        dict
            Dictionary containing filtered data and metadata
        """
        pass
