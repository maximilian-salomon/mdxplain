class ReduceDistanceMetrics:
    """Available reduce metrics for feature analysis."""
    
    @property
    def CV(self) -> str:
        """Coefficient of variation - measures relative variability"""
        return 'cv'
    
    @property
    def STD(self) -> str:
        """Standard deviation - measures absolute variability"""
        return 'std'
    
    @property
    def VARIANCE(self) -> str:
        """Variance - squared standard deviation"""
        return 'variance'
    
    @property
    def RANGE(self) -> str:
        """Range - difference between max and min values"""
        return 'range'
    
    @property
    def TRANSITIONS(self) -> str:
        """Transition analysis - detects conformational changes"""
        return 'transitions'

    