class ReduceContactMetrics:
    """Available reduce metrics for feature analysis."""

    @property
    def FREQUENCY(self) -> str:
        """Frequency - number of occurrences"""
        return "frequency"

    @property
    def STABILITY(self) -> str:
        """Stability - measure of conformational stability"""
        return "stability"
