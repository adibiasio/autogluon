from abc import abstractmethod
from typing import List, Callable

try:
    from strategies.AbstractPatienceStrategy import AbstractPatienceStrategy
except:
    from stopping_simulator.strategies.AbstractPatienceStrategy import AbstractPatienceStrategy

class MinDeltaStrategy(AbstractPatienceStrategy):
    """
    Minimum delta can be simulated by simply using patience functionaliy with patience
    set to zero and minimum delta set to the desired delta.
    """
    def __init__(self, min_delta: float = 0.05):
        super().__init__("min_delta")

        if not isinstance(min_delta, (float, int)):
            raise ValueError("Valid minimum delta must be specified to use a Minimum Delta Stopping Strategy")

        self.min_delta = min_delta


    def _patience_fn(self):
        def _get_patience(iter):
            return 0
        return _get_patience
