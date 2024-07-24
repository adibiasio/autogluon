from abc import abstractmethod
from typing import List, Callable

try:
    from strategies.AbstractStrategy import AbstractStrategy
except:
    from stopping_simulator.strategies.AbstractStrategy import AbstractStrategy

class FixedIterationStrategy(AbstractStrategy):
    def __init__(self, n_iter: int = 10):
        super().__init__("fixed_iterations")

        if not isinstance(n_iter, int):
            raise ValueError("Valid number of iterations must be specified to use a Fixed Iteration Stopping Strategy")

        self.n_iter = n_iter


    def _run(self, curve):
        return self._iterate(curve)


    def _iterate(self, curve):
        if len(curve) > self.n_iter:
            best_iter = len(curve) - 1
            total_iter = best_iter
            # raise ValueError("Length of Curve is longer than number of iterations set for Fixed Iteration Stopping Strategy")

        best_iter = self.n_iter
        total_iter = self.n_iter

        return best_iter, total_iter

