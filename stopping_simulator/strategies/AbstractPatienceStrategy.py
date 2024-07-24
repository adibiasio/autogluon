from abc import abstractmethod
from typing import List, Callable

try:
    from strategies.AbstractStrategy import AbstractStrategy
except:
    from stopping_simulator.strategies.AbstractStrategy import AbstractStrategy


class AbstractPatienceStrategy(AbstractStrategy):
    def __init__(self, name: str, sliding_window: int | None = None, min_delta: float | int | None = None):
        super().__init__(name)
        self.sliding_window = None
        self.min_delta = None

        if isinstance(sliding_window, int):
            self.sliding_window = sliding_window
            self.name = "sliding_" + self.name
        
        if isinstance(min_delta, (int, float)):
            self.min_delta = min_delta
            self.name = self.name + "_with_min_delta"


    def _run(self, curve):
        kwargs = {}

        if self.sliding_window is not None:
            kwargs["sliding_window"] = self.sliding_window
        
        if self.min_delta is not None:
            kwargs["min_delta"] = self.min_delta

        return self._iterate(curve, patience=self._patience_fn(), **kwargs)


    # TODO: ensure that for patience of 1 million and 2 million there are the same ranks
    def _iterate(self, curve: List[float], patience: Callable[[int], int], sliding_window: int | None = None, min_delta: int = 0):
        """
        Parameters:
        --------------
        curve

        patience
            patience function

        sliding
            window size for averaging last n scores

        min_delta

        Return:
        --------------
        best iteration (zero indexed)

        total iterations
        """
        counter = 0
        best_iter = 0
        best_score = None
        sliding_sum = 0
        for iter, score in enumerate(curve):
            if sliding_window is not None and sliding_window > 0:
                n = min(iter + 1, sliding_window)
                sliding_sum += score
                sliding_sum -= curve[iter - n] if iter - n >= 0 else 0
                score = sliding_sum / n
            if best_score is None:
                best_score = score
            elif score >= best_score - min_delta:
                counter += 1
                if counter >= patience(iter + 1):
                    break
            else:
                best_iter = iter
                best_score = score
                counter = 0

        return best_iter, iter


    @abstractmethod
    def _patience_fn(self):
        pass
