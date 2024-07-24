from typing import List, Tuple
from abc import ABC, abstractmethod


class AbstractStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.params = None
        self.param_configs = None


    # TODO: rename best iter to chosen iter
    def simulate(self, stopping_curve: List[float], eval_curve: List[float]):
        best_iter, total_iter = self._run(stopping_curve)
        opt_iter = eval_curve.index(min(eval_curve))
        score_diff = eval_curve[best_iter] - eval_curve[opt_iter]
        percent_score_diff = score_diff / max(eval_curve)
        percent_iter_diff = (total_iter - opt_iter) / (opt_iter if opt_iter != 0 else 1)
        return [total_iter + 1, best_iter + 1, opt_iter + 1, eval_curve[best_iter], eval_curve[opt_iter], score_diff, percent_score_diff, percent_iter_diff]


    @abstractmethod
    def _run(self, curve: List[float]) -> Tuple[int, int]:
        """
        Parameters:
        --------------
        curve: 


        Return:
        --------------
        best iteration (zero indexed)


        total iterations (zero indexed)
        """
        pass
