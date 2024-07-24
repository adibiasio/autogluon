try:
    from strategies.AbstractPatienceStrategy import AbstractPatienceStrategy
except:
    from stopping_simulator.strategies.AbstractPatienceStrategy import AbstractPatienceStrategy

class AdaptivePatienceStrategy(AbstractPatienceStrategy):
    def __init__(self, a: float | int = 0.2, b: float | int = 10, **kwargs):
        super().__init__("adaptive_patience", **kwargs)

        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError("Valid A and B values must be specified to use an Adaptive Patience Stopping Strategy")

        self.a = a
        self.b = b


    def _patience_fn(self):
        def _get_patience(iter):
            return self.a * iter + self.b
        return _get_patience

