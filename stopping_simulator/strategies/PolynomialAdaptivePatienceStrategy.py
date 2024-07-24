try:
    from strategies.AbstractPatienceStrategy import AbstractPatienceStrategy
except:
    from stopping_simulator.strategies.AbstractPatienceStrategy import AbstractPatienceStrategy

class PolynomialAdaptivePatienceStrategy(AbstractPatienceStrategy):
    def __init__(self, a: float | int = 0.2, degree: float | int = 1, b: float | int = 10, **kwargs):
        super().__init__("polynomial_adaptive_patience", **kwargs)

        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError("Valid A, N, and B values must be specified to use an Adaptive Patience Stopping Strategy")

        self.a = a
        self.degree = degree
        self.b = b


    def _patience_fn(self):
        def _get_patience(iter):
            return self.a * (iter ** self.degree) + self.b
        return _get_patience

