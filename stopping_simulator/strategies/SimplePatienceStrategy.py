try:
    from strategies.AbstractPatienceStrategy import AbstractPatienceStrategy
except:
    from stopping_simulator.strategies.AbstractPatienceStrategy import AbstractPatienceStrategy

class SimplePatienceStrategy(AbstractPatienceStrategy):
    def __init__(self, patience: int = 10, **kwargs):
        super().__init__("simple_patience", **kwargs)

        if not isinstance(patience, int):
            raise ValueError("A valid patience must be specified to use a Simple Patience Stopping Strategy")

        self.patience = patience


    def _patience_fn(self):
        def _get_patience(iter):
            return self.patience
        return _get_patience

