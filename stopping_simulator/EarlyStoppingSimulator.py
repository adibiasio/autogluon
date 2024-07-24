import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json, itertools, glob, os

from stopping_simulator.strategies.AbstractStrategy import AbstractStrategy
from stopping_simulator.strategies.SimplePatienceStrategy import SimplePatienceStrategy
from stopping_simulator.strategies.AdaptivePatienceStrategy import AdaptivePatienceStrategy
from stopping_simulator.strategies.PolynomialAdaptivePatienceStrategy import PolynomialAdaptivePatienceStrategy
from stopping_simulator.strategies.MinDeltaStrategy import MinDeltaStrategy


class StoppingSimulator:
    def __init__(self):
        self.tasks = []
        self.simulations = None
        self.ranks = None


        # ==> really simple and linear adaptive are just subsets of polynomial adaptive patience
        #       should just have polynomial patience and make simple / linear use polynomial in background
        #       essentially all patience will run through polynomial patience
        #       actually, it is binomial adaptive patience (bc only two terms)
        #       should we increase to polynomial?
        #       is there any merit?
        # base: simple, adaptive, polynomial, fixed
        # variations: min_delta, sliding
        self._strategy_class_map = {
            "simple_patience": SimplePatienceStrategy,
            "sliding_simple_patience": SimplePatienceStrategy,
            "sliding_simple_patience_with_min_delta": SimplePatienceStrategy,
            "adaptive_patience": AdaptivePatienceStrategy,
            "sliding_adaptive_patience": AdaptivePatienceStrategy,
            "sliding_adaptive_patience_with_min_delta": AdaptivePatienceStrategy,
            "polynomial_adaptive_patience": PolynomialAdaptivePatienceStrategy,
            "min_delta": MinDeltaStrategy,
        }


        p = { "patience": (0, 50, 5) }
        a = { "a": (0, 0.5, 0.05) }
        b = { "b": (0, 50, 5) }
        w = { "sliding_window": (0, 25, 5) }
        d = { "min_delta": (0, 0.1, 0.01) }
        n = { "degree": (0.1, 1.5, 0.1) }

        # p = { "patience": (0, 150, 5) } # 150
        # a = { "a": (0, 0.5, 0.05) } # 50
        # b = { "b": (0, 50, 5) } # 50
        # w = { "sliding_window": (0, 25, 5) } # 5
        # d = { "min_delta": (0, 0.1, 0.01) } # 20


        # simpler method is a | b | n, but not supported in python 3.8
        self.default_strategy_configs = {
            "simple_patience": { **p },
            # "sliding_simple_patience": { **p, **w },
            # "sliding_simple_patience_with_min_delta": { **p, **w, **d},
            "adaptive_patience": { **a, **b },
            # "sliding_adaptive_patience": { **a, **b, **w },
            # "sliding_adaptive_patience_with_min_delta": { **a, **b, **w, **d},
            # "min_delta": { **d },
        }


    def load_curves(self, path):
        if type(path) is list:
            self.tasks.append(path)
            return

        paths = []
        if os.path.isdir(path):
            paths = glob.glob(os.path.join(path, '**', '*.json'), recursive=True)
        elif os.path.isfile(path) and path.lower().endswith('.json'):
            paths.append(path)

        for file in paths:
            with open(file, "r") as f:
                self.tasks.append(json.load(f))


    def run(self, strategies: dict | None = None):
        if strategies is None:
            strategies = self.default_strategy_configs

        results = []
        for strategy, params in strategies.items():
            results.extend(self.param_search(strategy, **params))

        strategy_columns = ["strategy", "params"]
        task_columns = ["dataset", "tid", "fold", "framework", "problem_type"]
        curve_columns = ["model", "metric", "eval_set"]
        simulate_columns = ["total_iter", 'best_iter', 'opt_iter', "best_error", "opt_error", "error_diff", "percent_error_diff", "percent_iter_diff"]
        all_columns = strategy_columns + task_columns + curve_columns + simulate_columns
        self.simulations = pd.DataFrame(results, columns=all_columns)

        return self.simulations


    def param_search(self, strategy: str, **strategy_params):
        # define search space for simulation
        # either pass single param config, ranges of param configs (s, e, st), or direct list or combination
        for param, val in strategy_params.items():
            if isinstance(val, (float, int)):
                strategy_params[param] = [val]
            elif type(val) == tuple and len(val) == 3 and all(isinstance(num, (float, int)) for num in val):
                start, end, step = val
                strategy_params[param] = list(np.arange(start, end + step, step))
            elif type(val) == list and all(isinstance(num, (float, int)) for num in val):
                pass
            else:
                raise ValueError(f"Invalid strategy parameter value: {param}={val}")

        param_names = list(strategy_params.keys())
        param_vals = list(strategy_params.values())
        param_configs = list(itertools.product(*param_vals))

        results = []
        for config in param_configs:
            params = { name: param.item() for name, param in zip(param_names, config) }
            strategy_config = self._get_strategy_class(strategy=strategy)(**params)
            results.extend([[strategy_config.name, str(params)] + row for row in self.eval_config(strategy_config)])

        return results

    # (simple_configs + adaptive_configs) * tasks (6) * models (3) * metrics (5avg = 4m, 6b) * eval_sets (3)

    def eval_config(self, strategy: AbstractStrategy):
        results = []
        for task in self.tasks:
            t = task[0]
            task_metadata = [t["dataset"], t["tid"], t["fold"], t["framework"], t["problem_type"]]
            for model, info in task[1].items():
                metric_names, metric_curves = info
                for i, metric in enumerate(metric_names):
                    for j, eval_set in enumerate(["train", "val", "test"]):
                        if j >= len(metric_curves[i]): break
                        stopping_curve = metric_curves[i][1] # always stop on val
                        eval_curve = metric_curves[i][j]
                        results.append(task_metadata + [model, metric, eval_set] + strategy.simulate(stopping_curve=stopping_curve, eval_curve=eval_curve))

        return results


    def rank(self, eval_set: str = "val", strategies: dict | None = None):
        if self.simulations is None:
            self.run(strategies=strategies)

        ranks = self.simulations.copy()

        ranks = ranks[ranks["eval_set"] == eval_set]
        ranks['superscore'] = list(zip(ranks['percent_error_diff'], ranks['percent_iter_diff']))
        ranks["rank"] = ranks.groupby(["dataset", "fold", "model", "metric"])["superscore"].rank()
        strategy_name_mapping = ranks.groupby('params')['strategy'].unique().to_dict()

        ranks = ranks.groupby("params")["rank"].mean()
        ranks = ranks.sort_values().reset_index()

        ranks["strategy"] = ranks["params"].map(strategy_name_mapping).apply(lambda x: x[0])
        self.ranks = ranks

        return self.ranks


    def _rank(self, by: str = "error", eval_set: str = "val", strategies: dict | None = None):
        if self.simulations is None:
            self.run(strategies=strategies)

        ranks = self.simulations.copy()

        ranks = ranks[ranks["eval_set"] == eval_set]

        ranks['superscore'] = list(zip(ranks['percent_error_diff'], ranks['percent_iter_diff']))

        ranks["rank"] = ranks.groupby(["dataset", "fold", "model", "metric"])["superscore"].rank()
        strategy_name_mapping = ranks.groupby('params')['strategy'].unique().to_dict()

        ranks = ranks.groupby("params")["rank"].mean()
        ranks = ranks.sort_values().reset_index()

        ranks["strategy"] = ranks["params"].map(strategy_name_mapping).apply(lambda x: x[0])
        self.ranks = ranks

        return self.ranks


    def _rank_by_error(self, df: pd.DataFrame):
        return list(df['percent_error_diff'])

    def _rank_by_iter(self, df: pd.DataFrame):
        return list(df['percent_iter_diff'])

    def _rank_by_error_then_iter(self, df: pd.DataFrame):
        return list(zip(df['percent_error_diff'], df['percent_iter_diff']))


    def plot(self, strategy: str, eval_set, *params: str):
        for param in params:
            if param not in self._get_strategy_params(strategy=strategy):
                raise ValueError(f"Invalid Parameter: {param} for strategy {strategy}")

        if len(params) == 0:
            params = self._get_strategy_params(strategy=strategy)

        self.rank(eval_set="val")
        df = self.ranks.copy()
        df = df[df["strategy"] == strategy]
        df["params"] = df["params"].apply(lambda x: json.loads(x.replace("'", '"')))
        val = df

        self.rank(eval_set="test")
        df = self.ranks.copy()
        df = df[df["strategy"] == strategy]
        df["params"] = df["params"].apply(lambda x: json.loads(x.replace("'", '"')))
        test = df

        nparam = len(params)
        if nparam == 1:
            plot_func = self.plot_1d
        elif nparam == 2:
            plot_func = self.plot_2d
        else:
            raise ValueError("Plots with this many dimensions are not supported")

        return plot_func(val, test, strategy, eval_set, *params)


    def plot_1d(self, val: pd.DataFrame, test, strategy: str, eval_set, param: str):
        # df["params"] = df["params"].apply(lambda x: x[param])


        val["val"] = val["params"].apply(lambda x: x[param])
        test["test"] = test["params"].apply(lambda x: x[param])


        plt.ioff()
        fig, ax = plt.subplots()
        sns.lineplot(x='val', y='rank', data=val, ax=ax, label="val")
        sns.lineplot(x='test', y='rank', data=test, ax=ax, label="test")

        plt.legend()

        plt.title(f"Stopping Strategy Performance for {strategy}") #  on {eval_set} Dataset
        plt.xlabel(param)
        plt.ylabel("rank")
        plt.grid(True)
        plt.ion()
        return fig


    def plot_2d(self, df: pd.DataFrame, strategy: str, eval_set, x_param: str, y_param: str):
        xy = df["params"].apply(lambda r: (r[x_param], r[y_param])).tolist()
        z = df["rank"]

        x_values = sorted(set(c[0] for c in xy))
        y_values = sorted(set(c[1] for c in xy))

        heatmap = np.zeros((len(y_values), len(x_values)))

        for (x, y), rank in zip(xy, z):
            heatmap[y_values.index(y), x_values.index(x)] = rank

        plt.ioff()
        fig, ax = plt.subplots()

        plot = ax.imshow(heatmap, cmap='viridis', origin='lower', aspect='auto', extent=(min(x_values), max(x_values), min(y_values), max(y_values)))
        fig.colorbar(plot, ax=ax, label='rank')

        ax.set_title(f"Stopping Strategy Performance for {strategy}") #  on {eval_set} Dataset
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        plt.ion()

        return fig


    def plot_3d():
        # 2d heatmap with contour lines
        pass


    def _get_strategy_class(self, strategy: str):
        return self._strategy_class_map[strategy]


    # TODO: this should be an attribute of strategy classes
    # maybe as a class method, we can have the strategy names defined for
    # each of the strategy configurations
    def _get_strategy_params(self, strategy: str):
        return self.default_strategy_configs[strategy].keys()



"""
Considerations:
- Can't just compare metric error's directly (diff magnitudes): https://chatgpt.com/share/04b78338-7faa-48cb-9a88-cc096ba9f387
- Diff models have diff number of total iterations (i.e. gbm has ~9999 iterations while nn_torch has < 50), so want to test diff patience values for each model seperately
- How many max iterations / max epochs should we even set when generating learning curves for testing early stopping / during regular training when max iter/epoch is hit
- When choosing optimal parameter configurations, give more weight to the metrics used as defaults for stopping in AutoGluon (i.e. log_loss, mean_absolute_error)


"which stopping strategy to use":
---------------------------------
for each strategy:

    opt score = min(test curve)

    for each metric:
        best score = simulate strategy
        compare best score to opt score


This is a different problem entirely: "which stopping metric to use":
---------------------------------------------------------------------
count times metric beats others

opt score = min(test curve)

for each metric:
    best score = simulate strategy
    compare best score to opt score
    save best score to metric

best metric = min(best scores) # can't compare across metrics like this
"""


