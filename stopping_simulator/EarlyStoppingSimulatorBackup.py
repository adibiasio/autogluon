import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json, itertools, glob, os
from typing import List, Callable


class LearningCurveGenerator:
    """
    This function will need to be rerun when models are changed (but learning curve wouldn't be as sensitive to smaller changes)
    i.e. when new release of AG is out, models changed, so LC will reflect these changes
    and will need to be regenerated
    """
    pass

# goal is to find robust parameter settings that can be applied across different models
# and model families (so small changes on a model implementation should not impact LC that much)
# essentially: rerun once per major release

class EarlyStoppingSimulator:
    def __init__(self):
        self.tasks = []

        self._strategy_func_map = {
            "simple_patience": self._simple_patience,
            "sliding_simple_patience": self._sliding_simple_patience,
            "adaptive_patience": self._adaptive_patience,
            "sliding_adaptive_patience": self._sliding_adaptive_patience,
        }

        p = { "p": (0, 150, 5) }
        a = { "a": (0, 1, 0.1) }
        b = { "b": (0, 50, 5) }
        n = { "n": (0, 25, 5) }

        self.default_strategy_configs = {
            "simple_patience": { **p },
            # "sliding_simple_patience": { **p, **n },
            "adaptive_patience": { **a, **b },
            # "sliding_adaptive_patience": { **a, **b, **n }, # simpler is a | b | n, but not supported in python 3.8
        }


        """
        # self.curves format:
        {
            "XGB": {
                "log_loss": [
                    [curve1, curve2, curve3], # train
                    [curve1, curve2, curve3], # val
                    [curve1, curve2, curve3] # test
                ],
                "precision": [],
                "r2": []
            },


            "GBM": ...,
            "NN_TORCH": ...,
        }
        """



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
        simulate_columns = ["total_iter", 'best_iter', 'opt_iter', "best_score", "opt_score", "score_diff", "percent_score_diff", "percent_iter_diff"]
        all_columns = strategy_columns + task_columns + curve_columns + simulate_columns
        df = pd.DataFrame(results, columns=all_columns)

        return df


    def param_search(self, strategy: str, **strategy_params):
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
            params = { name: param for name, param in zip(param_names, config) }
            results.extend([[strategy, str(params)] + row for row in self.eval_config(strategy, **params)])

        return results


    def eval_config(self, strategy: str, **strategy_params):
        # runs simulations on all curves available for the specified model and stores results
        # runs across all models / strategies by default

        # {different datasets, different folds, different tasks on those folds} ==> FOR EACH OF THESE do the loop below

        results = []
        for task in self.tasks:
            t = task[0]
            task_metadata = [t["dataset"], t["tid"], t["fold"], t["framework"], t["problem_type"]]
            for model, info in task[1].items():
                metric_names, metric_curves = info
                for i, metric in enumerate(metric_names):
                    for j, eval_set in enumerate(["train", "val", "test"]):
                        if j >= len(metric_curves[i]): break
                        curve = metric_curves[i][j]
                        results.append(task_metadata + [model, metric, eval_set] + self.simulate(curve, strategy, **strategy_params))

        return results


    def simulate(self, curve: List[float], strategy: str, **strategy_params):
        strategy = self._get_strategy_func(strategy)
        best_iter, total_iter = strategy(curve, **strategy_params)
        opt_iter = curve.index(min(curve))
        score_diff = curve[best_iter] - curve[opt_iter] # make same index scheme ==> 0 indexed (special case handling for 0 iteration case)
        percent_score_diff = score_diff / max(curve)
        percent_iter_diff = (total_iter - opt_iter + 1) / (opt_iter + 1)
        return [total_iter, best_iter + 1, opt_iter + 1, curve[best_iter], curve[opt_iter], score_diff, percent_score_diff, percent_iter_diff]


    # get_patience_curves (flag for returning iterations graph in section 4.3.1 of design doc)
    def _patience(self, curve: List[float], patience: Callable[[int], int], sliding: int | None = None):
        """
        Parameters:
        --------------
        curve

        patience
            patience function

        sliding
            window size for averaging last n scores

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
            if sliding is not None and sliding > 0:
                n = min(iter + 1, sliding)
                sliding_sum += score
                sliding_sum -= curve[iter - n] if iter - n >= 0 else 0
                score = sliding_sum / n
            if best_score is None:
                best_score = score
            elif score >= best_score: # - min_delta
                counter += 1
                if counter >= patience(iter + 1):
                    break
            else:
                best_iter = iter
                best_score = score
                counter = 0

        return best_iter, iter + 1


    def _simple_patience_fn(self, p):
        def _get_patience(iter):
            return p
        return _get_patience


    def _adaptive_patience_fn(self, a, b):
        def _get_patience(iter):
            return a * iter + b
        return _get_patience


    def _simple_patience(self, curve, p: int = 10):
        return self._patience(curve, patience=self._simple_patience_fn(p))


    def _adaptive_patience(self, curve, a: float = 0.5, b: int = 10):
        return self._patience(curve, patience=self._adaptive_patience_fn(a, b))


    def _sliding_simple_patience(self, curve, p: int = 10, n: int = 10):
        return self._patience(curve, patience=self._simple_patience_fn(p), sliding=n)


    def _sliding_adaptive_patience(self, curve, a: float = 0.5, b: int = 10, n: int = 10):
        return self._patience(curve, patience=self._adaptive_patience_fn(a, b), sliding=n)


    def _get_strategy_func(self, strategy):
        return self._strategy_func_map[strategy]


    def rank(self, df: pd.DataFrame | None = None, eval_set: str = "val", strategies: dict | None = None):
        if df is None:
            df = self.run(strategies=strategies)
        
        ranks = df.copy()
        ranks = ranks[ranks["eval_set"] == eval_set]

        ranks['superscore'] = list(zip(ranks['percent_score_diff'], ranks['percent_iter_diff']))
        ranks["rank"] = ranks.groupby(["strategy", "model", "metric"])["superscore"].rank()

        strategy_name_mapping = ranks.groupby('params')['strategy'].unique().to_dict()

        ranks = ranks.groupby("params")["rank"].mean()
        ranks = ranks.sort_values().reset_index()

        ranks["strategy"] = ranks["params"].map(strategy_name_mapping).apply(lambda x: x[0])

        return ranks


    def plot_simple_patience(self, rank: pd.DataFrame):
        test = rank.copy()
        test = test[test["strategy"] == "simple_patience"]
        test["params"] = test["params"].apply(lambda x: json.loads(x.replace("'", '"'))["p"])

        plt.ioff()
        fig, ax = plt.subplots()
        sns.lineplot(x='params', y='rank', data=test, ax=ax)

        plt.title('Simple Patience Stopping Strategy Performance')
        plt.xlabel('patience')
        plt.ylabel("rank")
        plt.grid(True)
        plt.ion()
        return fig


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


"""
from functools import reduce
import json, itertools, glob, os
import numpy as np
import pandas as pd
from typing import List, Callable

class LearningCurveGenerator:

    This function will need to be rerun when models are changed (but learning curve wouldn't be as sensitive to smaller changes)
    i.e. when new release of AG is out, models changed, so LC will reflect these changes
    and will need to be regenerated

    pass

# goal is to find robust parameter settings that can be applied across different models
# and model families (so small changes on a model implementation should not impact LC that much)
# essentially: rerun once per major release

class EarlyStoppingSimulator:
    def __init__(self):
        self.tasks = []

        self._strategy_func_map = {
            "simple_patience": self._simple_patience,
            "sliding_simple_patience": self._sliding_simple_patience,
            "adaptive_patience": self._adaptive_patience,
            "sliding_adaptive_patience": self._sliding_adaptive_patience,
        }

        p = { "p": (0, 150, 5) }
        a = { "a": (0, 1, 0.1) }
        b = { "b": (0, 50, 5) }
        n = { "n": (0, 25, 5) }

        self.default_strategy_configs = {
            "simple_patience": { **p },
            # "sliding_simple_patience": { **p, **n },
            "adaptive_patience": { **a, **b },
            # "sliding_adaptive_patience": { **a, **b, **n }, # simpler is a | b | n, but not supported in python 3.8
        }


        # self.curves format:
        {
            "XGB": {
                "log_loss": [
                    [curve1, curve2, curve3], # train
                    [curve1, curve2, curve3], # val
                    [curve1, curve2, curve3] # test
                ],
                "precision": [],
                "r2": []
            },


            "GBM": ...,
            "NN_TORCH": ...,
        }

        # benchmark return format: --> display results as dataframe
        {
            "XGB": {
                "log_loss": [
                    [simulate1, simulate2, simulate3], # train
                    [simulate1, simulate2, simulate3], # val
                    [simulate1, simulate2, simulate3] # test
                ],
                "precision": [],
                "r2": []
            },


            # default: include all models
            "GBM": ...,
            "NN_TORCH": ...,
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


    def benchmark(self, strategies: dict | None = None):
        if strategies is None:
            strategies = self.default_strategy_configs

        results = []
        for strategy, params in strategies.items():
            results.extend(self.param_search(strategy, **params))

        strategy_columns = ["strategy", "params"]
        task_columns = ["dataset", "tid", "fold", "framework", "problem_type"]
        curve_columns = ["model", "metric", "eval_set"]
        simulate_columns = ["total_iter", 'best_iter', 'opt_iter', "best_score", "opt_score", "score_diff", "percent_score_diff", "percent_iter_diff"]
        all_columns = strategy_columns + task_columns + curve_columns + simulate_columns
        df = pd.DataFrame(results, columns=all_columns)


        # now we need ranking functionality
        # self.rank(df)

        return df


    def param_search(self, strategy: str, **strategy_params):
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
            params = { name: param for name, param in zip(param_names, config) }
            results.extend([[strategy, str(params)] + row for row in self.eval_config(strategy, **params)])

        return results


    def eval_config(self, strategy: str, **strategy_params):
        # runs simulations on all curves available for the specified model and stores results
        # runs across all models / strategies by default

        # {different datasets, different folds, different tasks on those folds} ==> FOR EACH OF THESE do the loop below

        results = []
        for task in self.tasks:
            t = task[0]
            task_metadata = [t["dataset"], t["tid"], t["fold"], t["framework"], t["problem_type"]]
            for model, info in task[1].items():
                metric_names, metric_curves = info
                for i, metric in enumerate(metric_names):
                    for j, eval_set in enumerate(["train", "val", "test"]):
                        if j >= len(metric_curves[i]): break
                        curve = metric_curves[i][j]
                        results.append(task_metadata + [model, metric, eval_set] + self.simulate(curve, strategy, **strategy_params))

        return results


    def simulate(self, curve: List[float], strategy: str, **strategy_params):
        strategy = self._get_strategy_func(strategy)
        best_iter, total_iter = strategy(curve, **strategy_params)
        opt_iter = curve.index(min(curve))
        score_diff = curve[best_iter] - curve[opt_iter]
        percent_score_diff = score_diff / max(curve)
        percent_iter_diff = (total_iter - opt_iter + 1) / (opt_iter + 1)
        return [total_iter, best_iter + 1, opt_iter + 1, curve[best_iter], curve[opt_iter], score_diff, percent_score_diff, percent_iter_diff]


    # get_patience_curves (flag for returning iterations graph in section 4.3.1 of design doc)
    def _patience(self, curve: List[float], patience: Callable[[int], int], sliding: int | None = None):

        Parameters:
        --------------
        curve

        patience
            patience function

        sliding
            window size for averaging last n scores

        Return:
        --------------
        best iteration (zero indexed)

        total iterations

        counter = 0
        best_iter = 0
        best_score = None
        sliding_sum = 0
        for iter, score in enumerate(curve):
            if sliding is not None and sliding > 0:
                n = min(iter + 1, sliding)
                sliding_sum += score
                sliding_sum -= curve[iter - n] if iter - n >= 0 else 0
                score = sliding_sum / n
            if best_score is None:
                best_score = score
            elif score >= best_score: # - min_delta
                counter += 1
                if counter >= patience(iter + 1):
                    break
            else:
                best_iter = iter
                best_score = score
                counter = 0

        return best_iter, iter + 1


    def _simple_patience_fn(self, p):
        def _get_patience(iter):
            return p
        return _get_patience


    def _adaptive_patience_fn(self, a, b):
        def _get_patience(iter):
            return a * iter + b
        return _get_patience


    def _simple_patience(self, curve, p: int = 10):
        return self._patience(curve, patience=self._simple_patience_fn(p))


    def _adaptive_patience(self, curve, a: float = 0.5, b: int = 10):
        return self._patience(curve, patience=self._adaptive_patience_fn(a, b))


    def _sliding_simple_patience(self, curve, p: int = 10, n: int = 10):
        return self._patience(curve, patience=self._simple_patience_fn(p), sliding=n)


    def _sliding_adaptive_patience(self, curve, a: float = 0.5, b: int = 10, n: int = 10):
        return self._patience(curve, patience=self._adaptive_patience_fn(a, b), sliding=n)


    def _get_strategy_func(self, strategy):
        return self._strategy_func_map[strategy]

    def rank(self, df, strategies: List[str] | None = None):
        if strategies is None:
            strategies = df["strategy"].unique().tolist()

        # rank param configs for each strategy

        # rank strategies by their best configs

        for strategy in strategies:
            filters = [
                df["strategy"] == strategy,
                df["eval_set"] == "val"
            ]
            data = df[reduce(lambda x, y: x & y, filters)]
            data = data.sort_values(by=['percent_score_diff', 'percent_iter_diff'], ascending=[True, True])

            

        # Can filter across all problem types or not
        # calculate test rank on diff metric (error from opt iter)
        # return

"""