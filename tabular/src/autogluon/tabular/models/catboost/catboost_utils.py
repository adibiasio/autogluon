import logging
import numpy as np

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

logger = logging.getLogger(__name__)


CATBOOST_QUANTILE_PREFIX = "MultiQuantile:"

# User defined Metrics:
# https://github.com/catboost/catboost/blob/24aceedd3abf0ddb2f4791d4b8e054deb6de916f/catboost/tutorials/custom_loss/custom_loss_and_metric_tutorial.ipynb


# TODO: Add weight support?
# TODO: Can these be optimized? What computational cost do they have compared to the default catboost versions?
class CustomMetric:
    """Calculating custom metrics.

    Helpful CatBoost tutorial notebook for reference:
    https://github.com/catboost/catboost/blob/24aceedd3abf0ddb2f4791d4b8e054deb6de916f/catboost/tutorials/custom_loss/custom_loss_and_metric_tutorial.ipynb

    Parameters
    ----------
    scorer : Scorer
       A metric, represented as a scorer object, to be computed.
    """
    def __init__(self, scorer, wrapper=None):
        self.metric = scorer
        self.name = self.metric.name
        self.is_higher_better = self.metric.greater_is_better
        self.needs_pred_proba = not self.metric.needs_pred

        if not wrapper:
            wrapper = self.metric

        self.wrapper = wrapper

    @staticmethod
    def get_final_error(error, weight):
        return error

    def is_max_optimal(self):
        return self.is_higher_better

    def evaluate(self, approxes, target, weight):
        return self.wrapper(approxes, target, weight)


def func_generator(metric, problem_type: str, error=False):
    """Create a custom metric compatible with Catboost"""
    needs_pred_proba = not metric.needs_pred
    compute = (metric.error if error else metric)

    if needs_pred_proba:

        def custom_wrapper(approxes, target, weight):
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])
            approx = approxes[0]
            proba = np.asarray([np.exp(a) / (1 + np.exp(a)) for a in approx], dtype=np.float32)
            y_true = target
            return compute(y_true, proba), 0

    else:

        if problem_type in [MULTICLASS, SOFTCLASS]:

            def custom_wrapper(approxes, target, weight):
                assert len(approxes) == 1
                assert len(target) == len(approxes[0])
                approx = approxes[0]
                proba = np.asarray([np.exp(a) / (1 + np.exp(a)) for a in approx], dtype=np.float32)
                y_true = target
                y_hat = proba.argmax(axis=1)
                return compute(y_true, y_hat), 0

        elif problem_type == BINARY:

            def custom_wrapper(approxes, target, weight):
                assert len(approxes) == 1
                assert len(target) == len(approxes[0])
                approx = approxes[0]
                proba = np.asarray([np.exp(a) / (1 + np.exp(a)) for a in approx], dtype=np.float32)
                y_true = target
                y_hat = np.round(proba)
                return compute(y_true, y_hat), 0

        else:

            def custom_wrapper(approxes, target, weight):
                assert len(approxes) == 1
                assert len(target) == len(approxes[0])
                approx = approxes[0]
                proba = np.asarray([np.exp(a) / (1 + np.exp(a)) for a in approx], dtype=np.float32)
                y_true = target
                return compute(y_true, proba), 0

    custom_metric = CustomMetric(metric, wrapper=custom_wrapper)
    custom_metric.__name__ = metric.name
    return custom_metric


def get_catboost_metric_from_ag_metric(metric, problem_type, quantile_levels=None):
    if problem_type == SOFTCLASS:
        from .catboost_softclass_utils import SoftclassCustomMetric

        if metric.name != "soft_log_loss":
            logger.warning("Setting metric=soft_log_loss, the only metric supported for softclass problem_type")
        return SoftclassCustomMetric(metric=None, is_higher_better=True, needs_pred_proba=True)
    elif problem_type == BINARY:
        metric_map = dict(
            log_loss="Logloss",
            accuracy="Accuracy",
            roc_auc="AUC",
            f1="Logloss",  # f1 uses Logloss because f1 in CatBoost is not reliable (causes errors between versions)
            f1_macro="Logloss",
            f1_micro="Logloss",
            f1_weighted="Logloss",
            balanced_accuracy="BalancedAccuracy",
            recall="Recall",
            recall_macro="Recall",
            recall_micro="Recall",
            recall_weighted="Recall",
            precision="Precision",
            precision_macro="Precision",
            precision_micro="Precision",
            precision_weighted="Precision",
        )
        metric_class = metric_map.get(metric.name, "Logloss")
    elif problem_type == MULTICLASS:
        metric_map = dict(
            log_loss="MultiClass",
            accuracy="Accuracy",
        )
        metric_class = metric_map.get(metric.name, "MultiClass")
    elif problem_type == REGRESSION:
        metric_map = dict(
            mean_squared_error="RMSE",
            root_mean_squared_error="RMSE",
            mean_absolute_error="MAE",
            median_absolute_error="MedianAbsoluteError",
            r2="R2",
        )
        metric_class = metric_map.get(metric.name, "RMSE")
    elif problem_type == QUANTILE:
        if quantile_levels is None:
            raise AssertionError(f"quantile_levels must be provided for problem_type = {problem_type}")
        if not all(0 < q < 1 for q in quantile_levels):
            raise AssertionError(f"quantile_levels must fulfill 0 < q < 1, provided quantile_levels: {quantile_levels}")
        quantile_string = ",".join(str(q) for q in quantile_levels)
        metric_class = f"{CATBOOST_QUANTILE_PREFIX}alpha={quantile_string}"
    else:
        raise AssertionError(f"CatBoost does not support {problem_type} problem type.")

    return metric_class