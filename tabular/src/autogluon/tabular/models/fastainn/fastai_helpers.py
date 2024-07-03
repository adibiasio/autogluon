import pickle
import warnings
from pathlib import Path
from typing import Any
from torch.nn import functional as F

import torch
import numpy as np
from fastai.metrics import AccumMetric, Metric, ActivationType, skm_to_fastai
from fastai.tabular.all import nn
from fastai.torch_core import flatten_check

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS, PROBLEM_TYPES_CLASSIFICATION

class CustomMetric(Metric):
    """FastAI Custom Metric Wrapper Class for Scorer Objects.

    Parameters
    ----------
    scorer : Scorer
       A metric, represented as a scorer object, to be computed.

    problem_type : str
        Current problem type (i.e. BINARY or REGRESSION)

    wrapper : func(y_true : np.ndarray, y_pred : np.ndarray)
        Wrapper function that adheres to sklearn's metric calculation api
    """
    def __init__(self, scorer, problem_type, wrapper=None): # error=True
        if not wrapper:
            wrapper = scorer
        self.metric = scorer
        self.wrapper = wrapper
        self.problem_type = problem_type
        self.needs_pred_proba = not self.metric.needs_pred
        self.reset()

    @property
    def name(self): 
        return self.metric.name

    def reset(self):
        self.y_pred = []
        self.y_true = []

    def accumulate(self, learn):
        """
        Main issues:
         - not working with roc_auc: 
         - not working with predict_proba fn
         - model can't be saved: pickle

        pred = learn.pred

        if self.needs_pred_proba:
            pred = F.softmax(pred, dim=1)
        else:
            if self.problem_type in [MULTICLASS, SOFTCLASS]:
                pred = F.softmax(pred, dim=1)
                pred = pred.argmax(dim=1)
            elif self.problem_type == BINARY:
                pred = torch.sigmoid(pred)
                pred = (pred >= 0.5)

        y_true = learn.to_detach(learn.y)
        y_pred = learn.to_detach(pred)
  
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        """

        y_true = learn.y.detach()
        y_pred = learn.pred.detach()      
  
        if self.needs_pred_proba:
            y_pred = torch.softmax(y_pred, dim=-1)
            if self.problem_type == BINARY:
                y_pred = y_pred[:, -1]
        else:
            if self.problem_type in [MULTICLASS, SOFTCLASS]:
                y_pred = torch.softmax(y_pred, dim=-1)
                y_pred = y_pred.argmax(dim=1)
            elif self.problem_type == BINARY:
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)
            else:
                y_pred = torch.softmax(y_pred, dim=-1)

        self.y_true.extend(y_true.cpu().numpy())
        self.y_pred.extend(y_pred.cpu().numpy())


    @property
    def value(self):
        return self.wrapper(np.array(self.y_true), np.array(self.y_pred))


def func_generator(metric, problem_type: str, error=False):
    """Create a custom metric compatible with FastAI, based on the FastAI 2.7+ API"""

    """
    class WrapperCustomMetric(AccumMetric):
        pass

    needs_pred_proba = not metric.needs_pred
    compute = metric.error if error else metric
    custom_metric = WrapperCustomMetric(compute, invert_arg=True, to_np=True, activation="sigmoid", thresh=0.5)
    custom_metric.__class__.__name__ = metric.name
    return custom_metric
    """

    compute = metric.error if error else metric
    needs_pred_proba = not metric.needs_pred

    params = {}

    # TODO: test pred_proba for multiclass, binary pred works, not pred_proba
    # FIXME: table for now and come back after simulation stuff is done

    if needs_pred_proba:
        params["activation"] = "softmax"
        if problem_type == BINARY:
            params["activation"] = "binarysoftmax"
    else:
        if problem_type in [MULTICLASS, SOFTCLASS]:
            params["activation"] = "softmax"
            params["dim_argmax"] = 1
        elif problem_type == BINARY:
            params["activation"] = "sigmoid"
            params["thresh"] = 0.5
        else:
            pass

    custom_metric = AccumMetric(
                        compute,
                        name=metric.name,
                        to_np=True,
                        invert_arg=True,
                        **params
                    )

    # custom_metric = CustomMetric(metric, problem_type, wrapper=compute)
    # print(custom_metric.name)
    return custom_metric


def export(model, filename_or_stream="export.pkl", pickle_module=pickle, pickle_protocol=2):
    import torch
    from fastai.torch_core import rank_distrib

    "Export the content of `self` without the items and the optimizer state for inference"
    if rank_distrib():
        return  # don't export if child proc
    model._end_cleanup()
    old_dbunch = model.dls
    model.dls = model.dls.new_empty()
    state = model.opt.state_dict() if model.opt is not None else None
    model.opt = None
    target = open(model.path / filename_or_stream, "wb") if is_pathlike(filename_or_stream) else filename_or_stream
    with warnings.catch_warnings():
        # To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        torch.save(model, target, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
    model.create_opt()
    if state is not None:
        model.opt.load_state_dict(state)
    model.dls = old_dbunch


def is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))


def medae(inp, targ):
    "Mean absolute error between `inp` and `targ`."
    inp, targ = flatten_check(inp, targ)
    e = torch.abs(inp - targ)
    return torch.median(e).item()