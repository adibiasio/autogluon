import os, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

import time


def runModel(problem_type="binary"):
    labels = {
        "binary": "class",
        "multiclass": "marital-status",
        "regression": "age",
        # "quantile": "age",
    }

    label = labels[problem_type]

    train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    subsample_size = None  # subsample subset of data for faster demo, try setting this to much larger values
    if subsample_size is not None and subsample_size < len(train_data):
        train_data = train_data.sample(n=subsample_size, random_state=0)
    test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

    # change _ag_params() method in cat & lgbm classes (xgb done)


    # "ag.test_data": test_data,


    hyperparameters = {
        #### CONFIRMED WORKS FOR BINARY/MULTICLASS CLASSIFICATION & REGRESSION ####
        "NN_TORCH": {
            "ag.generate_curves": True,
            # "ag.curve_metrics": ["pinball_loss"],
            "ag.curve_metrics": ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2'],
            # "ag.curve_metrics": ["accuracy", "recall_weighted", "f1_weighted", "precision_weighted"],
            "ag.use_error_for_curve_metrics": True,
        },

        ## CONFIRMED WORKS FOR BINARY/MULTICLASS CLASSIFICATION & REGRESSION ####
        ## issues with metrics that need pred_proba (not during epochs though)
        # "FASTAI": {
        #     "ag.generate_curves": True,
        #     # "ag.curve_metrics": ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2'],
        #     "ag.curve_metrics": ["roc_auc"], # "roc_auc", "accuracy", "precision", "recall", "f1", "log_loss"
        #     # "ag.curve_metrics": ["accuracy", "recall_weighted", "f1_weighted", "precision_weighted"],
        #     "ag.use_error_for_curve_metrics": True,
        # },

        "GBM": {
            "ag.generate_curves": True,
            # "ag.curve_metrics": ["accuracy", "log_loss", "roc_auc", "recall", "f1", "precision"],
            # "ag.curve_metrics": ["accuracy", "recall_weighted", "f1_weighted", "precision_weighted"],
            "ag.curve_metrics": ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2'],
            "ag.use_error_for_curve_metrics": True,
        },


        "XGB": {
            "ag.generate_curves": True,
            # "ag.curve_metrics": ["log_loss"],
            # "ag.curve_metrics": ["accuracy", "log_loss", "roc_auc", "recall", "f1", "precision"],
            "ag.curve_metrics": ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2'],
            "ag.use_error_for_curve_metrics": True,
        },


        # "CAT": {
        #     "ag.generate_curves": True,
        #     "ag.curve_metrics": ["log_loss", "f1", "precision"],
        #     "ag.use_error_for_curve_metrics": True,
        # },
    }

    # for fastai, error if eval_metric not specified
    init_args = dict(
        # eval_metric='log_loss', # Can we specify more here? #
    )

    # TODO: ideally we can specify in TabularPredictor
    # that generate_curves = True, all the curve_metrics
    # and the use_error flag
    # since custom metrics supported for everything, then we can use those metrics for all models

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        **init_args,
    )

    predictor = predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=600,
        verbosity=4,
    )

    predictor.leaderboard(test_data, display=True)


if __name__ == '__main__':
    s = time.time()
    # print(f"START TIME: {s}")
    runModel(problem_type="regression")
    # print(f"time taken: {time.time() - s}")


    """
    # save curve information in location relative to where current model artifact is saved
    # and then an object that can collect all of the data cached for the multiple models that we ran
    # something like predictor.collect_curves or something

    for dataset in datasets:
        runModel()    # this will export model data
        
        get exported model data from wherever it has been saved, then add it to the 
        current dataset organization / fold collection etc
    """

# switch to using all custom metrics for code simplicity

# Test data path
# predictor -> learner (most intensive preprocessing) -> trainer -> model

# sample_weights (can try or ignore it)

# ignore these for now
# stacking -> to enable with  predict on test data with prior layer of models to get layer 2 (not important rn)
# bagging -> a lot more complicated (we will probably ignore it)

# multiclass warning
# /opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1497: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))