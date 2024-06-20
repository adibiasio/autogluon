import os, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from autogluon.common.utils.log_utils import add_log_to_file


def runModel():
    label = 'class'
    train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    subsample_size = None  # subsample subset of data for faster demo, try setting this to much larger values
    if subsample_size is not None and subsample_size < len(train_data):
        train_data = train_data.sample(n=subsample_size, random_state=0)
    test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

    hyperparameters = {
        # "NN_TORCH": {},
        # "XGB": {},
        # "CAT": {},
        # "CAT": {
        #     "ag.extra_metrics": ["accuracy", "f1", "precision"]
        # },
        "GBM": {},
        # "FASTAI": {},
    }

    init_args = dict(
        eval_metric='log_loss', # Can we specify more here? #
    )

    predictor = TabularPredictor(
        label=label,
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

    """
    # save curve information in location relative to where current model artifact is saved
    # and then an object that can collect all of the data cached for the multiple models that we ran
    # something like predictor.collect_curves or something

    for dataset in datasets:
        runModel()    # this will export model data
        
        get exported model data from wherever it has been saved, then add it to the 
        current dataset organization / fold collection etc
    """


    runModel()

