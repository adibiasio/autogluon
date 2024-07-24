import os, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from stopping_simulator.EarlyStoppingSimulator import StoppingSimulator


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

    hyperparameters = {
        # TODO: set early_stop to maximum number of iterations supported by model
        "GBM": {
            "ag.early_stop": 999999,
            "num_boost_round": 1000,
        },
        "XGB": {
            "ag.early_stop": 999999,
            "n_estimators": 1000,
        },
        "NN_TORCH": {
            "epochs_wo_improve": 999999,
            "num_epochs": 100,
        },
        # "FASTAI": {}, # issues with metrics that need pred_proba (not during epochs though)
        # "CAT": {},
    }

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
    )

    predictor = predictor.fit(
        train_data=train_data,
        # tuning_data=test_data,
        test_data=test_data,
        learning_curves=True,
        # learning_curves={
        #     # "metrics": ["mean_absolute_error"],
        #     "use_error": True,
        # },
        hyperparameters=hyperparameters,
        # time_limit=600,
        # verbosity=4,
    )

    lc = predictor.learning_curves()
    return lc

    # import json
    # with open("curves.json", "w") as f:
    #     json.dump(lc, f)






if __name__ == '__main__':
    pass
    # runModel(problem_type="regression")
    pass
    # start = time.time()
    # curves = runModel(problem_type="regression")
    # ss.load_curves("curves.json")

    # oneConfig = {
    #     "simple_patience": { "p": 10 },
    #     "adaptive_patience": { "a": 0.2, "b": 20 },
    # }

    # manyConfigs = {
    #     "simple_patience": { "p": [0, 10, 50, 65] },
    #     "adaptive_patience": { "a": [0.01, 0.05, 0.1, 0.25, 0.5], "b": 20 },
    # }

    # allConfigs = {
    #     "simple_patience": { "p": (0, 150, 5) },
    #     "adaptive_patience": { "a": (0, 1, 0.1), "b": (0, 50, 5) },
    # }

    # ss = EarlyStoppingSimulator()
    # ss.load_curves("results_learning_curves/data/")
    # df = ss.benchmark(strategies=manyConfigs)
    # print(time.time() - start)
    # print(df)


    # import json
    # with open("curves.json", "r") as f:
    #     ss = EarlyStoppingSimulator()
    #     ss.load_curves(json.load(f))
    #     df = ss.benchmark()
    #     print(df)

        # curve = ss.curves[1]["XGB"][1][0][1]
        # adaptive = ss.param_search(curve, "adaptive_patience", a=(0, 2, 0.1), b = (0, 50, 5))
        # print(adaptive)

        # simple = ss.benchmark(strategy="simple_patience", patience=100)
        # adaptive = ss.benchmark(strategy="adaptive_patience", a=0.7, b=10)
        # print(simple)
        # print(adaptive)


    # runModel(problem_type="regression")


    # PLOT CURVES
    # import json
    # with open("curves.json", "r") as f:
    #     curves = json.load(f)

    #     predictor = TabularPredictor(
    #         label="c",
    #         problem_type="binary",
    #     )

    #     plot = predictor.plot_curves("XGB", "r2", learning_curves=curves)
    #     plot.savefig('seaborn_plot.png')


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


print("""
              params         rank           strategy
{'a': 0.23, 'b': 17}  1208.344017  adaptive_patience
{'a': 0.13, 'b': 20}  1211.114316  adaptive_patience
{'a': 0.09, 'b': 23}  1212.645299  adaptive_patience
{'a': 0.12, 'b': 21}  1213.504274  adaptive_patience
 {'a': 0.1, 'b': 23}  1214.552350  adaptive_patience
{'a': 0.22, 'b': 18}  1214.604701  adaptive_patience
 {'a': 0.1, 'b': 22}  1217.996795  adaptive_patience
{'a': 0.15, 'b': 19}  1218.295940  adaptive_patience
{'a': 0.25, 'b': 15}  1218.341880  adaptive_patience
{'a': 0.26, 'b': 15}  1218.425214  adaptive_patience
{'a': 0.03, 'b': 33}  1218.459402  adaptive_patience
{'a': 0.07, 'b': 28}  1218.727564  adaptive_patience
 {'a': 0.2, 'b': 17}  1218.740385  adaptive_patience
{'a': 0.24, 'b': 15}  1219.087607  adaptive_patience
{'a': 0.29, 'b': 14}  1219.300214  adaptive_patience
""")

# print("""
#               params            rank             strategy
# {'a': 0.21, 'b': 38}    21078.662393    adaptive_patience
# {'a': 0.23, 'b': 37}    21142.653846    adaptive_patience
#  {'a': 0.2, 'b': 39}    21154.518162    adaptive_patience
#  {'a': 0.2, 'b': 40}    21167.854701    adaptive_patience
# {'a': 0.22, 'b': 38}    21195.511752    adaptive_patience
# {'a': 0.19, 'b': 40}    21208.475427    adaptive_patience
# {'a': 0.19, 'b': 41}    21213.486111    adaptive_patience
# {'a': 0.21, 'b': 39}    21224.696581    adaptive_patience
# {'a': 0.23, 'b': 38}    21225.322650    adaptive_patience
# {'a': 0.19, 'b': 43}    21244.157051    adaptive_patience
# {'a': 0.32, 'b': 39}    21244.522436    adaptive_patience
# {'a': 0.19, 'b': 39}    21246.479701    adaptive_patience
#  {'a': 0.2, 'b': 41}    21251.603632    adaptive_patience
# {'a': 0.18, 'b': 43}    21259.795940    adaptive_patience
# {'a': 0.16, 'b': 45}    21263.338675    adaptive_patience
# """)

# print("""
#           params         rank         strategy
# {'patience': 84}  1252.101496  simple_patience
# {'patience': 83}  1254.683761  simple_patience
# {'patience': 85}  1256.551282  simple_patience
# {'patience': 82}  1256.705128  simple_patience
# {'patience': 76}  1257.429487  simple_patience
# {'patience': 87}  1257.631410  simple_patience
# {'patience': 90}  1257.666667  simple_patience
# {'patience': 88}  1258.211538  simple_patience
# {'patience': 75}  1259.335470  simple_patience
# {'patience': 80}  1260.239316  simple_patience
# {'patience': 97}  1260.358974  simple_patience
# {'patience': 77}  1260.909188  simple_patience
# {'patience': 86}  1261.183761  simple_patience
# {'patience': 91}  1262.183761  simple_patience
# {'patience': 98}  1262.628205  simple_patience
#              ...          ...              ...
#       """)


# print("""
# percent_score_diff   percent_iter_diff   rank
#                  0                   1      1
#                  0                   2      2
#                  0                   3      3.5
#                  0                   3      3.5
#                  1                   3      5
#                  1                   4      6
#                  2                   4      7
#                  3                   2      8
#                  4                   9      9
# """)

# print("""
#     total_iter  best_iter  opt_iter  best_score  opt_score  score_diff   percent_score_diff  percent_iter_diff             superscore  
# ...        135         29        29    0.014134   0.014134    0.000000             0.000000           3.785714        (0.0, 3.785714)  
# ...        159         34        34    0.165016   0.165016    0.000000             0.000000           3.787879        (0.0, 3.787879)  
# ...        159         34        34    0.166667   0.166667    0.000000             0.000000           3.787879        (0.0, 3.787879)  
# ...        159         34        34    0.204082   0.204082    0.000000             0.000000           3.787879        (0.0, 3.787879)  
# ...         92         20        20    0.383844   0.383844    0.000000             0.000000           3.789474        (0.0, 3.789474)
# ...        184        101       198    0.238889   0.233333    0.005556             0.017857          -0.071066  (0.017857, -0.071066)  
# ...        185        101       198    0.238889   0.233333    0.005556             0.017857          -0.065990   (0.017857, -0.06599)  
# ...        186        101       198    0.238889   0.233333    0.005556             0.017857          -0.060914  (0.017857, -0.060914)  
# ...        187        101       198    0.238889   0.233333    0.005556             0.017857          -0.055838  (0.017857, -0.055838)  
# ...        188        101       198    0.238889   0.233333    0.005556             0.017857          -0.050761  (0.017857, -0.050761)  
# ...        257        157      1005    0.052632   0.050439    0.002193             0.033333          -0.745020   (0.033333, -0.74502)  
# ...        258        157      1005    0.052632   0.050439    0.002193             0.033333          -0.744024  (0.033333, -0.744024)  
# ...        259        157      1005    0.052632   0.050439    0.002193             0.033333          -0.743028  (0.033333, -0.743028)  
# ...        260        157      1005    0.052632   0.050439    0.002193             0.033333          -0.742032  (0.033333, -0.742032)  
# ...        261        157      1005    0.052632   0.050439    0.002193             0.033333          -0.741036  (0.033333, -0.741036) 
# """)

# print("""
# ... total_iter best_iter  opt_iter  best_score  opt_score  score_diff  percent_score_diff  percent_iter_diff            super_score
# ...          3         2        96    0.027890   0.000000    0.027890            0.709319          -0.947917  (0.709319, -0.947917)
# ...          4         3       702    0.101967   0.089027    0.012940            0.112740          -0.991453  (0.112740, -0.991453)
# ...          3         2       916    0.086587   0.062818    0.023769            0.237288          -0.994541  (0.237288, -0.994541)
# ...          8         7        92    0.086364   0.000000    0.086364            0.086364          -0.891304  (0.086364, -0.891304)
# ...          5         4        44    0.267857   0.142857    0.125000            0.125000          -0.840909  (0.125000, -0.840909)
# ...        ...       ...       ...         ...        ...         ...                 ...                ...                    ...
# ...       1000        38        38    0.411252   0.411252    0.000000            0.000000          25.368421  (0.000000, 25.368421)
# ...       1000        58        58    0.370057   0.370057    0.000000            0.000000          16.275862  (0.000000, 16.275862)
# ...       1000       129       129    0.000000   0.000000    0.000000            0.000000           6.767442  (0.000000,  6.767442)
# ...       1000        25        25    0.408240   0.408240    0.000000            0.000000          39.080000  (0.000000, 39.080000)
# ...       1000        58        58    0.369128   0.369128    0.000000            0.000000          16.275862  (0.000000, 16.275862)
#       """)

# print("""
#                 strategy               params     dataset ... total_iter best_iter  opt_iter  best_score  opt_score  score_diff  percent_score_diff  percent_iter_diff
# 0        simple_patience             {'p': 0}  Australian ...          3         2        96    0.027890   0.000000    0.027890            0.709319          -0.947917
# 1        simple_patience             {'p': 0}  Australian ...          4         3       702    0.101967   0.089027    0.012940            0.112740          -0.991453
# 2        simple_patience             {'p': 0}  Australian ...          3         2       916    0.086587   0.062818    0.023769            0.237288          -0.994541
# 3        simple_patience             {'p': 0}  Australian ...          8         7        92    0.086364   0.000000    0.086364            0.086364          -0.891304
# 4        simple_patience             {'p': 0}  Australian ...          5         4        44    0.267857   0.142857    0.125000            0.125000          -0.840909
# ...                  ...                  ...         ... ...        ...       ...       ...         ...        ...         ...                 ...                ...
# 41035  adaptive_patience  {'a': 1.0, 'b': 50}       yeast ...       1000        38        38    0.411252   0.411252    0.000000            0.000000          25.368421
# 41036  adaptive_patience  {'a': 1.0, 'b': 50}       yeast ...       1000        58        58    0.370057   0.370057    0.000000            0.000000          16.275862
# 41037  adaptive_patience  {'a': 1.0, 'b': 50}       yeast ...       1000       129       129    0.000000   0.000000    0.000000            0.000000           6.767442
# 41038  adaptive_patience  {'a': 1.0, 'b': 50}       yeast ...       1000        25        25    0.408240   0.408240    0.000000            0.000000          39.080000
# 41039  adaptive_patience  {'a': 1.0, 'b': 50}       yeast ...       1000        58        58    0.369128   0.369128    0.000000            0.000000          16.275862
# """)

#                      params           rank             strategy
# 0          {'patience': 84}    1252.101496      simple_patience
# 1          {'patience': 83}    1254.683761      simple_patience
# 2          {'patience': 85}    1256.551282      simple_patience
# 3          {'patience': 82}    1256.705128      simple_patience
# 4          {'patience': 76}    1257.429487      simple_patience
# 5          {'patience': 87}    1257.631410      simple_patience
# 6          {'patience': 90}    1257.666667      simple_patience
# ...                     ...            ...                  ...
# 2745    {'a': 0.05, 'b': 0}   40713.998932    adaptive_patience
# 2746    {'a': 0.04, 'b': 0}   41108.266026    adaptive_patience
# 2747    {'a': 0.03, 'b': 0}   41412.545940    adaptive_patience
# 2748    {'a': 0.02, 'b': 0}   41618.414530    adaptive_patience
# 2749     {'a': 0.0, 'b': 1}   41813.946581    adaptive_patience
# 2750     {'a': 0.0, 'b': 0}   41813.946581    adaptive_patience
# 2751    {'a': 0.01, 'b': 0}   41813.946581    adaptive_patience