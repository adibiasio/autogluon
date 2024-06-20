import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# save curve information in location relative to where current model artifact is saved
# and then an object that can collect all of the data cached for the multiple models that we ran
# something like predictor.collect_curves or something

# data = {  # the full data for the project required for simulations
#     "dataset1": {  # dataset
#         2: {  # dataset fold (aka task)

#             # DONE #

#             "LightGBM_r123": {  # model (name tied to a specific hyperparameter setting)
#                 0: {  # iteration / epoch
#                     "accuracy": {  # metric
#                         "train": 0.45,  # train error
#                         "val": 0.46,  # val error
#                         "test": 0.49,  # test error
#                     },
#                     "log_loss": ...,
#                     ...,
#                 },
#                 1: ...,
#                 ...,
#             },

#         },
#     },
# }

# paths defined in these files:
# autogluon/core/src/autogluon/core/models/abstract/abstract_model.py
# autogluon/tabular/src/autogluon/tabular/predictor/predictor.py



# should be a part of autogluon/core/src/autogluon/core/models/abstract/abstract_model.py
def save_curve(model, metric, training_curve, validation_curve):
    base_path = f"learning_curves/{model}"
    os.makedirs(base_path, exist_ok=True)
    log_num = len(os.listdir(base_path))

    curve = {
        model : [
            # iteration data goes here
        ]
    }


    for i in range(len(training_curve[:3])): # TODO: remove the [:3]
        curve[model].append({
            metric : [
                training_curve[i], # "train": 
                validation_curve[i], # "val": 
                # "test": training_curve[i], # TODO: implement test curve data
            ]
        })
    
    print(curve)
    return curve



def plot_curve(path, metric, training_curve, validation_curve):
    # Plotting Curves
    curves = pd.DataFrame(
        {
            "Training Curve" : training_curve,
            "Validation Curve" : validation_curve,
        }
    )

    sns.set_theme(style="whitegrid")
    plot = sns.lineplot(data=curves, dashes=False)
    plot.set_ylabel(metric)
    plot.set_xlabel("Iterations")
    plot.set_title(f"Iterations vs {metric}")
    plot.set_ylim(0, 1)
    plt.savefig(f"{path}/learning_curves.png")




# class LearningCurves():
#     def __init__(self) -> None:
#         pass

#     def save_curve(self):




"""
Notes:

- have a local tmp file (JSON or database???) somewhere that we can use to build the files with all the epoch data, then later send entire file to s3 with s3 path specified by the user

"""
