import pandas as pd
from autogluon.tabular import TabularPredictor
train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

predictor = TabularPredictor(label="age", problem_type="regression")
predictor = predictor.fit(train_data=train_data, test_data=test_data, learning_curves=True)

print(predictor.learning_curves())