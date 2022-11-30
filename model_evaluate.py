# Run as: python model_evaluate.py <trained model> <input data csv>

import pandas as pd
from CorporateDefaultModel import CorporateDefaultModel
import dill
import pickle
import sys
from sklearn.metrics import roc_auc_score

test = pd.read_csv(sys.argv[2])

# Load the pre-trained model
filehandler = open(sys.argv[1], 'rb')
model = dill.load(filehandler)

# Use the model to make predictions on the input data
true, pi_adjusted = model.predict_harness(test)

AUC_score=roc_auc_score(true, pi_adjusted)

print('AUC score:', AUC_score)

print(pi_adjusted)

pi_adjusted.to_csv("predictions.csv")