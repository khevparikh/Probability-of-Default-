import pandas as pd
from CorporateDefaultModel import CorporateDefaultModel
import dill
import pickle
import sys
from sklearn.metrics import roc_auc_score

test = pd.read_csv(sys.argv[1])

# Load the pre-trained model
filehandler = open("trained_model.obj", 'rb') 
model = dill.load(filehandler)

# Use the model to make predictions on the input data
pi_adjusted = model.predict_harness(test)

print(pi_adjusted)

pi_adjusted.to_csv("predictions.csv")