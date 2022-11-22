import pandas as pd
from CorporateDefaultModel import CorporateDefaultModel
import dill
import pickle

train = pd.read_csv("train.csv")

# Train the model
model = CorporateDefaultModel()
model.train_harness(train)

# Save the trained model to a file
filehandler = open("trained_model.obj", 'wb') 
dill.dump(model, filehandler)
filehandler.close()