# Run as: python model_train.py <training data csv> <file to save trained model to>

import pandas as pd
from CorporateDefaultModel import CorporateDefaultModel
import dill
import pickle
import sys

train = pd.read_csv(sys.argv[1])

# Train the model
model = CorporateDefaultModel()
model.train_harness(train)

# Save the trained model to a file
filehandler = open(sys.argv[2], 'wb')
dill.dump(model, filehandler)
filehandler.close()