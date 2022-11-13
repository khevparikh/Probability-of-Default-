import datetime
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

class CorporateDefaultModel:
    def __init__(self, df):
        self.df = df
        print("Raw data:", self.df.shape)
    
    def preprocess_data(self):
        # pandas indexes the dataframe automatically, so we can remove the Unnamed column.
        # 
        # We remove the column eqty_corp_family_tot because it is all NaNs.
        self.df.drop(columns = ['Unnamed: 0', 'eqty_corp_family_tot'], inplace=True)
        
        # Sort the data by company ID so that all of a company's financial statement rows are together.
        # Then sort by fiscal year to make sure the data is in chronological order, and to prevent future
        # data leakage.
        self.df.sort_values(["id", "fs_year"], inplace=True)
        
        # TO-DO: preprocess industry ATECO sector (look at the manuals Roger gave us)
        pass
        
        # Dimensions of the new preprocessed data
        print("Preprocessed data:", self.df.shape)
    
    def engineer_features(self):
        # Create a dataframe of features that will be used by the model
        self.features = pd.DataFrame([])
        
        # Calculate features
        self.features['wc_ta'] = self.df.wc_net / self.df.asst_tot
        self.features["ebit_ta"] = self.df.ebitda / self.df.asst_tot
        self.features["leverage"] = 1 - self.df.eqty_tot / self.df.asst_tot
        self.features["cf_to_debt"] = self.df.cf_operations / (self.df.debt_st + self.df.debt_lt)
        
        # TO-DO: we may need to add taxes and net profit as features
        
        print(self.df.iloc[: , -4:].describe()[1:])
        return self.features
    
    # Walk-forward analysis
    def train(self):
        pass
    
    def predict(self):
        pass
#%%
print("Reading data...")
df = pd.read_csv("train.csv")
print("Reading data... DONE")

#%%
model = CorporateDefaultModel(df)
model.preprocess_data()

#%%
X = model.engineer_features()