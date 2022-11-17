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
        """
        pandas indexes the dataframe automatically, so we can remove the Unnamed column.
        
        We remove the column eqty_corp_family_tot because it is all NaNs.
        """
        self.df.drop(columns = ['Unnamed: 0', 'eqty_corp_family_tot'], inplace=True)
        
        """
        Sort the data by company ID so that all of a company's financial statement rows are together.
        Then sort by fiscal year to make sure the data is in chronological order, and to prevent future
        data leakage.
        """
        self.df.sort_values(["id", "fs_year"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Drop statement date because fs_year already contains that information
        # All statement dates are 12/31, which doesn't tell us anything new
        self.df.drop(columns=["stmt_date"], inplace=True)
        
        # Cast default date to datetime type
        self.df["def_date"] = pd.to_datetime(self.df["def_date"])
        
        # TO-DO: preprocess industry ATECO sector (look at the manuals Roger gave us)
        pass
    
        # Convert default dates into indicator targets
        self.target = self.date_to_target(df["def_date"])
        
        # Dimensions of the new preprocessed data
        print("Preprocessed data:", self.df.shape)
    
    # Function for encoding default dates into 1s and 0s
    def date_to_target(self, def_dates):        
        """
        In Italy, the fiscal period is 1/1 to 12/31. We assume that we sit
        down 1/1 and run the model to predict default probabilities.
        
        For example, after FY 2008, we sit down 1/1/2009, and we predict
        probability of default between 1/1/2009 to 12/31/2009. But because
        FY 2008 data won't be released until March or April 2009, we can't
        use FY 2008 data to predict probability of default between 1/1/2009
        to 12/31/2009. We'd only be able to use data from FY 2007 and earlier.
        
        If the company defaulted during the 12-month period after the fiscal
        year ended, then y_it = 1, otherwise 0.
        
        y_t = 1 if the firm defaulted by time t, 0 otherwise. Once the company
        defaults, it'll always remain in that state. In other words, over time,
        the moment that y_t changes from 0 to 1, it will always remain 1.
        """
        
        return np.logical_and(def_dates.notna(),
                              self.df["fs_year"] + 1 >= def_dates.dt.year).astype(int)
    
    def engineer_features(self):
        # Create a dataframe of features that will be used by the model
        self.features = pd.DataFrame([])
        
        # Calculate features
        self.features["wc_ta"] = self.df.wc_net / self.df.asst_tot
        self.features["ebit_ta"] = self.df.ebitda / self.df.asst_tot
        self.features["leverage"] = 1 - self.df.eqty_tot / self.df.asst_tot
        self.features["cf_to_debt"] = (self.df.cf_operations + self.df.taxes) / (self.df.debt_st + self.df.debt_lt)
        
        # TO-DO: we may need to add taxes and net profit as features
        pass
        
        return self.features, self.target
    
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
