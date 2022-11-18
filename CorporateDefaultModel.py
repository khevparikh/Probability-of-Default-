import datetime
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

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
        
        # Sort the data by company ID and fiscal year to prevent potential look-ahead bias
        self.df.sort_values(["fs_year", "id"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Drop statement date because fs_year already contains that information
        # All statement dates are 12/31, which doesn't tell us anything new
        self.df.drop(columns=["stmt_date"], inplace=True)
        
        # Cast default date to datetime type
        self.df["def_date"] = pd.to_datetime(self.df["def_date"])
        
        # Cast ATECO sector from float to integer
        df["ateco_sector"] = df["ateco_sector"].astype(int)
        
        # TO-DO: preprocess industry ATECO sector (look at the manuals Roger gave us)
        pass
    
        # Convert default dates into indicator targets
        self.target = self.date_to_target(df["def_date"])
        
        """
        Because the (fiscal year, company ID) pair uniquely identifies each row, make
        it the index.
        """
        self.df.set_index(["fs_year", "id"], inplace=True)
        
        # Dimensions of the new preprocessed data
        print("Preprocessed data:", self.df.shape)
    
    # Function for imputing (filling in NaN values)
    def impute(self):
        """
        WARNING: Since we're averaging over all years, there will be look-ahead
        bias because if we want to impute a NaN value in 2007, for example, we'd
        be replacing it with the mean of 2007, 2008, 2009, etc. - data that we
        wouldn't have available in 2007.
        """
        
        # Fields to impute
        fields = ['wc_net', 'asst_tot', 'ebitda', 'eqty_tot', 'cf_operations', 'taxes', \
                  'debt_st', 'debt_lt', 'debt_bank_lt', 'liab_lt', 'liab_lt_emp', 'AP_lt', \
                  'roe', 'roa', 'prof_financing', 'exp_financing']
        
        """
        For each ATECO sector, compute means of these fields for all companies
        in that ATECO sector.
        """
        ateco_codes = np.unique(self.df["ateco_sector"])
        ateco_means = self.df.groupby("ateco_sector").mean()[fields]
        
        # For each ATECO code
        for code in ateco_codes:
            for field in fields:                
                # Replace NaN with the average value of that field for that specific ATECO code
                fill_value = ateco_means.loc[code][0]
                
                tmp = self.df.loc[self.df["ateco_sector"] == code, field]
                tmp = tmp.fillna(fill_value)
                self.df.loc[self.df["ateco_sector"] == code, field] = tmp
    
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
        self.features = pd.DataFrame([], index=self.df.index)
        
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
        """
        Walk-forward algorithm:
            
            1) Train the model on all observations from the beginning up until year t.
            2) Use the trained model to make predictions on observations from year t + 1.
            3) Save the predictions in an array.
            4) Increment t by 1 and repeat steps 1 to 3.
        """
        fs_year = self.df.index.map(lambda pair: pair[0])
        train_years = np.unique(fs_year)[:-1]
        
        # Partition the dataframe into subsets for training and testing
        X_trains = list(map(lambda t: self.features[fs_year <= t], train_years))
        y_trains = list(map(lambda t: self.target[fs_year <= t], train_years))
        X_tests = list(map(lambda t: self.features[fs_year == t+1], train_years))
        y_tests = list(map(lambda t: self.target[fs_year == t+1], train_years))
        
        print(train_years)
        print(list(map(lambda X: X.size, y_trains))) 
        print(list(map(lambda X: X.size, y_tests)))
        
        # Train models
        models = list(map(lambda X, y: LogisticRegression().fit(X, y),
                          X_trains, y_trains))
        
        # Predictions
        predictions = list(map(lambda model, X: model.predict(X),
                          models, X_tests))
        
    def predict(self):
        pass
    
    def harness(self):
        fields = ['wc_net', 'asst_tot', 'ebitda', 'eqty_tot', 'cf_operations', 'taxes', 'debt_st', 'debt_lt', 'debt_bank_lt', 'liab_lt', 'liab_lt_emp', 'AP_lt', 'roe', 'roa', 'prof_financing', 'exp_financing']
        self.preprocess_data()
        print(np.count_nonzero(self.df[fields].isna()))
        self.impute()
        print(np.count_nonzero(self.df[fields].isna()))
        self.engineer_features()
        #self.train()

#%%

print("Reading data...")
df = pd.read_csv("train.csv", nrows=1000)
print("Reading data... DONE")

#%%

model = CorporateDefaultModel(df)
df_new = model.harness()