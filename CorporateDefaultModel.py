import datetime
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from statsmodels.gam.api import GLMGam, BSplines #GLMGam Generalized Additive Models (GAM)
from xgboost import XGBClassifier
from sklearn.svm import SVC
import statsmodels.api as sms
from sklearn.pipeline import make_pipeline

class CorporateDefaultModel:
    def __init__(self):
        # Store models here during walk-forward analysis
        self.models = None
        return
    
    def preprocess_data(self, data):
        # pandas indexes the dataframe automatically, so we can remove the Unnamed column.
        # We remove the column eqty_corp_family_tot because it is all NaNs.
        # Drop statement date because fs_year already contains that information
        # All statement dates are 12/31, which doesn't tell us anything new
        
        data.drop(columns = ['Unnamed: 0', 'eqty_corp_family_tot', 'stmt_date'], inplace=True)
        
        # Sort the data by company ID and fiscal year to prevent potential look-ahead bias
        data.sort_values(["fs_year", "id"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        # Cast default date to datetime type
        data["def_date"] = pd.to_datetime(data["def_date"])
        
        # Cast ATECO sector from float to integer
        data["ateco_sector"] = data["ateco_sector"].astype(int)
        
        # TO-DO: preprocess industry ATECO sector (look at the manuals Roger gave us)
        pass
    
        # Convert default dates into indicator targets
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
        
        data["target"] = np.logical_and(data["def_date"].notna(),
                                        data["fs_year"] + 1 >= data["def_date"].dt.year).astype(int)
        
        """
        Because the (fiscal year, company ID) pair uniquely identifies each row, make
        it the index.
        """
        data.set_index(["fs_year", "id"], inplace=True)
    
    # Function for imputing (filling in NaN values)
    def impute(self, data):
        # Fields to impute
        fields = ['wc_net', 'asst_tot', 'ebitda', 'eqty_tot', 'cf_operations', 'taxes', \
                  'debt_st', 'debt_lt', 'debt_bank_lt', 'liab_lt', 'liab_lt_emp', 'AP_lt', \
                  'roe', 'roa', 'prof_financing', 'exp_financing']
        
        company_index = data.index.map(lambda pair: pair[1])
        companies = np.unique(company_index)
        
        # Impute NaNs based on interpolation method
        map(lambda company: data.loc[company_index == company, fields].interpolate(
            limit_direction="forward", inplace=True), companies)
    
    def engineer_features(self, data):
        # Save the target in its own variable
        target = data["target"]
        
        # Create a dataframe of features that will be used by the model
        features = pd.DataFrame([], index=data.index)
        
        # Calculate features
        features["wc_ta"] = data.wc_net / data.asst_tot
        features["ebit_ta"] = data.ebitda / data.asst_tot
        features["leverage"] = 1 - data.eqty_tot / data.asst_tot
        features["cf_to_debt"] = data.cf_operations / (data.debt_st + data.debt_lt)
        features["log_AP_st"] = np.log(data.AP_st)
        features["ST_debt_to_cur_asst"] = data.debt_st / data.asst_current
        
        """
        Suggested by Khevna - she found a 16% difference in median profit
        between defaulting and nondefaulting firms.
        """
        features["financial_profit"] = data.prof_financing
        
        # TODO: We can't just delete rows, we need to deal with those companies
        # in specific ways. We must output a probability for every company.
        #
        # The mask is an array of booleans, such that if mask[i] = True, it means
        # we delete row i.
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        is_inf = np.isinf(features)
        is_nan = np.isnan(features)

        features.ffill(axis=0, inplace=True)
        mask = np.any(np.logical_or(is_inf, is_nan), axis=1)
        reordered_mask=pd.DataFrame(mask).reorder_levels(["id", "fs_year"]).sort_index()
        for col in list(reordered_mask.columns):
            reordered_mask[col]=reordered_mask.groupby('id')[col].transform(lambda x: x.ffill())
        
        return features, target

    # Walk-forward analysis
    def train(self, features, target):
        """
        Walk-forward algorithm:
            
            1) Train the model on all observations from the beginning up until year t.
            2) Use the trained model to make predictions on observations from year t + 1.
            3) Save the predictions in an array.
            4) Increment t by 1 and repeat steps 1 to 3.
        """
        fs_year = features.index.map(lambda pair: pair[0])
        self.years = np.unique(fs_year)[:-1] #list of all years besides last one for mapping purpose
        
        # Select part of the data for training
        X_trains = list(map(lambda t: features[fs_year <= t], self.years))
        y_trains = list(map(lambda t: target[fs_year <= t], self.years))
                
        # Train models
        self.models = map(lambda X, y: XGBClassifier().fit(X, y), X_trains, y_trains)
        #self.models = list(map(lambda X, y: LogisticRegression().fit(X, y), X_trains, y_trains))
        
        # Save our training data to the class
        self.train_features = features
        self.train_target = target
    
    # Make predictions using the trained models
    def predict(self, features, target):
        if self.models == None:
            raise Exception("Model not trained yet")
        
        #features: dataframe of features that will be used by the model
        fs_year = features.index.map(lambda pair: pair[0])
        
        # Select part of the data for testing
        X_tests = map(lambda t: features[fs_year == t+1], self.years)
        X_tests = list(X_tests)
        y_tests = map(lambda t: target[fs_year == t+1], self.years)
        y_tests = list(y_tests)
        
        # Aggregate ground truth values for each company
        # For each company, the last value tells us whether the company ultimately defaulted
        true = pd.concat(y_tests)
        true = true.reorder_levels(["id", "fs_year"]).sort_index()
        true = true.groupby(level="id").last()
        
        # Compute predictions for each row
        pred = map(lambda model, X: pd.DataFrame(model.predict_proba(X), index=X.index),
                   self.models, X_tests)
        pred = list(pred)
        pred = pd.concat(pred)
        
        # Aggregate these predictions to generate a default probability for each company
        # For each company, we just predict the most recent probability
        pred = pred.reorder_levels(["id", "fs_year"]).sort_index()
        pred = pred.groupby(level="id").last()
        
        return true, pred.iloc[:, 1]
        
    def calibration(self, pred):
        # ACPMIP p. 220
        #
        # Baseline default rate ranging from 0.5 to 15%
        # check if max(pred.iloc[:,1]) > pi_true/pi_sample. This is impossible because pi_adjusted <=1.
        pi_sample = self.train_target.mean()
        pi_true = 0.005

        #simple calibration adjustment
        pi_adjusted = pred.apply(lambda x: (pi_true/pi_sample)*x)
        
        # Elkan_calibration corrects for differences in base rates in different datasets
        pi_adjusted_el = pred.apply(lambda x: (pi_true)*(x - (x*pi_sample))/(pi_sample-(x*pi_sample)+(x*pi_true)-(pi_sample*pi_true)))
        
        print('Is max probability generated by model possible:', max(pred) > (pi_true/pi_sample))
        return pi_adjusted_el
    
    # A function that consolidates all steps needed for training
    def train_harness(self, train):
        self.preprocess_data(train)
        self.impute(train)
        features, target = self.engineer_features(train)
        
        # Now that the data has been transformed, we can use it for training
        self.train(features, target)
    
    # A function that consolidates all steps needed for testing
    def predict_harness(self, test):
        self.preprocess_data(test)
        self.impute(test)
        features, target = self.engineer_features(test)
        
        # Now that the data has been transformed, we can make predictions on it
        true, pred = self.predict(features, target)
        
        # Calibrate the probabilities
        return self.calibration(pred)