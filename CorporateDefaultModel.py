import datetime
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from statsmodels.gam.api import GLMGam, BSplines #GLMGam Generalized Additive Models (GAM)
#from xgboost import XGBClassifier
from sklearn.svm import SVC
import statsmodels.api as sms
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

class CorporateDefaultModel:
    def __init__(self):
        return
    
    def load_data(self, df):
        self.df = df
        print("Raw data:", self.df.shape)
    
    def preprocess_data(self):
        """
        This function ...
        
        
        """
        # pandas indexes the dataframe automatically, so we can remove the Unnamed column.
        # We remove the column eqty_corp_family_tot because it is all NaNs.
        # Drop statement date because fs_year already contains that information
        # All statement dates are 12/31, which doesn't tell us anything new
        
        self.df.drop(columns = ['Unnamed: 0', 'eqty_corp_family_tot', 'stmt_date'], inplace=True)
        
        # Sort the data by company ID and fiscal year to prevent potential look-ahead bias
        self.df.sort_values(["fs_year", "id"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
             
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
        
        print("Number of companies in dataset:", self.df.index.map(lambda pair: pair[1]).nunique())
        
        # Dimensions of the new preprocessed data
        print("Preprocessed data:", self.df.shape)
    
    # Function for imputing (filling in NaN values)
    def impute(self):
        # Fields to impute
        fields = ['wc_net', 'asst_tot', 'ebitda', 'eqty_tot', 'cf_operations', 'taxes', \
                  'debt_st', 'debt_lt', 'debt_bank_lt', 'liab_lt', 'liab_lt_emp', 'AP_lt', \
                  'roe', 'roa', 'prof_financing', 'exp_financing']
        print("Number of fields to impute:", len(fields))
        
        company_index = self.df.index.map(lambda pair: pair[1])
        companies = np.unique(company_index)
            
        # Impute NaNs based on interpolation method
        map(lambda company: self.df.loc[company_index == company, fields].interpolate(
            limit_direction="forward", inplace=True), companies)
    
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
        
        result = np.logical_and(def_dates.notna(),
                                self.df["fs_year"] + 1 >= def_dates.dt.year).astype(int)
        
        target_index = pd.MultiIndex.from_frame(self.df[["fs_year", "id"]])
        return pd.Series(result.values, index=target_index)
    
    def engineer_features(self):
        # Create a dataframe of features that will be used by the model
        self.features = pd.DataFrame([], index=self.df.index)
        
        # Calculate features
        self.features["wc_ta"] = self.df.wc_net / self.df.asst_tot
        self.features["ebit_ta"] = self.df.ebitda / self.df.asst_tot
        self.features["leverage"] = 1 - self.df.eqty_tot / self.df.asst_tot
        self.features["cf_to_debt"] = self.df.cf_operations / (self.df.debt_st + self.df.debt_lt)
        
        """
        Suggested by Khevna - she found a 16% difference in median profit
        between defaulting and nondefaulting firms.
        """
        self.features["profit"] = self.df.profit
        
        # Try adding some more features to see if they help
        self.features["log_AP_st"] = np.log(self.df.AP_st)
        self.features["log_current_asst"] = np.log(self.df.asst_current)
        
        # TODO: We can't just delete rows, we need to deal with those companies
        # in specific ways. We must output a probability for every company.
        #
        # The mask is an array of booleans, such that if mask[i] = True, it means
        # we delete row i.
        self.features.replace([np.inf, -np.inf], np.nan, inplace=True)
        is_inf = np.isinf(self.features)
        is_nan = np.isnan(self.features)

        self.features.ffill(axis=0, inplace=True)
        mask = np.any(np.logical_or(is_inf, is_nan), axis=1)
        reordered_mask=pd.DataFrame(mask).reorder_levels(["id", "fs_year"]).sort_index()
        for col in list(reordered_mask.columns):
            reordered_mask[col]=reordered_mask.groupby('id')[col].transform(lambda x: x.ffill())

        print(reordered_mask)
        #Impute the value for the specific company id; idea: forward fill (from the previous year)? 
        #Reorder index as id, fs_year to do this
        
        #print('line 136:', reordered_mask)
        
        #print("Dropping {} rows".format(np.count_nonzero(mask)))
        
        # Now that we have features for our model, we don't need the original data anymore
        self.df = None

        #include this after engineer_features function

#def transformation_ratio(self):    
#self.df
#N=len(df_afterengineerfeatures) #this is after dropping the isna or isinf values 
#ateco_transf=train.sort_values(by='leverage')
#ateco_transf
#estimate nonlinear curve
#Define k buckets
#Define a nonlinear curve to map quantiles

    # Walk-forward analysis
    def train(self):
        """
        Walk-forward algorithm:
            
            1) Train the model on all observations from the beginning up until year t.
            2) Use the trained model to make predictions on observations from year t + 1.
            3) Save the predictions in an array.
            4) Increment t by 1 and repeat steps 1 to 3.
        """
        fs_year = self.features.index.map(lambda pair: pair[0])
        self.years = np.unique(fs_year)[:-1] #list of all years besides last one for mapping purpose
        
        # Select part of the data for training
        X_trains = map(lambda t: self.features[fs_year <= t], self.years)
        y_trains = map(lambda t: self.target[fs_year <= t], self.years)
        
        # Train models
        self.models = map(lambda X, y: LogisticRegression().fit(X, y), X_trains, y_trains)

        
    # Make predictions using the trained models
    def predict(self):
        #self.features: dataframe of features that will be used by the model
        fs_year = self.features.index.map(lambda pair: pair[0])
        
        # Select part of the data for testing
        X_tests = map(lambda t: self.features[fs_year == t+1], self.years)
        y_tests = map(lambda t: self.target[fs_year == t+1], self.years)
        
        # Aggregate ground truth values for each company
        # For each company, the last value tells us whether the company ultimately defaulted
        true = pd.concat(y_tests).reorder_levels(["id", "fs_year"]).sort_index()
        true = true.groupby(level="id").last()
        
        # Compute predictions for each row
        pred = map(lambda model, X: pd.DataFrame(model.predict_proba(X), index=X.index), self.models, X_tests)
        pred = pd.concat(pred)
        
        # Aggregate these predictions to generate a default probability for each company
        # For each company, we just predict the most recent probability
        pred = pred.reorder_levels(["id", "fs_year"]).sort_index()
        pred = pred.groupby(level="id").last()
        
        return true, pred

    def calibration(self, pred):
        #baseline default rate ranging from 0.5 to 15%
        pi_sample=self.target.mean()
        print("Sample default rate:", pi_sample)
        pi_true=0.005
        #t,pred=self.predict()
        pi_adjusted=pred.apply(lambda x: (pi_true/pi_sample)*(x))

        #Elkan_calibration
        pi_adjusted_el=pred.apply(lambda x:(pi_true)*(x - (x*pi_sample))/(pi_sample-(x*pi_sample)+(x*pi_true)-(pi_sample*pi_true)))

        return pi_adjusted
    
    def harness(self):
        self.preprocess_data()
        self.impute()
        self.engineer_features()
        self.train()
        t,p=self.predict()
        pi_a=self.calibration(p)
        return t, pi_a

print("Reading data...")
path=r"/Users/anthonychen/Desktop/ML_Finance/train.csv"
#path="train.csv"
df = pd.read_csv(path)

print("Reading data... DONE")

model = CorporateDefaultModel()
model.load_data(df)
t, pi_adjusted= model.harness()

X, y = model.features, model.target

print("AUC =", roc_auc_score(t, pi_adjusted.iloc[:, 1]))

