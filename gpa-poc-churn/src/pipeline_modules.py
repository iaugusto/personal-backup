#  --------------------
import os 
import sys
import nltk 
import random
import joblib
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
from pathlib import Path
from datetime import datetime
from unidecode import unidecode
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
#  --------------------

#  --------------------
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
#  --------------------

#  --------------------
from feature_selection_module import feature_selection
#  --------------------


#  --------------------
def encode_sex(x):
    
    if x in ['f', 'F']:
        return 0
    
    elif x in ['m','M']:
        return 1
    
    else:
        return 0.5
#  --------------------


#  --------------------
class drop_temporary_columns(BaseEstimator, TransformerMixin):

    def __init__(self):
        
        self.columns = ['date','cidade','uf']
        
        return
        
    def fit(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame) 
        output_X = X.copy()
        output_X = output_X.drop(columns=self.columns)

        return output_X
#  --------------------


#  --------------------
class encode_sex_column(BaseEstimator, TransformerMixin):

    def __init__(self):
        
        self.columns = ['sexo']
        
        return
        
    def fit(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame) 
        output_X = X.copy()
        for var in self.columns:
            output_X[var] = output_X[var].str.lower()
            output_X[var] = output_X[var].apply(lambda x:encode_sex(x))

        return output_X
#  --------------------


#  --------------------
class handle_negative_values(BaseEstimator, TransformerMixin):

    def __init__(self, columns:list):
        
        self.columns = columns
        
        return
        
    def fit(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame) 
        output_X = X.copy()
        for var in self.columns:
            output_X[var] = [i if i>=0 else 0 for i in output_X[var]]

        return output_X
#  --------------------


#  --------------------
class fill_na_values_with_zero(BaseEstimator, TransformerMixin):

    def __init__(self, columns:list):
        
        self.columns = columns
        
        return
        
    def fit(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame) 
        output_X = X.copy()
        for var in self.columns:
            output_X[var] = output_X[var].fillna(0)

        return output_X
#  --------------------


#  --------------------
class fill_missing_numerical(BaseEstimator, TransformerMixin):
    
    """
    Replaces missing continuous values according to selected method.

    ---
    Variables:

    `columns (str,list)`: column or list of columns to fill. \n
    `method (str)`: name of the method. \n 
        * mean [default] - fill missing data with mean \n
    `indicator (bool)`: make a column that indicates, for each filled variable, 
    the rows that were modified. \n
    `classification (bool)`: indicates if objective is classification (True) 
    or regression (False) for predict method - default is False. \n
    """

    def __init__(
        self, 
        columns, 
        method:str='mean', 
        indicator:bool=False,
        classification:bool=False
        ):
        
        if type(columns) == list:
            self.columns = columns
        else:
            self.columns = [columns]

        self.method = method
        self.indicator = indicator
        self.var_dict = None
        self.var_df = None
        self.model_df = None
        self.params = None
        
        return
        
    def fit(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns
        
        self.var_dict = {
            'var':[],
            'rule':[],
            'correction':[]
        }

        if self.method == 'mean':
            for var in self.columns:
                self.var_dict['var'].append(var)
                self.var_dict['rule'].append(X[var][X[var].notnull()].mean())
                self.var_dict['correction'].append(0)
        else:
            print('wrong method entry')
            print('terminating fill_missing_continuous routine')
            return self

        self.var_df = pd.DataFrame(self.var_dict, columns=['var','rule','correction'])

        return self

    def transform(self, X):   

        assert isinstance(X, pd.DataFrame) 
        output_X = X.copy()

        for index, row in self.var_df.iterrows():
            replace_location = np.where(X[row['var']].isnull())
            output_X[row['var']].fillna(row['rule']+row['correction']*random.random(), inplace=True)
            if self.indicator==True:
                output_X[row['var']+'_dropind'] = 0
                output_X[row['var']+'_dropind'].iloc[replace_location] = 1

        return output_X
#  --------------------


#  --------------------
class encode_categorical(BaseEstimator, TransformerMixin):
    
    """
    Encode categorical variables according to selected method.

    ---
    Variables:

    `columns (str,list)`: column or list of columns to encode. \n
    `method (str)`: name of the method. \n 
        * onehot [default] - SKLearn's one-hot encoder \n
    `drop_columns (bool)`: drops original columns after encoding. \n
    """

    def __init__(
        self, 
        columns, 
        method:str='onehot', 
        drop_columns:bool=True
        ):

        if type(columns) == list:
            self.columns = columns
        else:
            self.columns = [columns]

        self.method = method
        self.drop_columns = drop_columns
        self.training_cols = {
            'variable':[],
            'columns':[]
        }
        self.training_cols_df = None

        return
        
    def fit(self, X, y=None):

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns

        if self.method == 'onehot':

            for var in self.columns:
                encoder = OneHotEncoder(
                    categories='auto',
                    drop='first',
                    sparse=False,
                    handle_unknown='error'
                )
                encoder.fit(X[var].values.reshape(-1,1))
                X_aux = pd.DataFrame(encoder.transform(X[var].values.reshape(-1,1)))
                column_names = [var+'_'+str(i) for i in list(X_aux.columns)]
                
                self.training_cols['variable'].append(var)
                self.training_cols['columns'].append(column_names)

            self.training_cols_df = pd.DataFrame(self.training_cols, columns=self.training_cols.keys())

        return self

    def transform(self, X):   

        assert isinstance(X, pd.DataFrame)

        if self.method == 'onehot':

            X.reset_index(drop=True, inplace=True)
            X_to_concat = [X]

            for var in self.columns:
                
                encoder = OneHotEncoder(
                    categories='auto',
                    drop='first',
                    sparse=False,
                    handle_unknown='error'
                )
                encoder.fit(X[var].values.reshape(-1,1))
            
                X_aux = pd.DataFrame(encoder.transform(X[var].values.reshape(-1,1)))
                X_aux.columns = [var+'_'+str(i) for i in list(X_aux.columns)]
                X_aux.reset_index(drop=True, inplace=True)

                # checking for divergence between new data and training data
                #----------
                reference_df = self.training_cols_df[self.training_cols_df['variable']==var]
                reference_values = reference_df['columns'].to_list()[0]
                missing_columns = set(reference_values).difference(set(X_aux.columns))
                unwanted_columns = set(X_aux.columns).difference(set(reference_values))

                for col in missing_columns:
                    X_aux[col] = 0

                if unwanted_columns:
                    X_aux.drop(columns=unwanted_columns, inplace=True)
                #----------

                X_to_concat.append(X_aux)

            output_X = pd.concat(X_to_concat, axis=1, ignore_index=False)
            output_X.drop(columns=self.columns, inplace=True)
            
        else:
            print('wrong method entry')
            print('returning original dataset')
            output_X = X.copy()

        return output_X
#  --------------------


#  --------------------
class data_transformation(BaseEstimator, TransformerMixin):
    
    """
    Transformation of numerical variables according to selected method.

    ---
    Variables:

    `columns (str,list)`: list of columns to fill. \n
    `method (str)`: name of the method. \n 
        * yeojohnson [default] - yeojohnson technique\n
    """

    def __init__(
        self, 
        columns, 
        method:str='yeojohnson'
        ):

        if type(columns) == list:
            self.columns = columns
        else:
            self.columns = [columns]

        self.method = method
        self.transform_ref = None
        self.transform_df = None
        
        return

    def fit(self, X, y=None):

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns
    
        return self
        
    def fit_transform(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        self.transform_ref = {
            'variable':[],
            'transformation':[],
            'parameters':[]
        }

        if self.method == 'yeojohnson':
            for var in self.columns:
                X[var], params = stats.yeojohnson(X[var])
                
                self.transform_ref['variable'].append(var)
                self.transform_ref['transformation'].append('yeojohnson')
                self.transform_ref['parameters'].append(params)

        self.transform_df = pd.DataFrame(self.transform_ref, columns=self.transform_ref.keys())

        return X

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        if self.transform_df.empty:
            print('Transformation dataframe is empty.\nReturning inputed dataset.')
            return X

        for index, row in self.transform_df.iterrows():
            if row['transformation'] == 'yeojohnson':
                X[row['variable']] = stats.yeojohnson(X[row['variable']],
                    lmbda=row['parameters'])

        return X
#  --------------------


#  --------------------
class outlier_handling(BaseEstimator, TransformerMixin):
    
    """
    Process dataset for outliers according to selected method.

    ---
    Variables:

    `columns (str,list)`: column or list of columns to process. \n
    `method (str)`: name of the method. \n
        * gauss - (-std, std) * band \n
    `action (str)`: action to take when a variable is out of the accepted interval. \n
        * mean [default] - replace outlier by the column's mean \n
    `band (float)`: bandwith for outlier definition according to Inter Quartile Technique, 
    or std correction for gaussian method. \n
    """

    def __init__(
        self, 
        columns,
        method:str='gauss',
        action:str='mean',
        band:float=1.5
        ):

        if type(columns) == list:
            self.columns = columns
        else:
            self.columns = [columns]

        self.action=action
        self.method = method
        self.band = band
        self.reference_params = None
        self.reference_df = None

        return

    def fit(self, X, y=None):

        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns
        
        self.reference_params = {
            'variable':[],
            'lower_boundary':[],
            'upper_boundary':[],
            'mean':[]
        }

        if self.method=='gauss':

            # (band = 2.69) in Gauss method is equivalent to (band = 1.5) on Robust method
            for var in self.columns:
                mean = X[var].mean()
                std = X[var].std()
                lower_boundary = mean-std*self.band
                upper_boundary = mean+std*self.band

                self.reference_params['variable'].append(var)
                self.reference_params['lower_boundary'].append(lower_boundary)
                self.reference_params['upper_boundary'].append(upper_boundary)
                self.reference_params['mean'].append(mean)

        else:
            print('No match for method argument.')
            return self
        
        self.reference_df = pd.DataFrame(self.reference_params, columns=self.reference_params.keys())

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)
        
        output_X = X.copy()

        if self.action=='mean':

            for index, row in self.reference_df.iterrows():
                
                output_X.reset_index(drop=True, inplace=True)
                reference_list = output_X.index[
                    (output_X[row['variable']] >= row['upper_boundary']) | (output_X[row['variable']]<= row['lower_boundary'])
                    ].to_list()
                if reference_list:
                    output_X[row['variable']].iloc[reference_list] = row['mean']
                output_X.reset_index(drop=True, inplace=True)


        if self.action=='max':

            for index, row in self.reference_df.iterrows():
                
                output_X.reset_index(drop=True, inplace=True)
                reference_list = output_X.index[
                    (output_X[row['variable']] >= row['upper_boundary']) | (output_X[row['variable']]<= row['lower_boundary'])
                    ].to_list()
                if reference_list:
                    output_X[row['variable']].iloc[reference_list] = row['upper_boundary']
                output_X.reset_index(drop=True, inplace=True)


        return output_X
#  --------------------


#  --------------------
class group_rare_categorical(BaseEstimator, TransformerMixin):
    
    """
    Check for variables with high cardinality and low term frequency.
    Categories with lower frequencies are grouped according to threshold.
    Should be used AFTER dropping constant values!
    ---
    Variables:

    `columns (str,list)`: column or list of columns to evaluate. \n
    `treshold (float)`: dropping treshold relative to percentage of valid inputs (default is 0.05). \n
    `rpl_str (str)`: label of grouped features (default is 'grouped'). \n

    """

    def __init__(
        self, 
        columns,
        threshold:float=0.05,
        rpl_str:str='grouped'
        ):

        if type(columns) == list:
            self.columns = columns
        else:
            self.columns = [columns]

        self.threshold = threshold
        self.grouping_df = None
        self.rpl_str = rpl_str
        self.columns_to_drop = []

        return
        
    def fit(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns

        grouping_dict = {
            'variable':[],
            'values':[]
        }

        for var in self.columns:
            
            X_aux = X.copy()
            frequencies = X_aux[var].value_counts(normalize=True)

            if len(frequencies) == 1: 
                print(var, ' has just one value. Dropping this column.')
                self.columns_to_drop.append(var)

            mapping = X_aux[var].map(frequencies)
            X_aux[var].mask(mapping<=self.threshold, self.rpl_str, inplace=True)
            values = list(X_aux[var].value_counts(normalize=True).keys())
            
            if self.rpl_str in values:
                values.remove(self.rpl_str)
                grouping_dict['variable'].append(var)
                grouping_dict['values'].append(values)
            
        self.grouping_df = pd.DataFrame(grouping_dict, columns=grouping_dict.keys())

        del grouping_dict

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        X.drop(columns=self.columns_to_drop, inplace=True)

        for var in self.grouping_df['variable']:
            valid_values = self.grouping_df[self.grouping_df['variable']==var]['values'].to_list()[0]
            found_values = list(X[var].value_counts().keys())
            invalid_values = set(found_values).difference(set(valid_values))
            for val in list(invalid_values):
                X[var] = X[var].str.replace(val,self.rpl_str)      

        return X
#  --------------------





#  --------------------
class drop_numerical_with_variance(BaseEstimator, TransformerMixin):
    
    """
    Selection of continuous variables according to a minimum variance threshold.
    Variables with variance bellow the threshold will be dropped.
    ---
    Variables:

    `columns (str,list)`: column or list of columns to evaluate. \n
    `threshold (float)`: variance threshold for dropping column. \n
    """

    def __init__(
        self, 
        columns,
        threshold:float=0.01
        ):

        if type(columns) == list:
            self.columns = columns 
        else:
            self.columns = [columns]

        self.columns_to_drop = None
        self.threshold = threshold

        return
        
    def fit(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame)

        available_columns = list(X.columns)
        if self.columns != available_columns:
            self.columns = [var for var in self.columns if var in available_columns]
        del available_columns

        X_aux = X[self.columns].copy()

        sel = VarianceThreshold(threshold=self.threshold)
        sel.fit(X_aux)
        X_aux = sel.transform(X_aux)
        feature_indices = sel.get_support(indices=True)

        columns_to_keep=[]
        X_aux = X[self.columns].copy()

        for index in feature_indices:
            columns_to_keep.append(X_aux.columns[index])

        self.columns_to_drop = set(self.columns).difference(set(columns_to_keep))

        del X_aux, columns_to_keep

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        var_original = list(X.columns)

        X.drop(columns=self.columns_to_drop, inplace=True)

        var_kept = list(X.columns)
        dropped_vars = set(var_original).difference(set(var_kept))
        
        if len(dropped_vars) != 0:
            print('Dropped numerical variables: \n', dropped_vars)

        del var_original, var_kept, dropped_vars

        return X
#  --------------------


#  --------------------
class select_with_correlation(BaseEstimator, TransformerMixin):
    
    """
    Feature selection according to a correlation threshold. Features with high correlation 
    are grouped and filtered according to predictive power.
    ---
    Variables:

    `threshold (float)`: correlation threshold for grouping columns. \n
    `method (str)`: method label. \n
        * recursive [default] - recursive feature selection \n
    `objective (str)`: model that will be applied to the grouped data. \n 
        * classification [default] \n
    """

    def __init__(
        self, 
        threshold:float=0.8,
        method:str='recursive',
        objective:str='classification'
        ):

        self.threshold = threshold
        self.method = method
        self.objective = objective

        return
        
    def fit(self, X, y):
        
        assert isinstance(X, pd.DataFrame)

        self.columns = list(X.columns)
        X_aux = X.copy()
        
        fs = feature_selection()

        if self.method == 'recursive':

            if self.objective == 'classification':

                removed_features = []

                status=True
                while status == True:

                    group, status = fs.recursive_correlation_term(
                        dataset=X_aux,
                        threshold=self.threshold
                    )

                    if status == False:
                        break

                    print('GROUPED FEATURES: ', group)

                    local_dataset = X_aux[set(group)]
                    rf = RandomForestClassifier(n_estimators=200, random_state=40, max_depth=6)
                    rf.fit(local_dataset.fillna(0), y)
                    important_feature_reference = list(rf.feature_importances_).index(np.max(rf.feature_importances_))
                    removed_features = [var for var in local_dataset.columns if var != local_dataset.columns[important_feature_reference]]
                    X_aux.drop(columns=removed_features, inplace=True)
                    print('removed features: ', removed_features)
                
        self.remaining_features = list(X_aux.columns)
        print(self.remaining_features)

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)
        X = X[self.remaining_features]

        return X
#  --------------------