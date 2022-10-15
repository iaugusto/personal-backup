import pandas as pd
import numpy as np
import datetime
import pickle
from datetime import datetime as dt
from math import ceil
from functools import reduce
import functools
from time import time
import great_expectations as ge

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

pd.options.mode.chained_assignment = None

#########################################################################################################
######### REFERENCES
#
# https://wkirgsn.github.io/2018/02/15/pandas-pipelines/
# https://github.com/jem1031/pandas-pipelines-custom-transformers/blob/master/code/custom_transformers.py
#
#
######### AVAILABLE CLASSES
#
# ApplyDtypes           : Apply schema to dataframe 
# ShiftDiff             : creates shif and diff features
# DaysToEvent           : creates features of days to event based on events dictionary
# SinCosFeaturizer      : extracts sin and cosin from selected columns
# DatetimeFeaturizer    : extracts features from datetime Series
# ValidateDF            : make validations based on fitted dataframe
# FilterDF              : applies filters in df 
# PandasTypeSelector    : select dtypes
# ColumnSelector        : select columns
# ColumnDropper         : drop columns
# PandasFeatureUnion    : similar to sklearn FeatureUnion but for pandas
# CustomFunction        : input custom transformation functions only!
# FillNa                : fill NaN 
# OneHotPandas          : dummy encoder
# TfIdfPandas           : Tf-Idf
# CountVectorizerPandas : Bag of words (count)
# LabelEncoderPandas    : categorical label encoding
# WindowFeatures        : make window features (file window_features.py)
#
#
#########################################################################################################

def timer(func):
    """ print the runtime of the decorated function """
    def wrapper_timer(*args, **kwargs):
        start_time = time()
        value = func(*args, **kwargs)
        end_time = time()
        run_time = end_time - start_time
        print(f'{func.__name__!r} took {run_time:.4f} seconds')
        
        return value
    return wrapper_timer


def log_name(self_):
    print(f'\n----------- {self_.__class__.__name__}\n')




class PandasFeatureUnion(TransformerMixin):
    """
    Sklearn FeatureUnion similar, but for pandas DataFrames
    Outer Join diferent dataframes by index
    """
    
    def __init__(self, transformer_list):
        
        self.transformer_list = transformer_list

        steps = []
        for i, pipe_ in enumerate(self.transformer_list):
            if  type(pipe_) == Pipeline:
                steps.append((f'pipeline_{i}', pipe_))
            else:
                steps.append(pipe_)


        self.steps = steps


    def fit(self, X, y=None):
        
        
        for pipe in self.transformer_list:      
            pipe.fit(X.copy())

        return self

    def transform(self, X):
        # assumes X is a DataFrame
        
        #print([(name,t) for  name, t in self.transformer_list])
        Xts = [pipe.transform(X.copy()) for  pipe in self.transformer_list]
        #print(Xts)
        
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True,  how='outer'), Xts)
        return Xunion




class ApplyDtypes(TransformerMixin):
    """
    Apply types to dataframe
    
    """
    def __init__(self, 
                float_columns = [], 
                int_columns = [], 
                binary_columns = [], 
                categorical_columns = [],
                datetime_columns = [] 
                ):
        
        log_name(self)
        
        self.float_columns = float_columns 
        self.int_columns = int_columns 
        self.binary_columns = binary_columns 
        self.categorical_columns = categorical_columns
        self.datetime_columns = datetime_columns

            
        return
    

    def fit(self, X, y=None):
        isinstance(X, pd.DataFrame)
        
        type_dict = {}
        
        for col in self.float_columns:
            type_dict[col] = 'float64'

        for col in self.int_columns:
            type_dict[col] = 'int64'

        for col in self.binary_columns:
            type_dict[col] = 'int8'

        for col in self.categorical_columns:
            type_dict[col] = 'category'

        for col in self.datetime_columns:
            type_dict[col] = 'datetime64[ns]'
            
        
        self.type_dict = type_dict
                
        
        return self
    

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        for col in X.columns:
            if col not in self.type_dict.keys():
                print(f'[{col}] [{X[col].dtype}]: tipo nao especificado, mantendo original')

        return X.astype(self.type_dict)
    

class ValidateDF(TransformerMixin):
    """
    Validate data with great-expectations
    Need to apply dtypes before!!
    https://docs.greatexpectations.io/en/latest/reference/glossary_of_expectations.html
    """
    
    def __init__(self):
        log_name(self)
            
        return
    

    def fit_transform(self, X, y=None):
        isinstance(X, pd.DataFrame)

        df = ge.from_pandas(X)
        
        for col in df.columns:
            
            # expera que valores de nulos nao sejam maiores do que os fitados
            df.expect_column_values_to_not_be_null(col, mostly=df[col].notnull().sum()/df.shape[0])
            
            if df[col].dtype in ['datetime64[ns]']:
                a = 1
            
            if df[col].dtype.name == 'category':
                df.expect_column_values_to_be_in_set(col, df[col].unique().tolist())
                
                
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8','float64', 'float32', 'float16']:
                mean = df[col].mean()
                std  = df[col].std()
                min_  = df[col].min()
                max_  = df[col].max()
                
                df.expect_column_mean_to_be_between(col, mean-std, mean+std)
                df.expect_column_max_to_be_between(col, min_, max_)
                df.expect_column_min_to_be_between(col, min_, max_)
                    
                if df[col].dtype in ['int64', 'int32', 'int16']:
                    pass
                    
                if df[col].dtype in ['int8']:                
                    df.expect_column_values_to_be_in_set(col, [0,1])
                
                if df[col].dtype in ['float64', 'float32', 'float16']:
                    pass
                
        self.expectations = df.get_expectation_suite()         
            
        return self
    

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        validation = (ge
                      .from_pandas(X, expectation_suite=self.expectations)
                      .validate()
                      .to_json_dict()
                    )
        
        self.validation = validation
        
        print('\n----- Validations Report -----\n')
        print(f"Validation success: {validation['success']}\n")
        for stat in validation['statistics'].keys():
            print(f"{stat}: {validation['statistics'][stat]:.2f}")
        
        dfr = pd.DataFrame(validation['results'])

        dfr['column'] = dfr['expectation_config'].apply(lambda x: x['kwargs']['column'])
        dfr['expectation'] = dfr['expectation_config'].apply(lambda x: x['expectation_type'])
        dfr['expectation_values'] = dfr['expectation_config'].apply(lambda x: {key:value for (key,value) in x['kwargs'].items() if key not in ['column']})

        results_df = (pd
                    .DataFrame(dfr['result'].values.tolist(),index=dfr.index)
                    .drop(['partial_unexpected_list', 'missing_count', 'unexpected_count'],axis  = 1)
        )

        dfr[results_df.columns] = results_df.values

        dfr = dfr[['success','column','expectation','expectation_values']+ results_df.columns.tolist()]


        self.errors_on_validation = dfr[dfr.success == False]

        return X



class FilterDF(TransformerMixin):
    """
    Apply filters to dataframe
    
    """

        
    def fit(self, X, y=None):
        log_name(self)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

    
        X = X[

            (X['equip'] == 'vibracao_ponto_4' )


        ].copy()


        return X
    
    def fit_transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        
        X = X[

            (X.velocidade_mean > 0.001)&
            (X.temperatura_mean > 10)&
            (X.corrente_mean > 0.001)&
            (X['equip'] == 'vibracao_ponto_4' )
            
        ].copy()
        

        return X
    
    
    
    
class GroupDate(TransformerMixin):
    """
    Resample
    
    """
    def __init__(self):
        log_name(self)
        return
        
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
    
        assert isinstance(X, pd.DataFrame)
        X = X.set_index('date').resample('5min').agg({

            'id': 'last',
            'equip': 'last',
            'temperatura':['mean', 'std'],
            'pressao':['mean', 'std'],
            'corrente':['mean', 'std'],
            'nivel':['mean', 'std'],
            'velocidade':['mean', 'std'],


        })

        X.columns = [f'{i}_{j}' for i, j in X.columns]

        return X.reset_index().rename(columns = {'id_last': 'id', 'equip_last': 'equip'})
    
    
    
    
    
    
    
    
class ShiftDiff(BaseEstimator, TransformerMixin):
    """

    """

    def __init__(self, groups, target, operations, date_column, periods ):
        log_name(self)
        
        self.groups = groups
        self.target = target
        self.operations = operations
        self.date_column = date_column
        self.periods = periods

        return
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = time()
        
        df = X.copy()
        
        for variables_to_group in self.groups:

            # cria uma lista com data + to_group
            list_group_date = variables_to_group + [self.date_column]
            # sort values
            df = df.sort_values(list_group_date).reset_index(drop = True)

            # conserta o problema do pandas com rolling e datas duplicadas
            grouped = df[list_group_date + [self.target]].groupby(list_group_date).sum().reset_index()

            # cria dataframe temporario com index
            df_index = grouped[list_group_date].copy()
            
            # cria o dataframe agrupado
            grouped = grouped.groupby(variables_to_group)


            #para cada timerange
            for period in self.periods:

                # para cada funcao
                for operation in self.operations:
                    t1 = time()
                    
                    # feature ficará com nome 'VOLUME_30D_PDV_mean', por exemplo
                    column_name = '_'.join([self.target, str(period), '_'.join(variables_to_group), operation])
                    
                    
                    if operation == 'shift':
                        df_index[column_name] = grouped[self.target].shift(period)
                    if operation == 'diff':
                        df_index[column_name] = grouped[self.target].diff(period)

                        
                    print("[{:.2f}] {} {} {}".format(time()- t1, variables_to_group, period, operation ))

            df = pd.merge(

                left = df,
                right = df_index.reset_index(),
                left_on = list_group_date,
                right_on = list_group_date,
                how = 'left'

            )
    
        
        print('shift diff', time() - t)

        return df


class DaysToEvent(BaseEstimator, TransformerMixin):
    """
    creates features of days to event based on events dictionary in the format

    dict_events = {

     '2018-04-10':'pascoa',
     '2018-11-28':'bfriday',
     '2018-12-25':'natal',
     '2019-04-06':'pascoa',
     '2019-11-30':'bfriday',
     '2019-12-25':'natal'

    }

    """

    def __init__(self, date_col, events_dict):
        log_name(self)
        self.events_dict = events_dict
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

            
        Xsc = events_features(X, self.date_col, self.events_dict)
        
        return Xsc



class SinCosFeaturizer(BaseEstimator, TransformerMixin):
    """
    Perform sin and cosin transformations on columns
    """
    def __init__(self):
        log_name(self)
        return
        
    def fit(self, X, y=None):

        self.Xmax = X.max()
        self.original_columns = X.columns.tolist()

        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
            
        Xsc = X.clip(upper = self.Xmax, axis = 1).div(self.Xmax)
        
        Xsc[[col +'_sen' for col in self.original_columns]] = Xsc[self.original_columns].apply(lambda x: np.sin(2*np.pi*x))
        Xsc[[col +'_cos' for col in self.original_columns]] = Xsc[self.original_columns].apply(lambda x: np.cos(2*np.pi*x))
    
        
        return  Xsc.drop(self.original_columns, axis = 1).sort_index(axis=1)





class DatetimeFeaturizer(BaseEstimator, TransformerMixin):
    """
    Create Datetime features 
    args:
        - datetime_features (list of strings): list of features to extract
            see all available features with DatetimeFeaturizer().available_features
    """
    def __init__(self, datetime_features = None):
        log_name(self)
        self.datetime_features = datetime_features
        
        self.available_features = [

                    'hora', 
                    'dia', 
                    'mes', 
                    'ano', 
                    'dia_semana',
                    'dia_ano', 
                    'semana_mes',
                    'quarto_mes', 
                    'semana_ano', 
                    'fds', 
                    'periodo_dia'

        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assert isinstance(X, pd.DataFrame)
        assert self.datetime_features != None
        
        Xdt = make_date_features(X.iloc[:, 0], self.datetime_features)
        
        return Xdt

   


    
class ColumnSelector(TransformerMixin):
    """
    Select dataframe columns
    args:
        - cols(list of strings): columns to select   
    """
    
    def __init__(self, cols):
        log_name(self)
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        assert isinstance(X, pd.DataFrame)
        
        return X[self.cols]
    
    
    
    
class ColumnDropper(TransformerMixin):
    """
    Drop dataframe columns
    args:
        - cols: columns to drop   
    """
    
    def __init__(self, cols):
        log_name(self)
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        assert isinstance(X, pd.DataFrame)

        return X.drop(self.cols, axis = 1)
    
    
    


         
    
class FillNa(BaseEstimator, TransformerMixin):
    """
    Fill NaN values with specified value
    args:
        - fill_with: value to fill NaNs
    """
    def __init__(self,fill_with):
        log_name(self)
        self.fill_with = fill_with

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        X = X.fillna(self.fill_with)
        
        return X
        
                  
    
    
class OneHotPandas(BaseEstimator, TransformerMixin):
    """
    Performs one hot encoding to multiple features, ignores unknown
    transforms all the dataframe passed as categorical
    """
    
    def __init__(self):
        log_name(self)
        self.ohe = None
        
    def fit(self, X, y=None):
        
        X = X.fillna('nan_ohe')
        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', dtype = np.int32)
        self.ohe.fit(X)

        self.original_cols_names = X.columns.tolist()
        self.cols_map = {f'x{i}': col for i,col in enumerate(self.original_cols_names)}
        
        cols_ohe = self.ohe.get_feature_names()
        cols_ohe = [i.split('_') for i in cols_ohe]
        cols_ohe = ['_'.join([self.cols_map[i[0]]]+ i[1:]) for i in cols_ohe]
        self.cols_ohe = cols_ohe
        
        
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.fillna('nan_ohe')

        Xt = self.ohe.transform(X)
        Xdum = pd.DataFrame(Xt, index = X.index, columns = self.cols_ohe )
        
        return Xdum.fillna(0) 
    


class TfIdfPandas(BaseEstimator, TransformerMixin):
    """
    Performs tf-idf on a column
    args:
        max_features: umero maximo de features
    
    """  
    
    def __init__(self, max_features):
        log_name(self)
        self.tfidf = None
        self.max_features = max_features
        
    def fit(self, X, y=None):
        
        self.col = X.columns.tolist()[0]
        

        self.tfidf =  TfidfVectorizer(dtype=np.float32, 
                                      max_features=self.max_features, 
                                      sublinear_tf=True, 
                                      use_idf=True, 
                                      smooth_idf=True
                                     )
                           
        self.tfidf.fit(X[self.col])
        
        return self


    def transform(self, X):
        #assert isinstance(X, pd.DataFrame)
        
        Xt = self.tfidf.transform(X[self.col]).todense()
        Xt = pd.DataFrame(Xt, columns = self.tfidf.get_feature_names(), index=X.index)
        
        return Xt  

    
    
class CountVectorizerPandas(BaseEstimator, TransformerMixin):
    """
    Performs bag of words on a column
    args:
        max_features: maximum number of features
    
    """  
    
    def __init__(self, max_features):
        log_name(self)
        self.cv = None
        self.max_features = max_features
        
    def fit(self, X, y=None):
        
        self.col = X.columns.tolist()[0]
        

        self.cv =  CountVectorizer(max_features=self.max_features)
                           
        self.cv.fit(X[self.col])
        
        return self


    def transform(self, X):
        #assert isinstance(X, pd.DataFrame)
        
        Xt = self.cv.transform(X[self.col]).todense()
        Xt = pd.DataFrame(Xt, columns = self.cv.get_feature_names(), index=X.index)
        
        return Xt  

    
    
class LabelEncoderPandas(BaseEstimator, TransformerMixin):
    
    """
    Performs a robust label encoder on column
    col: column to be applied Label Encoder
    
    """  
        
    def __init__(self, col):
        log_name(self)
        self.le = None
        self.col = col
    
        
    def fit(self, X, y=None):
        
        self.le = LabelEncoder()
        self.le.fit(X[self.col])
        
        self.le.classes_ = np.append(self.le.classes_, 'NaN')
        
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        #X[self.col] = X[self.col].fillna('NaN')
           
        X[self.col] = X[self.col].apply(lambda s: 'NaN' if s not in self.le.classes_ else s)
        X[self.col] = self.le.transform(X[self.col])
        X[self.col] = X[self.col].astype('category')
        
        return X  



class TargetEncoderPandas(BaseEstimator, TransformerMixin):
    
    """
    Performs a robust target encoder on column
    col: column to be applied Label Encoder
    ytrain: Pandas series, the target
    operation: operation to be performed. options: 'mean', 'median'
    fill_unknown: fill method for new classes. options: 'mean', 'median', 'none'
    """  
        
    def __init__(self, col, ytrain, operation = 'mean', fill_unknow = 'none'):
        log_name(self)
        self.col = col
        self.ytrain = ytrain
        self.fill_unknow = fill_unknow
        self.operation = operation
        
        
    def fit(self, X, y=None):
        
        map_ = X[[self.col]].copy().assign(label = self.ytrain).groupby(self.col)
        
        if self.operation == 'mean':
            map_ = map_.mean()
        elif self.operation == 'median':
            map_ = map_.median()
        else: raise ValueError(f'operation {self.operation} not found')
            

        map_ = map_.to_dict()['label']
        
        
        self.map_ = map_
        self.ytrain = None
        
        return self

    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        X = X.copy()
        X[self.col] = X[self.col].astype('str').map(self.map_)
        
        if self.fill_unknow == 'mean':
            fill_num = np.mean([x for x in self.map_.values()])
            X[self.col] = X[self.col].fillna(fill_num)
            
        elif self.fill_unknow == 'median':
            fill_num = np.median([x for x in self.map_.values()])
            X[self.col] = X[self.col].fillna(fill_num)
            
            
        elif self.fill_unknow == 'none': pass
        else: raise ValueError(f'fill_unknow {self.fill_unknow} not found')
        
        return X  
    
    
    

class PandasTypeSelector(BaseEstimator, TransformerMixin):
    """
    Select columns in a pandas dataframe based on their dtype
    :param include: types to be included in the dataframe
    :param exclude: types to be excluded in the dataframe
    """

    def __init__(self, include=None, exclude=None):
        log_name(self)
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        """
        Saves the column names for check during transform
        :param X: pandas dataframe to select dtypes out of
        :param y: not used in this class
        """
        self._check_X_for_type(X)
        self.X_dtypes_ = X.dtypes
        self.feature_names_ = list(
            X.select_dtypes(include=self.include, exclude=self.exclude).columns
        )

        if len(self.feature_names_) == 0:
            raise ValueError(f"Provided type(s) results in empty dateframe")

        return self

    def get_feature_names(self, *args, **kwargs):
        return self.feature_names_

    def transform(self, X):
        """
        Transforms pandas dataframe by (de)selecting columns based on their dtype
        :param X: pandas dataframe to select dtypes for
        """
        #check_is_fitted(self, ["X_dtypes_", "feature_names_"])

        try:
            if (self.X_dtypes_ != X.dtypes).any():
                raise ValueError(
                    f"Column dtypes were not equal during fit and transform. Fit types: \n"
                    f"{self.X_dtypes_}\n"
                    f"transform: \n"
                    f"{X.dtypes}"
                )
        except ValueError as e:
            raise ValueError(f"Columns were not equal during fit and transform") from e

        self._check_X_for_type(X)
        transformed_df = X.select_dtypes(include=self.include, exclude=self.exclude)

        return transformed_df
        
    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")



class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Apply some custom function in the dataframe
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        X = X.head(2)
        
        return X.copy()




############################################################################################
##################################### HELPER FUNCTIONS #####################################
############################################################################################




def make_date_features(data_col, arg):
    '''
    data_col: pandas.Series com datas em datetime
    arg: lista contendo os atributos. 
        Ex: ['hora', 'dia', 'mes', 'ano', 'dia_semana', 'dia_ano', 'semana_mes', 'quarto_mes', 'semana_ano', 'fds', 'periodo_dia']
    
    return: dataframe com as colunas desejadas
    '''
    
    time_df = pd.DataFrame(index = data_col.index)
    
    
    if 'hora' in arg:
        time_df['hora'] = data_col.dt.hour
    
    if 'dia' in arg:
        time_df['dia'] = data_col.dt.day
        
    if 'mes' in arg:
        time_df['mes'] = data_col.dt.month
        
    if 'ano' in arg:
        time_df['ano'] = data_col.dt.year
        
    if 'dia_semana' in arg:
        # segunda dia 1, domingo dia 7
        time_df['dia_semana'] = data_col.dt.weekday + 1
        
    if 'dia_ano' in arg:
        time_df['dia_ano'] = data_col.dt.dayofyear
        
    if 'semana_mes' in arg:
        def week_of_month (dt):
            """ Returns the week of the month for the specified date.
            """
            first_day = dt.replace(day=1)
            dom = dt.day
            adjusted_dom = dom + first_day.weekday()
            return int(ceil(adjusted_dom/7.0))
        
        time_df['semana_mes'] = data_col.apply(week_of_month)
        
    if 'quarto_mes' in arg:
        b = [1,9,17,24,31]
        l = [1, 2, 3, 4]
        time_df['quarto_mes'] = pd.cut(data_col.dt.day, bins=b, labels=l, include_lowest=True).astype('int64')
        
    if 'semana_ano' in arg:
        time_df['semana_ano'] = data_col.dt.isocalendar().week.astype('int64')
        
    if 'fds' in arg:
        time_df['fds'] = np.where((data_col.dt.weekday + 1) >= 6, 1, 0).astype('int64')
        
    if 'periodo_dia' in arg:
        b = [0,6,12,18,24]
        l = ['madrugada', 'manha','tarde','noite']
        time_df['periodo_dia'] = pd.cut(data_col.dt.hour, bins=b, labels=l, include_lowest=True)
        
    return time_df


def events_features(event, data_col, dict_event):
    '''
    event: dataframe
    data_col: [str] coluna com datas em datetime
    dict_event: dicionario de feriados no esquema {'data': 'feriado'}.'data' no modelo 'AAAA-MM-DD'
    return: dataframe com as colunas de diferença em dias para o proximo feriado
    data_col
    '''

    # TODO: reformular essa funcao, está dando warning e vem cheio de erros
    # TODO separar em duas funcoes: dias para eventos, evento...

    colunas_geradas = []
    for i in [*dict_event.keys()]:
        d_ = pd.to_datetime(i, format='%Y-%m-%d')
        nome_coluna = 'dias_para_'+dict_event[i]+str(d_.year)
        
        event[nome_coluna] = d_ - event[data_col]
        event[nome_coluna] = event[nome_coluna].dt.days + 1
        colunas_geradas.append(nome_coluna)

    #event[event < 0] = np.nan
    event['dias_prox_evento'] = event[event[colunas_geradas] >= 0].min(axis=1)
    
    grouped = {}

    for key, value in sorted(dict_event.items()):
        grouped.setdefault(value, []).append(key)
        
    for j in [*grouped.keys()]:
        event[j] = event[data_col].dt.strftime('%Y-%m-%d').isin(grouped[j])
    
    return event



        