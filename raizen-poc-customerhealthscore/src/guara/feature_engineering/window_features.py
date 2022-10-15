from scipy import stats
from time import time
import pandas as pd
import numpy as np
import itertools
from sklearn.base import BaseEstimator, TransformerMixin


class WindowFeatures(BaseEstimator, TransformerMixin):
    """
    Create window features

    Input: pd.Dataframe with date column, target column, grouped variables

    args:
        - date_column (str): timestamp columns where will be applied the rolling windows
        - target (str): target column that rolling windows will perform operations on
        - time_ranges (list of ints): number of lines of the rolling window. ex : [4,8]
        - rol_functions (list of strings): functions that will be applied. ex ['max', 'min']
        - grouped_variables (list of lists of strings): all the groupings that will be apllied. ex. [['SKU'], ['PDV', 'SKU']]
        - print: print for debug purposes
    """



    def __init__(self, date_column, target, time_ranges, rol_functions, grouped_variables, print_ = True ):
        
        self.date_column = date_column
        self.target = target
        self.time_ranges = time_ranges
        self.rol_functions = rol_functions
        self.grouped_variables = grouped_variables
        self.print_ = print_
        
        self.available_functions = [
                
            #'shift',
            #'diff',
            'mean',
            'median',
            'sum',
            'std',
            'max',
            'min',
            'kurtosis',
            'skewness',
            'quant_90',
            'quant_10',
            'trend',
            'longest_period_above_mean',
            'longest_period_below_mean',
            'longest_consecutive_decreases',
            'longest_consecutive_increases',
            'last_max',
            'last_min',
            'size',
            'list',
            'count_above_zero'

        ]

        not_implemented = [i for i in self.rol_functions if i not in self.available_functions]
        assert len(not_implemented) == 0, f'As funcoes {not_implemented} ainda não foram implementadas\nfuncoes implementadas {self.available_functions}'


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assert isinstance(X, pd.DataFrame)
        t = time()
        
        Xrw = rolling_window(
  
            df = X, 
            date_column = self.date_column, 
            target = self.target, 
            time_ranges = self.time_ranges,
            rol_functions = self.rol_functions,
            groups = self.grouped_variables,
            print_ = self.print_

        )
        
        print('window', time() - t)
        return Xrw




def rolling_window(df, date_column, target, time_ranges, rol_functions, groups, print_):

    for variables_to_group in groups:

        # cria uma lista com data + to_group
        list_group_date = variables_to_group + [date_column]
        
        df = df.sort_values(list_group_date).reset_index(drop = True)

        # conserta o problema do pandas com rolling e datas duplicadas
        grouped = df[list_group_date + [target]].groupby(list_group_date).sum().reset_index()
        
        # cria dataframe temporario com index
        df_index = grouped[list_group_date].copy().set_index(list_group_date)

        tg = time()
        grouped = grouped.groupby(variables_to_group)
        if print_:
            print("[{:.2f}] GROUPBY {}".format(time()- tg, variables_to_group ))

        #para cada timerange
        for period in time_ranges:
            
            tr = time()
            rolled = grouped[[target]].rolling(period)
            if print_:
                print("[{:.2f}] ROLLING {}".format(time()- tr, period )) 
                
                
            # para cada funcao
            for operation in rol_functions:

                t1 = time()
                try:

                    # feature ficará com nome 'VOLUME_30D_PDV_mean', por exemplo
                    column_name = '_'.join([target, str(period), '_'.join(variables_to_group), operation])
                    
                    # TODO
                    # verificar tirar o .values daqui (garante indexacao
                    df_index[column_name] = get_rolling_amount(rolled, operation).values
                    

                    if print_:
                        print("[{:.2f}] {} {} {}".format(time()- t1, variables_to_group, period, operation ))
                    
                except Exception as e:
                    print('erro ao computar',operation, period )
                    print(e)
                    pass

        df = pd.merge(

            left = df,
            right = df_index.reset_index(),
            left_on = list_group_date,
            right_on = list_group_date,
            how = 'left'

        )

               
    return df


def get_rolling_amount(grp, operation):
    '''
    grp = objeto groupedBy.rolling 
    operation = operação a ser realizada
    
    '''
    
        
    if operation == 'mean': 
        r = grp.mean()
        
      
    elif operation == 'median':
        r = grp.median()
        
    elif operation == 'sum':
        r = grp.sum()
        
        
    elif operation == 'std':
        r = grp.std()
        
    elif operation == 'max':
        r = grp.max()
        
    elif operation == 'min':
        r = grp.min() 
        
    elif operation == 'kurtosis':
        r = grp.kurt()
        
    elif operation == 'skewness':
        r = grp.skew()
        
    elif operation == 'quant_90':
        r = grp.quantile(0.9)
        
    elif operation == 'quant_10':
        r = grp.quantile(0.1)
        
    elif operation == 'trend':
        r = grp.apply(trend)
        
    elif operation == 'longest_period_above_mean':
        r = grp.apply(longest_period_above_mean)
 
    elif operation == 'longest_period_below_mean':
        r = grp.apply(longest_period_below_mean)
    
    elif operation == 'longest_consecutive_decreases':
        r = grp.apply(longest_consecutive_decreases)

    elif operation == 'longest_consecutive_increases':
        r = grp.apply(longest_consecutive_increases)
 
    elif operation == 'last_max':
        r = grp.apply(last_max)
    
    elif operation == 'last_min':
        r = grp.apply(last_min)

    elif operation == 'size':
        r = grp.apply(size)    
        
    elif operation == 'list':
        r = grp.apply(lambda x: x.tolist())
        
    elif operation == 'minmax_std':
        r = grp.apply(minmax_std) 
        
    elif operation == 'count_above_zero':
        r = grp.apply(count_above_zero)

       
        
    #elif operation == 'app_entr':
    #    r = grp.set_index(date_column)[[target]].rolling(freq).apply(ApEn, raw=False) 
    
    #elif operation == 'c3_score':
    #    r = grp.set_index(date_column)[[target]].rolling(freq).apply(C3, raw=False)
    
    return r



#####################################################################################
################################# FEATURE FUNCTIONS #################################
#####################################################################################

def trend(arr):
    slope, _, _, _, _ = stats.linregress( np.linspace(0, len(arr)-1, num=len(arr)), arr)
    return slope

def count_above_zero(arr):
    return np.sum(np.array(arr) > 0)



def longest_period_above_mean(arr):
    arr = np.array(arr)
    return max(_get_length_sequences_where(arr > arr.mean()))


def longest_period_below_mean(arr):
    arr = np.array(arr)
    return max(_get_length_sequences_where(arr < arr.mean()))
    
    
def longest_consecutive_decreases(arr):
    arr = np.array(arr)
    return max(_get_length_sequences_where(arr[:-1] > arr[1:]))


def longest_consecutive_increases(arr):
    arr = np.array(arr)
    return max(_get_length_sequences_where(arr[:-1] < arr[1:]))


def last_max(arr):
    return len(arr) - np.argmax(np.array(arr)) #+ 1


def last_min(arr):
    return len(arr) - np.argmin(np.array(arr)) #+ 1


def size(arr):
    return (~np.isnan(arr)).sum()
        


def minmax_std(arr):
    
    large_std = arr.std() > (arr.max()-arr.min())/2

    return large_std


def _get_length_sequences_where(x):
    
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.

    Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]
    



