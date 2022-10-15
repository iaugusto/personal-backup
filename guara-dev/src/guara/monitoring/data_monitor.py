import os
import sys
import pickle
import operator
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.utils import shuffle


class DataMonitor():


    def __init__(self, monitor):
        
        self.gen_date = datetime.now()
        self.dir = os.getcwd() 
        self.available_monitors = ['input','distribution']
        self.column_types_baseline = None
        self.column_types = None
        self.input_status_df = None
        self.df_dist_baseline = None
        self.df_dist = None
        self.dist_verification_df = None
        self.numericals = ['float','int']
        self.categoricals = ['str', 'category']
        self.dates = ['date-time']
        self.verbose_list = []
        self.warning_list = []
        self.shifted_features = []
        self.warning_threshold = {
            'input':[operator.le,1],
            'distribution':[operator.ge,4]
        }
        
        if type(monitor) == list:
            self.selected_monitors = monitor
        else:
            self.selected_monitors = [monitor]

        for single_monitor in self.selected_monitors:
            if single_monitor not in self.available_monitors:
                print('No match for monitor. \nTerminating sequence.')
                exit()

        return

    
    def create_input_baseline(self, X):

        """
        Creates a reference for input format and features' dtypes.
        Required to compare reference/training data to validation/production data. 
        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        """
        
        self.column_types_baseline = get_dtypes(X)

        return


    def compare_input_to_baseline(self, X):
        
        """
        Confronts input's format and features' dtypes with the reference/baseline 
        for inconsistencies.
        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        """
        
        self.column_types = get_dtypes(X)
        
        if self.column_types == self.column_types_baseline:
            print('Production input matches baseline from training.')
        else:
            
            input_status_dict = {
                'feature':[],
                'type':[],
                'status':[]
            }

            for ctype in self.column_types_baseline.keys():
                missing = set(self.column_types_baseline[ctype]).difference(set(self.column_types[ctype]))
                new = set(self.column_types[ctype]).difference(set(self.column_types_baseline[ctype]))
                if missing:
                    for var in missing:
                        input_status_dict['feature'].append(var)
                        input_status_dict['type'].append(ctype)
                        input_status_dict['status'].append('missing')
                if new:
                    for var in new:
                        input_status_dict['feature'].append(var)
                        input_status_dict['type'].append(ctype)
                        input_status_dict['status'].append('new')
                    
            self.input_status_df = pd.DataFrame(input_status_dict)
            self.input_status_df['version'] = [self.gen_date]*len(self.input_status_df)

            self.verbose_list = self.verbose_list + \
                ['-'*60] + \
                ['Production input doesn\'t match baseline from training.'] + \
                [self.input_status_df.head(5)] + \
                ['Use \'DataMonitor().input_status_df\' to see detailed information.'] + \
                ['-'*60]

        return


    def get_distribution(self, X) -> pd.DataFrame:

        """
        Creates a reference for features' distributions.
        Required to compare reference/training data to validation/production data. 
        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        """

        if self.column_types == None:
            self.column_types = get_dtypes(X)

        input_dataset = apply_dtypes(
            X=X,
            var_type=self.column_types
            )

        dist_dict = {
            'feature':[],
            'type':[],
            'mean':[],
            'std':[],
            'min':[],
            'max':[],
            '2nd-quantile':[],
            '3rd-quantile':[],
            'skew':[],
            'kurtosis':[],
            'category':[],
            'frequency':[],
            'date-format':[]
        }

        for ctype in self.numericals:
            for var in self.column_types[ctype]:
                dist_dict['feature'].append(var)
                dist_dict['type'].append(ctype)
                dist_dict['mean'].append(input_dataset[var].dropna(axis=0).mean(skipna=True))
                dist_dict['std'].append(input_dataset[var].dropna(axis=0).std(skipna=True))
                dist_dict['min'].append(input_dataset[var].dropna(axis=0).min())
                dist_dict['max'].append(input_dataset[var].dropna(axis=0).max())
                dist_dict['2nd-quantile'].append(input_dataset[var].dropna(axis=0).quantile(0.50))
                dist_dict['3rd-quantile'].append(input_dataset[var].dropna(axis=0).quantile(0.75))
                dist_dict['skew'].append(input_dataset[var].dropna(axis=0).skew(skipna=True))
                dist_dict['kurtosis'].append(input_dataset[var].dropna(axis=0).kurtosis(skipna=True))
                
                unused_keys = list(dist_dict.keys())[10:13]
                for key in unused_keys:
                    dist_dict[key].append(np.nan)

        for ctype in self.categoricals:
            for var in self.column_types[ctype]:
                dist_dict['feature'].append(var)
                dist_dict['type'].append(ctype)
                dist_dict['category'].append(list(input_dataset[var].value_counts(normalize=True).keys()[:]))
                dist_dict['frequency'].append(list(input_dataset[var].value_counts(normalize=True).values))

                unused_keys = list(dist_dict.keys())[2:10] + ['date-format']
                for key in unused_keys:
                    dist_dict[key].append(np.nan)

        for ctype in self.dates:
            for var in self.column_types[ctype]:
                dist_dict['feature'].append(var)
                dist_dict['type'].append(ctype)
                
                try:
                    aux = datetime.strptime(input_dataset[var].iloc[0], '%Y-%m-%d %H:%M:%S')
                    dist_dict['date-format'].append('american')
                except:   
                    try:
                        aux = datetime.strptime(input_dataset[var].iloc[0], '%d-%m-%Y %H:%M:%S')
                        dist_dict['date-format'].append('brazilian')
                    except:
                        dist_dict['date-format'].append('custom')
                
                unused_keys = list(dist_dict.keys())[2:12]
                for key in unused_keys:
                        dist_dict[key].append(np.nan)

        dist_df = pd.DataFrame(dist_dict)
        dist_df['version'] = [self.gen_date]*len(dist_df)

        return dist_df

    
    def create_distribution_baseline(self, X):

        """
        Creates a baseline reference of the features' distributions.
        Required to compare reference/training data to validation/production data. 
        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        """
        
        self.df_dist_baseline = self.get_distribution(X)

        return

    
    def compare_distribution_to_baseline(
        self, 
        X, 
        threshold:float=0.15
        ):

        """
        Confronts input's distributions with the reference/baseline 
        for inconsistencies.
        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        - `threshold (float)`: deviation threshold for distribution shift. \n
        """

        if isinstance(self.df_dist_baseline, pd.DataFrame) == False:
            print('-'*60)
            print('Must create distribution baseline before running distribution verification.')
            print('Use \'distribution_baseline(dataset)\' function to configure baseline.')
            print('-'*60)
            return
        
        numericals_check_list = [
            'mean',
            'std',
            'min',
            'max',
            '2nd-quantile',
            '3rd-quantile',
            'skew',
            'kurtosis'
        ]

        dist_verification_dict = {
            'feature':[],
            'property':[],
            'baseline':[],
            'current':[],
            'ratio':[],
            'distance':[]
        }

        self.df_dist = self.get_distribution(X)

        # Running verification for numerical variables
        df_list = []
        for ctypes in self.numericals:
            df_list.append(self.df_dist[self.df_dist['type']==ctypes])
        
        if len(df_list) > 0:
            df_dist_numericals = pd.concat(df_list, axis=0)
            for index, row in df_dist_numericals.iterrows():
                df_ref = self.df_dist_baseline[self.df_dist_baseline['feature'] == row['feature']]
                for item in numericals_check_list:
                    rel_distance = (row[item] - df_ref[item].values[0])/(df_ref[item].values[0])
                    if abs(rel_distance) > threshold:   
                        dist_verification_dict['feature'].append(row['feature'])
                        dist_verification_dict['property'].append(item)
                        dist_verification_dict['baseline'].append(df_ref[item].values[0])
                        dist_verification_dict['current'].append(row[item])
                        dist_verification_dict['ratio'].append(row[item]/df_ref[item].values[0])
                        dist_verification_dict['distance'].append(rel_distance)

        # Running verification for categorical variables
        df_list = []
        for ctypes in self.categoricals:
            df_list.append(self.df_dist[self.df_dist['type']==ctypes])

        if len(df_list) > 0:
            df_dist_categoricals = pd.concat(df_list, axis=0)
            for index, row in df_dist_categoricals.iterrows():
                
                df_ref = self.df_dist_baseline[self.df_dist_baseline['feature'] == row['feature']]

                # Checking for category deviations from baseline
                categories = set(row['category'])
                baseline_categories = set(df_ref['category'].to_list()[0])

                missing = baseline_categories.difference(categories)
                new = categories.difference(baseline_categories)
                intersec = baseline_categories.intersection(categories)

                if missing | new:
                    dist_verification_dict['feature'].append(row['feature'])
                    dist_verification_dict['property'].append('category')
                    dist_verification_dict['baseline'].append(baseline_categories)
                    dist_verification_dict['current'].append(categories)
                    dist_verification_dict['ratio'].append(np.nan)
                    dist_verification_dict['distance'].append(np.nan)
                    
                # Checking frequency shift for persistent data
                if intersec:
                    max_rel_dist = 0
                    ref_counter = 0

                    cross_ref_dict = {
                        'category':[],
                        'baseline':[],
                        'current':[]
                    }

                    for cat in intersec:
                        cross_ref_dict['category'].append(cat)
                        baseline_index = df_ref['category'].to_list()[0].index(cat)
                        current_index = row['category'].index(cat)
                        cross_ref_dict['baseline'].append(df_ref['frequency'].to_list()[0][baseline_index])
                        cross_ref_dict['current'].append(row['frequency'][current_index])   

                    for counter in range(len(cross_ref_dict['category'])):
                        rel_distance = (cross_ref_dict['current'][counter] - cross_ref_dict['baseline'][counter] ) # /(cross_ref_dict['baseline'][counter])
                        if abs(rel_distance) >= abs(max_rel_dist):
                            max_rel_distance = rel_distance
                            ref_counter = counter

                    if abs(max_rel_distance) > threshold:
                        dist_verification_dict['feature'].append(row['feature'])
                        dist_verification_dict['property'].append('frequency')
                        dist_verification_dict['baseline'].append(df_ref['frequency'].to_list()[0])
                        dist_verification_dict['current'].append(row['frequency'])
                        dist_verification_dict['ratio'].append(cross_ref_dict['current'][ref_counter]/cross_ref_dict['baseline'][ref_counter])
                        dist_verification_dict['distance'].append(max_rel_distance)

        # Running verification for datetime variables
        df_list = []
        for ctypes in self.dates:
            df_list.append(self.df_dist[self.df_dist['type']=='date-time'])
        
        if len(df_list) > 0:
            df_dist_dates = pd.concat(df_list, axis=0)
            for index, row in df_dist_dates.iterrows():
                df_ref = self.df_dist_baseline[self.df_dist_baseline['feature'] == row['feature']]
                if row['date-format'] != df_ref['date-format'].values[0]:
                    dist_verification_dict['feature'].append(row['feature'])
                    dist_verification_dict['property'].append('date-format')
                    dist_verification_dict['baseline'].append(df_ref['date-format'].values[0])
                    dist_verification_dict['current'].append(row['date-format'])
                    dist_verification_dict['ratio'].append(np.nan)
                    dist_verification_dict['distance'].append(np.nan)
        
        if dist_verification_dict:
            self.dist_verification_df = pd.DataFrame(dist_verification_dict)
            self.dist_verification_df['version'] = [self.gen_date]*len(self.dist_verification_df)
            self.verbose_list = self.verbose_list + ['-'*60] + \
                ['Production feature distribution doesn\'t match baseline from training.'] + \
                [self.dist_verification_df.head(5)] + \
                ['Use \'DataMonitor().dist_verification_df\' to see detailed information.'] + ['-'*60]
            return
        else:
            self.verbose_list = self.verbose_list + ['-'*60] + \
                ['Production feature distribution matches baseline from training.'] + ['-'*60]
            return


    def make_baseline(self, X):

        """
        Creates baseline for all selected monitors.

        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        """

        for monitor in self.selected_monitors:
            if monitor == 'input':
                self.create_input_baseline(X)
            elif monitor == 'distribution':
                self.create_distribution_baseline(X)

        print('Successfully created baseline for data monitor.')

        return


    def run_monitor(
        self, 
        X,
        threshold:float=0.15,
        verbose:bool=True
        ):

        """
        Confronts input's informations to the baseline 
        for all selected monitors.
        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        - `threshold (float)`: deviation threshold for distribution shift. \n
        - `verbose (bool)`: True if user wants to print the warning list. \n
        """
        
        for monitor in self.selected_monitors:
            if monitor == 'input':
                self.compare_input_to_baseline(X)
            elif monitor == 'distribution':
                self.compare_distribution_to_baseline(X, threshold=threshold)

        if verbose == True:
            print_lists(self.verbose_list)

        return

    
    def probe(
        self,
        X:pd.DataFrame,
        keep_ratio:float=1.0,
        shuffle_bool:bool=True,
        path:str=os.path.dirname(os.getcwd()) + '/artifacts/monitor/data'
        ):

        """
        Creates a probe point to collect data in validation/production sets.
        The saved data can then be used to confront production and training data
        and generate monitoring reports with self.run_verification_from_probe()
        function.
        ---
        Variables:

        - `X (pd.DataFrame)`: input dataset. \n
        - `keep_ratio (float)`: normalized percentage of data to keep. \n
        - `shuffle_bool (bool)`: True if user wants to get samples at random. \n
        - `path (str)`: path to directory where output dataframe will be saved. \n
        """

        self.probe_dir = path

        if (keep_ratio!=1.0) & (shuffle_bool==True):
            rows_to_keep = int(len(X)*keep_ratio)
            df_keep = shuffle(X)
            df_keep = df_keep.iloc[0:rows_to_keep]
        else:
            df_keep = X

        if os.path.exists(path) == False:
            Path(path).mkdir(parents=True, exist_ok=True)

        label = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        df_keep.to_parquet(path+'/data-monitor-'+label+'.parquet', compression='gzip')

        return


    def run_monitor_with_probe_data(
        self, 
        threshold:float=0.15,
        verbose:bool=True
        ):

        """
        Confronts data collected by probe to the baseline 
        for all selected monitors.
        ---
        Variables:

        - `threshold (float)`: deviation threshold for distribution shift. \n
        - `verbose (bool)`: True if user wants to print the warning list. \n
        """

        if not self.probe_dir:
            print('DataMonitor().probe_dir doesn\'t exist.')
            return 

        files = os.listdir(self.probe_dir)
        files = [self.probe_dir+'/'+file for file in files if 'data-monitor' in file]
        
        if not files:
            print('No references in DataMonitor().probe_dir = ', self.probe_dir)
            return

        df_list = []
        for file in files:
            df_list.append(pd.read_parquet(file))
        dataset = pd.concat(df_list, axis=0)

        self.run_monitor(
            X=dataset,
            threshold=threshold,
            verbose=verbose
            )

        return


    def save_artifacts(
        self,
        artifact_id:str='default',
        path:str=os.path.dirname(os.getcwd()) + '/artifacts/monitor',
        keep_old:bool=False
        ):

        """
        Save class artifacts.

        ---
        Variables:

        - `artifact_id(str)`: artifact identification - default as generated time. \n
        - `path (str)`: path to directory where output file will be saved. \n
        - `keep_old (bool)`: use True if user wants to keep previous artifacts. \n
        """

        if os.path.exists(path) == False:
            Path(path).mkdir(parents=True, exist_ok=True)

        if keep_old == False:
            old_artifacts = os.listdir(path)
            old_artifacts = [file for file in old_artifacts if 'data-monitor' in file]
            for file in old_artifacts:
                os.remove(path+'/'+file)

        if artifact_id == 'default':
            artifact_id = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')

        with open(path+'/data-monitor-'+artifact_id+'.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

        return


    def load_artifacts(
        self,
        artifact_id:str='latest',
        path:str=os.path.dirname(os.getcwd()) + '/artifacts/monitor'
        ):

        """
        Load class artifacts.
        
        ---
        Variables:

        - `artifact_id(str)`: artifact identification - default as latest file in path. \n
        - `path (str)`: path to directory from where input file is loaded. \n
        """

        if os.path.exists(path) == False:
            print('No reference for data monitor in ', path, '.')

        files = os.listdir(path)
        files = [path+'/'+file for file in files if 'data-monitor' in file]
        if artifact_id == 'latest':
            selected_file = max(files, key=os.path.getctime)
        else:
            selected_file = [file for file in files if artifact_id in file]
            selected_file = selected_file[0]
            if selected_file == False:
                print('No file obtained used artifact_id = ', artifact_id, 'in dir ', path)
                return 

        with open(selected_file, 'rb') as file:
            class_obj = pickle.load(file)

        return class_obj


    def list_warnings(self):

        """
        Collect and list all warnings from data monitor.

        """

        self.shifted_features = []
        self.warning_list.append('-'*60)

        for monitor in self.selected_monitors:
            if monitor == 'input':
                obj = self.input_status_df
                self.warning_list.append('INPUT MONITOR')
            elif monitor == 'distribution':
                obj = self.dist_verification_df
                self.warning_list.append('DISTRIBUTION MONITOR')

            if isinstance(obj, pd.DataFrame):
                for var in set(obj['feature']):
                    if self.warning_threshold[monitor][0](len(obj[obj['feature']==var]),self.warning_threshold[monitor][1]):
                        self.warning_list.append('* '+str(var))
                        self.shifted_features.append(var)

            self.warning_list.append('-'*60)

        for i in self.warning_list:
            print(i)

        if self.warning_list:
            print('Adjustments or model retraining might be advised.')
            print('For the detailed information use DataMonitor().input_status_df (input monitor) and DataMonitor().dist_verification_df (distribution monitor).')
            print('A list containing all shifted features can be accessed with \'DataMonitor().shifted_features\'.')

        self.shifted_features = list(sorted(set(self.shifted_features)))
        
        return


############################################################################################
################################### AUXILIARY FUNCTIONS ####################################
############################################################################################


def get_dtypes(X) -> dict:

    """
    Creates dictionary containing features' dtypes.

    ---
    Variables:

    - `X (pd.DataFrame)`: input dataset. \n
    """

    assert isinstance(X, pd.DataFrame)

    var_type = {
        'float':[],
        'int':[],
        'str':[],
        'category':[],
        'date-time':[]
    }

    for column in list(X.columns):
        if X[column].dtype in ['float64', 'float32', 'float16']:
            var_type['float'].append(column)
        elif X[column].dtype in ['int64','int32','int16','int8']:
            var_type['int'].append(column)
        elif X[column].dtype in ['str','string_','object']:
            var_type['str'].append(column)
        elif X[column].dtype.name in ['category']:
            var_type['category'].append(column)
        if X[column].dtype in ['datetime64[ns]']:
            var_type['date-time'].append(column)

    return var_type


def apply_dtypes(
    X:pd.DataFrame,
    var_type:dict
    ) -> pd.DataFrame:

    """
    Apply dtypes from dictionary - referenced from 'get_dtypes' function.

    ---
    Variables:

    - `X (pd.DataFrame)`: input dataset. \n
    - `var_type (dict)`: dictionary containing variables' dtypes reference. \n
    """

    output_X = X.copy()

    for col_type in ['float','int']:
        if var_type[col_type]: 
            output_X[var_type[col_type]] = X[var_type[col_type]].astype(col_type)

    for col_type in ['str']:
        if var_type[col_type]: 
            output_X[var_type[col_type]] = output_X[var_type[col_type]].fillna('missing', axis=1)
            output_X[var_type[col_type]] = X[var_type[col_type]].astype(col_type)

    return output_X


def print_lists(content:list):

    """
    Prints all items in a list of arguments.

    ---
    Variables:

    - `content (list)`: list containing information that will be printed sequentially. \n
    """

    for i in content:
        print(i)

    return