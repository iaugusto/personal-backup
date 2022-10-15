import os
import sys
import pickle
import joblib
import operator
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.metrics import explained_variance_score, mean_squared_log_error, median_absolute_error
from sklearn.metrics import balanced_accuracy_score, average_precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, log_loss, precision_score,  jaccard_score, roc_auc_score, brier_score_loss


class ModelMonitor():


    def __init__(self, objective):
        
        self.gen_date = datetime.now()
        self.dir = os.getcwd()
        self.objective = objective
        self.available_objectives = ['regression','binary']

        if self.objective not in self.available_objectives:
            print('No match for objective. \nTerminating sequence.')
            exit()

        self.processed_baseline_features = None
        self.baseline_features = None
        self.baseline_target = None
        self.baseline_gen_time = None
        self.pipeline_list = None
        self.baseline_model = None
        self.metrics_df = None
        self.variant_comparison_dict = None
        self.variant_comparison_df = None
        self.best_variant = None
        self.model_registry = pd.DataFrame(columns=['id','model'])
        
        if self.objective == 'regression':

            self.metrics = {
                'model_id':[],
                'version':[],
                'r2':[],
                'rsme':[],
                'sme':[],
                'mae':[],
                'exp_var':[],
                'max_error':[],
                'mean_squared_log':[],
                'median_abs_error':[],
                'mean_poisson_deviance':[],
                'mean_gamma_deviance':[]
            }

            self.metrics_score_basis = {
                'r2':[operator.ge,1],
                'rsme':[operator.le,1],
                'sme':[operator.le,1],
                'mae':[operator.le,1],
                'exp_var':[operator.ge,1],
                'max_error':[operator.le,1],
                'mean_squared_log':[operator.le,1],
                'median_abs_error':[operator.le,1],
                'mean_poisson_deviance':[operator.le,1],
                'mean_gamma_deviance':[operator.le,1]
            }

            self.metrics_descriptions = {
                'r2':[
                    'Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).',
                    'A constant model that always predicts the expected value of y, disregarding the input features, would get a R2 score of 0.0.'
                ],
                'rsme':[
                    'Square root of mean squared error regression loss.'
                ],
                'sme':[
                    'Mean squared error regression loss.'
                ],
                'mae':[
                    'Mean absolute error regression loss.'
                ],
                'exp_var':[
                    'Explained variance regression score function.',
                    'Best possible score is 1.0, lower values are worse.'
                ],
                'max_error':['max_error metric calculates the maximum residual error.'],
                'mean_squared_log':[
                    'Mean squared logarithmic error regression loss.'
                ],
                'median_abs_error':[
                    'Median absolute error regression loss.',
                    'Median absolute error output is non-negative floating point. The best value is 0.0.'
                ],
                'mean_poisson_deviance':[
                    'Mean Poisson deviance regression loss.',
                    'Poisson deviance is equivalent to the Tweedie deviance with the power parameter power=1.'
                ],
                'mean_gamma_deviance':[
                    'Mean Gamma deviance regression loss.',
                    'Gamma deviance is equivalent to the Tweedie deviance with the power parameter power=2.',
                    'It is invariant to scaling of the target variable, and measures relative errors.'
                ]
            }

        elif self.objective == 'binary':

            self.metrics = {
                'model_id':[],
                'version':[],
                'accuracy':[],
                'balanced_accuracy':[],
                'precision':[],
                'averaged_precision':[],
                'micro_recall':[],
                'macro_recall':[],
                'weighted_recall':[],
                'micro_f1':[],
                'macro_f1':[],
                'weighted_f1':[],
                'roc_auc':[],
                'micro_jaccard':[],
                'macro_jaccard':[],
                'weighted_jaccard':[],
                'log_loss':[],
                'neg_brier_loss':[]
            }

            self.metrics_score_basis = {
                'accuracy':[operator.ge,1],
                'balanced_accuracy':[operator.ge,1],
                'precision':[operator.ge,1],
                'averaged_precision':[operator.ge,1],
                'micro_recall':[operator.ge, 1],
                'macro_recall':[operator.ge,1],
                'weighted_recall':[operator.ge,1],
                'micro_f1':[operator.ge,1],
                'macro_f1':[operator.ge,1],
                'weighted_f1':[operator.ge,1],
                'roc_auc':[operator.ge,1],
                'micro_jaccard':[operator.ge,1],
                'macro_jaccard':[operator.ge,1],
                'weighted_jaccard':[operator.ge,1],
                'log_loss':[operator.le,1],
                'neg_brier_loss':[operator.le,1]
            }

            self.metrics_descriptions = {
                'accuracy':[
                    'In multilabel classification, this function computes subset accuracy:',
                    'The set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.', 
                    'In binary classification, this function is equal to the jaccard_score function.',
                    '',
                    'balanced_accuracy: The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.',
                    'It is defined as the average of recall obtained on each class.'
                ],
                'precision':[
                    'The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.', 
                    'The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0.',
                    '',
                    'averaged_precision: Compute average precision (AP) from prediction scores.',
                    'AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight'
                ],
                'recall':[
                    'The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.',
                    'The recall is intuitively the ability of the classifier to find all the positive samples.',
                    '',
                    'micro_recall: Calculate metrics globally by counting the total true positives, false negatives and false positives.',
                    'macro_recall: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.',
                    'weighted_recall: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).'
                ],
                'f1_score':[
                    'The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.',
                    '',
                    'micro_f1: Calculate metrics globally by counting the total true positives, false negatives and false positives.',
                    'macro_f1: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.',
                    'weighted_f1: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).'
                ],
                'roc_auc':[
                    'Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores'
                ],
                'jaccard_score':[
                    'The Jaccard index, or Jaccard similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets.',
                    'It\'s used to compare set of predicted labels for a sample to the corresponding set of labels in y_true.',
                    '',
                    'micro_jaccard: Calculate metrics globally by counting the total true positives, false negatives and false positives.',
                    'macro_jaccard: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.',
                    'weighted_jaccard: Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).'
                ],
                'log_loss':[
                    'This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks.',
                    'Defined as the negative log-likelihood of a logistic model that returns y_pred probabilities for its training data y_true.'
                ],
                'neg_brier_loss':[
                    'The smaller the Brier score loss, the better, hence the naming with “loss”.',
                    'The Brier score measures the mean squared difference between the predicted probability and the actual outcome.'
                ]
            }

        self.available_metrics = list(self.metrics_score_basis.keys())
        self.selected_metrics = self.available_metrics

        return


    def create_data_baseline(
        self,
        X:pd.DataFrame,
        Y:pd.DataFrame,
        pipeline:list=[],
        update_metrics:bool=False
        ):

        """
        Creates a baseline reference of the ground truth data for model performance comparison.
        Required to compare new variants to the baseline model. 
        ---
        Variables:

        - `X (pd.DataFrame)`: raw or processed features. \n
        - `Y (pd.DataFrame)`: targets. \n
        - `pipeline (list)`: list containing all pipelines used in the data preparation if X is raw data, 
            if user entered processed features in the X argument, leave empty - default = [ ]. \n
        """
        
        self.baseline_features = X
        self.baseline_target = Y

        if pipeline:
            self.pipeline_list = pipeline
            self.processed_baseline_features = apply_pipeline_to_baseline(
                dataset=self.baseline_features,
                pipeline=self.pipeline_list
            )
        else:
            self.processed_baseline_features = self.baseline_features

        self.baseline_gen_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if (update_metrics == True) & (len(self.model_registry) > 0):
            self.run_model_registry()

        return


    def get_performance_metrics(
        self, 
        ytrue, 
        ypred, 
        version:str='default',
        model_id:str='default',
        binary_threshold:float=0.5
        ):

        """
        Calculates all selected metrics from the model using the 
        self.processed_baseline_features data.
        ---
        Variables:

        - `ytrue (pd.DataFrame[column],list,np.array)`: groundtruth target. \n
        - `ypred (pd.DataFrame[column],list,np.array)`: predicted target. \n
        - `version (str)`: label assigned to the model monitor version. \n
        - `model_id (str)`: label assigned to the model. \n
        - `binary_threshold (float)`: threshold for binary classification positive, 
            if user wants to explicitly select one outside of the model object. \n
        """

        if model_id == 'default':
            model_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if version == 'default':
            version = self.baseline_gen_time

        if self.objective == 'regression':
            
            self.metrics['r2'].append(round(r2_score(ytrue, ypred),6))
            self.metrics['rsme'].append(round(mean_squared_error(ytrue, ypred, squared=False),6))
            self.metrics['sme'].append(round(mean_squared_error(ytrue, ypred, squared=True),6))
            self.metrics['mae'].append(round(mean_absolute_error(ytrue, ypred),6))
            self.metrics['exp_var'].append(round(explained_variance_score(ytrue, ypred),6))
            self.metrics['max_error'].append(round(max_error(ytrue, ypred),6))
            self.metrics['mean_squared_log'].append(round(mean_squared_log_error(ytrue, ypred),6))
            self.metrics['median_abs_error'].append(round(median_absolute_error(ytrue, ypred),6))
            self.metrics['mean_poisson_deviance'].append(round(mean_poisson_deviance(ytrue, ypred),6))
            self.metrics['mean_gamma_deviance'].append(round(mean_gamma_deviance(ytrue, ypred),6))

        elif self.objective == 'binary':

            ypred_classes = [1 if i>=binary_threshold else 0 for i in list(ypred)]

            self.metrics['accuracy'].append(round(accuracy_score(ytrue, ypred_classes),6))
            self.metrics['balanced_accuracy'].append(round(balanced_accuracy_score(ytrue, ypred_classes),6))
            self.metrics['precision'].append(round(precision_score(ytrue, ypred_classes),6))
            self.metrics['averaged_precision'].append(round(average_precision_score(ytrue, ypred_classes),6))
            self.metrics['micro_recall'].append(round(recall_score(ytrue, ypred_classes, average='micro'),6))
            self.metrics['macro_recall'].append(round(recall_score(ytrue, ypred_classes, average='macro'),6))
            self.metrics['weighted_recall'].append(round(recall_score(ytrue, ypred_classes, average='weighted'),6))
            self.metrics['micro_f1'].append(round(f1_score(ytrue, ypred_classes, average='micro'),6))
            self.metrics['macro_f1'].append(round(f1_score(ytrue, ypred_classes, average='macro'),6))
            self.metrics['weighted_f1'].append(round(f1_score(ytrue, ypred_classes, average='weighted'),6))
            self.metrics['roc_auc'].append(round(roc_auc_score(ytrue, ypred_classes),6))
            self.metrics['micro_jaccard'].append(round(jaccard_score(ytrue, ypred_classes,average='micro'),6))
            self.metrics['macro_jaccard'].append(round(jaccard_score(ytrue, ypred_classes,average='macro'),6))
            self.metrics['weighted_jaccard'].append(round(jaccard_score(ytrue, ypred_classes,average='weighted'),6))
            self.metrics['log_loss'].append(round(log_loss(ytrue, ypred_classes),6))
            self.metrics['neg_brier_loss'].append(round(brier_score_loss(ytrue, ypred),6))

        self.metrics['version'].append(version)
        self.metrics['model_id'].append(model_id)

        return


    def create_performance_reference(
        self,
        model,
        version:str='default',
        model_id:str='default',
        binary_threshold:float=0.5
        ):

        """
        Creates a reference of the inputed variant's performance.
        Required to compare new variants to the baseline model. 
        ---
        Variables:

        - `model`: variant. \n
        - `version (str)`: label assigned to the model monitor version. \n
        - `model_id (str)`: label assigned to the model. \n
        - `binary_threshold (float)`: threshold for binary classification positive, 
        if user wants to explicitly select one outside of the model object. \n
        """

        predictions = model.predict(self.processed_baseline_features)
        try:
            self.get_performance_metrics(
                self.baseline_target, 
                predictions, 
                version=version, 
                model_id=model_id,
                binary_threshold=model.treshold
                )
        except:
            self.get_performance_metrics(
                self.baseline_target, 
                predictions, 
                version=version, 
                model_id=model_id,
                binary_threshold=binary_threshold
                )

        self.metrics_df = pd.DataFrame(self.metrics)
        self.metrics_df.drop_duplicates(
            subset=self.metrics_df.columns[1:], 
            keep='first',
            inplace=True, 
            ignore_index=True
        )
        self.metrics = self.metrics_df.to_dict('list')              
                
        return

    
    def create_performance_baseline(
        self,
        model,
        X:pd.DataFrame,
        Y:pd.DataFrame,
        pipeline:list=[],
        binary_threshold:float=0.5
        ):

        """
        Creates a baseline reference of the production model performance.
        Required to compare new variants to the baseline model. 
        ---
        Variables:

        - `model`: baseline model. \n
        - `X (pd.DataFrame)`: raw or processed features. \n
        - `Y (pd.DataFrame)`: targets. \n
        - `pipeline (list)`: list containing all pipelines used in the data preparation if X is raw data, 
            if user entered processed features in the X argument, leave empty - default = [ ]. \n
        - `binary_threshold (float)`: threshold for binary classification positive, 
            if user wants to explicitly select one outside of the model object. \n
        """
        
        self.create_data_baseline(
            X=X,
            Y=Y,
            pipeline=pipeline,
            update_metrics=False
            )

        self.input_to_model_registry(
            model=model,
            model_id='baseline'
        )

        self.create_performance_reference(
            model=model,
            version=self.baseline_gen_time,
            model_id='baseline',
            binary_threshold=binary_threshold
            )

        return


    def input_to_model_registry(
        self,
        model,
        model_id:str='default'
        ):

        """
        Inputs model to model monitor registry.

        ---
        Variables:

        - `model`: model object to be inputed to model monitor's registry. \n
        - `model_id (str)`: label attributed to model object in the registry. \n
        """

        if model_id == 'default':
            model_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        sample = {
            'id':[model_id],
            'model':[model]
        }
        
        self.model_registry = pd.concat([self.model_registry,pd.DataFrame(sample)], axis=0, ignore_index=True)
        self.model_registry.drop_duplicates(keep='first', inplace=True, ignore_index=True)
        self.model_registry.reset_index(inplace=True, drop=True)          
                
        return


    def input_variants_from_artifacts(
        self,
        path:str=os.path.dirname(os.getcwd())+'/artifacts/variant/model'
        ):

        """
        Inputs models from `path` directory to model monitor.

        ---
        Variables:

        - `path (str)`: path to directory from where input file is loaded. \n
        - `sufix (str)`: sufix string which can be used for extension definition. \n
        """

        model_obj_list = os.listdir(path)
        for model_id in model_obj_list:
            model_loop = joblib.load(path+'/'+model_id)
            self.input_to_model_registry(
                model=model_loop,
                model_id=model_id
            )
                
        return


    def run_model_registry(self):

        """
        Creates performance references for all registered models.
        """

        if len(self.model_registry) == 0:
            print('self.model_registry is empty.')
            print('Please input models to the model registry using self.input_to_model_registry()')
            print('or self.input_models_from_artifacts()')
            return 

        for index, row in self.model_registry.iterrows():
            self.create_performance_reference(
                model=row['model'],
                model_id=row['id']
            )

        print('Model performance reference successfully updated with model registry.')

        return


    def select_metrics(self, metric_list):

        """
        Selects metrics to be used in the performance scoring system for the 
        variants' comparisons and the selection of the best overall model.
        ---
        Variables:

        - `metric_list (list)`: list containing metrics to be watched by the model performance monitor. \n
        """

        if type(metric_list)!=list:
            metric_list=[metric_list]

        local_metric_list = [i for i in metric_list if i in self.available_metrics]
        
        if not local_metric_list:
            print('None of the metrics listed in the input argument are available.')
            print('Metric system kept as default.')
            return 

        elif local_metric_list != metric_list:
            print('Some metrics in metric_list are not available.')
        
        self.selected_metrics = local_metric_list

        print('Updated performance monitor metrics to: \n', local_metric_list)

        return


    def select_best_model(self, version:str='latest'):

        """
        Selects best model according to the self.metric_score_basis dict and the 
        variants' selected metrics - self.selected_metrics object. 
        ---
        Variables:

        - `version (str)`: model monitor's baseline version. \n
        """

        if version == 'latest':
            version = sorted(set(self.metrics_df['version'].to_list()))[-1]
        
        if 'baseline' not in set(self.metrics_df['model_id']):
            print('No baseline entry found in self.metrics_df[\'model_id\'].')
            return None

        local_metrics_df = self.metrics_df[self.metrics_df['version']==version]

        ref_score = 0
        ref_variant = 'baseline'

        variant_list = set(local_metrics_df['model_id'].to_list())
        variant_list = [variant for variant in variant_list if variant != 'baseline']
        metric_list = self.selected_metrics

        for variant in variant_list:
            score = 0
            for metric in metric_list:
                baseline_value = local_metrics_df[local_metrics_df['model_id']==ref_variant][metric].values[0]
                variant_value = local_metrics_df[local_metrics_df['model_id']==variant][metric].values[0]
                if self.metrics_score_basis[metric][0](variant_value, baseline_value):
                    score=score+self.metrics_score_basis[metric][1]      
                else:
                    score=score-self.metrics_score_basis[metric][1] 
            if score >= ref_score:
                ref_variant = variant                

        self.best_variant = ref_variant

        return ref_variant


    def update_baseline_model(self, model_id:str='default'):

        """
        Updates baseline to specific variant or to the best selected variant if called without
        arguments.
        ---
        Variables:

        - `model_id (str)`: label associated to wanted variant in self.metrics_df object. \n
        """

        if (model_id!='default'):
            local_id = model_id 
        elif (model_id=='default') & (self.best_variant!=None):
            local_id = self.best_variant
            if local_id == 'baseline':
                print('Baseline was best variant.')
                print('No update needed.')
        else:
            print('Updated baseline to variant ', self.best_variant)
            return

        # updating model registry
        old_baseline_pos = self.model_registry.index[self.model_registry['id'] == 'baseline'].tolist()
        new_baseline_pos = self.model_registry.index[self.model_registry['id'] == local_id].tolist()
        self.model_registry['id'].iloc[old_baseline_pos] = self.baseline_gen_time
        self.basline_gen_time = local_id
        self.model_registry['id'].iloc[new_baseline_pos] = 'baseline'
        self.metrics = self.model_registry.to_dict('list')
        print('Updated model registry baseline to version ', local_id)
        
        # updating metrics_df
        old_baseline_pos = self.metrics_df.index[self.metrics_df['model_id'] == 'baseline'].tolist()
        new_baseline_pos = self.metrics_df.index[self.metrics_df['model_id'] == local_id].tolist()
        self.metrics_df['model_id'].iloc[old_baseline_pos] = self.baseline_gen_time
        self.basline_gen_time = local_id
        self.metrics_df['model_id'].iloc[new_baseline_pos] = 'baseline'
        self.metrics = self.metrics_df.to_dict('list')
        print('Updated metrics baseline to version ', local_id)

        return

    
    def describe_metrics(self):

        for key in self.metrics_descriptions.keys():
            print(key)
            description = self.metrics_descriptions[key]
            for sentence in description:
                print(sentence)
            print('-'*60)

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
            old_artifacts = [file for file in old_artifacts if 'model-monitor' in file]
            for file in old_artifacts:
                os.remove(path+'/'+file)

        if artifact_id == 'default':
            artifact_id = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')

        with open(path+'/model-monitor-'+artifact_id+'.pkl', 'wb') as file:
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
            print('No reference for model monitor in ', path, '.')

        files = os.listdir(path)
        files = [path+'/'+file for file in files if 'model-monitor' in file]
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


############################################################################################
################################### AUXILIARY FUNCTIONS ####################################
############################################################################################


def apply_pipeline_to_baseline(dataset, pipeline):

    """
    Applies pipeline to the features.
    Both objects are defined in create_performance_baseline function.
    """

    processed_dataset = dataset

    for pipe in pipeline:
        processed_dataset = pipe.transform(processed_dataset)

    return processed_dataset