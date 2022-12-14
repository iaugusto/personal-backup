# modelagem 
import lightgbm as lgb
#import optuna.integration.lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_log_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score , roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from statsmodels.api import OLS

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
import datetime
import pickle
import warnings
import os
from time import time
from tabulate import tabulate
import itertools

import plotly.express as px
import shap
import optuna

## MSF: model selection function
##      serve para direcionar para funcoes especificas de cada modelo
##      sao as funcoes principais chamadas



class SupervisedModelz():
    
    def __init__(self, model_type, objective):
        
        sns.set_style("darkgrid")

        # available evaluation functions
        self.eval_functions = {

            'MAE':mean_absolute_error,
            'MSE':mean_squared_error,
            'MeAE':median_absolute_error,
            'R2':r2_score,
            'MSLE':mean_squared_log_error,
            'MAPE':self.mean_absolute_percentage_error, 
            'Accuracy':accuracy_score,
            'Recall':recall_score,
            'Precision':precision_score,
            'F1':f1_score,
            'ROC_AUC':roc_auc_score 
        }

        self.available_models = ['lgbm', 'random_forest', 'linear']
        self.available_objectives = ['binary', 'regression']


        assert (objective in self.available_objectives), f'objetivo nao encontrado! objetivos disponiveis:{self.available_objectives}'
        assert (model_type in self.available_models), f'modelo nao encontrado! modelos disponiveis:{self.available_models}'

        # set objective of the model (regression, binary)
        self.objective = objective
        # set model tipe (lgbm, xgboost,...)
        self.model_type = model_type


        
        
        return
    
    
    ###############################################################
    ########### TRAIN
    ###############################################################
    
    def fit(self, Xtrain, Xval, ytrain, yval, params, verbose=True):
        """
        Main function to call model-specific fit functions
        Xtrain: pd.DataFrame
        Xval: pd.DataFrame
        ytrain: pd.Series
        yval: pd.Series
        params: dict of hiperparameters for the model
        return: fitted model (original)
        """

        # MSF
        if verbose:
            print(f'Treinando o modelo {self.model_type.upper()}, objetivo {self.objective.upper()}... \n')
            t_i = time()
        
        # calls diferent training functions in the format train_{model_type}
        self.params = params
        if self.model_type == 'lgbm':
            self.model = self.train_lgbm(Xtrain, Xval, ytrain, yval, self.params)

        elif self.model_type == 'random_forest':
            if self.objective == 'regression':
                self.model = self.train_rf_regressor(Xtrain, Xval, ytrain, yval, self.params)
            elif self.objective == 'binary' or 'multiclass':
                self.model = self.train_rf_classifier(Xtrain, Xval, ytrain, yval, self.params)
        
        elif self.model_type == 'linear':
            if self.objective == 'regression':
                self.model = self.train_linear_regression(Xtrain, ytrain, self.params)
            elif self.objective == 'binary':
                self.model = self.train_logistic_regression(Xtrain, ytrain, self.params)
        
            else: print('n??o existem modelos para essa configuracao')
        
        
        
        else: print('n??o existem modelos para essa configuracao')
        
        if verbose:
            train_time = round((time()-t_i)/60,2)
            print(f'Modelo treinado em {train_time} minutos \n')

        return self.model
    
       
    def train_lgbm(self, Xtrain, Xval, ytrain, yval, params):
        """
        fit LightGBM model
        Xtrain: pd.DataFrame
        Xval: pd.DataFrame
        ytrain: pd.Series
        yval: pd.Series
        params: dict of hiperparameters for the model
        return: fitted model
        """

        # makes lgbm specific data type
        lgbtrain = lgb.Dataset(Xtrain, 
                           label=ytrain)

        lgbval = lgb.Dataset(Xval, 
                         label=yval)
        
        # make params an atribute
        self.params = params
        # lgbm needs the objective as parameter
        self.params['objective'] = self.objective
        # train info dict
        self.train_info = dict()
        
        model = lgb.train(
            
            self.params, 
            lgbtrain,
            #num_boost_round = 5_000, 
            #fobj=loglikelihood,
            valid_sets=[lgbtrain, lgbval], 
            early_stopping_rounds=100,
            evals_result = self.train_info,
            verbose_eval=False
            
            #feval = rmsle_eval,
            
        )
        
        return model


    def train_rf_regressor(self, Xtrain, Xval, ytrain, yval, params):
            """
            fit RandomForestRegressor
            Xtrain: pd.DataFrame
            Xval: pd.DataFrame
            ytrain: pd.Series
            yval: pd.Series
            params: dict of hiperparameters for the model
            return: fitted model
            """

            # make params an atribute
            self.params = params
            
            model = RandomForestRegressor(random_state=0, **params).fit(Xtrain, ytrain)
            
            return model


    def train_rf_classifier(self, Xtrain, Xval, ytrain, yval, params):
            """
            fit RandomForestClassifier
            Xtrain: pd.DataFrame
            Xval: pd.DataFrame
            ytrain: pd.Series
            yval: pd.Series
            params: dict of hiperparameters for the model
            return: fitted model
            """

            # make params an atribute
            self.params = params
            
            model = RandomForestClassifier(random_state=0, **params).fit(Xtrain, ytrain)
            
            return model


    def train_logistic_regression(self, Xtrain, ytrain, params):
            """
            fit sklearns LogisticRegression
            Xtrain: pd.DataFrame
            Xval: pd.DataFrame
            ytrain: pd.Series
            yval: pd.Series
            params: dict of hiperparameters for the model
            return: fitted model
            """

            # make params an atribute
            self.params = params
            
            model = LogisticRegression(random_state=0, **params).fit(Xtrain, ytrain)
            
            return model


    def train_linear_regression(self, Xtrain, ytrain, params):
            """
            fit statsmodels.regression.linear_model.OLS
            Xtrain: pd.DataFrame
            ytrain: pd.Series
            params: dict of hiperparameters for the model
            return: fitted model
            """

            # make params an atribute
            self.params = params
            
            model = OLS(ytrain, Xtrain, **params).fit()
            
            return model


    ###############################################################
    ############# PREDICT
    ###############################################################
    
    def predict(self, X):
        """
        Main function to call model-specific predict functions
        X: pd.DataFrame
        return: predictions
        """
        # MSF
        if self.model_type == 'lgbm':
            pred = self.model.predict(X, num_iteration= self.model.best_iteration)
        
        elif self.model_type == 'random_forest':
            if self.objective == 'regression':
                pred = self.model.predict(X)
            elif self.objective == 'binary' or 'multiclass':
                pred = self.model.predict_proba(X)
                pred = np.array([i[1] for i in pred])
        
        elif self.model_type == 'linear':
            if self.objective == 'regression':
                pred = self.model.predict(X).values
            elif self.objective == 'binary' or 'multiclass':
                pred = self.model.predict_proba(X)
                pred = np.array([i[1] for i in pred])
    
        return pred
    
    
    ################################################################
    ############## PLOTS TRAIN PERFORMANCE
    ################################################################
    
    def plot_train_info(self):
        """
        Main function to call model specific training info
        return:
        """

        # MSF 
        if self.model_type == 'lgbm':
            self.plot_train_info_lgbm()

        elif self.model_type == 'random_forest':
            print(f'nao ha plots disponiveis para {self.model_type}')
        
        elif self.model_type == 'linear' and self.objective == 'regression':
            print(self.model.summary())
                
        elif self.model_type == 'linear' and self.objective == 'binary':
            print(f'nao ha plots disponiveis para regressao logistica')
        
        return    
            
        
        
    def plot_train_info_lgbm(self):
        """
        LGBM training plots
        return:
        """
        
        # get loss name, training and val lists from lgbm returns
        loss_name = list(self.train_info['training'].keys())[0]
        train = list(self.train_info.keys())[0]
        val = list(self.train_info.keys())[1]
        best_iteration = self.model.best_iteration
        
        print(f'Model best iteration: {best_iteration}')
        print(f'\n{loss_name.upper()} loss:')
        print(f'Train: {round(self.train_info[train][loss_name][best_iteration-1], 4)}')
        print(f'Val  : {round(self.train_info[val][loss_name][best_iteration-1], 4)}')
        
        # plot train losses
        plt.figure(figsize = (7,5))
        plt.plot(self.train_info[train][loss_name])
        plt.plot(self.train_info[val][loss_name])

        #plt.title('', fontsize= 20)
        plt.ylabel(loss_name, fontsize= 13)
        plt.xlabel('epoch', fontsize= 13)
        plt.legend([train, val], loc='upper right', fontsize= 13)

        return 
    
    
    ####################################################################
    ######### SHAP VALUES
    ####################################################################

    def init_shap(self, X):
        '''
        Get SHAP values dataframe, only available for tree-based models
        X: pd.DataFrame - the data wich will be generated SHAP dataframe
        
        return: SHAP dataframe
        '''
        # init shap
        shap.initjs()

        # get shap explainer
        explainer = shap.TreeExplainer(self.model)
        # make shap values matrix
        shap_values = explainer.shap_values(X)


        if self.objective == 'binary':
            # when applying shap to binary data, a list of two matrix is produced
            # the shap values matrix with respect to label 0
            # the shap values matrix with respect to label 1
            # in this case we select the last in order to facilitate analisys
            self.shap_ = shap_values[1]      
        elif self.objective == 'regression':
            # the shap matrix for regression is straightfoward
            self.shap_ = shap_values

        # makes shap dataframe
        self.df_shap = pd.DataFrame(self.shap_, columns = X.columns)
        
        return self.df_shap


    def shap_importance(self, X):
        '''
        Plots 2 shap importances plots: scatter and bar
        X: pd.DataFrame - the data wich will be generated SHAP dataframe
        
        return: pd.Series - SHAP importances
        '''

        shap.summary_plot(self.shap_, X, plot_size = 1, alpha = 0.5)
        shap.summary_plot(self.shap_, X, plot_type="bar")

        return self.df_shap.abs().mean().sort_values(ascending = False)


    def shap_dependence(self, X, var_1, var_2):
        '''
        Plots shap dependence, to analyse the shap values on the interaction of two variables
        X: pd.DataFrame - the data wich will be generated SHAP dataframe
        var_1: str - primary variable (will be displayed shap values)
        var_2: str - secondary variable
        
        return: 
        '''
        shap.dependence_plot(var_1, self.shap_, X, interaction_index=var_2)

    #####################################################################
    ######### EVALUATION TABLE
    #####################################################################

    def metrics_print(self, true, estimation, metrics):
        '''
        Print metrics values for some true, estimation arrays
        true: np.array true values
        estimation: np.array estimation of the true values from the model (predict)
        metrics: list of strings for the metrics available 
        '''
        assert type(true) == np.ndarray
        assert type(estimation) == np.ndarray
        assert type(metrics) == list

        
        results = {}
        for metric in metrics:
            if self.objective == 'regression': 
                results[metric] = self.eval_functions[metric](true, estimation)
            elif self.objective == 'binary': 
                results[metric] = self.eval_functions[metric](true, np.where(estimation < self.treshold , 0, 1))
                
            
        results = pd.DataFrame(results, index = [''])

        print(tabulate(results,
                       tablefmt="github",
                       floatfmt = '.4f', 
                       headers="keys", 
                       showindex= False, 
                       stralign = 'center'
                      ))

        return results


    #####################################################################
    ######### EVALUATION PLOTS
    #####################################################################
    
    def evaluation_plots(self, ytrain, ytrain_pred, yval, yval_pred):
        '''
        directs the plotting functions depending on the objective settled.
        ytrain: np.array training response
        ytrain_pred: np.array predicted training response
        yval: np.array validation response
        yval_pred: np.array predicted validation response
        
        '''
        
        # MSF

        assert type(ytrain) == np.ndarray
        assert type(ytrain_pred) == np.ndarray
        assert type(yval) == np.ndarray
        assert type(yval_pred) == np.ndarray


        if self.objective == 'regression':
            self.eval_regression(ytrain, ytrain_pred, yval, yval_pred)

        elif self.objective == 'binary':
            self.eval_binary(ytrain, ytrain_pred, yval, yval_pred)

        elif self.objective == 'multiclass':
            self.eval_multiclass()


        return


    def eval_regression(self, ytrain, ytrain_pred, yval, yval_pred):
        
        nrows = 4
        ncols = 2
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(15,18))

        
        print('Remocao de nans')
        erro_treino = ytrain - ytrain_pred
        ltr = len(erro_treino)
        erro_treino = erro_treino[~np.isnan(erro_treino)]
        print('Treino: {}'.format(ltr - len(erro_treino)))
        ltr = len(erro_treino)

        erro_teste = yval - yval_pred
        lte = len(erro_teste)
        erro_teste = erro_teste[~np.isnan(erro_teste)]
        print('Validacao: {}'.format(lte - len(erro_teste)))
        lte = len(erro_teste)

        max_ = max(max(erro_treino), max(erro_teste))
        min_ = min(min(erro_treino), min(erro_teste))
        
        # training residual
        plt.subplot(nrows,ncols, 1)
        self.plot_residuals(erro_treino, 'Treino', min_, max_)
        
        # validation residual
        plt.subplot(nrows,ncols, 2)
        self.plot_residuals(erro_teste, 'Validacao', min_, max_)
        
        # distribui????o dos residuais do treino
        plt.subplot(nrows,ncols, 3)
        self.plot_residuals_distribution(erro_treino, min_, max_)

        # distribui????o dos residuais da validacao
        plt.subplot(nrows,ncols, 4)
        self.plot_residuals_distribution(erro_teste, min_, max_)

        # plot true vs predicted for train
        plt.subplot(nrows,ncols, 5)
        self.plot_true_x_predicted(ytrain, ytrain_pred)

        # plot true vs predicted for validation
        plt.subplot(nrows,ncols, 6)
        self.plot_true_x_predicted(yval, yval_pred)
        
        # plot distribuctions
        plt.subplot(nrows,ncols, 7)
        self.plot_distribuctions(ytrain, ytrain_pred)

        # plot distribuctions
        plt.subplot(nrows,ncols, 8)
        self.plot_distribuctions(yval, yval_pred)
             
        
        plt.show()

        return


    def eval_binary(self, ytrain, ytrain_pred, yval, yval_pred):

        nrows = 2
        ncols = 2
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(15,8))
        
        # AUC_ROC
        plt.subplot(nrows,ncols, 1)
        self.plot_roc_auc(ytrain, ytrain_pred, yval, yval_pred)

        # F1 treshold
        plt.subplot(nrows,ncols, 2)
        self.plot_f1_treshold(yval, yval_pred)
        
        # plot distribuctions
        plt.subplot(nrows,ncols, 3)
        plt.title('Treino', fontsize = 16)
        self.plot_distribuctions(ytrain, ytrain_pred)

        # plot distribuctions
        plt.subplot(nrows,ncols, 4)
        plt.title('Validacao', fontsize = 16)
        self.plot_distribuctions(yval, yval_pred)
        
        plt.tight_layout(pad=3.0)
        
        # create new grid for unbalanced graph sizes
        plt.figure(figsize = (15,5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 
        
        # binarize acording to treshold
        yval_pred_ = np.where(yval_pred < self.treshold , 0, 1)
        
        # plot confusion matrix
        plt.subplot(gs[0])
        self.plot_confusion_matrix(yval, yval_pred_)
        
        # plot distribuction for each class
        plt.subplot(gs[1])
        self.plot_labeled_distributions(yval, yval_pred)
        
        
        plt.show()
        print('')
        print(classification_report(yval, yval_pred_))

        return


    def eval_multiclass(self):
        #TODO

        return


    #####################################################################
    ##### PLOT FUNCTIONS
    #####################################################################
    
    def frange(self, start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step
            
    def plot_roc_auc(self, ytrain, ytrain_pred, yval, yval_pred):
        
        ns_probs = [0 for _ in range(len(yval))]
        ns_auc = roc_auc_score(yval, ns_probs)
        val_auc = roc_auc_score(yval, yval_pred)
        train_auc = roc_auc_score(ytrain, ytrain_pred)


        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(yval, ns_probs)
        train_fpr, train_tpr, _ = roc_curve(ytrain, ytrain_pred)
        val_fpr, val_tpr, _ = roc_curve(yval, yval_pred)

        plt.plot(ns_fpr, ns_tpr, label='Random')
        plt.plot(val_fpr, val_tpr, label=f'Validation ( AUC = {round(val_auc, 4)} )')
        plt.plot(train_fpr, train_tpr, label=f'Train ( AUC = {round(train_auc, 4)} )')

        plt.title('ROC Curve', fontsize = 16)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(fontsize = 12)
        
    def plot_f1_treshold(self, true, prediction):
        
        valores_f1 = []
        passo = [x for x in self.frange(0.1, 1, 0.02)]

        for i in passo:
            prediction_ = np.where(prediction < i , 0, 1)
            valores_f1.append(f1_score(true, prediction_))

        self.max_f1 = max(valores_f1) 
        self.treshold = passo[valores_f1.index(self.max_f1)]
         
        plt.title('Sensibilidade do treshold no F1 Score', fontsize = 16)
        plt.ylabel('F1 Score', fontsize = 12)
        plt.xlabel('Treshold', fontsize = 12)
        plt.tick_params(labelsize=12)

        
        plt.plot(passo, valores_f1)
        plt.axvline(x=self.treshold,  color = 'green', ls = '--' )
        #plt.ylim((0,1))
        #plt.xlim((0,1))

        plt.legend([f'F1 Score max: {round(self.max_f1, 3)}', 
                    f'Melhor treshold: {round(self.treshold, 2)}'], 
                   loc='upper right', fontsize= 12)
        
    def plot_residuals(self, erros, title, min_, max_):
        
        sns.scatterplot(x=range(len(erros)), y=erros, label="modelo", alpha = 0.7)
        sns.lineplot(x=range(len(erros)), y=[0]*len(erros), label="0", color ='red' )
        plt.ylim((min_,max_))
        plt.title(title, fontsize= 16)
        plt.ylabel('real - pred', fontsize= 12)
        
    def plot_residuals_distribution(self, erros, min_, max_):
        sns.distplot(erros, label='mean {:.2f}\nstd {:.2f}'.format(np.mean(erros), np.std(erros)))
        plt.xlim((min_,max_))
        plt.legend(fontsize= 12)

    def plot_true_x_predicted(self, true, predicted):

        sns.scatterplot(x = predicted, 
                        y = true, 
                        alpha = 0.1)

        min_ = min([min(true), min(predicted)])
        max_ = max([max(true), max(predicted)])

        sns.lineplot(x=np.linspace(min_, max_),
                     y=np.linspace(min_, max_),
                     label="x=y", 
                     color ='red' 
                    )

        plt.xlabel('Predicted')
        plt.ylabel('True')

        return
    
    def plot_distribuctions(self, true, predicted):
        
        sns.distplot(predicted,kde = False, label = 'pred')
        sns.distplot(true, kde = False, label = 'true')
        plt.legend()
        return
    
    def plot_confusion_matrix(self, true, pred,  normalize=False):
        
        classes = np.unique(true)
        cm = confusion_matrix(true, pred)
        

        # Configure Confusion Matrix Plot Aesthetics (no text yet) 
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix\ntreshold = {self.treshold}', fontsize=16)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)

        # Calculate normalized values (so all cells sum to 1) if desired
        if normalize:
            cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis]

        # Place Numbers as Text on Confusion Matrix Plot
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)

        plt.grid(b=None)
        


        return

    def plot_labeled_distributions(self, yval, yval_pred):
        j = pd.DataFrame({'true': yval, 'pred': yval_pred})
        sns.distplot(j[j['true'] == 0]['pred'], label = 'true = 0')
        sns.distplot(j[j['true'] == 1]['pred'], label = 'true = 1')
        plt.axvline(x=self.treshold,  color = 'green', ls = '--' )
        plt.legend(fontsize = 12)
        
        return
    
    
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
