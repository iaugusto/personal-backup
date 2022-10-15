import pandas as pd
import pytz
from datetime import datetime as dt
import os
import joblib

from guara.modeling.supervised_modelz import *
from guara.feature_engineering.pipeline_modules import *
from guara.feature_engineering.window_features import *
from guara.feature_selection.feature_selection import *

import utils
import get_data


class Train:
    
    def __init__(self):
        self.exp_name = dt.now(tz=pytz.timezone('America/Sao_Paulo')).strftime('%Y%m%d-%H%M%S')
        
        utils.log(f'\n================ TRAIN ================\n')
        utils.log(f'Train started with id {self.exp_name} \n')
    
    
    def get_raw_data(self, date_min, date_max):
        utils.log(f'GET DATA ----------------\n ')
        utils.log(f'Retrieving data with {date_min} <= date < {date_max}')
        
        getdata = get_data.GetData(date_min, date_max)

        X = getdata.get_raw()
        y = getdata.get_target()
        
        utils.log(f'date_min: {y.date.min()}, date_max: {y.date.max()}')
        
        getdata.close_connection()
    
        return X, y
    
    
    def make_train_validation(self, X, split_date):
        utils.log(f'make_train_validation ----------------\n ')
        
        Xtrain = X[(X.date <  split_date)].copy()
        Xval   = X[(X.date >= split_date)].copy()
        
        perc_train = Xtrain.shape[0]/X.shape[0]*100
        perc_val = Xval.shape[0]/X.shape[0]*100
        
        utils.log(f'Xtrain raw shape: {Xtrain.shape} ({round(perc_train, 2)} %)' )
        utils.log(f'Xval raw shape:   {Xval.shape}   ({round(perc_val, 2)} %)' )
        
        return Xtrain, Xval
        
    
    def fit_transform_pipeline(self, Xtrain, Xval):
        utils.log(f'fit_transform_pipeline ----------------\n ')
        
        
        self.pipeline = Pipeline([
    
            #('validate', ValidateDF()),
            ('group 5min', GroupDate()),
            ('filtra', FilterDF()),
            
                        
        ])

        Xtrain_processed = self.pipeline.fit_transform(Xtrain)
        Xval_processed = self.pipeline.transform(Xval)

        utils.log(f'Xtrain processed shape: {Xtrain_processed.shape}')
        utils.log(f'Xtrain processed shape: {Xval_processed.shape}')
        
        return Xtrain_processed, Xval_processed
     
    
    def organize_features_and_targets(self, Xtrain, Xval, y):
        utils.log(f'organize_features_and_targets ----------------\n ')
        
        # group target same way as features
        y = (
            y.set_index('date')
            .resample('5min')
            .agg({
                    'id': 'last',
                    'vibracao':'mean',
            })
            .reset_index()
        )

        y['vibracao'] = y['vibracao'].shift(1)

        # split according to X
        ytrain = (
            y
            .set_index('date')
            .reindex(index=Xtrain['date'])
            .reset_index()
            .query('vibracao > 0.1')
            .dropna()
        )

        yval = (
            y
            .set_index('date')
            .reindex(index=Xval['date'])
            .reset_index()
            .query('vibracao > 0.1')
            .dropna()
        )




        Xtrain = (
            
            Xtrain
            .set_index('date')
            .reindex(index=ytrain['date'])
            .reset_index()
        )


        Xval = (
            
            Xval
            .set_index('date')
            .reindex(index=yval['date'])
            .reset_index()
        )



        utils.log(f'Xtrain shape {Xtrain.shape}')
        utils.log(f'Xval shape   {Xval.shape}')
        utils.log(f'ytrain shape {ytrain.shape}')
        utils.log(f'yval shape   {yval.shape}')
        

        assert ytrain.shape[0] == Xtrain.shape[0]
        assert yval.shape[0] == Xval.shape[0]

        return Xtrain, Xval, ytrain, yval
    
    
    def hyperparameter_search(self, X, y):
        pass
    
    
    def fit_model(self, Xtrain, Xval, ytrain, yval):
        utils.log(f'fit model ----------------\n ')
        
        params = {

            'random_state': 501,
            'boosting_type': 'gbdt',
            'device_type': 'cpu',
            'learning_rate': 0.005,
            'colsample_bytree': 0.298,
            'max_depth': 890,
            'n_estimators': 1800,
            'sub_sample': 0.95,
            'num_leaves': 3,
            'min_child_samples': 164,
            
        }
        
        self.md = SupervisedModelz('lgbm', 'regression')
        self.model = self.md.fit(Xtrain, Xval, ytrain, yval, params)
        
        utils.log(f'Model fitted')
        
        self.md.plot_train_info()
        
        return self.model
        
        
    def evaluate(self, Xtrain, Xval, ytrain, yval):
        
        metrics = ['MAE', 'MeAE', 'R2', 'MSLE', 'MAPE']
        ytrain_pred = self.md.predict(Xtrain)
        yval_pred = self.md.predict(Xval)

        if type(ytrain) == pd.Series:
            ytrain = ytrain.values
        if type(yval) == pd.Series:
            yval = yval.values

        print('\n ============== Resumo metricas ============== \n')

        print('TREINO:\n')

        metrics_train = self.md.metrics_print(
            ytrain.clip(0, None), 
            ytrain_pred.clip(0, None), 
            metrics
        )

        print('\nVALIDACAO:\n')

        metrics_val = self.md.metrics_print(
            yval.clip(0, None), 
            yval_pred.clip(0, None), 
            metrics

        )
        
        print('\n')
        
        #self.md.evaluation_plots(ytrain, ytrain_pred, yval, yval_pred)
            
            
        return
        
    
    def save_pipe_and_model(self, artifacts_path = '/usr/app/artifacts/experiments/'):
        utils.log(f'SAVING ----------------\n ')
        
        
        path_to_experiment = artifacts_path + self.exp_name
        model_path = f'{path_to_experiment}/model_{self.exp_name}.joblib'
        pipeline_path = f'{path_to_experiment}/pipeline_{self.exp_name}.joblib'
        
        utils.log(f'Creating experiment directory in {path_to_experiment}')
        os.mkdir(path_to_experiment)
        
        
        
        joblib.dump(self.pipeline, pipeline_path )
        utils.log(f'Pipeline saved in: {pipeline_path}')
        
        joblib.dump(self.model, model_path )
        utils.log(f'Model saved in: {model_path}')
        
        

        
    
###########################################################


if __name__ == '__main__':
    
    ids = ['date', 'id', 'equip']
    train = Train()
    
    X, y = train.get_raw_data(date_min = '2015-01-01', date_max = '2020-01-01')
    
    Xtrain_, Xval_ = train.make_train_validation(X, split_date = '2019-09-01')
    
    Xtrain_, Xval_ = train.fit_transform_pipeline(Xtrain_, Xval_)
    
    Xtrain_, Xval_, ytrain, yval = train.organize_features_and_targets(Xtrain_, Xval_, y)

    
    Xtrain = Xtrain_.drop(ids, axis = 1).copy()
    Xval   = Xval_.drop(ids, axis = 1).copy()

    model = train.fit_model(Xtrain, Xval, ytrain['vibracao'], yval['vibracao'])
    
    train.evaluate(Xtrain, Xval, ytrain['vibracao'], yval['vibracao'])
    
    train.save_pipe_and_model(artifacts_path = '/usr/app/artifacts/experiments/')
    
    utils.log('\n=================== END ===================\n')