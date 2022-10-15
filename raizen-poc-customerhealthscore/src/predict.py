import pandas as pd
import pytz
from datetime import datetime as dt
import os
import joblib
from glob import glob
from time import time
import uuid
from sqlalchemy import create_engine

from guara.modeling.supervised_modelz import *
from guara.feature_engineering.pipeline_modules import *
from guara.feature_engineering.window_features import *
from guara.feature_selection.feature_selection import *

import utils
import get_data

class Predictor():
    def __init__(self):
        return
    
    def get_prediction_data(self, date_min, date_max):
        utils.log(f'Retrieving data with {date_min} <= date < {date_max}')
        
        getdata = get_data.GetData(date_min, date_max)

        X = getdata.get_raw()
        getdata.close_connection()
        
        return X
    
    def load_artifacts(self):
        try:
            production_id = glob('/usr/app/artifacts/production/*')[0].split('/')[-1]
            utils.log(f'Production Experiment: {production_id}')
        except: 
            utils.log('No experiment in production, exiting...')
            exit()
            
            
        model_path = f'/usr/app/artifacts/production/{production_id}/model_{production_id}.joblib'
        pipe_path = f'/usr/app/artifacts/production/{production_id}/pipeline_{production_id}.joblib'    
        
        self.model = joblib.load(model_path)
        utils.log(f'Model Loaded')
        
        self.pipeline = joblib.load(pipe_path)
        utils.log(f'Pipeline Loaded')
        
        self.production_id = production_id
        return
    
    def make_predictions(self, X, ids):
        md = SupervisedModelz('lgbm', 'regression')
        md.model = self.model
        
        initial_shape = X.shape
        ti = time()
        X = self.pipeline.transform(X)
        self.t_pipe = time() - ti
        utils.log(f'pipeline applied in {round(self.t_pipe, 5)} seconds')
        utils.log(f'X transformed: {initial_shape} -> {X.shape}')
        
        
        ti = time()
        initial_shape = X.drop(ids, axis = 1).shape
        self.pred = md.predict(X.drop(ids, axis = 1))
        self.t_model = time() - ti
        utils.log(f'model applied in {round(self.t_model, 5)} seconds')
        utils.log(f'y transformed: {initial_shape} -> {self.pred.shape}')
        
        self.Xids = X[ids].copy()
    
        return self.pred
    
    def make_predictions_table(self):
        
        Xpred = self.Xids.copy()
        Xpred['predict'] = self.pred
        Xpred['predict_to'] = Xpred['date'] + pd.Timedelta(minutes = 5)
        Xpred['id_model'] = f'model_{self.production_id}'
        Xpred['id_pipeline'] = f'pipeline_{self.production_id}'
        Xpred['id_pred'] = [uuid.uuid4() for _ in range(Xpred.shape[0])]
        Xpred['ts_run'] = dt.now(tz=pytz.timezone('America/Sao_Paulo'))
        Xpred['model_runtime'] = self.t_model
        Xpred['pipeline_runtime'] = self.t_pipe
        Xpred['id_batch'] = uuid.uuid4()
        Xpred['model_status'] = 'production'
        

        return Xpred
    
    def save_predicions_table(self, Xpred):
        
        engine = create_engine(f'postgresql://medicao_user:medicao_pass@127.0.0.1:5432/medicao_indireta_db')
        Xpred.to_sql('predicoes', engine, if_exists='replace', index=False)
        #Xpred.to_sql('predicoes', engine, if_exists='append', index=False)
        
        #Xpred.to_csv('/usr/app/data/pred.csv', index = False)
        utils.log('predictions saved')
        return
    


if __name__ == '__main__':
    
    date_min = '2019-06-01'
    date_max = '2019-08-01'
    ids = ['date', 'id', 'equip']
    
    predictor = Predictor()
    X = predictor.get_prediction_data(date_min, date_max)
    predictor.load_artifacts()
    _ = predictor.make_predictions(X, ids)
    Xpred = predictor.make_predictions_table()
    predictor.save_predicions_table(Xpred)
    
    utils.log('=============== END ===============')
    
    
    
    

    
    
    
    
    
