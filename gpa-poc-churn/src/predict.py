import os
import sys
import pytz
import uuid
import glob
import joblib
import datetime
import pandas as pd
from sklearn.pipeline import Pipeline

from guara.modeling.supervised_modelz import *

import utils
import get_data
import pipeline_modules


class Predictor():
    
    
    def __init__(self):
        return
    
    
    def get_raw_data(self):
        
        utils.log(f'GET DATA ----------------\n ')
        utils.log(f'Retrieving data from poc-gpa/data/processed folder')
        X= get_data.get_prediction_data()

        return X
    
    
    def load_artifacts(self):
        
        try:
            parent_dir = os.getcwd()
            print(f'parent dir: {parent_dir}/artifacts/staging/')
            production_id = glob.glob(parent_dir+'/artifacts/staging/*')[0].split('/')[-1]
            print(f'Production Experiment: {production_id}')
        except: 
            print('No experiment in production, exiting...')
            exit()
        
        model_path = parent_dir + f'/artifacts/staging/{production_id}/model.joblib'
        fe_pipe_path = parent_dir+ f'/artifacts/staging/{production_id}/fe_pipeline.joblib'
        fs_pipe_path = parent_dir+ f'/artifacts/staging/{production_id}/fs_pipeline.joblib'   
        
        self.model = joblib.load(model_path)
        utils.log(f'Model Loaded')
        
        self.fe_pipeline = joblib.load(fe_pipe_path)
        utils.log(f'Feature engineering pipeline loaded')
        
        self.fs_pipeline = joblib.load(fs_pipe_path)
        utils.log(f'Feature selection pipeline loaded')
        
        self.production_id = production_id
        return
    
    
    def make_predictions(self, X):
        
        md = SupervisedModelz('lgbm', 'binary')
        md.model = self.model
        utils.log('Successfully loaded model')

        initial_shape = X.shape
        X = self.fe_pipeline.transform(X)
        X = self.fs_pipeline.transform(X)
        utils.log('Successfully applied pipelines')
        utils.log(f'X transformed: {initial_shape} -> {X.shape}')

        self.pred = md.model.predict(X)
        
        return
    
    
    def make_predictions_table(self, X):
        
        Xpred = pd.DataFrame()
        Xpred['cod_cliente'] = X['cod_cliente'].copy()
        Xpred['churn_prediction'] = self.pred
        Xpred['prediction_time'] = datetime.datetime.now().strftime(format='%Y-%m-%d %H:%M:%S')
        Xpred['variant'] = f'{self.production_id}'
        Xpred['batch_id'] = str(uuid.uuid4())
        Xpred['model_status'] = 'poc'
        
        self.output_df = Xpred.copy()

        return
    
    
    def save_predicions_table(self):
        
        output_path = 'gs://gpa-churn/data/processed/output/predictions.parquet'
        self.output_df.to_parquet(
            output_path, 
            compression='gzip', 
            index = False
        )
        utils.log(f'predictions saved in {output_path}')
        return




###########################################################


if __name__ == '__main__':
    
    predictor = Predictor()
    X = predictor.get_raw_data()
    predictor.load_artifacts()
    predictor.make_predictions(X)
    predictor.make_predictions_table(X)
    predictor.save_predicions_table()
    
    utils.log('=============== END ===============')