import os 
import sys
import boto3
import joblib
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Pipeline de engenharia de features
#  --------------------
class feature_engineering_pipeline(BaseEstimator, TransformerMixin):
    
    """
    Classe para a construção e aplicação dos pipelines de engenharia e seleção de features.
    
    """
    
    
    # função de inicialização - contém informações gerais sobre a classe de engenharia de features
    def __init__(self):
        self.gen_time = datetime.now()
        self.class_name = 'feature engineering pipeline'
        return
    
    
    # método para extração de variáveis base
    # variáveis que serão usadas como referência nas transformações
    # sempre extrair informações apenas dos datasets de treino
    def fit(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame)
        
        self.target = 'weather'
        self.features = [i for i in X.columns if i != self.target]
        
        if 'date' in X.columns:
            self.initial_date = min(pd.to_datetime(X['date']))
        else:
            self.initial_date = X.iloc[0].name
            
        return self

    
    # método para aplicação das transformações nos dados - compreende casos de treinamento e validação
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)

        X = set_date_index(X)
        X = create_temperature_delta(X)
        X = create_temperature_mean(X)
        X = create_cyclical_from_time(X, initial_date=self.initial_date)
        X = encode_target(X)
        X = select_features(X)
        X = X[['weather','precipitation','wind','temp_delta','temp_mean','cyclical_date']]
        
        return X
#  --------------------


def save_pipeline(
    pipeline,
    bucket:str,
    path:str='framework-overview/artifacts/fe_pipeline/pipeline.joblib'
    ):
    
    """
    Função para salvar pipelines no Simple Storage Service com o formato joblib.

    ---

    Variáveis:
    - pipeline (sklearn pipeline obj): objeto do sklearn contendo o pipeline. \n
    - bucket (str): nome do bucket. \n
    - path (str): caminho complementar até o arquivo - incluí formato do arquivo (sufixo .csv). \n
    
    """
    
    s3_client = boto3.client('s3')
    
    with tempfile.TemporaryFile() as tf:
        joblib.dump(pipeline, tf)
        tf.seek(0)
        s3_client.put_object(Body=tf.read(), Bucket=bucket, Key=path)
    
    return


def read_pipeline(
    bucket:str,
    path:str='framework-overview/artifacts/fe_pipeline/pipeline.joblib'
    ):
    
    """
    Função para ler pipelines do Simple Storage Service com o formato joblib.

    ---

    Variáveis:
    - bucket (str): nome do bucket. \n
    - path (str): caminho complementar até o arquivo - incluí formato do arquivo (sufixo .csv). \n
    
    ---
    
    Outputs:
    - pipeline (sklearn pipeline obj): objeto do sklearn contendo o pipeline. \n
    
    """
    
    s3_client = boto3.client('s3')
    
    with tempfile.TemporaryFile() as tf:
        s3_client.download_fileobj(Fileobj=tf, Bucket=bucket, Key=path)
        tf.seek(0)
        pipeline = joblib.load(tf)
    
    return pipeline


# Funções criadas na etapa de análise exploratória - específicas do caso de uso
#  --------------------
def set_date_index(X):
    X['date'] = pd.to_datetime(X['date'])
    X = X.set_index('date')
    return X

def create_temperature_delta(X):
    X['temp_delta'] = X['temp_max'] - X['temp_min']
    return X

def create_temperature_mean(X):
    X['temp_mean'] = (X['temp_max'] + X['temp_min'])/2
    return X

def create_cyclical_from_time(X, initial_date):
    X_aux = X.copy()
    X_aux['time_delta'] = X_aux.index -  initial_date
    X['cyclical_date'] = X_aux['time_delta']
    X['cyclical_date'] = X['cyclical_date'].apply(lambda x:x.days)
    X['cyclical_date'] = np.cos(X['cyclical_date']*(2*np.pi/(365.25)))
    return X

def encode_target(Y):    
    translation_dict = {
        'snow':0,
        'rain':1,
        'drizzle':2,
        'fog':3,
        'sun':4
    }
    Y['weather'] = Y['weather'].apply(lambda x:translation_dict[x])
    return Y

def select_features(X):
    X = X.drop(columns=['temp_max','temp_min'])
    return X
#  --------------------