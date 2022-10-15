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

from preprocess import preprocess_module
preproc = preprocess_module()


# Pipeline de engenharia de features
#  --------------------
class feature_engineering_pipeline(BaseEstimator, TransformerMixin):
    
    """
    Classe para a construção e aplicação dos pipelines de engenharia e seleção de features.
    
    """
    
    
    # função de inicialização - contém informações gerais sobre a classe de engenharia de features
    def __init__(
        self, 
        stage:str='dev',
        ):
        self.gen_time = datetime.now()
        self.class_name = 'feature engineering pipeline'
        self.description = 'feature engineering pipeline for the entity recognition engine - proj. Manobras'
        self.bucket = f'ons-ds-{stage}-entityrecognition'
        self.processed_path = 'mpo/mvp/data/processed/'
        self.incremental_path = 'mpo/mvp/data/raw/incremental/'
        return
    
    
    # método para extração de variáveis base
    # variáveis que serão usadas como referência nas transformações
    # sempre extrair informações apenas dos datasets de treino
    def fit(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame)
        
        create_equipments_samples(X)
        create_manualactions_samples(X)
        create_mathoperator_samples(X)
        create_operativestate_samples(X)
        create_powerplant_samples(X)
        create_substation_samples(X)
        create_valuewithunit_samples(X)
        create_inequalities_samples(X)
        
        return self
    
    
    # método para aplicação das transformações nos dados - compreende casos de treinamento e validação
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)

        output = create_trainset(
            bucket,
            self.processed_path,
            self.incremental_path
        )
        
        return output
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
def save_sample_to_s3(
    X,
    bucket:str,
    filepath:str
    ):

    sample_path = f's3://{bucket}/{filepath}'
    X.to_parquet(sample_path, compression='gzip')
    print(f'saved samples in {sample_path}')
    
    return

def create_equipments_samples(X):
    
    n_samples = len(X)
    n_batches = np.ceil(n_samples/100 + 1)
    batch_size = np.floor(n_samples/n_batches)
    intervals = list(np.arange(0, n_samples, batch_size).tolist())
    if intervals[-1] != len(X):
        intervals[-1] = len(X)
    
    samples_list = []
    for i in range(len(intervals)-1):
        text = X.copy().iloc[int(intervals[i]):int(intervals[i+1])]
        text = preproc.find_equipments(df=text, col='processed')
        text.astype('str').drop_duplicates(
            subset=['MpoCentro', 'processed', 'EQUIPAMENTO'], 
            keep='last', 
            inplace=True)
        samples = preproc.create_synthetic_samples(
            text,
            entity_col='EQUIPAMENTO',
            text_col='processed'
            )
        samples.drop_duplicates(inplace=True)
        samples_list.append(samples)
    print('Created samples_eq dataframe')
        
    samples = pd.concat(samples_list, axis=0)
    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/equipment_samples.parquet'
    )
    
    return

def create_manualactions_samples(X):
    
    df_manualactions = preproc.get_manual_actions(
        bucket='ons-ds-dev-entityrecognition',
        path='mpo/mvp/data/raw/manual_actions'
        )
    df_manualactions = preproc.preprocess_fragments(df_manualactions, 'actions')
    print('Preprocessed manual actions')

    text = X[['processed']].copy()
    text = preproc.find_manual_actions(
        text,
        df_manualactions
        )
    print('Found manual actions in fragments\' text')

    samples = preproc.create_synthetic_samples(
        text,
        entity_col='ACAO_MANUAL',
        text_col='processed'
        )
    samples.drop_duplicates(inplace=True)
    print('Created samples dataframe for manual actions entity')

    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/manualactions_samples.parquet'
    )
    
    return

def create_mathoperator_samples(X):
    
    text = X[['processed']].copy()
    text = preproc.find_mathoperator(df=text, col='processed')
    print('Found math operator in fragments\' text')

    samples = preproc.create_synthetic_samples(
        text,
        entity_col='OPERADOR_MATEMATICO',
        text_col='processed'
        )
    samples.drop_duplicates(inplace=True)
    print('Created samples dataframe for math operator entity')
    
    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/mathoperator_samples.parquet'
    )
    
    return

def create_operativestate_samples(X):
    
    text = X[['processed']].copy()
    text = preproc.find_operativestate(df=text, col='processed')
    print('Found operative state in fragments\' text')

    samples = preproc.create_synthetic_samples(
        text,
        entity_col='ESTADO_OPERATIVO',
        text_col='processed'
        )
    samples.drop_duplicates(inplace=True)
    print('Created samples dataframe for operative state entity')
    
    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/operativestate_samples.parquet'
    )
    
    return

def create_powerplant_samples(X):
    
    text = X[['processed']].copy()
    text = preproc.find_powerplant(df=text, col='processed')
    print('Found power plant in fragments\' text')

    samples = preproc.create_synthetic_samples(
        text,
        entity_col='USINA',
        text_col='processed'
        )
    samples.drop_duplicates(inplace=True)
    print('Created samples dataframe for power plant entity')
    
    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/powerplant_samples.parquet'
    )
    
    return

def create_substation_samples(X):
    
    text = X[['processed']].copy()
    text = preproc.find_substation(df=text, col='processed')
    print('Found substation in fragments\' text')

    samples = preproc.create_synthetic_samples(
        text,
        entity_col='SUBESTACAO',
        text_col='processed'
        )
    samples.drop_duplicates(inplace=True)
    print('Created samples dataframe for substation entity')
    
    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/substation_samples.parquet'
    )
    
    return

def create_valuewithunit_samples(X):
    
    text = X[['processed']].copy()
    text = preproc.find_valuewithunit(df=text, col='processed')
    print('Found values with unit in fragments\' text')

    samples = preproc.create_synthetic_samples(
        text,
        entity_col='VALOR_COM_UNID.MEDIDA',
        text_col='processed'
        )
    samples.drop_duplicates(inplace=True)
    print('Created samples dataframe for value with unit entity')
    
    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/valuewithunit_samples.parquet'
    )
    
    return

def create_inequalities_samples(X):
    
    text = X[['processed']].copy()
    text = preproc.find_inequality(df=text, col='processed')
    text = text[text['INEQUACAO'].astype(bool)]
    print('Found values with unit in fragments\' text')

    samples = preproc.create_synthetic_samples(
        text,
        entity_col='INEQUACAO',
        text_col='processed'
        )
    samples.drop_duplicates(inplace=True)
    print('Created samples dataframe for inequality entity')
    
    save_sample_to_s3(
        samples,
        bucket='ons-ds-dev-entityrecognition',
        filepath='mpo/mvp/data/processed/inequality_samples.parquet'
    )
    
    return

def create_trainset(
    bucket:str,
    processed_path:str,
    incremental_path:str,
    predefined_filepath:str='mpo/mvp/data/train/predefined/dataset.parquet',
    inequalities_filepath:str='mpo/mvp/data/train/inequalities/dataset.parquet'
    ):
    
    samples = preproc.get_processed_samples(
        bucket=bucket,
        path=processed_path
    )
    samples_incremental = preproc.get_processed_samples(
        bucket=bucket,
        path=incremental_path
    )
    samples = pd.concat([samples, samples_incremental], axis=0)
    print('created samples dataframe')
    
    save_sample_to_s3(
        samples,
        bucket=bucket,
        filepath=predefined_filepath
    )
    
    samples_inequalities = preproc.get_processed_samples_inequalities(
        bucket=bucket,
        path=processed_path
    )
    samples_inequalities_incremental = preproc.get_processed_samples_inequalities(
        bucket=bucket,
        path=incremental_path
    )
    samples_inequalities = pd.concat([samples_inequalities, samples_inequalities_incremental], axis=0)
    print('created samples dataframe for the inequality recognition model')
    
    save_sample_to_s3(
        samples_inequalities,
        bucket=bucket,
        filepath=inequalities_filepath
    )
    print('created samples dataframe')
    
    return samples
#  --------------------