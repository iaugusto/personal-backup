import sys
import time
import boto3
import numpy as np
import pandas as pd
from io import StringIO
import awswrangler as wr


def read_csv(
    bucket:str,
    path:str
    ):
        
    """
    Função para leitura de csv's do Simple Storage Service.

    ---

    Variáveis:
    - bucket (str): nome do bucket. \n
    - path (str): caminho complementar até o arquivo - incluí formato do arquivo (sufixo .csv). \n
    """
        
    obj_uri = 's3://{}/{}'.format(bucket, path)
    df = wr.s3.read_csv(obj_uri)
        
    return df


def read_excel(
    bucket:str,
    path:str
    ):
        
    """
    Função para leitura de arquivos xlsx (excel) do Simple Storage Service.

    ---

    Variáveis:
    - bucket (str): nome do bucket. \n
    - path (str): caminho complementar até o arquivo - incluí formato do arquivo (sufixo .xlsx). \n
    """
        
    obj_uri = 's3://{}/{}'.format(bucket, path)
    df = wr.s3.read_excel(obj_uri)
        
    return df


def run_sql_query(
    sql:str,
    database:str
    ):
    
    """
    Função para leitura de dados do Athena.

    ---

    Variáveis:
    - sql (str): query SQL para filtrar os dados do Athena. \n
    - database (str): label do banco de dados. \n
    """
    
    df = wr.athena.read_sql_query(sql=sql, database=database)
    
    return df