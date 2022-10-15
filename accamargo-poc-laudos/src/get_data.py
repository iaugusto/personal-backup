import pandas as pd
from sqlalchemy import create_engine
import psycopg2 as p
import boto3
import awswrangler as wr

import utils


class GetData:
    
    
    def __init__(self):        
        self.data_with_label_path = 's3://br-accamargo-rawdata-dev/Laudo_classificacao_2020.xlsx'
        self.bucket_name = 'br-accamargo-rawdata-dev'

        
    def get_data_with_labels(self):
        
        df =  wr.s3.read_excel(path=self.data_with_label_path, engine = 'openpyxl')
        df.columns = df.columns.str.lower()
        df.rename(columns={'(1) câncer / (2) não câncer':'cancer', 'texto_diagnostico_resumo':'resumo'}, inplace=True)
        
        return df
    
    
    def get_laudos_raw(self):
        
        # lendo datasets atualizados
        laudos_list=[]

        s3 = boto3.resource('s3', region_name='us-east-1', verify=False)
        bucket = s3.Bucket(self.bucket_name)

        for files in bucket.objects.filter(Prefix='laudos_novo/LAUDOS'):
            laudos_list.append(wr.s3.read_parquet(path='s3://br-accamargo-rawdata-dev/'+str(files.key)))
            
        # concatenação dos datasets atualizados
        laudos = pd.concat(laudos_list, axis=0)
        laudos.columns = laudos.columns.str.lower()
        laudos.rename(columns={'testodiagnosi':'texto','codigo_interno':'laudo'}, inplace=True)
        laudos.drop_duplicates(inplace=True)
        
        return laudos
    
    def get_input_data(self):
        
        df = self.get_data_with_labels()
        laudos = self.get_laudos_raw()
        
        # juntar df e laudos pela coluna laudo
        full_df = pd.merge(left=df, right=laudos, on=['laudo'], how='inner')
        
        # filtrar o dataframe: colunas = [df.id, laudo.laudo, df.rotulo]
        full_df = full_df#[['laudo','texto','cancer']]
        full_df['cancer'] = abs(full_df['cancer']-2)
        
        return full_df