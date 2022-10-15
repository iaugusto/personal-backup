import os
import gc
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from google.cloud import storage
from sqlalchemy import create_engine


# creating client list from items in storage
# creates json file in LOCAL - HAS TO BE UPLOADED TO STORAGE MANUALLY
#-----------------------------------------------------
def create_client_list(
    chunksize:int=2*10**7,
    initial_date:str='2021-10-01',
    prefix:str='gs://gpa-churn/',
    file_name:str='STAGING/relevanc_item_20220425.zip'
    ):
    
    ref_ids = []

    for chunk in pd.read_csv(
        prefix + file_name, 
        chunksize=chunksize, 
        usecols=['cod_cliente','dat_venda'],
        sep=';', 
        iterator=True
        ):

        chunk = chunk[chunk['dat_venda']>=initial_date]
        chunk.dropna(subset=['cod_cliente', 'dat_venda'], inplace=True)
        chunk['cod_cliente'] = chunk['cod_cliente'].astype(int)

        local_ids = list(chunk['cod_cliente'])
        ref_ids = ref_ids + local_ids
        ref_ids= list(set(ref_ids))

        print(f'len of ref_ids: {len(ref_ids)}')
        
    client_list = {
        'cod_cliente':ref_ids
    }
    
    with open('full_cod_cliente.json', 'w') as file:
        json.dump(client_list, file)
        
    return


# reading client list from storage
#-----------------------------------------------------
def read_client_list():
    
    with open('full_cod_cliente.json', 'r') as file:
        client_list = json.load(file)
    
    return client_list


# reading 'chunk' million lines as reference and searching for it's ids in the rest of the data
#-----------------------------------------------------
def process_items_from_codcliente(
    ref_ids:list,
    step:int=int(1*10**5),
    initial_date:str='2021-10-01',
    chunksize:int=50*10**6,
    prefix:str='gs://gpa-churn/',
    file_name:str='STAGING/relevanc_item_20220425.zip'
    ):
    
    columns = [
        'cod_cliente',
        'cod_loja',
        'dat_venda',
        'cod_interno_prod',
        'num_cupom',
        'val_venda_bruta_cupom',
        'qtd_item_venda'
        ]
    
    initial_dtypes = {
        'cod_cliente':'float32',
        'cod_loja':'float32',
        'data_venda':'str',
        'cupom':'float32',
        'cod_produto':'float32',
        'val_venda_bruta_cupom':'float32',
        'qtd_item_venda':'float32'
        }
    
    n_steps = np.ceil(len(ref_ids)/step)
    step_count = 0
    while step_count < n_steps:
        
        print('-'*20)
        print('STEP: ', step_count)
        id_list = ref_ids[int(step_count*step):int((step_count+1)*step)]
    
        df_list = []
        n_rows = 0
        chunk_number = 0
        for chunk in pd.read_csv(
            prefix + file_name, 
            chunksize=chunksize, 
            usecols=columns,
            dtype=initial_dtypes,
            sep=';', 
            iterator=True
            ):

            chunk = chunk[chunk['dat_venda']>=initial_date]
            chunk.dropna(subset=['cod_cliente', 'dat_venda'], inplace=True)
            chunk['cod_cliente'] = chunk['cod_cliente'].astype(int)

            chunk = chunk[chunk['cod_cliente'].isin(id_list)]


            chunk_number = chunk_number + 1
            n_rows = n_rows + len(chunk)
            df_list.append(chunk)
            print(f'Processed {chunk_number} chunks, number of rows: {n_rows}')

        df = pd.concat(df_list, axis=0)
        del df_list, chunk

        items_cols_to_rename = {
            'cod_interno_prod':'cod_produto',
            'num_cupom':'cupom',
            'dat_venda':'data_venda'
        }
        df.rename(columns=items_cols_to_rename, inplace=True)
        df = df.drop_duplicates()
        df.reset_index(drop=True, inplace=True)

        dtypes = {
            'cod_cliente':'int32',
            'cod_loja':'int16',
            'data_venda':'str',
            'cupom':'int32',
            'cod_produto':'int32',
            'val_venda_bruta_cupom':'float32',
            'qtd_item_venda':'int8'
            }
        for var, tp in dtypes.items():
            df[var] = df[var].astype(tp)
            
        df.to_parquet(f'gs://gpa-churn/data/raw/items/items_{step_count}.parquet', compression='gzip')
        step_count = step_count+1
    
    return


# getting activations from ref_ids - obtained from items' df
#-----------------------------------------------------
def process_activations_from_codcliente(
    ref_ids:list,
    step:int=int(1*10**5),
    chunksize:int=int(20*10**6),
    prefix:str='gs://gpa-churn/',
    file_name:str='STAGING/V_DW_FM16_MOV_VEND_CRM.TXT.zip'
    ):
    
    names=[
        'unknown',
        'cupom',
        'cod_produto',
        'cod_loja',
        'cod_cliente',
        'val_gross_margin_cupom',
        'val_vend_bruta_mercad',
        'flg_vend_meu_desct',
        'valor_desconto',
        'tipo_promo',
        'flag_dev'
    ]
    
    columns = [
        'cupom',
        'cod_produto',
        'cod_loja',
        'cod_cliente',
        'val_gross_margin_cupom',
        'val_vend_bruta_mercad',
        'flg_vend_meu_desct',
        'valor_desconto',
        'tipo_promo',
        'flag_dev'
    ]
    
    dtypes = {
        'cupom':'int32',
        'cod_produto':'int32',
        'cod_loja':'int16',
        'cod_cliente':'int32',
        'val_gross_margin_cupom':'float32',
        'val_vend_bruta_mercad':'float32',
        'flg_vend_meu_desct':'int8',
        'valor_desconto':'float32',
        'tipo_promo':'str',
        'flag_dev':'int8'
    }
    
    n_steps = np.ceil(len(ref_ids)/step)
    step_count = 0
    while step_count < n_steps:
    
        print('-'*20)
        print('STEP: ', step_count)
        id_list = ref_ids[int(step_count*step):int((step_count+1)*step)]
    
        df_list = []
        n_rows = 0
        chunk_number = 0
        for chunk in pd.read_csv(
            prefix + file_name,
            chunksize=chunksize, 
            sep=';',
            names=names,
            usecols=columns,
            dtype=dtypes,
            iterator=True
            ):

            chunk = chunk[chunk['cod_cliente'].isin(id_list)]
            chunk = chunk.drop_duplicates()
            df_list.append(chunk)

            chunk_number = chunk_number + 1
            n_rows = n_rows + len(chunk)
            print(f'Processed {chunk_number} chunks, number of rows: {n_rows}')

        df = pd.concat(df_list, axis=0)
        del df_list, chunk
        df = df.drop_duplicates()
        
        df.to_parquet(f'gs://gpa-churn/data/raw/activations/activations_{step_count}.parquet', compression='gzip')
        step_count = step_count+1
    
    return


# Calling functions
#-----------------------------------------------------

process_items = True
process_activations = True

if process_items == True:
    print('-'*10)
    print('Processing items')
    print('Creating client list')
    ref_ids = create_client_list()
    print('Successfully created client list')
    print('Generating samples')
    process_items_from_codcliente(
        ref_ids=ref_ids
        )
    print('Successfully generated items in storage')
    print('-'*10)

if process_activations == True:
    
    if process_items == False:
        print('Reading client list from local')
        client_list = read_client_list()
        print('Successfully read client list from local')
    
    print('-'*10)
    print('Processing activations from client list')
    process_activations_from_codcliente(
        ref_ids=client_list['cod_cliente']
        )
    print('Successfully generated items in storage')
    print('-'*10)