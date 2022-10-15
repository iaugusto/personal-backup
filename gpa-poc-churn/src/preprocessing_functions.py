import os
import gc
import sys
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sn
from datetime import datetime
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.pipeline import Pipeline
from dateutil.relativedelta import relativedelta

from geopy.geocoders import Nominatim
from pycep_correios import get_address_from_cep, WebService

sys.path.append(os.path.dirname(os.getcwd())+'/src')
import pipeline_modules


# get object list in blob storage bucket (STAGING)
#-----------------------------------------------------
def get_obj_list():
    
    storage_client = storage.Client()
    obj_list = storage_client.list_blobs('gpa-churn')
    obj_list = [i.name for i in obj_list if 'STAGING/' in i.name]

    tables_dict = {
        'register':[i for i in obj_list if 'cadastro' in i],
        'stix':[i for i in obj_list if 'stix_optin' in i],
        'email':[i for i in obj_list if 'optin_email' in i],
        'items':[i for i in obj_list if 'relevanc_item' in i],
        'stores':[i for i in obj_list if 'relevanc_store' in i],
        'products':[i for i in obj_list if 'relevanc_product' in i],
        'activations':[i for i in obj_list if 'mov_vend_crm' in i]
    }
    
    return tables_dict


# read generic information from files in staging according to an obj_list
#-----------------------------------------------------
def read_files_from_staging(
    obj_list:list,
    prefix:str='gs://gpa-churn/'
    ):
    
    df_list = []
    for file in obj_list:
        try:
            local_df = pd.read_csv(prefix + file, sep=';')
        except:
            local_df = pd.read_csv(prefix + file, sep='\n')
            column_list = local_df.columns[0].split(';')
            local_df = local_df[local_df.columns[0]].str.split(';', expand=True)
            local_df.columns = column_list
        df_list.append(local_df)
        print('added file: ', file)
    print('-'*10)
    df = pd.concat(df_list, axis=0)
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    
    return df


# reading 'chunk' million lines as reference and searching for it's ids in the rest of the data
#-----------------------------------------------------
def get_items_dataframe(
    initial_date:str='2021-10-10',
    chunksize:int=0.5*10**6,
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
    
    dtypes = {
        'cod_cliente':'float32',
        'cod_loja':'float32',
        'data_venda':'str',
        'cupom':'float32',
        'cod_produto':'float32',
        'val_venda_bruta_cupom':'float32',
        'qtd_item_venda':'float32'
        }
    
    df_list = []
    n_rows = 0
    chunk_number = 0
    for chunk in pd.read_csv(
        prefix + file_name, 
        chunksize=chunksize, 
        usecols=columns,
        dtype=dtypes,
        sep=';', 
        iterator=True
        ):
        
        chunk = chunk[chunk['dat_venda']>=initial_date]
        chunk.dropna(subset=['cod_cliente', 'dat_venda'], inplace=True)
        chunk['cod_cliente'] = chunk['cod_cliente'].astype(int)
        
        if chunk_number==0:
            ref_ids = set(chunk['cod_cliente'])
            print(f'len of ref_ids: {len(ref_ids)}')
        else:
            chunk = chunk[chunk['cod_cliente'].isin(ref_ids)]
        
        
        chunk_number = chunk_number + 1
        n_rows = n_rows + len(chunk)
        df_list.append(chunk)
        
        if (chunk_number % 20) == 0:
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
    
    return df, ref_ids  


# getting activations from ref_ids - obtained from items' df
#-----------------------------------------------------
def match_activations_dataframe(
    ref_ids:list,
    chunksize:int=15*10**6,
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
    
    chunk_number = 0
    n_rows = 0
    df_list = []
    for chunk in pd.read_csv(
        prefix + file_name,
        chunksize=chunksize, 
        sep=';',
        names=names,
        usecols=columns,
        dtype=dtypes,
        iterator=True
        ):
        
        chunk = chunk[chunk['cod_cliente'].isin(ref_ids)]
        chunk = chunk.drop_duplicates()

        chunk_number = chunk_number + 1
        n_rows = n_rows + len(chunk)
        df_list.append(chunk)
        
        if (chunk_number % 20) == 0:
            print(f'Processed {chunk_number} chunks, number of rows: {n_rows}')
        
    df = pd.concat(df_list, axis=0)
    del df_list, chunk
    df = df.drop_duplicates()
    
    return df


# get products data from input
#-----------------------------------------------------
def get_products_data(
    tables_dict:dict
    ):
    
    products = read_files_from_staging(
        tables_dict['products']
        )

    products = products[products['ind_ativo']==True]
    products_cols_to_drop = [
        'cod_subcategoria',
        'cod_categoria',
        'desc_subcategoria',
        'desc_subcategoria',
        'cod_grupo',
        'desc_grupo',
        'cod_subgrupo',
        'desc_subgrupo',
        'cod_departamento',
        'ind_ativo',
        'desc_plu' # product names - comment to validate
    ]
    products.drop(columns=products_cols_to_drop, inplace=True)

    products_cols_to_rename = {
        'cod_plu':'cod_produto',
        #'desc_plu':'produto', # product names - uncomment to validate
        'desc_categoria':'categoria',
        'desc_departamento':'departamento'
    }
    products.rename(columns=products_cols_to_rename, inplace=True)

    products_cols_to_lower = ['categoria', 'departamento'] #,'produto'] # product names - uncomment to validate
    for var in products_cols_to_lower:
        products[var] = products[var].str.lower()
    
    return products


# get register data from input
#-----------------------------------------------------
def get_register_data(
    tables_dict:dict
    ):
    
    register = read_files_from_staging(
        tables_dict['register']
        )

    register.sort_values(by=['idcliente','datultatual'], inplace=True, ascending=True)
    register.drop_duplicates(subset=['idcliente','datultatual'], keep='last')

    register_cols_to_drop = ['codorigcliente','codorigcliente']
    register = register.drop(columns=register_cols_to_drop)
    register = register.dropna(subset=['idcliente','cidadecli','ufcli'], how='any', axis=0)

    register_cols_to_rename = {
        'idcliente':'cod_cliente',
        'datcadorigem':'data_cadastro',
        'codsexo':'sexo',
        'cidadecli':'cidade',
        'ufcli':'uf',
        'datnasccli':'data_nascimento',
        'datultatual':'data_registro'
    }
    register.rename(columns=register_cols_to_rename, inplace=True)

    register_cols_to_lower = ['cidade', 'uf']
    for var in register_cols_to_lower:
        register[var] = register[var].str.lower()

    register = register.drop_duplicates(subset=['cod_cliente','sexo','cidade','uf','data_nascimento'])
    register.reset_index(drop=True, inplace=True)
    
    return register


# get stix data from input
#-----------------------------------------------------
def get_stix_data(
    tables_dict:dict
    ):
    
    stix = read_files_from_staging(
        tables_dict['stix']
        )
    
    stix.drop(columns=['dt_nascimento'], inplace=True)
    stix.sort_values(by=['id_cliente','dt_cadastro_origem'], ascending=True, inplace=True)
    stix = stix.drop_duplicates(subset=['id_cliente','ind_email'], keep='first')
    stix['ind_email'] = abs(stix['ind_email']-2)
    stix.reset_index(drop=True, inplace=True)
    
    stix_cols_to_rename = {
        'id_cliente':'cod_cliente',
        'dt_cadastro_origem':'data_registro_stix'
    }
    stix.rename(columns=stix_cols_to_rename, inplace=True)
    
    return stix


# get sales data from input
#-----------------------------------------------------
def get_items_data(
    lower_boundary:int=int(0),
    upper_boundary:int=int(1*10**5),
    initial_date:str='2021-10-01'
    ):
    
    print('-'*10)
    print('get_items_data')
    items = pd.read_parquet('gs://gpa-churn/data/raw/items/items.parquet')
    items = items[items['data_venda']>=initial_date]
    id_list = list(set(items['cod_cliente']))
    print(f'original number of clients: {len(id_list)}')
    id_list = id_list[lower_boundary:upper_boundary]
    items = items[items['cod_cliente'].isin(id_list)]
    print(f'adjusted number of clients: {len(id_list)}')
    print(items.info())
    print('-'*10)
    
    return items, id_list


# get activations data from input
#-----------------------------------------------------
def get_activations_data(
    id_list:list
    ):

    print('-'*10)
    print('get_activations_data')
    activations = pd.read_parquet('gs://gpa-churn/data/raw/activations/activations.parquet')
    activations.drop(columns=['val_vend_bruta_mercad', 'val_gross_margin_cupom'], inplace=True)
    activations = activations[activations['cod_cliente'].isin(id_list)]
    print(activations.info())
    print('-'*10)
    
    return activations


# split date columns to year and month columns
#-----------------------------------------------------
def split_date(
    df,
    column:str='data_venda'
    ):

    print('-'*10)
    print('split_date')
    
    df[['ano','mes','dia']] = df['data_venda'].str.split('-', expand=True)
    df.drop(columns=['dia'], inplace=True)
    
    print('-'*10)
    
    return df


# aggregate numeric data to get total monthly quantitative info
#-----------------------------------------------------
def aggregate_sum(df):

    print('-'*10)
    print('aggregate_sum')
    
    df = df.drop(columns=['cod_loja','data_venda','cupom'])
    df = df.groupby(['cod_cliente','ano','mes']).sum()
    df.reset_index(inplace=True)
    
    print('-'*10)
    
    return df


# aggregate numerical data to get monthly interactions
#-----------------------------------------------------
def aggregate_count(df):
    
    print('-'*10)
    print('aggregate_count')
    
    df = df[['cod_cliente','cupom','ano','mes']]
    df = df.groupby(['cod_cliente','ano','mes']).count()
    rename_dict = {
        'cupom': 'compras_mes'
    }
    df = df.rename(columns=rename_dict)
    df.reset_index(inplace=True)
    
    print('-'*10)
    
    return df


# create samples from aggreagates with customizable time lengths
#-----------------------------------------------------
def create_samples_from_aggregates(
    df,
    initial_date:str='2021-10-01',
    unitary_increment:int=1,
    feature_sample_range:int=2,
    target_sample_range:int=1
    ):
    
    # sample range turns into sample_range + 1 as its definition
    
    print('-'*10)
    print('create_samples_from_aggregates')
    
    initial_date = datetime.strptime(initial_date, '%Y-%m-%d')
    final_date = df['date'].max()
    print(f'initial date: {initial_date}')
    print(f'final date: {final_date}')

    if target_sample_range == 1:
        sum_target_df_rows = False
    else:
        sum_target_df_rows = True

    total_time_delta = final_date - initial_date
    n_months = np.ceil(total_time_delta/np.timedelta64(1, 'M')) + 1
    n_samples = (n_months-(unitary_increment+feature_sample_range+target_sample_range))+1
    print(f'total span in months: {n_months}')
    print(f'number of samples generated: {n_samples}')

    unitary_increment = relativedelta(months=+unitary_increment)
    print(f'unitary_increment: {unitary_increment}')

    feature_sample_range = relativedelta(months=+feature_sample_range)
    print(f'feature_sample_range: {feature_sample_range}')

    target_sample_range = relativedelta(months=+target_sample_range)
    print(f'target_sample_range: {target_sample_range}')
    
    print('Generating samples from segmented dataframe')
    print('-'*10)
    current_date = initial_date+feature_sample_range
    samples_list = []
    step = 0
    while step < n_samples:

        print('REFERENCE DATE: ', current_date)

        # reading features range
        #--------------------------------
        lower_mask = df['date']>=current_date-feature_sample_range
        upper_mask = df['date']<=current_date
        features_df = df[lower_mask & upper_mask]
        features_df = features_df.sort_values(by=['cod_cliente','date'])
        print('features:')
        print(features_df['date'].value_counts())

        # aggregating 3 month data
        #--------------------------------
        current_month_mask = df['date']==current_date
        features_m0 = features_df[current_month_mask].drop(columns=['date','ano','mes'])
        features_m3 = features_df.drop(columns=['date','ano','mes']).groupby(['cod_cliente']).sum()
        features_m3.columns = ['agg_l3m_' + i for i in list(features_m3.columns)]
        features_m3.reset_index(inplace=True)

        # creating sample
        #--------------------------------
        sample = pd.merge(features_m0, features_m3, on='cod_cliente')

        # reading target range
        #--------------------------------
        lower_mask = df['date']>=current_date+target_sample_range
        upper_mask = df['date']<current_date+target_sample_range+unitary_increment
        target_df = df[lower_mask & upper_mask]
        target_df = target_df.sort_values(by=['cod_cliente','date'])
        print('target:')
        print(target_df['date'].value_counts())

        # aggregates within target range definition
        #--------------------------------
        if sum_target_df_rows == True:
            target_df = target_df[['cod_cliente','qtd_item_venda']].groupby(['cod_cliente']).sum()
            target_df.reset_index(inplace=True)

        # selecting customer ids that churned in the target range
        #--------------------------------
        churn_clientids_1 = list(target_df[target_df['qtd_item_venda']==0]['cod_cliente'])
        churn_clientids_2 = list(set(features_df['cod_cliente']).difference(set(target_df['cod_cliente'])))
        churn_clientids = list(set(churn_clientids_1 + churn_clientids_2))

        # selecting customer ids that churned in the target range
        #--------------------------------
        sample['date'] = current_date
        sample['target'] = [1 if i in churn_clientids else 0 for i in list(sample['cod_cliente'])]    
        samples_list.append(sample)

        current_date+=unitary_increment
        step+=1
        
    del target_df, features_m0, features_m3, sample
    gc.collect()

    print('-'*5)
    
    output_df = pd.concat(samples_list)
    output_df.drop_duplicates(inplace=True)
    
    del samples_list
    gc.collect()
    
    print('')
    print('Successfully generated samples')
    print('-'*10)
    
    return output_df


# get region from federal unit
#-----------------------------------------------------
def get_region(x):    
    
    region_dict = {
        'se':['sp','mg','rj','es'],
        's':['pr','sc','rs'],
        'ne':['ma','ce','rn','pb','pe','al','se','ba','pi'],
        'n':['rr','ap','am','pa','ac','ro','to'],
        'co':['mt','df','go','ms']
        }
    
    for key in region_dict.keys():
        try:
            if x in region_dict[key]:
                return key
        except:
            pass
              
    return 'null'


# get percapit pib from federal unit
#-----------------------------------------------------
def get_pib_from_uf(x):
    
    pib_dict = {
        'df':90742,
        'sp':51140,
        'rj':45174,
        'sc':45118,
        'rs':42406,
        'pr':40788,
        'mt':40787,
        'ms':38482,
        'es':34177,
        'mg':30794,
        'go':29732,
        'ro':26497,
        'am':26101,
        'to':25022,
        'rr':23594,
        'pa':20734,
        'pe':20702,
        'ap':20688,
        'rn':20342,
        'ba':19716,
        'se':19441,
        'ce':17912,
        'ac':17722,
        'al':17667,
        'pb':16919,
        'pi':16125,
        'ma':13757
    }
    
    try:
        return pib_dict[x]
    except:
        return np.nan
    
    
# get latitude and longitude
#-----------------------------------------------------
def get_lat_long(municipio):

    location = geolocator.geocode(municipio)
    return location.latitude, location.longitude


# using functions to prepare raw data for modeling
#-----------------------------------------------------
if __name__ == '__main__':
    
    
    # local definitions
    initial_date = '2021-10-01'
    reprocess_bigdata = False
    save_before_aggregations = True
    save_after_aggregations = True
    save_after_registrationmerge = True
    save_after_stixmerge = True
    
    # required for get_lat_long function
    geolocator = Nominatim(user_agent="get_lat_long", timeout=1)
    
    
    # reprocess items and activations chunks
    #-----------------------------------------------------
    if reprocess_bigdata == True:
        items, id_list = get_items_dataframe()
        items.to_parquet('gs://gpa-churn/data/raw/items/items.parquet', compression='gzip')
        del items
        gc.collect()

        activations = match_activations_dataframe(ref_ids=id_list)
        activations.to_parquet('gs://gpa-churn/data/raw/activations/activations.parquet', compression='gzip')
        del activations, id_list
        gc.collect()
    
    
#     # lower and upper boundary lists for sample generation
#     #-----------------------------------------------------
#     lower_boundary_list = [0,10**5,2*10**5]
#     upper_boundary_list = [10**5,2*10**5,2.5*10**5]
#     training_set = True
    
    
#     for chunk_number in range(len(lower_boundary_list)):
    
#         sufix = str(chunk_number)
    
#         # reading items and activations
#         #-------------------------------------------------
#         items, id_list = get_items_data(
#             lower_boundary=int(lower_boundary_list[chunk_number]),
#             upper_boundary=int(upper_boundary_list[chunk_number]),
#             initial_date = initial_date
#             )

#         activations = get_activations_data(
#             id_list=id_list
#             )
    
    training_set = False
    for i in range(29,32):
        
        sufix = str(i)
        
        items = pd.read_parquet(f'gs://gpa-churn/data/raw/items/items_{i}.parquet')
        activations = pd.read_parquet(f'gs://gpa-churn/data/raw/activations/activations_{i}.parquet')

        # merging items and activations
        #-------------------------------------------------
        df = pd.merge(items, activations, on=['cod_cliente','cupom','cod_loja','cod_produto'], how='outer')
        del items, activations
        gc.collect()

        dropna_list = ['data_venda']
        df = df.dropna(subset=dropna_list)

        fillna_numerical = [
            'val_venda_bruta_cupom',
            'qtd_item_venda',
            'flg_vend_meu_desct', 
            'valor_desconto', 
            'flag_dev'
        ]
        df[fillna_numerical] = df[fillna_numerical].fillna(0)

        fillna_categorical = ['tipo_promo']
        df[fillna_categorical] = df[fillna_categorical].fillna('missing')

        convert_dtypes = {
            'val_venda_bruta_cupom':'float32',
            'qtd_item_venda':'float32',
            'flg_vend_meu_desct':'int8',
            'valor_desconto':'float32',
            'flag_dev':'int8'
        }

        for var, tp in convert_dtypes.items():
            df[var] = df[var].astype(tp)


        # encoding promotions' column
        #-------------------------------------------------
        if training_set == True:

            pipe_encode_promo = Pipeline([
                ('group_rare_promo', pipeline_modules.group_rare_categorical(columns=['tipo_promo'], threshold=0.002)),
                ('encode_promo', pipeline_modules.encode_categorical(columns=['tipo_promo']))
            ])

            df = pipe_encode_promo.fit_transform(df)
            pipeline_modules.save_pipeline(pipeline=pipe_encode_promo,artifact_id='pipe_encode_promo')

        else:

            pipe_encode_promo = pipeline_modules.read_pipeline(artifact_id='pipe_encode_promo')[0]
            df = pipe_encode_promo.transform(df)

        columns_to_convert = [i for i in list(df.columns) if 'tipo_promo' in i]
        convert_dtypes = {key:'int8' for key in columns_to_convert}
        for var, tp in convert_dtypes.items():
            df[var] = df[var].astype(tp)


        # get object list from staging (cloud storage)
        #-------------------------------------------------
        tables_dict = get_obj_list()


        # preparing products data
        #-------------------------------------------------
        products = get_products_data(tables_dict)

        if training_set == True:

            pipe_encode_products = Pipeline([
                ('group_rare_products', pipeline_modules.group_rare_categorical(columns=['categoria','departamento'])),
                ('encode_products', pipeline_modules.encode_categorical(columns=['categoria','departamento']))
            ])

            products = pipe_encode_products.fit_transform(products)
            pipeline_modules.save_pipeline(pipeline=pipe_encode_products,artifact_id='pipe_encode_products')

        else:

            pipe_encode_products = pipeline_modules.read_pipeline(artifact_id='pipe_encode_products')[0]
            products = pipe_encode_products.transform(products)

        columns_to_convert_1 = [i for i in list(products.columns) if 'categoria' in i]
        columns_to_convert_2 = [i for i in list(products.columns) if 'departamento' in i]
        columns_to_convert = columns_to_convert_1 + columns_to_convert_2
        del columns_to_convert_1, columns_to_convert_2
        gc.collect()

        convert_dtypes = {key:'int8' for key in columns_to_convert}
        for var, tp in convert_dtypes.items():
            products[var] = products[var].astype(tp)


        # merging products to sales data (items and activations)
        #-------------------------------------------------
        df = pd.merge(df, products, on=['cod_produto'])
        del products
        gc.collect()
        df.drop(columns=['cod_produto'], inplace=True)


        # splitting date into independent month and year columns
        #-------------------------------------------------
        df = split_date(
            df,
            column='data_venda'
            )


        # splitting date into independent month and year columns
        #-------------------------------------------------
        if save_before_aggregations == True:
            df.to_parquet(f'gs://gpa-churn/data/processed/steps/before_agg_{sufix}.parquet', compression='gzip')


        # perform aggregations
        #-------------------------------------------------
        sum_m0 = aggregate_sum(df)

        df = pd.read_parquet(f'gs://gpa-churn/data/processed/steps/before_agg_{sufix}.parquet')
        count_m0 = aggregate_count(df)

        df = pd.merge(sum_m0, count_m0, on=['cod_cliente','ano','mes'])
        del sum_m0, count_m0
        gc.collect()

        df['date'] = df[['ano','mes']].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m')

        if save_after_aggregations == True:
            df.to_parquet(f'gs://gpa-churn/data/processed/steps/after_agg_{sufix}.parquet', compression='gzip')


        # create samples from quantitative data
        #-------------------------------------------------
        df = create_samples_from_aggregates(
            df=df,
            initial_date=initial_date
            )


        # preparing registration data
        #-------------------------------------------------
        register = get_register_data(tables_dict)
        register = register[register['data_nascimento']>='1900']
        register['region'] = register['uf'].apply(lambda x:get_region(x))
        register['pib_percapita'] = register['uf'].apply(lambda x:get_pib_from_uf(x))
        #register['lat'], register['long'] = zip(*register['uf'].map(get_lat_long))


        # merging registration data - items, activations and registration
        #-------------------------------------------------
        df = pd.merge(df, register, on=['cod_cliente'], how='left')
        del register
        gc.collect()
        df = df.sort_values(by=['cod_cliente','data_registro'], ascending=True)
        df = df.drop_duplicates(subset=['cod_cliente','date'],keep='last')


        # creating time-related features (age, time since registration)
        #-------------------------------------------------
        df['data_nascimento'] = pd.to_datetime(df['data_nascimento'])
        df['idade'] = df['date']-df['data_nascimento']
        df['idade'] = df['idade'].apply(lambda x:x.days)

        df['data_cadastro'] = pd.to_datetime(df['data_cadastro'])
        df['delta_de_cadastro'] = df['date']-df['data_cadastro']
        df['delta_de_cadastro'] = df['delta_de_cadastro'].apply(lambda x:x.days)

        df = df.drop(columns=['data_nascimento', 'data_registro', 'data_cadastro'])

        if save_after_registrationmerge == True:
            df.to_parquet(f'gs://gpa-churn/data/processed/steps/after_registration_{sufix}.parquet', compression='gzip')


        # reading stix data
        #-------------------------------------------------
        stix = get_stix_data(tables_dict)


        # mergin stix data - items, activations, registration and email contact info
        #-------------------------------------------------
        df = pd.merge(df, stix, on=['cod_cliente'], how='left', indicator=True)
        del stix
        gc.collect()
        df.rename(columns={'_merge':'cadastro_stix'}, inplace=True)
        df['cadastro_stix'] = [0 if i=='left_only' else 1 for i in list(df['cadastro_stix'])]
        df = df.sort_values(by=['cod_cliente','data_registro_stix'], ascending=True)
        df = df.drop_duplicates(subset=['cod_cliente','date'],keep='last')


        # creating time-related features (time in stix)
        #-------------------------------------------------
        df['data_registro_stix'] = pd.to_datetime(df['data_registro_stix'])
        df['delta_de_stix'] = df['date']-df['data_registro_stix']
        df['delta_de_stix'] = df['delta_de_stix'].apply(lambda x:x.days)

        df = df.drop(columns=['data_registro_stix'])

        if save_after_stixmerge == True:
            df.to_parquet(f'gs://gpa-churn/data/processed/input/after_stix_{sufix}.parquet', compression='gzip')
            
        training_set = False