import os
import sys
import numpy as np
import pandas as pd
import awswrangler as wr
from preprocess import preprocess_module


preproc = preprocess_module()
preproc.gen_time, preproc.class_name
print('Loaded preproc module')

df_fragments = preproc.get_fragments()
df_fragments = preproc.preprocess_fragments(df_fragments, 'Texto Limpo')
print('Preprocessed fragments')

df_fragments_eq = preproc.find_equipments(df=df_fragments, col='processed')
df_fragments_eq.astype('str').drop_duplicates(subset=['MpoCentro', 'processed', 'EQUIPAMENTO'], keep='last', inplace=True)
print('Found equipments in fragments\' text')

samples_eq = preproc.create_synthetic_samples(
    df_fragments_eq,
    entity_col='EQUIPAMENTO',
    text_col='processed'
    )
samples_eq.drop_duplicates(inplace=True)
print('Created samples_eq dataframe')

bucket = 'ons-ds-dev-entityrecognition'
samples_eq_path = f's3://{bucket}/mpo/mvp/data/processed/equipment_samples.parquet'
samples_eq.to_parquet(samples_eq_path, compression='gzip')
print(f'saved samples in {samples_eq_path}')