import os
import sys
import numpy as np
import pandas as pd
import awswrangler as wr
from preprocess import preprocess_module


preproc = preprocess_module()
print('Loaded preproc module')

df_fragments = preproc.get_fragments()
text = df_fragments[['Texto Limpo']].drop_duplicates()
text = preproc.preprocess_fragments(text, 'Texto Limpo')
print('Preprocessed fragments')

text = pd.concat([text[['processed']].iloc[0:500].copy(), text[['processed']].iloc[1000:1500].copy()], axis=0)
text = preproc.find_inequality(df=text, col='processed')
text = text[text['INEQUACAO'].astype(bool)]
print('Found power plant in fragments\' text')

samples = preproc.create_synthetic_samples(
    text,
    entity_col='INEQUACAO',
    text_col='processed'
    )
samples.drop_duplicates(inplace=True)
print('Created samples dataframe for inequality entity')

bucket = 'ons-ds-dev-entityrecognition'
samples_path = f's3://{bucket}/mpo/mvp/data/processed/inequality_samples.parquet'
samples.to_parquet(samples_path, compression='gzip')
print(f'saved samples in {samples_path}')