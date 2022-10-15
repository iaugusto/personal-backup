import os
import pytz
import joblib
import pandas as pd
from datetime import datetime as dt
from sklearn.pipeline import Pipeline
from guara.modeling.supervised_modelz import *

import pipeline_modules

import utils
import get_data
import pipeline_modules


class Train:
    
    
    def __init__(self):
        
        self.exp_name = dt.now(tz=pytz.timezone('America/Sao_Paulo')).strftime('%Y%m%d-%H%M%S')
        
        utils.log(f'\n================ TRAIN ================\n')
        utils.log(f'Train started with id {self.exp_name} \n')
    
    
    def get_raw_data(self):
        utils.log(f'GET DATA ----------------\n ')
        utils.log(f'Retrieving data from poc-gpa/data/processed folder')
        
        Xtrain, Xval, ytrain, yval = get_data.get_train_data()
    
        return Xtrain, Xval, ytrain, yval
    
    
    def createapply_feature_engineering_pipeline(self, Xtrain, Xval):
        utils.log(f'createapply_feature_engineering_pipeline ----------------\n ') 
        
        # training set
        #-------------------------------------------------------
        
        numerical_columns = [
            'val_venda_bruta_cupom',
            'qtd_item_venda',
            'flg_vend_meu_desct',
            'valor_desconto',
            'flag_dev',
            'tipo_promo_0',
            'tipo_promo_1',
            'tipo_promo_2',
            'tipo_promo_3',
            'tipo_promo_4',
            'tipo_promo_5',
            'categoria_0',
            'categoria_1',
            'categoria_2',
            'categoria_3',
            'categoria_4',
            'categoria_5',
            'categoria_6',
            'categoria_7',
            'departamento_0',
            'compras_mes',
            'agg_l3m_val_venda_bruta_cupom',
            'agg_l3m_qtd_item_venda',
            'agg_l3m_flg_vend_meu_desct',
            'agg_l3m_valor_desconto',
            'agg_l3m_flag_dev',
            'agg_l3m_tipo_promo_0',
            'agg_l3m_tipo_promo_1',
            'agg_l3m_tipo_promo_2',
            'agg_l3m_tipo_promo_3',
            'agg_l3m_tipo_promo_4',
            'agg_l3m_tipo_promo_5',
            'agg_l3m_categoria_0',
            'agg_l3m_categoria_1',
            'agg_l3m_categoria_2',
            'agg_l3m_categoria_3',
            'agg_l3m_categoria_4',
            'agg_l3m_categoria_5',
            'agg_l3m_categoria_6',
            'agg_l3m_categoria_7',
            'agg_l3m_departamento_0',
            'agg_l3m_compras_mes',
        ]

        outlier_columns_mean = [
            'pib_percapita',
            'idade',
            'delta_de_cadastro',
            'delta_de_stix'
        ]
        
        yeojohnson_columns = [
            'val_venda_bruta_cupom',
            'qtd_item_venda',
            'flg_vend_meu_desct',
            'valor_desconto',
            'compras_mes',
            'agg_l3m_val_venda_bruta_cupom',
            'agg_l3m_qtd_item_venda',
            'agg_l3m_flg_vend_meu_desct',
            'agg_l3m_valor_desconto',
            'agg_l3m_compras_mes',
            'pib_percapita',
            'idade',
            'delta_de_cadastro'
        ]
        
        self.fe_pipeline = Pipeline([
            ('drop_temporary_columns', pipeline_modules.drop_temporary_columns()),
            ('drop_with_low_variance', pipeline_modules.drop_numerical_with_variance(columns=numerical_columns)),
            ('encode_sex_column', pipeline_modules.encode_sex_column()),
            ('group_rare_regions', pipeline_modules.group_rare_categorical(columns=['region'], threshold=0.002)),
            ('encode_regions', pipeline_modules.encode_categorical(columns=['region'])),
            ('handle_outliers_max', pipeline_modules.outlier_handling(
                columns=numerical_columns, 
                method='gauss', 
                band=2.8, 
                action='max')),
            ('handle_outliers_mean', pipeline_modules.outlier_handling(
                columns=outlier_columns_mean, 
                method='gauss', 
                band=2.5, 
                action='mean')),
            ('handle_negative_values', pipeline_modules.handle_negative_values(columns=numerical_columns)),
            ('fill_missing_numerical_zero', pipeline_modules.fill_na_values_with_zero(
                columns=['ind_email','cadastro_stix','delta_de_cadastro','delta_de_stix'])),
            ('fill_missing_numerical_mean', pipeline_modules.fill_na_values_with_zero(
                columns=['pib_percapita','idade'])),
            ('transform_yeojohnson', pipeline_modules.data_transformation(
                columns=yeojohnson_columns, 
                method='yeojohnson'))
        ])
        
        Xtrain = self.fe_pipeline.fit_transform(Xtrain)
        
        # validation set
        #-------------------------------------------------------
        
        Xval = self.fe_pipeline.transform(Xval)

        utils.log(f'Xtrain processed shape: {Xtrain.shape}')
        utils.log(f'Xtrain processed shape: {Xval.shape}')
        
        return Xtrain, Xval
    
    
    def createapply_feature_selection_pipeline(self, Xtrain, Xval, ytrain):
        utils.log(f'createapply_feature_selection_pipeline ----------------\n ') 
        
        # training set
        #-------------------------------------------------------
        
        self.fs_pipeline = Pipeline([
            ('select_with_correlation', pipeline_modules.select_with_correlation(
                threshold=0.82, 
                method='recursive',
                objective='classification'))
        ])
        
        Xtrain = self.fs_pipeline.fit_transform(Xtrain, ytrain)
        
        # validation set
        #-------------------------------------------------------
        
        Xval = self.fs_pipeline.transform(Xval)

        utils.log(f'Xtrain processed shape: {Xtrain.shape}')
        utils.log(f'Xtrain processed shape: {Xval.shape}')
        
        return Xtrain, Xval
    
    
    def hyperparameter_search(self, X, y):
        pass
    
    
    def fit_model(self, Xtrain, Xval, ytrain, yval):
        utils.log(f'fit model ----------------\n ')
        
        scale_pos_weight=ytrain.value_counts(normalize=True)[0]/ytrain.value_counts(normalize=True)[1]
        
        params = {
            'random_state':501, 
            'boosting_type':'gbdt', 
            'device_type':'cpu',
            'scale_pos_weight':scale_pos_weight,
            'sub_sample':0.8,
            'min_child_samples':24,
            'learning_rate':0.38832846505493473,
            'colsample_bytree':0.31177546084715557,
            'n_estimators':499,
            'max_depth':4,
            'num_leaves':10
        }
        
        self.md = SupervisedModelz('lgbm', 'binary')
        self.model = self.md.fit(Xtrain, Xval, ytrain, yval, params)
        
        utils.log(f'Model fitted')
        
        self.md.plot_train_info()
        
        return self.model
    
    
    def evaluate(self, Xtrain, Xval, ytrain, yval):
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
        ytrain_pred = self.md.predict(Xtrain)
        yval_pred = self.md.predict(Xval)
        
        self.md.eval_binary(ytrain['target'], ytrain_pred, yval['target'], yval_pred)

        if type(ytrain) == pd.Series:
            ytrain = ytrain.values
        if type(yval) == pd.Series:
            yval = yval.values

        print('\n ============== Resumo metricas ============== \n')
        print('TREINO:\n')
        metrics_train = md.metrics_print(
            ytrain.values.clip(0, None), 
            ytrain_pred.clip(0, None), 
            metrics
        )

        print('\nVALIDACAO:\n')
        metrics_val = md.metrics_print(
            yval.values.clip(0, None), 
            yval_pred.clip(0, None), 
            metrics
        )

        print('\n')  
        return
        
    
    def save_pipe_and_model(self):
        utils.log(f'SAVING ----------------\n ')
        
        path_to_experiment = os.getcwd() + f'/artifacts/staging/{self.exp_name}'
        utils.log(f'Creating experiment directory in {path_to_experiment}')
        os.mkdir(path_to_experiment)
        
        model_path = f'{path_to_experiment}/model.joblib'
        fe_pipeline_path = f'{path_to_experiment}/fe_pipeline.joblib'
        fs_pipeline_path = f'{path_to_experiment}/fs_pipeline.joblib'
        
        joblib.dump(self.fe_pipeline, fe_pipeline_path)
        utils.log(f'Feature Engineering Pipeline saved in: {fe_pipeline_path}')
        
        joblib.dump(self.fs_pipeline, fs_pipeline_path)
        utils.log(f'Feature Selection Pipeline saved in: {fs_pipeline_path}')
        
        joblib.dump(self.model, model_path)
        utils.log(f'Model saved in: {model_path}')




###########################################################


if __name__ == '__main__':
    
    train = Train()    
    Xtrain, Xval, ytrain, yval = train.get_raw_data()
    Xtrain, Xval = train.createapply_feature_engineering_pipeline(Xtrain, Xval)
    Xtrain, Xval = train.createapply_feature_selection_pipeline(Xtrain, Xval, ytrain)

    model = train.fit_model(Xtrain, Xval, ytrain, yval)
    train.evaluate(Xtrain, Xval, ytrain, yval)
    train.save_pipe_and_model()
    
    utils.log('\n=================== END ===================\n')