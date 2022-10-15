import sys
import os
import os.path as path
import unittest
import pandas as pd
import pandas.testing as pd_testing
pd.options.mode.chained_assignment = None


path_to_guara = '/usr/app/src'
sys.path.insert(1, path_to_guara)

from guara.feature_engineering.pipeline_modules import *
from guara.feature_engineering.window_features import *
                                                                                                                                  

class TestPipeline(unittest.TestCase):

    def test_DatetimeFeaturizer(self):
        dates = pd.DataFrame(pd.date_range(start = '01/01/2018', periods = 22, freq = '430H'), columns = ['datetime'])

        true_resp = pd.DataFrame({

            'hora': [0, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 22, 20, 18, 16, 14, 12, 10, 8, 6],
            'dia': [1, 18, 5, 23, 13, 31, 18, 6, 24, 11, 29, 17, 4, 21, 8, 26, 14, 1, 19, 7, 25, 12],
            'mes': [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12, 1], 
            'ano': [2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2019], 
            'dia_semana': [1, 4, 1, 5, 2, 6, 3, 7, 4, 1, 5, 2, 6, 2, 6, 3, 7, 4, 1, 5, 2, 6], 
            'dia_ano': [1, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 233, 251, 269, 287, 305, 323, 341, 359, 12], 
            'semana_mes': [1, 3, 2, 4, 3, 5, 4, 1, 4, 3, 5, 4, 1, 4, 2, 5, 2, 1, 4, 2, 5, 2], 
            'quarto_mes': [1, 3, 1, 3, 2, 4, 3, 1, 3, 2, 4, 2, 1, 3, 1, 4, 2, 1, 3, 1, 4, 2], 
            'semana_ano': [1, 3, 6, 8, 11, 13, 16, 18, 21, 24, 26, 29, 31, 34, 36, 39, 41, 44, 47, 49, 52, 2], 
            'fds': [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1], 
            'periodo_dia': ['madrugada', 'noite', 'noite', 'tarde', 'tarde', 'tarde', 'manha', 'manha', 'manha', 'madrugada', 'madrugada', 'madrugada', 'madrugada', 'noite', 'noite', 'tarde', 'tarde', 'tarde', 'manha', 'manha', 'manha', 'madrugada']
            
            }).astype({
                'periodo_dia':pd.CategoricalDtype(categories=['madrugada', 'manha', 'tarde', 'noite'], ordered=True)
                }) 


        feat = ['hora', 'dia', 'mes', 'ano', 'dia_semana', 'dia_ano', 'semana_mes', 'quarto_mes', 'semana_ano', 'fds', 'periodo_dia']

        pipe = Pipeline([ ('date_feat', DatetimeFeaturizer(feat)) ])

        resp = pipe.fit_transform(dates)
        
        pd_testing.assert_frame_equal(resp, true_resp)
        
        
    def test_SinCosFeaturizer(self):
        
        df_fit = pd.DataFrame({
                    'range_4': [1,2,3,4],
                    'equal_0': [0,0,0,0],
                    'equal_1': [1,1,1,1],
                    'nan':[np.nan,np.nan,np.nan,1],
                    'inf':[np.inf,np.inf,-np.inf,0]
                })


        df_transform = pd.DataFrame({
                    'range_4': [1,2,3,5],
                    'equal_0': [0,0,0,1],
                    'equal_1': [1,1,1,2],
                    'nan':[np.nan,np.nan,np.nan,2],
                    'inf':[np.inf,np.inf,-np.inf,1]
                })

        true_resp = pd.DataFrame({
                    'equal_0_cos': [np.nan, np.nan, np.nan, np.nan], 
                    'equal_0_sen': [np.nan, np.nan, np.nan, np.nan], 
                    'equal_1_cos': [1.0, 1.0, 1.0, 1.0], 
                    'equal_1_sen': [0.0, 0.0, 0.0, 0.0], 
                    'inf_cos': [np.nan, np.nan, np.nan, 1.0], 
                    'inf_sen': [np.nan, np.nan, np.nan, 0.0], 
                    'nan_cos': [np.nan, np.nan, np.nan, 1.0], 
                    'nan_sen': [np.nan, np.nan, np.nan, 0], 
                    'range_4_cos': [0.0, -1.0, 0.0, 1.0],
                    'range_4_sen': [1.0, 0.0, -1.0, 0.0]
                })

        pipe = Pipeline([ 
            ('sin_cos', SinCosFeaturizer())
        ])

        r_fit = pipe.fit_transform(df_fit)
        r_transform = pipe.transform(df_transform)
        
        pd_testing.assert_frame_equal(r_fit, true_resp)
        pd_testing.assert_frame_equal(r_transform, true_resp)


    def test_PandasTypeSelector(self):
        
        df_fit = pd.DataFrame({
                        'int': [1,2,3,5],
                        'float': [0.0,0.0,0.0,1.0],
                        'categorical': ['a', 'b', 'c', 'd'],
                        'object': ['o1', 'o2', 'o3', 'o4'],
                        'nan':[np.nan,np.nan,np.nan,np.nan],
                        'inf':[np.inf,np.inf,np.inf,np.inf]

                    }).astype({
                        'categorical':pd.CategoricalDtype()
                        })

        ########## INT

        resp_int = pd.DataFrame({
                    'int': [1, 2, 3, 5]
                })

        pipe = Pipeline([ ('type_selector', PandasTypeSelector(include = ['int64'])) ])
        
        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_int)


        ########## FLOAT

        resp_float = pd.DataFrame({
                    'float': [0.0, 0.0, 0.0, 1.0], 
                    'nan': [np.nan, np.nan, np.nan, np.nan], 
                    'inf': [np.inf, np.inf, np.inf, np.inf]
                })

        pipe = Pipeline([ ('type_selector', PandasTypeSelector(include = ['float'])) ])
        
        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_float)


        ########## CATEGORICAL

        resp_cat = pd.DataFrame({
                    'categorical': ['a', 'b', 'c', 'd']
                }).astype({
                        'categorical':pd.CategoricalDtype()
                        })

        pipe = Pipeline([ ('type_selector', PandasTypeSelector(include = ['category'])) ])
        
        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_cat)


        ########## OBJECT

        resp_obj = pd.DataFrame({
                    'object': ['o1', 'o2', 'o3', 'o4']
                })

        pipe = Pipeline([ ('type_selector', PandasTypeSelector(include = ['object'])) ])
        
        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_obj)
        



        return


    def test_ColumnSelector(self):
        df_fit = pd.DataFrame({

                    'int': [1,2,3,5],
                    'float': [0.0,0.0,0.0,1.0],
                    'object': ['o1', 'o2', 'o3', 'o4'],
                    'nan':[np.nan,np.nan,np.nan,np.nan]

                })
        

        ####### ONE COLUMN

        resp_1 = pd.DataFrame({
                    'int': [1, 2, 3, 5]
                })
        

        pipe = Pipeline([ 
            ('column_selector', ColumnSelector(['int']))
        ])

        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_1)
        

        ####### MULTIPLE COLUMNS

        resp_multi = pd.DataFrame({
                    'int': [1, 2, 3, 5], 
                    'float': [0.0, 0.0, 0.0, 1.0]
                })
        


        pipe = Pipeline([ 
            ('column_selector', ColumnSelector(['int', 'float']))
        ])

        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_multi)


        return


    def test_ColumnDropper(self):
        
        
        df_fit = pd.DataFrame({

                    'int': [1,2,3,5],
                    'float': [0.0,0.0,0.0,1.0],
                    'object': ['o1', 'o2', 'o3', 'o4'],
                    'nan':[np.nan,np.nan,np.nan,np.nan]

                })
        


        resp_1 = pd.DataFrame({
                    'int': [1, 2, 3, 5]
                })
        

        pipe = Pipeline([ 
            ('column_dropper', ColumnDropper(['float','object','nan']))
        ])

        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_1)


        return


    def test_FillNa(self):

        df_fit = pd.DataFrame({

                    'x': [1,2,np.nan, np.nan],
                    'nan':[np.nan,np.nan,np.nan,np.nan]

                })
        


        resp_1 = pd.DataFrame({
                    'x': [1.0 ,2.0, -1.0, -1.0],
                    'nan':[-1.0, -1.0, -1.0, -1.0]
                })
        

        pipe = Pipeline([ 
            ('fillna', FillNa(-1))
        ])

        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_1)

        return



    def test_OneHotPandas(self):

        df_fit = pd.DataFrame({

                    'nulos': ['a','a','b','b',np.nan],
                    'cat': ['a','a','b','b','c'],
                    'num': [1,1,1,1,1]

                })


        df_trans = pd.DataFrame({

                    'nulos': ['c'],
                    'cat': [np.nan],
                    'num': [1]

                })




        resp_fit = pd.DataFrame({
                    'nulos_a': [1, 1, 0, 0, 0], 
                    'nulos_b': [0, 0, 1, 1, 0], 
                    'nulos_nan_ohe': [0, 0, 0, 0, 1], 
                    'cat_a': [1, 1, 0, 0, 0], 
                    'cat_b': [0, 0, 1, 1, 0], 
                    'cat_c': [0, 0, 0, 0, 1], 
                    'num_1': [1, 1, 1, 1, 1]
                }).astype('int32')


        resp_transform = pd.DataFrame({
                    'nulos_a': [0], 
                    'nulos_b': [0], 
                    'nulos_nan_ohe': [0],
                    'cat_a': [0], 
                    'cat_b': [0], 
                    'cat_c': [0],
                    'num_1': [1]
                }).astype('int32')


        pipe = Pipeline([ 
            ('one_hot', OneHotPandas())
        ])


        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_fit)
        pd_testing.assert_frame_equal(pipe.transform(df_trans), resp_transform)

        return


    def test_TfIdfPandas(self):

        df_fit = pd.DataFrame({

                    'texto': ['aqui tem um texto','texto nao explicativo','nada a ver'],

                })


        df_trans = pd.DataFrame({

                    'texto': ['texto','nunca vi essa frase'],

                })




        resp_fit = pd.DataFrame({
                    'aqui': [0.795960545539856, 0.0, 0.0], 
                    'explicativo': [0.0, 0.795960545539856, 0.0], 
                    'texto': [0.605348527431488, 0.605348527431488, 0.0]
                }).astype('float32')


        resp_transform = pd.DataFrame({
                   'aqui': [0.0, 0.0], 
                   'explicativo': [0.0, 0.0], 
                   'texto': [1.0, 0.0]
                }).astype('float32')


        pipe = Pipeline([ 
            ('tf_idf', TfIdfPandas(max_features = 3))
        ])


        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_fit)
        pd_testing.assert_frame_equal(pipe.transform(df_trans), resp_transform)


        return



    def test_CountVectorizerPandas(self):

        df_fit = pd.DataFrame({

                    'texto': ['aqui aqui tem um texto texto','texto nao explicativo','nada a ver'],

                })


        df_trans = pd.DataFrame({

                'texto': ['nunca vi essa frase mas tem texto']

                })




        resp_fit = pd.DataFrame({
                    'aqui': [2, 0, 0], 
                    'explicativo': [0, 1, 0], 
                    'nada': [0, 0, 1], 
                    'nao': [0, 1, 0],
                    'texto': [2, 1, 0]
                }).astype('int64')


        resp_transform = pd.DataFrame({
                    'aqui': [0], 
                    'explicativo': [0], 
                    'nada': [0], 
                    'nao': [0], 
                    'texto': [1]
                }).astype('int64')


        pipe = Pipeline([ 
            ('bow', CountVectorizerPandas(max_features = 5))
        ])

        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_fit)
        pd_testing.assert_frame_equal(pipe.transform(df_trans), resp_transform)

        return
    

    def test_LabelEncoderPandas(self):


        df_fit = pd.DataFrame({

                    'cat1': ['a','b','b'],
                    'cat2': ['a','c','c'],

                })

        df_trans = pd.DataFrame({

                    'cat1': ['K', 'Z'],
                    'cat2': ['a', 'a'],

                })


        resp_fit = pd.DataFrame({
                    'cat1': [0, 1, 1], 
                    'cat2': ['a', 'c', 'c']
                }).astype({'cat1':'category'})


        resp_transform = pd.DataFrame({
                    'cat1': [2, 2],
                    'cat2': ['a', 'a']
                }).astype({'cat1':'category'})


        pipe = Pipeline([ 
            ('LE', LabelEncoderPandas(col = 'cat1'))
        ])      
        
        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_fit)
        pd_testing.assert_frame_equal(pipe.transform(df_trans), resp_transform)       

        return



    def test_PandasFeatureUnion(self):
        df_fit = pd.DataFrame({
            'cat': ['a','a','a','b','b','b','b'],
            'int': [1,1,1,1,1,1,1],
            'float': [1.0,1.0,1.0,1.0,1.0,1.0,1.0]

        })

        
        resp_fit = pd.DataFrame({
            'cat': [0, 0, 0, 1, 1, 1, 1], 
            'int': [1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan], 
            'float': [1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan]

        }).astype({'cat': 'category'})       

        df_trans = pd.DataFrame({
            'cat': ['a','a','a','b','b','b','b'],
            'int': [1,1,1,1,1,1,1],
            'float': [1.0,1.0,1.0,1.0,1.0,1.0,1.0]

        })

        pipe1 = Pipeline([
                    ('categoricals',PandasTypeSelector(include = ['object']) ),
                    ('le', LabelEncoderPandas('cat')),
                ])
                
        pipe2 = Pipeline([
                    ('int',PandasTypeSelector(include = ['int64']) ),
                    ('head_2', CustomTransformer()),
                ])
                
        pipe3 = Pipeline([
                    ('float',PandasTypeSelector(include = ['float64']) ),
                    ('head_21', CustomTransformer()),
                ])

        pipe = Pipeline([ 
            
            ('union', PandasFeatureUnion([ pipe1, pipe2, pipe3 ]))

        ])

        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_fit)


        return

    


    def test_WindowFeatures(self):
        
        df_fit = pd.DataFrame({
            'dt': pd.date_range(start='1/1/2018', periods=8).tolist(),
            'sku':['XXX','XXX','XXX','XXX','ZZZ','ZZZ','ZZZ','ZZZ'],
            'vol':[1,2,3,4,4,3,2,1]
                    })

        

        resp_fit = pd.DataFrame({
            'dt': ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'], 
            'sku': ['XXX', 'XXX', 'XXX', 'XXX', 'ZZZ', 'ZZZ', 'ZZZ', 'ZZZ'], 
            'vol': [1, 2, 3, 4, 4, 3, 2, 1], 
            'vol_3_sku_mean': [np.nan, np.nan, 2.0, 3.0, np.nan, np.nan, 3.0, 2.0], 
            'vol_3_sku_median': [np.nan, np.nan, 2.0, 3.0, np.nan, np.nan, 3.0, 2.0], 
            'vol_3_sku_sum': [np.nan, np.nan, 6.0, 9.0, np.nan, np.nan, 9.0, 6.0], 
            'vol_3_sku_std': [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 1.0, 1.0], 
            'vol_3_sku_max': [np.nan, np.nan, 3.0, 4.0, np.nan, np.nan, 4.0, 3.0], 
            'vol_3_sku_min': [np.nan, np.nan, 1.0, 2.0, np.nan, np.nan, 2.0, 1.0], 
            'vol_3_sku_kurtosis': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'vol_3_sku_skewness': [np.nan, np.nan, 0.0, 0.0, np.nan, np.nan, 0.0, 0.0],
            'vol_3_sku_quant_90': [np.nan, np.nan, 2.8, 3.8, np.nan, np.nan, 3.8, 2.8], 
            'vol_3_sku_quant_10': [np.nan, np.nan, 1.2, 2.2, np.nan, np.nan, 2.2, 1.2],
            'vol_3_sku_trend': [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, -1.0, -1.0], 
            'vol_3_sku_longest_period_above_mean': [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 1.0, 1.0], 
            'vol_3_sku_longest_period_below_mean': [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 1.0, 1.0], 
            'vol_3_sku_longest_consecutive_decreases': [np.nan, np.nan, 0.0, 0.0, np.nan, np.nan, 2.0, 2.0], 
            'vol_3_sku_longest_consecutive_increases': [np.nan, np.nan, 2.0, 2.0, np.nan, np.nan, 0.0, 0.0], 
            'vol_3_sku_last_max': [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 3.0, 3.0], 
            'vol_3_sku_last_min': [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 1.0, 1.0], 
            'vol_3_sku_size': [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 3.0, 3.0]
        
        }).astype({'dt':'datetime64'})



        date_column = 'dt'
        target = 'vol'
        time_ranges = [3]
        grouped_variables = [['sku']]
        rol_functions = [

            'mean',
            'median',
            'sum',
            'std',
            'max',
            'min',
            'kurtosis',
            'skewness',
            'quant_90',
            'quant_10',
            'trend',
            'longest_period_above_mean',
            'longest_period_below_mean',
            'longest_consecutive_decreases',
            'longest_consecutive_increases',
            'last_max',
            'last_min',
            'size',
            #'list'

        ]

        

        pipe = Pipeline([ 
            ('rolling', WindowFeatures(date_column, target, time_ranges, rol_functions, grouped_variables, print_ = False))
        ])
        
        pd_testing.assert_frame_equal(pipe.fit_transform(df_fit), resp_fit)


        return



    ############## NOT IMPLEMENTED 
    
    # def test_DaysToEvent(self):
    #     return
    # def test_ValidateDF(self):
    #     return
    # def test_FilterDF(self):
    #     return
    # def test_CustomFunction(self):
    #     return

    




if __name__ == '__main__':
    unittest.main(verbosity=2)