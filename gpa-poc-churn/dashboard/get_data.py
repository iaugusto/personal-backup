import pandas as pd
from sqlalchemy import create_engine
import psycopg2 as p

#import utils


class GetData:
    def __init__(self, date_min, date_max):
        
        
        user = 'medicao_user'
        password = 'medicao_pass'
        predict_table = 'predicoes'
        raw_table = 'medicoes'
        target_table = 'variavel_alvo'
        db = 'medicao_indireta_db'        
        
        #engine = create_engine(f'postgresql://{user}:{password}@127.0.0.1:5432/{db}')
        engine = create_engine(f'postgresql://{user}:{password}@192.168.0.240:5432/{db}')

        self.conn = engine.connect()
        
        ##utils.log('Connection successfull')
        
        self.target_table = target_table
        self.predict_table = predict_table
        self.raw_table = raw_table
        self.date_min = date_min
        self.date_max = date_max
        
    def get_raw(self):
                
        query =  f'''
            
            SELECT * FROM {self.raw_table} 
            WHERE  date >= '{self.date_min}'
            AND date < '{self.date_max}'
            
            '''   
            
        r = self.conn.execute(query)

        df = pd.DataFrame(
            r.fetchall(),
            columns = ['id', 
                    'equip', 
                    'date', 
                    'temperatura',
                    'pressao', 
                    'corrente',
                    'nivel',
                    'velocidade'
                    ]
            ).astype({
                    
                    'id':'str', 
                    'equip':'str', 
                    'date':'datetime64[ns]', 
                    'temperatura':'float32',
                    'pressao':'float32', 
                    'corrente':'float32',
                    'nivel':'float32',
                    'velocidade':'float32'

        })

        #utils.log(f'Got features dataset with shape {df.shape} ')
        
        return df

    def get_target(self):
        
        query = f'''
            SELECT raw.id, raw.date, target.vibracao 
            FROM {self.raw_table} AS raw
            LEFT JOIN {self.target_table} AS target
            ON raw.id = target.id
            WHERE  raw.date >= '{self.date_min}'
            AND date < '{self.date_max}'
            
            '''

        r = self.conn.execute(query)

        df = pd.DataFrame(
            
            r.fetchall(),
            
            columns = [
                    'id', 
                    'date',
                    'vibracao'
                    ]
            
            ).astype({
                    
                    'id':'str', 
                    'date':'datetime64[ns]',
                    'vibracao':'float32'

                }
        )

        #utils.log(f'Got target dataset with shape {df.shape} ')
        return df
    
    
    def get_prediction(self):
        query =  f'''
            SELECT date, predict FROM {self.predict_table} 
            WHERE date >= '{self.date_min}'
            AND date < '{self.date_max}'
            '''   
            
        r = self.conn.execute(query)

        df = pd.DataFrame(
            r.fetchall(),
            columns = ['date', 'predict']
            ).astype({
                    'date':'datetime64[ns]', 
                    'predict':'float32'
        })

        #utils.log(f'Got prediction dataset with shape {df.shape} ')

        return df

    def close_connection(self):
        self.conn.close()
        #utils.log('Connection closed')
        return
    
    
            
if __name__ == '__main__':
    
    getdata = GetData(date_min = '2019-01-01')

    print('Conectado!!!!111!!!!!')