# Libraries
import os
import re
import sys
import boto3 
import openpyxl
import itertools
import pandas as pd
import awswrangler as wr
from datetime import datetime


class preprocess_module():
    
    
    def __init__(self):
        
        self.gen_time = datetime.now()
        self.class_name = 'preprocessing module'
        
        self.valuewithunit_list = [
            '[0-9]+ kv',
            '[0-9]+ , [0-9]+ kv',
            '[0-9]+, [0-9]+ kv',
            '[0-9]+,[0-9]+ kv',
            '[0-9]+ . [0-9]+ kv',
            '[0-9]+. [0-9]+ kv',
            '[0-9]+.[0-9]+ kv',
            '[0-9]+ kva',
            #'[0-9]+ kvb',
            #'[0-9]+ kvt',
            '[0-9]+ kw',
            '[0-9]+ mw',
            '[0-9]+ mvar',
            '[0-9]+ , [0-9]+ hz',
            '[0-9]+, [0-9]+ hz',
            '[0-9]+,[0-9]+ hz',
            '[0-9]+ . [0-9]+ hz',
            '[0-9]+. [0-9]+ hz',
            '[0-9]+.[0-9]+ hz',
            '[0-9]+ hz',
            '[0-9]+ \ , [0-9]+ hz',
            '[0-9]+\,[0-9]+ hz',
            '[0-9]+ ug',
            '[0-9]+ ugs',
            '[0-9]+ \( \w+ \) ugs',
            '[0-9]+°',
            '[0-9]+ °',
            '[0-9]+º',
            '[0-9]+ º',
            '[0-9]+ graus',
            '[0-9]+,[0-9]+ graus',
            '[0-9]+ , [0-9]+ graus'
        ]
        
        self.mathoperator_list = [
            '+',
            '>',
            '<',
            '=',
            '≥',
            '≤',
            'inferior ou igual à',
            'superior ou igual à',
            'igual ou superior à',
            'igual ou inferior à',
            'inferior ou igual a',
            'superior ou igual a',
            'igual ou superior a',
            'igual ou inferior a',
            'superior ou igual',
            'inferior ou igual',
            'igual ou superior',
            'igual ou inferior',
            'menor ou igual à',
            'maior ou igual à',
            'igual ou menor à',
            'igual ou maior à',
            'menor ou igual a',
            'maior ou igual a',
            'igual ou menor a',
            'igual ou maior a',
            'igual ou menor',
            'igual ou maior',
            'menor ou igual',
            'maior ou igual',
            'superior à',
            'inferior à',
            'superior a',
            'inferior a',
            'superior',
            'inferior',
            'no mínimo',
            'no máximo',
            'no minimo',
            'no maximo',
            'entre'
        ]
        
        return
    
    
    def get_fragments(
        self,
        path:str='s3://ons-ds-dev-entityrecognition/mpo/mvp/data/raw/fragments/fragmentos_ecm_1_5000.xlsx',
        cols:list=['Texto Limpo', 'MpoCentro'],
        sample_size:int=500
        ):
        
        df = wr.s3.read_excel(path, engine='openpyxl', usecols=cols)
        df = df.iloc[0:sample_size]
        #df['Texto Limpo'] = df['Texto Limpo'].str.lower()
        df['MpoCentro'] = df['MpoCentro'].str.lower()
        df = df.dropna(subset=['Texto Limpo'], axis=0)
        
        return df
    
    
    def get_equipments(
        self, 
        sql:str='select guid_equipamento, nom_longoequipamento from e_equipamento',
        database:str='consumo'
        ):
        
        df = wr.athena.read_sql_query(sql=sql, database=database)
        df['nom_longoequipamento'] = df['nom_longoequipamento'].str.lower()
        df = df.dropna(subset=['nom_longoequipamento'], axis=0)
        
        return df
    
    
    def get_equipments_fameq(
        self,
        sql:str='SELECT guid_tipoequipamentosgi, nom_tipoeqp  FROM e_tipoequipamentosgi',
        database:str='consumo'
        ):

        df = wr.athena.read_sql_query(sql=sql, database=database)
        df['nom_tipoeqp'] = df['nom_tipoeqp'].str.lower()
        df = df.dropna(subset=['nom_tipoeqp'], axis=0)
        
        return df
    
    
    def regex_subs_fragments(
        self,
        text
        ):
        
        # seções numéricas
        rule_list = [
            '[0-9]+\.[0-9]+.[0-9]+.[0-9]+'
            '[0-9]+\.[0-9]+.[0-9]+',
            '[0-9]+\.[0-9]+.[0-9]+',
            '[0-9]+\.[0-9]+. '
            '[0-9]\.[0-9]+'
        ]
        for rule in rule_list:
            text = re.sub(f'{rule}', '', text)

        # seções alfabéticas
        text = re.sub(r'(\s[abcde]\)\s+)','. ',text)
        text = re.sub(r'([0-9]\)\s+)|([0-9][0-9]\)\s+)','. ',text)

        # módulos alfanuméricos
        text = re.sub(r'(módulo\s[0-9])|(modulo\s[0-9])|(submódulo\s[0-9].[0-9][0-9].)|(submodulo\s[0-9].[0-9][0-9].)','',text)
        text = re.sub(r'(módulo\s[0-9])|(modulo\s[0-9])|(submódulo\s[0-9])|(submodulo\s[0-9])','',text)
        text = re.sub(r'(módulo)|(submódulo)|(modulo)|(submodulo)','',text)

        # caracteres especiais
        text = re.sub(r'(•)', "", text)
        text = re.sub(r'(\s-\s)', " ", text)
        text = re.sub(r'(-\s)', " ", text)

        # palavras específicas
        text = re.sub(r'((objetivo)[1-9].|(considerações gerais)[1-9].)',(''),text)

        # separação de termos - concatenação indevida
        text = re.sub(' +', ' ', text)
        text = re.sub(r'([a-zzáàâãéèêíïóôõöúçñ][a-zzáàâãéèêíïóôõöúçñ])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])(energização)', r'\1 \2', text)
        text = re.sub(r'([0-9])(desenergização)', r'\1 \2', text)
        text = re.sub(r'([a-rzáàâãéèêíïóôõöúçñ])(energização)', r'\1 \2', text)
        text = re.sub(r'([t-zzáàâãéèêíïóôõöúçñ])(energização)', r'\1 \2', text)
        text = re.sub(r'([a-rzáàâãéèêíïóôõöúçñ])(desenergização)', r'\1 \2', text)
        text = re.sub(r'([t-zzáàâãéèêíïóôõöúçñ])(desenergização)', r'\1 \2', text)
        text = re.sub(r'([a-zzáàâãéèêíïóôõöúçñ])(manobra)', r'\1 \2', text)
        text = re.sub(r'([a-zzáàâãéèêíïóôõöúçñ])(lt)', r'\1 \2', text)
        #text = re.sub(r'[^\S]?(\(.*?\))[^\S]?', r' \1 ', text)
        text = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", text)
        text = re.sub('(&#xa0;)', ' ', text)
        text = re.sub('(\\xa0)', ' ', text)
        text = re.sub('(color:#ffffff)', ' ', text)
        text = re.sub('<\span>', ' ', text)
        text = re.sub('\uf0b7', ' ', text)
        text = re.sub('\uf044', ' ', text)

        # substituir padrões de colunas temporárias para separadores
        text = re.sub(r'-[0-9][a-z][a-z][a-z][a-z]-[a-z][a-z][a-z]','#',text)
        text = re.sub(r'-[0-9][a-z][a-z][a-z][a-z]-[a-z][a-z]','#',text)
        text = re.sub(r'[a-z][a-z][a-z][a-z]-[a-z][a-z][a-z]','#',text)
        text = re.sub(r'[a-z][a-z][a-z][a-z]-[a-z][a-z]','#',text)
        text = re.sub(r'[a-z][a-z][a-z]-[a-z][a-z]','#',text)
        text = re.sub(r'chesf (cron)','#',text)
        text = re.sub(r'execuçãoprocedimentoitem / [a-z][a-z][a-z]-[a-z][a-z]','#',text)
        text = re.sub(r'execuçãoprocedimentoitem / [a-z][a-z][a-z]-[a-z][a-z][a-z]','#',text)
        text = re.sub(r'passocoordenaçãocontrolecomando','#',text)
        text = re.sub(r'execuçãoprocedimentosobjetivo','#',text)
        text = re.sub(r'execuçãoprocedimento','#',text)
        text = re.sub('#', '', text)

        #substituir acentuação
#         text = re.sub(r'[àáâãäå]', 'a', text)
#         text = re.sub(r'[èéêë]', 'e', text)
#         text = re.sub(r'[ìíîï]', 'i', text)
#         text = re.sub(r'[òóôõö]', 'o', text)
#         text = re.sub(r'[ùúûü]', 'u', text)
#         text = re.sub(r'[ç]', 'c', text)
        
        # remover excesso de espaços
        text = re.sub(' +', ' ', text)
        
        # remoção de pontos repetidos
        text = text.split('.')
        text = [i for i in text if i!='']
        text = '.'.join(text)
        
        return text
    
    
    def regex_subs_equipments(
        self, 
        text
        ):
        
        #substituir acentuação
#         text = re.sub(r'[àáâãäå]', 'a', text)
#         text = re.sub(r'[èéêë]', 'e', text)
#         text = re.sub(r'[ìíîï]', 'i', text)
#         text = re.sub(r'[òóôõö]', 'o', text)
#         text = re.sub(r'[ùúûü]', 'u', text)
#         text = re.sub(r'[ç]', 'c', text)
        
        # remover excesso de espaços
        text = re.sub(' +', ' ', text)
        
        # remover letras de sequências numéricas - lt
        text = re.sub(r'(lt) ([0-9]+)([a-z])([0-9]+) (kv)', r'\1 \2\4 \5', text)
        
        # remover letras de sequências numéricas - tr
        text = re.sub(r'(tr) ([0-9]+)([a-z])([0-9]+) (kv)', r'\1 \2\4 \5', text)
        text = re.sub(r'(tr) ([0-9]+)([a-z])([0-9]+) (/) ([0-9]+) (kv)', r'\1 \2\4 \5 \6 \7', text)
        text = re.sub(r'(tr) ([0-9]+) (/) ([0-9]+)([a-z])([0-9]+) (kv)', r'\1 \2 \3 \4\6 \7', text)
        text = re.sub(r'(tr) ([0-9]+)([a-z])([0-9]+) (/) ([0-9]+)([a-z])([0-9]+) (kv)', r'\1 \2\4 \5 \6\8 \9', text)
        
        # acrescentar espaço entre caracteres especiais ou pontuação
        text = re.sub(r'[^\S]?(\(.*?\))[^\S]?', r' \1 ', text)
        
        return text
    
    
    def preprocess_fragments(
        self, 
        df, 
        col
        ):
        
        df['processed'] = df[col].apply(lambda x:self.regex_subs_fragments(x))
        
        return df
    
    
    def preprocess_equipments(
        self, 
        df, 
        col
        ):
        
        df['processed_equipments'] = [x.split(' ')[:-14] for x in df[col]]
        df['processed_equipments'] = df['processed_equipments'].apply(lambda x:[i for i in x if i != ' '])
        df['processed_equipments'] = df['processed_equipments'].apply(lambda x:[i for i in x if i != ''])
        df['processed_equipments'] = df['processed_equipments'].apply(lambda x: ' '.join(x))
        df['processed_equipments'] = df['processed_equipments'].apply(lambda x:self.regex_subs_equipments(x))
        df = df.drop_duplicates()
        
        return df
    
    
    def find_expressions_in_str(
        self,
        text:str,
        rule_list:list,
        exclude_rule_list=None,
        exclude_result_list=None
        ):

        import re

        expressions_list = []
        for rule in rule_list:
            try:
                expressions_list = expressions_list + re.findall(rule, text, re.IGNORECASE)
            except:
                expressions_list = expressions_list

        expressions_list = list(set(expressions_list))
        
        if exclude_rule_list != None:
            for rule in exclude_rule_list:
                expressions_list = [re.sub(rule, '', text, flags=re.IGNORECASE) for text in expressions_list]
        
        if exclude_result_list != None:
            expressions_list = [i for i in expressions_list if i not in exclude_result_list]
                
        return expressions_list
    
    
    def find_equipments_lt(
        self,
        text
        ):
        
        prefix = ['lt [0-9]+ kv', 'lts [0-9]+ kv']
        word_sep = ['.',' ']
        location_sep = ['/']
        rule_list = []
        for rule in list(itertools.product(prefix,word_sep, location_sep)):
            prefix_ = rule[0]
            word_sep_ = rule[1]
            location_sep_ = rule[2]
            rule_list.append(f'{prefix_} \w+ {location_sep_} \w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+) {location_sep_} \w+')
            rule_list.append(f'{prefix_} \w+ {location_sep_} \w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+ {location_sep_} \w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+{word_sep_}\w+ {location_sep_} \w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+ {location_sep_} \w+{word_sep_}\w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+{word_sep_}\w+ {location_sep_} \w+{word_sep_}\w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+{word_sep_}\w+ {location_sep_} \w+{word_sep_}\w+{word_sep_}\w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+{word_sep_}\w+{word_sep_}\w+ {location_sep_} \w+{word_sep_}\w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+{word_sep_}\w+{word_sep_}\w+ {location_sep_} \w+{word_sep_}\w+{word_sep_}\w+{word_sep_}\w+')
            
        exclude_rule_list = [
            'com derivação para',
            'com derivação',
            'a partir da',
            'a partir',
            ' com',
            'e manter o',
            'e manter',
            'fique submetida a',
            'fique submetida',
            ' fique'
        ]
        
        equipment_list = self.find_expressions_in_str(
            text=text,
            rule_list=rule_list,
            exclude_rule_list=exclude_rule_list
            )
            
        for i in range(len(equipment_list)):
            for keyword in ['desenergizacao', 'desenergização', 'Desenergização', 'energizacao', 'energização', 'Energização']:
                if keyword in equipment_list[i]:
                    equipment_list[i] = equipment_list[i][0:equipment_list[i].index(keyword)]
        
        equipment_list = list(set(equipment_list))
        
        return equipment_list
    
    
    def find_equipments_tr(
        self,
        text
        ):
        
        # 1st pattern
        # ---------------------------
        prefix = [
            'atr [0-9]+ / [0-9]+ kv',
            'tr [0-9]+ / [0-9]+ kv',
            'atr [0-9]+ / [0-9]+',
            'tr [0-9]+ / [0-9]+',
            'atr [0-9]+/[0-9]+ kv',
            'tr [0-9]+/[0-9]+ kv',
            'atr [0-9]+/[0-9]+',
            'tr [0-9]+/[0-9]+',
            'atr [0-9]+ / [0-9]+ / [0-9]+ kv',
            'tr [0-9]+ / [0-9]+ / [0-9]+ kv',
            'tr [0-9]+ / [0-9]+ / [0-9]+',
            'atr [0-9]+/[0-9]+/[0-9]+ kv',
            'tr [0-9]+/[0-9]+/[0-9]+ kv',
            'tr [0-9]+/[0-9]+/[0-9]+',
            'atrs [0-9]+ / [0-9]+ kv',
            'trs [0-9]+ / [0-9]+ kv',
            'atrs [0-9]+/[0-9]+ kv',
            'trs [0-9]+/[0-9]+ kv',
            'atrs [0-9]+ / [0-9]+ / [0-9]+',
            'trs [0-9]+ / [0-9]+ / [0-9]+',
            'atrs [0-9]+/[0-9]+/[0-9]+'
            'trs [0-9]+/[0-9]+/[0-9]+'
        ]
        word_sep = [
            '.',
            ' '
        ]
        rule_list = []
        for rule in list(itertools.product(prefix, word_sep)):
            prefix_ = rule[0]
            word_sep_ = rule[1]
            rule_list.append(f'{prefix_} \w+')
            rule_list.append(f'{prefix_} \w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+')
            rule_list.append(f'{prefix_} \w+{word_sep_}\w+{word_sep}(\w+)')
            
        # 2nd pattern
        # ---------------------------
        prefix = [
            'autotransformadores',
            'autotransformador',
            'transformadores',
            'transformador',
            'atrs',
            'atr',
            'trs',
            'tr'
        ]
        specifications = [
            '[0-9]+ / [0-9]+ / [0-9]+ , [0-9]+ kv',
            '[0-9]+ / [0-9]+ / [0-9]+, [0-9]+ kv',
            '[0-9]+ / [0-9]+ / [0-9]+,[0-9]+ kv',
            '[0-9]+ / [0-9]+ / [0-9]+ . [0-9]+ kv',
            '[0-9]+ / [0-9]+ / [0-9]+. [0-9]+ kv',
            '[0-9]+ / [0-9]+ / [0-9]+.[0-9]+ kv',
            '[0-9]+ / [0-9]+ / [0-9]+ kv',
            '[0-9]+/[0-9]+/[0-9]+ , [0-9]+ kv',
            '[0-9]+/[0-9]+/[0-9]+, [0-9]+ kv',
            '[0-9]+/[0-9]+/[0-9]+,[0-9]+ kv',
            '[0-9]+/[0-9]+/[0-9]+ . [0-9]+ kv',
            '[0-9]+/[0-9]+/[0-9]+. [0-9]+ kv',
            '[0-9]+/[0-9]+/[0-9]+.[0-9]+ kv',
            '[0-9]+/[0-9]+/[0-9]+ kv',
            '[0-9]+ / [0-9]+ , [0-9]+ kv',
            '[0-9]+ / [0-9]+, [0-9]+ kv',
            '[0-9]+ / [0-9]+,[0-9]+ kv',
            '[0-9]+ / [0-9]+ . [0-9]+ kv',
            '[0-9]+ / [0-9]+. [0-9]+ kv',
            '[0-9]+ / [0-9]+.[0-9]+ kv',
            '[0-9]+ / [0-9]+ kv',
            '[0-9]+/[0-9]+ , [0-9]+ kv',
            '[0-9]+/[0-9]+, [0-9]+ kv',
            '[0-9]+/[0-9]+,[0-9]+ kv',
            '[0-9]+/[0-9]+ . [0-9]+ kv',
            '[0-9]+/[0-9]+. [0-9]+ kv',
            '[0-9]+/[0-9]+.[0-9]+ kv',
            '[0-9]+/[0-9]+ kv',
            
        ]
        sufix = [
            '- [0-9]+ mva',
            ''
        ]
        for rule in list(itertools.product(prefix, specifications, sufix)):
            prefix_ = rule[0]
            specifications_ = rule[1]
            sufix_ = rule[2]
            rule_list.append(f'{prefix_} {specifications_} {sufix_}')

        # marking in text
        # ---------------------------
        equipment_list = self.find_expressions_in_str(
            text=text,
            rule_list=rule_list
            )

        if equipment_list:
            for i in range(len(equipment_list)):
                for keyword in ['desenergizacao', 'desenergização', 'Desenergização', 'energizacao', 'energização', 'Energização']:
                    if keyword in equipment_list[i]:
                        equipment_list[i] = equipment_list[i][0:equipment_list[i].index(keyword)]

        equipment_list = list(set(equipment_list))            

        return equipment_list
    
    
    def find_equipments_cp(
        self,
        text
        ):
        
        prefix = [
            'bancos de capacitores', 
            'capacitores', 
            'cp'
        ]
        word_sep = [
            'de'
        ]
        sufix = [
            '[0-9]+ kv', 
            '[0-9]+ kv, [0-9]+ kv', 
            '[0-9]+ kv , [0-9]+ kv', 
            '[0-9]+ kv e [0-9]+ kv', 
            '[0-9]+ kv, [0-9]+ kv',
            '[0-9]+ kv, [0-9]+ kv e [0-9]+,[0-9]+ kv',
            '[0-9]+ kv, [0-9]+ kv e [0-9]+ kv',
        ]
        rule_list = []
        for rule in list(itertools.product(prefix,word_sep,sufix)):
            prefix_ = rule[0]
            word_sep_ = rule[1]
            sufix_ = rule[2]
            rule_list.append(f'{prefix_} {word_sep_} {sufix_}')
        
        equipment_list = self.find_expressions_in_str(
            text=text,
            rule_list=rule_list
            )
            
        for i in range(len(equipment_list)):
            for keyword in ['desenergizacao', 'desenergização', 'Desenergização', 'energizacao', 'energização', 'Energização']:
                if keyword in equipment_list[i]:
                    equipment_list[i] = equipment_list[i][0:equipment_list[i].index(keyword)]
        
        equipment_list = list(set(equipment_list))
        
        return equipment_list
    
    
    def find_equipments(
        self,
        df,
        col:str='processed'
        ):
        
        local_df = df.copy()
        local_df['tr'] = local_df[col].apply(lambda x:self.find_equipments_tr(x))
        local_df['lt'] = local_df[col].apply(lambda x:self.find_equipments_lt(x))
        local_df['cp'] = local_df[col].apply(lambda x:self.find_equipments_cp(x))
        df['EQUIPAMENTO'] = local_df['tr'] + local_df['lt'] + local_df['cp']
        
        return df
    
    
    def get_manual_actions(
        self,
        bucket:str,
        path:str
        ):

        import boto3
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(bucket)

        file_list = []
        for file in my_bucket.objects.all():
            file_list.append(file.key)
        file_list = [i for i in file_list if path in i]
        file_list = [i for i in file_list if '.xlsx' in i]

        df_list = []
        uri_prefix = f's3://{bucket}/'
        for file_path in file_list:
            df_list.append(pd.read_excel(uri_prefix+file_path, engine='openpyxl'))

        df = pd.concat(df_list)
        df.drop_duplicates(inplace=True)
        df = df.astype('str')
        df['actions'] = df['acoes'].str.lower()
        df.drop(columns=['acoes'], inplace=True)

        return df


    def create_manual_actions_rules(
        self,
        df,
        col:str='actions'
        ):

        self.manualactions_rules = []
        rule_list = []
        for word in df[col]:
            rule_list.append(f'{word}')

        self.manualactions_rules = list(set(rule_list))
        if 'nan' in self.manualactions_rules:
            self.manualactions_rules.remove('nan')

        return self.manualactions_rules


    def find_manual_actions(
        self,
        df,
        df_manualactions,
        col:str='processed'
        ):

        rule_list = self.create_manual_actions_rules(df_manualactions)
        df['ACAO_MANUAL'] = df[col].apply(lambda x:self.find_expressions_in_str(x, rule_list))         

        return df
    
    
    def create_operative_state_rules(
        self
        ):
        
        self.operative_state_list = [
            #'Reator de linha da SE (\w+) (\w+) (\w+) conectado',
            #'Reator de linha da SE (\w+) (\w+) conectado',
            #'Reator de linha da SE (\w+) conectado',
            'condições especificadas na \w+-\w+ . \w+ . \w+',
            'condicoes especificadas na \w+-\w+ . \w+ . \w+',
            'condições especificadas na \w+-\w+.\w+.\w+',
            'condicoes especificadas na \w+-\w+.\w+.\w+',
            'indisponível',
            'indisponivel',
            'indisponíveis',
            'indisponiveis',
            'indisponibilidade',
            'ltc na posição [0-9]+\w+',
            'ltc na posicao [0-9]+\w+',
            'com indisponibilidade',
            'fluxo de potência ativa',
            'fluxo de potencia ativa',
            'indisponibilidade simples',
            'fluxo nulo no transformador',
            'independente das condições operativas da ute',
            'independente das condicoes operativas da ute',
            'ligado',
            'ligada',
            'fechado',
            'fechada',
            'fechadas',
            'fechados',
            'desligado',
            'desligada',
            'desligados',
            'desligadas',
            'conectado',
            'conectada',
            'conectados',
            'conectadas',
            'em operação',
            'em operacao',
            'desconectado',
            'desconectada',
            'desconectados',
            'desconectadas',
            'sincronizada',
            'sincronizado',
            'sincronizados',
            'sincronizadas',
            'fora de operação',
            'fora de operacao',
            'fora da operação',
            'fora da operacao',
            'conectado para ser energizado',
            'carga ativa em pelo menos um dos circuitos',
            'restrição de regime permanente',
            'restricao de regime permanente',
            'interligado',
            'interligada',
            'interligados',
            'interligadas',
            'inserido',
            'inserida',
            'baipassado',
            'energizado em vazio',
            'em operação automática',
            'em operacao automatica',
            'sobrecarga',
            'sobrecarregado',
            'sobrecarregada',
            'sobrecarregados',
            'sobrecarregadas',
            'aberto',
            'aberta',
            'abertos',
            'abertas',
            'sem carga',
            'em carga',
            'em vazio',
            'fluxo nulo',
            'com folga',
            'em série',
            'em serie',
            'sob carga',
            'modo automático',
            'modo automatico',
            'em carregamento',
            'em barramento',
            'em um barramento',
            'energizado',
            'energizada',
            'energizados',
            'energizadas',
            'desenergizado',
            'desenergizada',
            'desenergizados',
            'desenergizados',
            'opere sozinha em um barramento',
            'energizado juntamente com a ltb',
            'energizado em pelo menos um circuito',
            'ajustado',
            'ajustada',
            'ajustados',
            'ajustadas',
            'bloqueado',
            'bloqueada',
            'bloqueados',
            'bloqueadas',
            'disponível para ser sincronizada',
            'disponível para ser sincronizado',
            'disponivel para ser sincronizada',
            'disponivel para ser sincronizado'
        ]

        rule_list = []
        for expression in self.operative_state_list:
            rule_list.append(f'{expression}')

        self.operative_state_list = list(set(rule_list))
        if 'nan' in self.operative_state_list:
            self.operative_state_list.remove('nan')

        return self.operative_state_list
    
    
    def find_operativestate(
        self,
        df,
        col:str='processed'
        ):

        rule_list = self.create_operative_state_rules()
        df['ESTADO_OPERATIVO'] = df[col].apply(lambda x:self.find_expressions_in_str(x, rule_list))         

        return df
    
    
    def create_valuewithunit_rules(
        self
        ):
        
        rule_list = []
        for expression in self.valuewithunit_list:
            rule_list.append(f'{expression}')

        self.valuewithunit_list = list(set(rule_list))
        if 'nan' in self.valuewithunit_list:
            self.valuewithunit_list.remove('nan')

        return self.valuewithunit_list
    
    
    def find_valuewithunit(
        self,
        df,
        col:str='processed'
        ):

        rule_list = self.create_valuewithunit_rules()
        df['VALOR_COM_UNID.MEDIDA'] = df[col].apply(lambda x:self.find_expressions_in_str(x, rule_list))         

        return df
    
    
    def create_substation_rules(
        self
        ):
        
        self.substation_list = [
            'se \w+',
            'se \w+ [0-9]+',
            'se \w+ \w+',
            'se \w+ \w+ [0-9]+',
            'se \w+ \w+ \w+',
            'se \w+ \w+ \w+ [0-9]+',
            'se \w+ \w+ \w+ \w+ [0-9]+'
        ]
        
        rule_list = []
        for expression in self.substation_list:
            rule_list.append(f'{expression}')

        self.substation_list = list(set(rule_list))
        if 'nan' in self.substation_list:
            self.substation_list.remove('nan')

        return self.substation_list
    
    
    def find_substation(
        self,
        df,
        col:str='processed'
        ):

        rule_list = self.create_substation_rules()
        exclude_rule_list = [
            'se especificados são iguais à [0-9]+',
            'se especificados são iguais a [0-9]+',
            'se especificados sao iguais a [0-9]+',
            'se especificados são iguais à',
            'se especificados são iguais a',
            'se especificados sao iguais a',
            'se especificados são iguais',
            'se especificados sao iguais',
            'se especificados são',
            'se especificados sao',
            'se especificados',
            'se as condições especificadas',
            'se as condicoes especificadas',
            'se as condicões',
            'se as condicoes',
            'se auto iniciando pelo',
            'se tratar de manobras',
            'se  iniciando pelo',
            'se  pelo terminal',
            'se auto iniciando',
            'se  iniciando',
            'se  pelo',
            'se onde',
            'se tratar de',
            'se tratar',
            'se transformador e realizado',
            'se transformador em sentido',
            'se transformador em',
            'transformador 2',
            'transformador',
            'autotransformador',
            'ângulo máximo em',
            'ângulo máximo para',
            'ângulo máximo',
            'angulo maximo em',
            'angulo maximo para',
            'angulo maximo',
            'inferior ou igual',
            'inferior ou i',
            'inferior ou',
            'sentido do [1-9]',
            'sentido do',
            'sentido da',
            'sentido',
            'esteja',
            'energização',
            'energizacao',
            'tensão na',
            'tensao na',
            'e para',
            'kv [1-9]+',
            'lt [1-9]+',
            ' [1-9][1-9][1-9]',
            'sobrecarga nas 0',
            'superior à [0-9]+',
            'superior a [0-9]+',
            'com folga',
            ' e proibida',
            ' de forma',
            ' auto ljat [0-9]+',
            ' auto ljat',
            'se houver'
        ]
        
        exclude_result_list = [
            '',
            ' ',
            's',
            'se',
            'se as',
            'se ',
            'se  ',
            ' se',
            'sa',
            'sis',
            'se  e'
        ]
        
        df['SUBESTACAO'] = df[col].apply(
            lambda x:self.find_expressions_in_str(
                x, 
                rule_list, 
                exclude_rule_list, 
                exclude_result_list)
        )         

        return df
    
    
    def create_mathoperator_rules(
        self
        ):
        
        rule_list = []
        for expression in self.mathoperator_list:
            rule_list.append(f'{expression}')

        self.mathoperator_list = list(set(rule_list))
        if 'nan' in self.mathoperator_list:
            self.mathoperator_list.remove('nan')

        return self.mathoperator_list
    
    
    def find_mathoperator(
        self,
        df,
        col:str='processed'
        ):

        rule_list = self.create_mathoperator_rules()
        df['OPERADOR_MATEMATICO'] = df[col].apply(lambda x:self.find_expressions_in_str(x, rule_list))         

        return df
    
    
    def create_powerplant_rules(
        self
        ):
        
        prefix_list = [
            'uhe',
            'pch',
            'ufv',
            'utn',
            'ute',
            'usina',
            'usinas',
            'uhe / se'
        ]

        powerplant_list = [
            '[a-zA-Z]+',
            '[a-zA-Z]+ [a-zA-Z]+',
            '[a-zA-Z]+ [a-zA-Z]+ [a-zA-Z]+',
        ]

        sufix_list = [
            '[0-9] ',
            'i',
            'ii',
            'iii',
            'iv'
        ]
        
        additional_powerplant_list = [
            'uhe \w+ [0-9]+ e [0-9]+',
            'uhes \w+ [0-9]+ e [0-9]+',
            'utn \w+ [0-9]+ e [0-9]+',
            'utns \w+ [0-9]+ e [0-9]+',
            'ute \w+ [0-9]+ e [0-9]+',
            'utes \w+ [0-9]+ e [0-9]+',
            'ufv \w+ [0-9]+ e [0-9]+',
            'ufvs \w+ [0-9]+ e [0-9]+',
            'usina eolica',
            'usinas eolicas'
        ]
        
        rule_list = []
        for rule in list(itertools.product(prefix_list,powerplant_list, sufix_list)):
            rule_list.append(f'{rule[0]} {rule[1]}')
            rule_list.append(f'{rule[0]} {rule[1]} {rule[2]}')
        
        self.powerplant_list = rule_list + additional_powerplant_list

        self.powerplant_list = list(set(self.powerplant_list))
        if 'nan' in self.powerplant_list:
            self.powerplant_list.remove('nan')

        return self.powerplant_list
    
    
    def find_powerplant(
        self,
        df,
        col:str='processed'
        ):
        
        rule_list = self.create_powerplant_rules()
        exclude_rule_list = [
            '[0-9]+ hz',
            '[0-9]+ kv',
            '[0-9][0-9]',
            '[0-9][0-9][0-9]',
        ]
        df['USINA'] = df[col].apply(
            lambda x:self.find_expressions_in_str(x, rule_list, exclude_rule_list)
        )

        return df
    
    
    def create_inequalities_rules(
        self
        ):
        
        prefix_list = [
            'SE \w+ \w+ \w+',
            'SE \w+ \w+',
            'SE \w+ ',
            ' δ',
            ' V',
            'Tensão',
            'Tensão pré-manobra \w+ SE \w+ \w+ \w+ \w+',
            'Tensão pré-manobra \w+ SE \w+ \w+ \w+',
            'Tensão pré-manobra \w+ SE \w+ \w+',
            'Tensão pré-manobra \w+ SE \w+',
            'Tensão \w+ SE onde se inicia a manobra',
            'Tensão \w+ SE que inicia a manobra',
            'Tensão de pré-energização',
            'Tensão \w+ SE \w+ \w+ \w+ \w+',
            'Tensão \w+ SE \w+ \w+ \w+',
            'Tensão \w+ SE \w+ \w+',
            'Tensão \w+ SE \w+',
            'UHE \w+ \w+ \w+ \w+',
            'UHE \w+ \w+ \w+',
            'UHE \w+ \w+',
            'UHE \w+',
            'ΔV',
            'Δϴ',
            'Δδ',
            'V\(\w+\)',
            'V \(\w+\)',
            'Diferença angular de',
            'Diferença angular',
            'Defasagem angular de',
            'Defasagem angular',
        ]
        
        middle_list = [
            '>',
            '<',
            '=',
            '≥',
            '≤',
            'inferior ou igual \w+',
            'superior ou igual \w+',
            'igual ou superior \w+',
            'igual ou inferior \w+',
            'superior \w+',
            'inferior \w+',
            'igual \w+',
            'no mínimo',
            'no máximo',
            'entre'
        ]
        
        sufix_list = [
            '[0-9]+ kv',
            '[0-9]+ kw',
            '[0-9]+ mw',
            '[0-9]+ mvar',
            '[0-9]+ hz',
            '[0-9]+°',
            '[0-9]+ °',
            '[0-9]+º',
            '[0-9]+ º',
            '[0-9]+ graus',
            '[0-9]+ %', 
            '[0-9]+%'
        ]
        
        additional_inequality_list = [
            '[0-9]+ kV ≤ V(\w+)',
            '[0-9]+ kV ≤ V (\w+)',
            '[0-9]+ kV ≤ V(\w+ \w+)',
            '[0-9]+ kV ≤ V (\w+ \w+)',
            '[0-9]+ kV ≤ V(\w+ \w+ \w+)',
            '[0-9]+ kV ≤ V (\w+ \w+ \w+)',
            '\w+ - VBarra entre + [0-9]+ kV e - [0-9]+ kV',
            '\w+ - VBarra entre [0-9]+ kV e [0-9]+ kV',
            'variação de potência reativa de até + [0-9]+ Mvar',
            'variação de potência reativa de até - [0-9]+ Mvar',
            'variação de potência reativa de até [0-9]+ Mvar',
            '[0-9]+ kV ≤ Tensão \w+ SE \w+ \w+ \w+ \w+ ≤ [0-9]+ kV',
            '[0-9]+ kV ≤ Tensão \w+ SE \w+ \w+ \w+ ≤ [0-9]+ kV',
            '[0-9]+ kV ≤ Tensão \w+ SE \w+ \w+ ≤ [0-9]+ kV',
            '[0-9]+ kV ≤ Tensão \w+ SE \w+ ≤ [0-9]+ kV',
            '[0-9]+ kV ≤ V \(\w+ \w+\) ≤ [0-9]+ kV V \(\w+ \w+\) ≥ [0-9]+ kV',
            '[0-9]+ kV ≤ V\(\w+ \w+\) ≤ [0-9]+ kV V\(\w+ \w+\) ≥ [0-9]+ kV',
            '[0-9]+ kV ≤ V \(\w+\) ≤ [0-9]+ kV V \(\w+ \w+\) ≥ [0-9]+ kV',
            '[0-9]+ kV ≤ V\(\w+\) ≤ [0-9]+ kV V\(\w+ \w+\) ≥ [0-9]+ kV',
            '[0-9]+ kV ≤ V \(\w+ \w+\) ≤ [0-9]+ kV V \(\w+\) ≥ [0-9]+ kV',
            '[0-9]+ kV ≤ V\(\w+ \w+\) ≤ [0-9]+ kV V\(\w+\) ≥ [0-9]+ kV',
            '[0-9]+ kV ≤ V \(\w+\) ≤ [0-9]+ kV V \(\w+\) ≥ [0-9]+ kV',
            '[0-9]+ kV ≤ V\(\w+\) ≤ [0-9]+ kV V\(\w+\) ≥ [0-9]+ kV',
            'UHE \w+ \w+ \w+ \w+ deve estar com o mínimo de [0-9]+ unidades sincronizadas',
            'UHE \w+ \w+ \w+ deve estar com o mínimo de [0-9]+ unidades sincronizadas',
            'UHE \w+ \w+ deve estar com o mínimo de [0-9]+ unidades sincronizadas',
            'UHE \w+ deve estar com o mínimo de [0-9]+ unidades sincronizadas',
            'SE \w+ \w+ \w+ \w+ com folga para um fornecimento de no mínimo [0-9]+ Mvar',
            'SE \w+ \w+ \w+ com folga para um fornecimento de no mínimo [0-9]+ Mvar',
            'SE \w+ \w+ com folga para um fornecimento de no mínimo [0-9]+ Mvar',
            'SE \w+ com folga para um fornecimento de no mínimo [0-9]+ Mvar'
        ]
        
        rule_list = []
        for rule in list(itertools.product(prefix_list, middle_list, sufix_list)):
            rule_list.append(f'{rule[0]} {rule[1]} {rule[2]}')
        
        rule_list = rule_list + additional_inequality_list
        self.inequalities_rules = list(set(rule_list))
        if 'nan' in self.inequalities_rules:
            self.inequalities_rules.remove('nan')

        return self.inequalities_rules
    
    
    def find_inequalities(
        self,
        df,
        col:str='processed'
        ):

        rule_list = self.create_inequalities_rules()
        df['INEQUACAO'] = df[col].apply(lambda x:self.find_expressions_in_str(x, rule_list))         

        return df
    
    
    def create_synthetic_samples(
        self,
        df,
        entity_col:str='EQUIPAMENTO',
        text_col:str='processed'
        ):
        
        marking_dict = {
            'text':[],
            'entity':[],
            'start':[],
            'end':[]
        }
        
        for index, row in df.iterrows():
            for entity in row[entity_col]:
                start_list = [i for i in range(len(row[text_col])) if row[text_col].startswith(entity, i)]
                end_list = [i+len(entity) for i in start_list]
                marking_dict['text'].append(row[text_col])
                marking_dict['entity'].append(entity)
                marking_dict['start'].append(start_list)
                marking_dict['end'].append(end_list)
        
        marking_df = pd.DataFrame(marking_dict)
        marking_df['label'] = entity_col
        marking_df = marking_df.set_index(['text','entity','label']).apply(pd.Series.explode).reset_index()
        marking_df.sort_values(by=['text','start','end'], axis=0, ascending=True, inplace=True)
        marking_df.drop_duplicates(subset=['text','label','start'], keep='last', inplace=True)
        
        return marking_df
    
    
    def get_samples(
        self,
        bucket:str='ons-ds-dev-entityrecognition',
        path:str='mpo/mvp/data/raw/incremental/predefined/'
        ):

        import boto3
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(bucket)
        df = pd.DataFrame()

        file_list = []
        for file in my_bucket.objects.all():
            file_list.append(file.key)
        file_list = [i for i in file_list if path in i]
        file_list = [i for i in file_list if '.parquet' in i]

        if file_list:
            df_list = []
            uri_prefix = f's3://{bucket}/'
            for file_path in file_list:
                df_local = pd.read_parquet(uri_prefix+file_path)
                df_local = df_local.drop_duplicates()
                df_local = df_local.dropna()
                df_list.append(df_local)

            df = pd.concat(df_list)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df
    
    
    def get_processed_samples(
        self,
        bucket:str='ons-ds-dev-entityrecognition',
        path:str='mpo/mvp/data/processed/'
        ):

        import boto3
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(bucket)
        df = pd.DataFrame()

        file_list = []
        for file in my_bucket.objects.all():
            file_list.append(file.key)
        file_list = [i for i in file_list if path in i]
        file_list = [i for i in file_list if '.parquet' in i]
        file_list = [i for i in file_list if 'inequality' not in i]

        if file_list:
            df_list = []
            uri_prefix = f's3://{bucket}/'
            for file_path in file_list:
                print(uri_prefix+file_path)
                df_local = pd.read_parquet(uri_prefix+file_path)
                df_local = df_local.drop_duplicates()
                df_local = df_local.dropna()
                df_list.append(df_local)

            df = pd.concat(df_list)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df
    
    
    def get_processed_samples_inequalities(
        self,
        bucket:str='ons-ds-dev-entityrecognition',
        path:str='mpo/mvp/data/processed/'
        ):

        import boto3
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(bucket)
        df = pd.DataFrame()
        
        file_list = []
        for file in my_bucket.objects.all():
            file_list.append(file.key)
        file_list = [i for i in file_list if path in i]
        file_list = [i for i in file_list if '.parquet' in i]
        file_list = [i for i in file_list if 'inequality' in i]

        if file_list:
            df_list = []
            uri_prefix = f's3://{bucket}/'
            for file_path in file_list:
                print(uri_prefix+file_path)
                df_local = pd.read_parquet(uri_prefix+file_path)
                df_local = df_local.drop_duplicates()
                df_local = df_local.dropna()
                df_list.append(df_local)

            if len(df_list)==1:
                df = df_list[0]
            else:
                df = pd.concat(df_list)

            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df