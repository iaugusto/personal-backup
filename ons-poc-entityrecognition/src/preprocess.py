import nltk
import boto3
import numpy as np
import regex as re
import pandas as pd
import awswrangler as wr
from datetime import datetime
from bleach.sanitizer import Cleaner

nltk.download('punkt')


class preprocess_data():
    
    
    # ------------------
    def __init__(self):
        self.name = 'Preprocessing module'
        self.gen_date = datetime.now()
        print(self.name)
        print('started at ', self.gen_date)
        return
    # ------------------
    
    
    # ------------------
    def read_from_s3(self):
        
        bucket = 'ons-dl-dev-landing'
        file = 'ons/mpo_ds_nlp/raw/mpo_baseline.xlsx'
        obj_uri = 's3://{}/{}'.format(bucket, file)

        df = wr.s3.read_excel(path=obj_uri, engine = 'openpyxl')
    
        return df
    # ------------------
    
    
    # ------------------
    def remove_noise_with_regex(self, texto):
        
        texto = re.sub('(&#xa0;)', ' ', texto)
        texto = re.sub('(\\xa0)', ' ', texto)
        texto = re.sub('(color:#ffffff)', ' ', texto)
        texto = re.sub('<\span>', ' ', texto)
        texto = re.sub('\uf0b7', ' ', texto)
        texto = re.sub('\uf044', ' ', texto)
    
        return texto
    # ------------------
    
    
    # ------------------
    def clean_text(
        self,
        dataset:pd.DataFrame,
        columns
        ):
        
        # asserting that columns' variable type is list
        
        if type(columns) != list:
            columns = [columns]
        
        # removing known tags with regex
        
        for col in columns:
            dataset[col] = dataset[col].apply(lambda x:self.remove_noise_with_regex(x))
            
        # defining cleaner agent from bleach
        
        cleaner = Cleaner(tags=[], attributes=[], styles=[], protocols=[], strip=True, strip_comments=True, filters=None)
        
        for col in columns:
            dataset[col] = dataset[col].apply(lambda x:cleaner.clean(x))
    
        return dataset
    # ------------------
    
    
    # ------------------
    def get_baseline_text(
        self,
        columns:str='texto'
        ):
        
        dataset = self.read_from_s3()
        dataset = self.clean_text(dataset, columns=columns)
        
        return dataset
    # ------------------
    
    
    # ------------------
    def get_subsections(
        self,
        dataset:pd.DataFrame,
        column:str='texto',
        get_actions:bool=True
        ):

        subsection_dict = {
            'reference':[],
            'subsection_list':[],
            'subsection_txt':[]
        }

        for index, row in dataset.iterrows():

            base_text = row[column]
                       
            # find subsection structure
            subitems = re.findall('[0-99].[0-99].?.?.', base_text)
            # remove samples containing spaces and punctuation
            subitems = [item for item in subitems if ('.' in item) & (')' not in item) & ('(' not in item)]
            # remove samples containing letters 
            subitems = [item for item in subitems if re.search('[a-zA-Z]', item) == None]
            # remove noise from other numeric entries
            subitems = [subitems[i] for i in range(len(subitems)-1) if int(subitems[i][0]) >= int(subitems[i-1][0])]
            
            if len(subitems) == 0:
                
                subsection_dict['reference'].append(index)
                subsection_dict['subsection_list'].append('')
                subsection_dict['subsection_txt'].append(base_text)
            
            elif len(subitems) == 1:
                
                beg = base_text.index(subitems[0])
                end = len(base_text)
                subsection_dict['reference'].append(index)
                subsection_dict['subsection_list'].append(str(subitems[0]))
                subsection_dict['subsection_txt'].append(base_text[beg:end])
            
            else:
                
                for n in range(len(subitems)-1):
                    
                    beg = base_text.index(subitems[n])
                    end = base_text.index(subitems[n+1])
                    subsection_dict['reference'].append(index)
                    subsection_dict['subsection_list'].append(str(subitems[n]))
                    subsection_dict['subsection_txt'].append(base_text[beg:end])
                
                beg = base_text.index(subitems[-1])
                end = len(base_text)
                subsection_dict['reference'].append(index)
                subsection_dict['subsection_list'].append(str(subitems[n]))
                subsection_dict['subsection_txt'].append(base_text[beg:end])

        output_dataset = pd.DataFrame(subsection_dict, columns=['reference','subsection','subsection_txt'])
        
        if get_actions:
            output_dataset['actions'] = output_dataset['subsection_txt'].apply(
                lambda x:self.get_actions_from_subsection_text(txt=x))

        return output_dataset 
    # ------------------
    
    
    # ------------------
    def get_actions_from_subsection_text(
        self,
        txt:str
        ):      
        
        agent="      "
            
        if agent in txt:
            actions = txt.split(agent)
            actions = [action for action in actions if action !='']
        else:
            actions = None
            
        return actions 
    # ------------------
    
    
    # ------------------
    def get_region(
        self,
        txt:str
        ):
    
        if 'Região Sul' in txt:
            return 'SUL'

        elif ('Sudedeste/Centro Oeste' in txt) | ('Sudeste/Centro Oeste' in txt):
            return 'SUDESTE'

        elif 'Região NE' in txt:
            return 'NORDESTE'

        elif 'Região Norte' in txt:
            return 'NORTE'

        else:
            return 'GERAL'
    # ------------------
    
    
    # ------------------
    def separate_df_by_region(
        self,
        dataset:pd.DataFrame,
        column:str
        ):
    
        dataset[column] = dataset[column].astype(str)
        dataset['region'] = dataset[column].apply(lambda x:self.get_region(x))

        dataset_list = []
        unique_values = set(dataset['region'].to_list())
        for region in unique_values:
            dataset_list.append([region, dataset[dataset['region']==region]])

        return dataset_list
    # ------------------
    
    
    # ------------------
    def separate_df_by_region_with_centerinfo(
        self,
        dataset:pd.DataFrame,
        column:str
        ):
    
        dataset[column] = dataset[column].astype(str)

        dataset_list = []
        unique_values = set(dataset[column].to_list())
        for region in unique_values:
            dataset_list.append([region, dataset[dataset[column]==region]])

        return dataset_list
    # ------------------
    
    
    # ------------------
    def split_into_sentences(
        self,
        text,
        len_threshold:int=3
        ):
        
        alphabets= "([A-Za-z])"
        upperstr = "([A-Z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov)"
        
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = text.replace(")", " ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        sentences = [i for i in sentences if len(i)>len_threshold]
        
        return sentences
    # ------------------
    
    
    # ------------------
    def get_split_entity_sequence(
        self,
        dataset:pd.DataFrame,
        column:str
        ):
        
        sentence_list = []
        if 'equipment' in dataset.columns:
            equipment_list = []
        
        for index, row in dataset.iterrows():
            sentences = self.split_into_sentences(row[column])
            sentence_list = sentence_list + sentences
            if 'equipment' in dataset.columns:
                equipment_list = equipment_list + [row['equipment']]*len(sentences)
        output_dataset = pd.DataFrame()
        output_dataset[column] = sentence_list
        
        if 'equipment' in dataset.columns:
                output_dataset['equipment'] = equipment_list
        
        return output_dataset
    # ------------------