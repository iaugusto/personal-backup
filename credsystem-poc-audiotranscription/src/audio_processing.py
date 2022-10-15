import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
import speech_recognition as sr


class audio_processing():

    def __init__(
        self,
        input_directory:str=os.getcwd()
        ):

        self.description = 'Classe para processamento de arquivos de audio no formato mp3.'
        self.dir = input_directory + '/'
        self.temp_file_name = 'temp_file.wav'
        self.file_name_separator = ' - '
        self.chunks = None
        self.transcription = None
        self.file_list = None
        self.field_dict = None
        self.output_df = None
        
        return None

    def divide_chunks(
        self, 
        audio, 
        chunksize
        ):

        for i in range(0, len(audio), chunksize):
            yield audio[i:i + chunksize]

    def ingest_file(
        self,
        file:str, 
        chunksize:int=20000
        ):

        audio = AudioSegment.from_mp3(self.dir + file)
        self.chunks = list(self.divide_chunks(audio, chunksize))
        
        return self.chunks

    def transcribe_chunks(
        self,
        cunks
        ):

        self.transcription = ""
        r = sr.Recognizer()
        for index, chunk in enumerate(self.chunks):
            chunk.export(self.temp_file_name,format='wav')
            with sr.AudioFile(self.temp_file_name) as source:
                audio_chunk = r.record(source)
                try:
                    transcription_chunk = r.recognize_google(audio_chunk, language='pt-BR')
                    self.transcription = self.transcription + (transcription_chunk) + " "
                except Exception as e:
                    pass
        
        return self.transcription

    def process_mp3_file(
        self,
        file:str, 
        chunksize:int=20000
        ):

        self.ingest_file(file, chunksize)
        self.transcribe_chunks(self.chunks)
        os.remove(self.temp_file_name)
        
        return self.transcription

    def get_audio_files(self):
        
        self.file_list = os.listdir(self.dir)
        self.file_list = [i for i in self.file_list if len(i.split(self.file_name_separator))==8]
        
        return self.file_list

    def structure_data_from_file_list(self):

        self.get_audio_files()
        
        if self.file_list == False:
            return None

        self.field_dict = {
            'nome_cliente':[],
            'codigo_cliente':[],
            'cpf':[],
            'data_atendimento':[],
            'codigo_de_atendimento':[],
            'unknown_1':[],
            'unknown_2':[],
            'unknown_3':[],
            'transcribed_audio':[]
        }

        for file in self.file_list:
            fields = file.split(self.file_name_separator)
            self.field_dict['nome_cliente'].append(fields[0])
            self.field_dict['codigo_cliente'].append(fields[1])
            self.field_dict['cpf'].append(fields[2])
            self.field_dict['data_atendimento'].append(fields[3])
            self.field_dict['codigo_de_atendimento'].append(fields[4])
            self.field_dict['unknown_1'].append(fields[5])
            self.field_dict['unknown_2'].append(fields[6])
            self.field_dict['unknown_3'].append(fields[7])
            self.field_dict['transcribed_audio'].append(self.process_mp3_file(file=file))

        self.output_df = pd.DataFrame(self.field_dict)

        return self.output_df