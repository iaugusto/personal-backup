import os
import sys
import boto3
import subprocess
from zipfile import ZipFile

stage = 'prod'
bucket = f'ons-ds-{stage}-entityrecognition'
path = 'mpo/mvp/python-modules'
file = 'ons-mvp-entity-recognition'

s3 = boto3.resource('s3')
file_path = f'{path}/{file}.zip'
print(file_path)
s3.Bucket(bucket).download_file(file_path, 'file.zip')

with ZipFile('file.zip', 'r') as zipObj:
    zipObj.extractall('root')
print('extracted files to root folder')

print('')
print('current dir: \n', os.getcwd())
print('')
print('files in dir: \n', os.listdir(os.getcwd()))
print('')
print('files in root dir: \n', os.listdir(os.getcwd()+'/root'))

result = subprocess.run('pip install spacy', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result)

result = subprocess.run('python -m spacy download pt_core_news_sm', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result)

result = subprocess.run('pip install sagemaker-studio-image-build', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result)

command = f'python3 {os.getcwd()}/root/ons-mvp-entity-recognition/src/train.py {stage}'
print('running command: \n', command)
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result)