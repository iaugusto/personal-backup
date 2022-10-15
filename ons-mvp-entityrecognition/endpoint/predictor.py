import os
import json
import flask
import spacy
import logging
from flask import Flask

#Load models
nlp_predefined = spacy.load('/opt/program/artifacts/predefined')
nlp_inequalities = spacy.load('/opt/program/artifacts/inequalities')
app = Flask(__name__)

# The flask app for getting status
@app.route('/ping', methods=['GET'])
def ping():
    
    # Check if the classifier was loaded correctly
    health = (nlp_predefined is not None) & (nlp_inequalities is not None)
    
    # response from nlp model
    # testar - retornar string ao inves de flask.response
    preset_entity_list = ['LOC', 'MISC', 'ORG', 'PER']
    known_labels = nlp_predefined.get_pipe('ner').labels + nlp_inequalities.get_pipe('ner').labels
    known_labels = [i for i in known_labels if i not in preset_entity_list]
    if health:
        return flask.Response(response=json.dumps(known_labels), status=200, mimetype='application/json')
    else:
        return flask.Response(response='\n', status=404, mimetype='application/json')

# The flask app for serving predictions
@app.route('/invocations', methods=['POST'])
def transformation():
    
    # Process/Parse input
    input_json = flask.request.get_json()
    fragment = input_json.get('fragmento', None)
    get = input_json.get('get', None)
    
    # response for fragments
    if fragment:
        
        recognition_dict = {
            'entidades':[]
        }

        doc = nlp_predefined(fragment)
        for i in range(len(doc.ents)):
            entity_obj = {
                'tipo_entidade':doc.ents[i].label_,
                'entidade':str(doc.ents[i])
            }
            recognition_dict['entidades'].append(entity_obj)
        
        doc = nlp_inequalities(fragment)
        for i in range(len(doc.ents)):
            entity_obj = {
                'tipo_entidade':doc.ents[i].label_,
                'entidade':str(doc.ents[i])
            }
            recognition_dict['entidades'].append(entity_obj)
        
        resultjson = json.dumps(recognition_dict)
        return flask.Response(response=resultjson, status=200, mimetype='application/json')
    
    # response for get
    elif get:
        if get == 'get_model_info':
            preset_entity_list = ['LOC', 'MISC', 'ORG', 'PER']
            known_labels = nlp_predefined.get_pipe('ner').labels + nlp_inequalities.get_pipe('ner').labels
            known_labels = [i for i in known_labels if i not in preset_entity_list]
            return flask.Response(
                response=json.dumps(known_labels), status=200, mimetype='application/json')
        else:
            return flask.Response(
                response=json.dumps('protocolo desconhecido'), status=207, mimetype='application/json')
    
    # response or anything else
    else:
        
        return flask.Response(
                response=json.dumps('protocolo desconhecido'), status=207, mimetype='application/json')