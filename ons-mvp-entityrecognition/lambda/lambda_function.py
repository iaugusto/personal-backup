import os
import sys
import json
import boto3


def lambda_handler(event, context):
    
    runtime_sm_client = boto3.client(service_name='sagemaker-runtime')
    content_type = "application/json"
    
    #Serialize data for endpoint
    print('event: \n', event)
    
    if event['auth'] !='987cc44828f553dd1c37910843ab2eb7':
        print('access denied')
        return None
    else:
        print('access granted')
    
    payload = json.dumps(event)
    
    #Endpoint invocation
    response = runtime_sm_client.invoke_endpoint(
        EndpointName='mpo-ner-prod-endpoint',
        ContentType=content_type,
        Body=payload)
    
    #Parse results
    result = json.loads(response['Body'].read().decode())['entidades']
    
    return result