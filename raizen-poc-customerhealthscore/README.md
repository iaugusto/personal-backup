# TENBU DATA SCIENCE PROJECT GUIDELINES :microscope:

### TODOs 

- [X] atualizar README.md
- [X] colocar datasets para exemplos dentro do s3 para ser baixado para exemplos
- [X] transferir códigos de exemplo para examples/
- [X] adaptar examples - **Feature Selection quebrado**
- [X] adaptar codigos de exemplo
- [ ] criar um git submodule para guara
- [ ] Fazer plotly funcionar no jupyter lab
- [ ] complemetar tests/
- [ ] criar notebook regressao completo e classificacao completo
- [ ] docker-compose para postgres - salvar resultado modelo e monitoramento (?)
- [ ] implementar MLFlow
- [ ] fazer função para mandar direto para reports os graficos do modelo em produção
- Feature Selection
    - [ ] dividir o unico arquivo .py nos modelos (Cat-Cat, Cat-num, etc) e criar novo .py para busca geral
    - [ ] Notebook de exemplo quebrado
    - [ ] quando ele deleta uma feature, acusa no proximo selector que nao existe (não colocar quais sao categoricas como argumento, mas reconhecer por dtype)
    - [ ] ter lista de procedimentos de FS
    - [ ] select_best_features: consertar remove_colinearity
    - [ ] feature_permutation: atualizar (colocar guara/optuna)
    - [ ] feature permutation: variaveis aleatorias
    - [ ] codigo: pensar em melhorar velocidade/otimizar
    - [ ] logs: melhorar
    - [ ] tentar substituir minepy
- Model monitoring:
    - [ ] modelo discriminante dados treino vs produção
    - [ ] variables shift 
    - [ ] new categories
    - [ ] NaNs
    - [ ] time monitoring?
    - [ ] prediction monitoring
    - [ ] comparisson predicted x real
- Feature Engineering:
    - [ ] basear novas FE em [feature engine](https://github.com/solegalli/feature_engine)


---------------------


## Project Organization :pencil2:

      ├── artifacts                <- Trained and serialized models, pipelines, etc. 
      │   ├── experiments          <- training experiments
      │   │   └── 20210329-161340  <- directory of experiment id
      │   │       ├── model_20210329-161340.joblib
      │   │       └── pipeline_20210329-161340.joblib
      │   │
      │   ├── production           <- experiment active in production
      │   │   └── 20210329-161340  <- directory of experiment id
      │   │       ├── model_20210329-161340.joblib
      │   │       └── pipeline_20210329-161340.joblib
      │   │
      │   └── variant          <- offlne model in production
      │ 
      │
      ├── container-aws        <- structure of aws container
      │   ├── ml
      │   └── service
      │       ├── nginx.conf
      │       ├── predictor.py <- script where model and pipeline are used
      │       ├── serve
      │       ├── train
      │       └── wsgi.py
      │
      ├── dashboard      <- dashboard scripts
      │
      ├── data
      │   ├── external   <- Data from third party sources.
      │   ├── processed  <- The final, canonical data sets for modeling.
      │   └── raw        <- The original, immutable data dump.
      │
      ├── notebooks      <- Jupyter notebooks. Naming convention is:
      |                     a number (for ordering), the creator's initials, 
      │                     and a short `-` delimited description.
      │                     e.g.
      │                     <sequence_num>-<version>-<name_inicials>-<description>
      │                     1-1-jqp-initial-data-exploration.
      │
      ├── src
      │  ├── guara_feature_selection.py   <- Guara's feature selection class
      │  ├── guara_pipeline_modules.py    <- Guara's pipeline classes for generic features.
      │  ├── guara_supervised_modelz.py   <- Guara's supervised modeling class.
      │  ├── guara_unsupervised_modelz.py <- Guara's unsupervised modeling class.
      │  ├── guara_window_features.py     <- Guara's pipeline class for window features.
      │  ├── __init__.py                  <- just for test pourposes
      │  ├── get_data.py                  <- Script to download or generate data
      │  ├── predict.py                   <- Script to use trained models to make predictions
      │  ├── train.py                     <- Script to train models and pipelines
      │  └── utils.py                     <- Script for helper functions.
      │
      ├── references  <- Data dictionaries, manuals, and all other explanatory materials.
      │
      ├── reports     <- Generated analysis as HTML, PDF, LaTeX, etc.
      │   └── figures <- Generated graphics and figures to be used in reporting.
      │
      ├── examples    <- notebooks for Guara examples.
      |
      ├── tests       <- .py files for testing code
      |
      ├── build_and_push_aws.sh <- script to push container to AWS ECR.
      ├── docker-compose.yml    <- instructions to build and run containers/services.
      ├── Dockerfile            <- Dockerfile for base-image.
      ├── Dockerfile_aws        <- Dockerfile for aws-image.
      ├── Makefile              <- Wrapper for the project cli instructions.
      ├── project_info.yml      <- infos about the project, such as creator, name, date, etc.
      ├── requirements.txt      <- The requirements file with installed libraries.
      └── README.md             <- The top-level README for developers using this project. 
      

---------------------

## Quick Start :clock10:

This is a template repository. [Follow this link for instructions to create a repository from a template](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template#creating-a-repository-from-a-template).

First, make sure `make`, `git`, `docker` and `docker-compose` are installed in your system.

These are requirements for your local machine, ideally a Debian Linux OS (the following can be achieved in Widowns by downloading WSL, that starts a Linux VM inside Windows :confused:):

#### [docker](https://docs.docker.com/engine/install/)

Follow the [instructions in the docker docs](https://docs.docker.com/engine/install/linux-postinstall/) to ensure that $USER has root access to docker.

#### [docker-compose](https://docs.docker.com/compose/install/)

Install docker compose to declarativelly manage creation of docker images and containers. The docker compose statements are wrapped in `Makefile`


#### VS Code

In your local machine:

1. [install VS Code](https://code.visualstudio.com/docs/setup/linux),

2. use Crtl+Shift+P to access commands

3. In commands, search `Extensions: install extensions`and install:

   - Docker:`ms-azuretools.vscode-docker`

   - Remote SSH - to connect via SSH to Tenbu's Data Science Server

   - Python

   - [`ms-vscode-remote.remote-containers`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

   - [extra] Dracula is a good theme extension :wink:

   - [extra] Edit csv (good for handling .csv files)



#### [git](https://git-scm.com/download/linux)

```
sudo apt-get git
```

#### make

```
sudo apt-get update
sudo apt-get install build-essential
```

The development work is performed via `make` commands.

To see the most up to date list of available commands run

```bash
$ make help
```


---------------

## Detailed information :blue_book:

------------------

### Data

1. Don't ever edit your raw data, especially not manually, and especially not in Excel. 
2. Don't overwrite your raw data. 
3. Don't save multiple versions of the raw data. 
4. Treat the data (and its format) as immutable. The code you write should move the raw data through a pipeline to your final analysis. 
5. You shouldn't have to run all of the steps every time you want to make a new figure (see Analysis is a DAG), but anyone should be able to reproduce the final products with only the code in src and the data in data/raw.

Also, if data is immutable, it doesn't need source control in the same way that code does. Therefore, by default, the data folder is included in the .gitignore file. If you have a small amount of data that rarely changes, you may want to include the data in the repository. Github currently warns if files are over 50MB and rejects files over 100MB. Some other options for storing/syncing large data include AWS S3 with a syncing tool (e.g., s3cmd), Git Large File Storage, Git Annex, and dat.





------------------

### Artifacts

This directory is the place where serialized models go to. files should be in the formats ```.joblib```  for now.

Each experiment will have ```model_<id>.joblib``` and ```pipeline_<id>.joblib``` files inside a directory named with the same ```id```

In ```/artifacts/experiments```, MLFlow experiments can be saved too.


------------------

### Notebooks

Since notebooks are challenging objects for source control (e.g., diffs of the json are often not human-readable and merging is near impossible), we recommended not collaborating directly with others on Jupyter notebooks, they must be used only in development phase, not in production. The code developed in the notebooks that will run in production must be transferred to ```.py``` files.

**Naming convention:** We use the format ```<step>-<version>-<ghuser>-<description>.ipynb``` (e.g., 1-1-ggb-visualize-distributions.ipynb).


Refactor the good parts. Don't write code to do the same task in multiple notebooks. If it's a data preprocessing task, put it in the pipeline at src/data/make_dataset.py and load data from data/interim. If it's useful utility code, refactor it to src.

To load functions from the src files in the notebook:

```python
# Load the "autoreload" extension so that code can change
%load_ext autoreload
%autoreload 2

# insert path of src files
sys.path.insert(1, '../src/')

from supervised_modelz import *
import get_data
```

------------------


### Src

It is important to use as much logs as possible here, in order to have visibility of the process.

1. ```get_data.py``` is the script with class to get data, usually with a connection to a database, a method to get feature data and another to get target data, for supervised problems.

2. ```train.py``` is the script to  train models and pipelines. This must contain the training of the best model and its hyperparameters, obtained in the modeling optimizaiton notebook. 

3. ```predict.py``` is the script to use trained models/pipelines and make predictions

4. ```__init__.py``` pytest require this file to teste functions. Empty.

5. ```utils.py``` collection of small Python functions and classes which make common patterns shorter and easier.


### Guara
TODO: put image

**Guara** is Tenbu's Machine Learning library, developed by the Data Science Team and consists of:

- **Guara Feature Engineering**:
   - ```guara_pipeline_modules.py```: classes for robust feature engineering to use in ```sklearn.Pipeline``` pipelines.
   - ```guara_window_features.py```: classes for robust statistics on time windows to use in ```sklearn.Pipeline``` pipelines.

- **Guara Modeling**:
   - ```guara_supervised_models.py```: wrapper for regression and classification modeling. It includes supervised models as RandomForest, LightGBM, linear models, features as plots of training rouns, metrics, explainability (SHAP values), Bayesian optimization.

   - ```guara_unsupervised_models.py```: wrapper for unsupervised modeling. Includes features such as Gaussian Mixture models, Kprototypes (numerical and categorical clustering), Isolation Forest. Dimensionality reduction can be performed with TSNe algorithm. Includes plots to visualize training rounds and metrics.

- **Guara Feature Selection**:
   - ```guara_feature_selection.py```: class to perform feature selection in a dataset. Includes methods and plots that help to select the best features for modeling.

------------------

### Tests

```tests/``` directory must include python files to test source code using pytest library.

Tests can be performed using the command:
```bash 
$ make test
``` 

------------------

### Requirements

The first step in reproducing an analysis is always reproducing the computational environment it was run in. You need the same tools, the same libraries, and the same versions to make everything play nicely together.

One effective approach to this is use ilsolated containers, such as Docker. By listing all of your requirements in the repository (we include a ```requirements.txt``` file) you can easily track the packages needed to recreate the analysis inside the container. 


--------------
### Docker

The project is made to run on isolated environments and Docker comes in hand. You can search on the internet for Docker courses or texts to enhance your knowledge on this subject. In a nutshell, there are 2 files that are related to handling with Docker:

- `Dockerfile and Dockerfile_aws`: these files are instructions to build Docker images for the general project and for the AWS Sagemaker, respectively.  They get a previously constructed image from the Docker cloud and perform Linux scripts on them.

- `docker-compose`: An declarative way to configure docker services, with specific instructions on what network the containers are going to run in, ports that will be opened, instructions to run when up, etc. 
Makefile wraps the docker-compose CLI to make commands more standardized.

--------------
### Makefile

```bash
USAGE

    make <command>
    Include 'sudo' when necessary.
    To avoid using sudo, follow the steps in
    https://docs.docker.com/engine/install/linux-postinstall/


COMMANDS

    base-image           build image using cache
    base-image-no-cache  build image from scratch, and not from cache
    bash                 container bash, suitable for debugging
    python3              access Python inside container
    train                runs src/train.py file
    predict              runs src/predict.py file
    jupyter              access Python through the Jupyter Notebook
    test                 run all tests using pytest
    aws-image            build an AWS ready to deploy container
    aws-jupyter          run AWS image and starts Jupyterlab

```


To build the base image (as it is), you first need to run:

```bash
$ make base-image
```


To access Jupyter in your local browser:

```bash
$ make jupyter

[ServerApp] Jupyter Server 1.5.1 is running at:
[ServerApp] http://tbudatascience:8889/lab?token=ed1...
[ServerApp]  or http://127.0.0.1:8889/lab?token=ed1...
[ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

```


Next, you simply need to follow the instructions printed out on your own terminal.


In the generic example above, I would paste the following on my browser:

```bash
http://...:8888/lab?token=...
```


Any changes made in the files within the Jupyter interface, for example saved changes in `.rs`, `.ipynb`, and `.py` files, will be reflected in the original files you store locally, and vice-versa. This is ensured by the fact that the whole greenhouse directory is set as a `volume` in the `docker-compose.yml` configuration file.



--------------

### Env

You really don't want to leak your AWS secret key or Postgres username and password on Github. Enough said — see the Twelve [Factor App principles](https://12factor.net/config) on this point. Here's one way to do this:

Store your secrets and config variables in the special file ```.env``` file in the project root folder. Thanks to the .gitignore, this file should never get committed into the version control repository. Here's an example:

```python
# example .env file
DATABASE_URL=postgres://username:password@localhost:5432/dbname
AWS_ACCESS_KEY=myaccesskey
AWS_SECRET_ACCESS_KEY=mysecretkey
OTHER_VARIABLE=something
```


***Use a package to load these variables automatically.***

You should use a package called ```python-dotenv``` to load up all the entries in this file as environment variables so they are accessible with os.environ.get. Here's an example snippet adapted from the python-dotenv documentation:

```python
import os
from dotenv import load_dotenv, find_dotenv

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

database_url = os.environ.get("DATABASE_URL")
other_variable = os.environ.get("OTHER_VARIABLE")
```

------------------
## Others



## Github conection and code versioning

### Github Synchronization 

After repository creation, presented in the Quickstart session, it is necessary to clone the repository on your local machine (or remote machine)

```bash
$ git clone https://github.com/tenbu-proj/<git-project-name>.git
```
### Sending code improvements to Github

Either because you have finished improving the code or want to checkpoint your progress, you will have to commit these improvements on your git repository. To accomplish this, you will first need to **add the changes** to git:

```bash
$ git add .      #add everything not ignored by .gitgnore
$ git add <file> #add specific file
```
After that, you will need to **commit those changes** on the local git repository

```bash
$ git commit -m "message" 
```
Here, it is important to standardize these commit messages. The messages should be structured as follows (more on [convention commits](https://www.conventionalcommits.org/en/v1.0.0/)):

- `fix:` patches a bug in your codebase
- `feat:` new feature to the codebase
- `docs:` correct documentation

a commit example could be

```bash
$ git commit -m "feat: model X added, fix: predictions in production"
```

In the final step, you should **send this commit** to Github

```bash
$ git push origin <branch>
```

If you are working on the main branch, you fill \<branch> with main

### Pull code from Github
If you are working on a team of people, it is expected that you will not always have the newest code on your machine.To import code from Github to your machine:
```bash
$ git pull origin <branch>
```

----------------
## POSTGRES
usar pgadmin para ver o banco inteiro

- ```\dt``` -  list tables
- ```\du``` -  list users
- ```\l```  -  list databases
- ```\c <banco_de_dados>``` -  connect on database

------------
## AWS



### Lambda Function
This is an example of a lambda function that invokes the endpoint created with the containerized model.


```python
from __future__ import print_function

import base64
import json
import csv, sys,os,time
import boto3

# nome do bucket aonde estão os dados
bucket = 'exdeploy-bucket-guilherme'

# pasta onde estão os dados
key = 'iris/payload.csv'

# nome do ednpoint criado
endpointName = 'iris2'

print('conectando s3...')
s3 = boto3.client('s3')
# Talk to SageMaker

print('clint sagemaker...')
client = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    payload = s3.get_object(Bucket=bucket,Key=key)['Body'].read()
    print('PAYLOAD:\n',str(payload))

    
    response = client.invoke_endpoint(
            EndpointName=endpointName,
            Body=payload,
            ContentType='text/csv'
    
        )
    
    result = response['Body'].read().decode('ascii')
    
    print(str(result))
```

#### [Deprecated] Register in aws (cli)

```python
aws ecr get-login-password --region us-east-1| docker login --username AWS --password-stdin 191448642801.dkr.ecr.us-east-1.amazonaws.com
```

TAG the Docker container (necessary when pushing to AWS)
```python
docker tag aws-teste:latest 633647267168.dkr.ecr.us-east-1.amazonaws.com/datascience:aws-teste
```

Pushing the container
```python
docker push 633647267168.dkr.ecr.us-east-1.amazonaws.com/datascience:aws-teste
```



