
BUILD = docker-compose build
RUN = docker-compose run
JUPYTER_PORT = 8889

help:
	@echo ""
	@echo "USAGE"
	@echo
	@echo "    make <command>"
	@echo "    Include 'sudo' when necessary."
	@echo "    To avoid using sudo, follow the steps in"
	@echo "    https://docs.docker.com/engine/install/linux-postinstall/"
	@echo
	@echo
	@echo "COMMANDS"
	@echo
	@echo "    base-image           build image using cache"
	@echo "    base-image-no-cache  build image from scratch"
	@echo "    bash                 container bash, suitable for debugging"
	@echo "    python3              access Python inside container"
	@echo "    train                runs src/train.py file"
	@echo "    predict              runs src/predict.py file"
	@echo "    jupyter              starts Jupyterlab inside container"
	@echo "    test                 runs all tests using pytest"
	@echo "    aws-image            build an AWS ready-to-deploy container"
	@echo "    aws-jupyter          runs AWS image and starts Jupyterlab"
	@echo ""


#################
# User Commands #
#################

base-image:
	$(BUILD) base

base-image-no-cache:
	$(BUILD) --no-cache base 

bash:
	$(RUN) bash

python3:
	$(RUN) python3

train:
	$(RUN) train

predict:
	$(RUN) predict

jupyter:
	$(RUN) --service-ports jupyter
	
test:
	$(RUN) test


aws-image:
	$(BUILD) --no-cache aws 


aws-jupyter:
	$(RUN) --service-ports aws-jupyter

dash:
	$(RUN) --service-ports dash