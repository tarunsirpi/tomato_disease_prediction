# Tomato Disease Classification

  Dataset contains images of Tomato leaves labelled in 10 categories. A tensorflow CNN model is trained to classify the images uploaded as one of the categories. Finally, a FastAPI application uses the saved model to make predictions.

  * ```training``` - contains files required for training the model (here data directory is hidden).
  * ```templates``` - contains frontend template.
  * ```saved_models``` - contains models saved in  *.keras* format.

## Setup for training the model:

Run the following to setup python environment for training the model,
```
conda create -p env_name python=3.9 -y
```
```
conda activate env_name
```
``` 
pip install -r training-requirements.txt
```

(Note: 
1. Here the model is trained using GPU. So, some of the packages used are dependent of the hardware.
2. Use ```requirements-common.txt``` to use packages without hardware dependency.)

## Setup git repo:

```
git init
git add .
git commit -m "commit messsage"
git branch -M main
git remote add origin <repo url>
git push -u origin main
```

## Model training:
1. Download the data from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
2. Only keep folders related to Tomato Disease Classification.
3. Run Jupyter Notebook (/training/tomato_training.ipynb).

## Setup environment for FastAPI application:
```
conda create -p env_name python=3.9 -y
```
```
conda activate env_name
```
``` 
pip install -r requirements.txt
```

## Docker setup:

Run the following commands to create a docker image and run the FastAPI application on a container,

```
docker build -t image-name .
```
```
docker run -p 8010:8010 --name container-name image-name
```

check the link : http://localhost:8010/


For running the application without container, run the ```main.py``` file to start the FastAPI application.