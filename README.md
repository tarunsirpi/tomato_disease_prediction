# Tomato Disease Classification

## Setup for training the model:

Run the following to setup python environment for training the model,
```
conda create -p env_name python=3.9 -y
```
```
conda activate env_name
```
``` pip install -r requirements.txt ```

(Note: Here the model is trained using GPU. So, some of the packages used are dependent of the hardware.)

## setup git repo

```
git init
git add .
git commit -m "commit messsage"
git branch -M main
git remote add origin <repo url>
git push -u origin main
```

## Model training:
Download the data from kaggle (Data is downloaded from the following link: https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
Only keep folders related to Tomato Disease Classification.
Run Jupyter Notebook.