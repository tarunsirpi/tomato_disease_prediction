from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_version = "test2-2-epochs"
loaded_model_2 = tf.keras.models.load_model(os.path.join(os.getcwd(), 'saved_models',f"{model_version}.keras"))


def read_file_as_image(data) -> np.ndarray:
  data = np.array(Image.open(BytesIO(data)))
  return data

@app.get("/", response_class= HTMLResponse)
async def home_page(request: Request):
   return templates.TemplateResponse("index.html", {'request':request})


@app.get("/predict", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
  bytes = await file.read()
  image = read_file_as_image(bytes)
  image_batch = np.expand_dims(image, axis=0)

  labels = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

  prediction = loaded_model_2.predict(image_batch)

  prediction_class = np.argmax(prediction[0])
  prediction_label = labels[prediction_class]
  pred_shape = prediction.shape
  pred_shape2 = prediction[0].shape
  confidence = "{:.2f}".format(max(prediction[0])*100)

  return templates.TemplateResponse("results.html", {"request": request, "prediction_label": prediction_label, 'confidence':str(confidence)})


if __name__ == '__main__':
  uvicorn.run(app, host='localhost', port=8010 )
