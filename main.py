from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

import os

app = FastAPI()

#print(os.path.join(os.getcwd(), 'saved_models', 'testing.keras'))


model_version = "test2-2-epochs"
loaded_model_2 = tf.keras.models.load_model(os.path.join(os.getcwd(), 'saved_models',f"{model_version}.keras"))

# loaded_model_2 = tf.keras.models.load_model(os.path.join(os.getcwd(), 'saved_models',"test_save.h5"))
print(os.path.join(os.getcwd(), 'saved_models',f"{model_version}.keras"))
print(loaded_model_2)
# "D:\data_science\ml_projects\tomato_disease_prediction\saved_models\testing.keras"
print(os.path.join(os.getcwd()))

@app.get("/")
async def ping():
  return "Pinging successfully!"


def read_file_as_image(data) -> np.ndarray:
  data = np.array(Image.open(BytesIO(data)))
  return data


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
  confidence = max(prediction[0])
  return {'prediction_label':prediction_label, 'confidence':str(confidence)} 

  # return templates.TemplateResponse("form.html", {"request": prediction_label, "response_message": str(confidence)})




if __name__ == '__main__':
  uvicorn.run(app, host='localhost', port=8010 )
  