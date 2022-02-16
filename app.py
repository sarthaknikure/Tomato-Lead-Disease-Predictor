from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# keras
from tensorflow.keras.applications. resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


from flask.templating import render_template_string
# Define a flask app
app = Flask(__name__)

# MODEL saved with Keras model.save()
MODEL_PATH = "model_resnet50.h5"

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path,model):
  print(img_path)
  img = image.load_img(img_path, target_size=(224,224))

  # Preprocessing the image
  x = image.img_to_array(img)
  x = x/255
  # Expand the shape of array
  x = np.expand_dims(x, axis=0)

  # Be careful how your trained model deals with the input
  # otherwise, it won't make correct prediction

  preds = model.predict(x)
  preds=np.argmax(preds, axis=1)
  if preds==0:
    preds="The Disease is Pepper Bell Bacterial Spot"
  elif preds==1:
    preds="The Disease is Papper Bell Healthy"
  elif preds==2:
    preds="The Disease is Potato Early Blight"
  elif preds==3:
    preds="The Disease is Potato Healthy"
  elif preds==4:
    preds="The Disease is Potato Late Blight"
  elif preds==5:
    preds="The Disease is Tomato Mosaic Virus"
  elif preds==6:
    preds="The Disease is Tomato Yellow Leaf Curl Virus"
  elif preds==7:
    preds="The Disease is Tomato Bacterial Spot"
  elif preds==8:
    preds="The Disease is Tomato Early Blight"
  elif preds==9:
    preds="The Disease is Papper Bell Bacterial Spot"
  elif preds==10:
    preds="The Disease is Papper Bell Bacterial Spot"
  elif preds==11:
    preds="The Disease is Papper Bell Bacterial Spot"
  elif preds==12:
    preds="The Disease is Papper Bell Bacterial Spot"
  elif preds==9:
    preds="The Disease is Papper Bell Bacterial Spot"
  
  return preds


@app.route('/', methods=['GET'])
def index():
  #Main page
  return render_template_('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    
    # MAke prediction
    preds = model_predict(file_path, model)
    result=preds
    return result
  return None


if __name__ == '__main__':
  app.run(debug=True)
  
