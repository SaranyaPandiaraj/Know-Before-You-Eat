import os
import json
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from flask_pymongo import PyMongo



# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = os.path.join("models","keras_models", "model-mobilenet-RMSprop0.0002-001-0.930507-0.647776.h5")

# Load your trained model
model = load_model(MODEL_PATH)
print("Model loaded. Check http://127.0.0.1:5000/")

with open(os.path.join("static","food_list", "food_list.json"), "r", encoding="utf8") as f:
    food_labels = json.load(f)
class_names = sorted(food_labels.keys())
label_dict = dict(zip(range(len(class_names)), class_names))

food_calories = pd.read_csv(os.path.join("static","food_list", "Food_calories.csv"))

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img) / 255
    x = np.expand_dims(x, axis=0)
    return x

# Use PyMongo to establish Mongo connection
#mongo = PyMongo(app, uri="mongodb://localhost:27017/Food")
#mongo.db.collection.remove( { } );

@app.route("/", methods=["GET"])
def index():
    # Main page
    #Food = mongo.db.collection.find_one()
    return render_template('Know_Before_You_Eat.html')

@app.route("/predict", methods=["GET", "POST"])
def upload():
    data = {}
    if request.method == "POST":
        # Get the file from post request
        f = request.files["image"]

        # Save the file to ./upload_image
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "upload_image", secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        image = prepare_image(file_path)
        preds = model.predict(image)
        predictions = preds.argmax(axis=-1)[0]
        pred_label = label_dict[predictions]

        food_retrieve = food_calories[food_calories["name"]==pred_label]

        food_nutrional_min = food_retrieve["nutritional value min,kcal"]
        food_nutrional_min=np.array(food_nutrional_min)
        food_nutrional_min = str(food_nutrional_min)


        food_nutrional_max = food_retrieve["nutritional value max,kcal"]
        food_nutrional_max=np.array(food_nutrional_max)
        food_nutrional_max = str(food_nutrional_max)

        Unit = food_retrieve["unit"]
        Unit=np.array(Unit)
        Unit = str(Unit)

        Calories = food_retrieve["average cal"]
        Calories=np.array(Calories)
        Calories = str(Calories)

        #Food_Data = {"food_nutrional_min":food_nutrional_min}

        #mongo.db.collection.update({}, Food_Data, upsert=True)
        
        return pred_label+" => "+"Probability :"+str(preds.max(axis=-1)[0]) + '\n' + \
        "                                       <br> Nutrional Value - Min (kcal) :" + food_nutrional_min + '\n' + \
        "                                       Nutrional Value - Max (kcal):" + food_nutrional_max + '\n' + \
        "                                       Unit" + Unit + '\n' + \
        "                                       Average Calories" + Calories
        

    return None


if __name__ == "__main__":
    # Serve the app with gevent
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()