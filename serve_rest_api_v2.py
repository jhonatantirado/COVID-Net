import os
import urllib.request
import cloudinary.uploader
from flask import Flask, jsonify
from keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np

app = Flask(__name__)
model = None

inception_v3_model_name = 'output_graph.pb'

bruise_width = 224
bruise_height = 224
bruise_channels = 3

cloudinary.config(
    cloud_name="digrubrgw",
    api_key="953684376456813",
    api_secret="ch1haynm_MVry9wbQrK84UgIdr0"
)

def load_model_for_prediction(model_path):
    global model
    model = load_model(model_path)
    model._make_predict_function()
    print (model)

with open('labels') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]

def get_preprocess_fn(type = 0):
    if type == 0:
        from keras.applications.inception_v3 import preprocess_input
        func = preprocess_input

    elif type == 1:
        from keras.applications.resnet50 import preprocess_input
        func = preprocess_input

    elif type == 2:
        from keras.applications.mobilenet import preprocess_input
        func = preprocess_input

    return func

def load_image_for_prediction(image_path, preprocess_input):
    url = cloudinary.utils.cloudinary_url(image_path)
    urllib.request.urlretrieve(url[0], image_path)

    img = image.load_img(image_path, target_size=(bruise_width, bruise_height, bruise_channels))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img

def predict(model_index, model_name, image_path):
    print ('Prediction with ' + model_name)
    preprocess_function = get_preprocess_fn(model_index)
    img = load_image_for_prediction(image_path, preprocess_function)
    classes = model.predict(img)
    return classes

@app.route('/todo/api/v1.0/prediagnosis/<string:image_url>', methods=['GET'])
def get_prediagnosis(image_url):
    result = predict(0, inception_v3_model_name, image_url)
    diagnosis = labels[np.argmax(result)]
    os.remove(image_url)
    return jsonify({'probability_distribution': result.tolist()}, {'diagnosis': diagnosis})

if __name__ == '__main__':
    load_model_for_prediction(inception_v3_model_name)
    app.run(host="0.0.0.0", debug=True)
